#!/usr/bin/env python3
"""
run_gpt2_ioi.py

CLI entrypoint for OASR-style joint multi-particle circuit discovery --
edge repulsion AND node repulsion applied simultaneously to N particles
trained together in one loop, not the sequential find-a-circuit-then-repel-
from-it pattern of pilot1.py/pilot2.py -- against a CircuitGPT (gpt2-small /
gpt2-medium) backbone on the IOI task.

This is the direct GPT2/IOI analogue of run_hubert.py. It imports its
particle/repulsion logic from circuit_discovery/ioi_discovery_setup.py
exactly the way run_hubert.py imports from
circuit_discovery/tasks/discovery_setup.py -- both delegate the
architecture-agnostic repulsion math and multi-GPU dispatch to
circuit_discovery/multi_particle.py.

Run from the OASR repo root. Unlike HuBERT's Articulatory Index tasks, IOI
needs no separate dataset-prep or head-training step first -- utils.py's
load_task_dataset loads the existing saved `ioi_dataset` directly.

Examples:
    python run_gpt2_ioi.py
    python run_gpt2_ioi.py --n_particles 3 --lambda_edge_max 0.66 --lambda_node_max 10.0

    # spread 6 particles round-robin over two GPUs, combine repulsion on cuda:0
    python run_gpt2_ioi.py \\
        --n_particles 6 --devices cuda:0 cuda:1 --repulsion_device cuda:0

Turning a repulsion term off: pass --lambda_edge_max 0 or --lambda_node_max 0.
The ramp schedule returns 0 for the whole run in that case -- there's no
separate --disable_* flag, one fewer thing to keep in sync.

Parallelism: IMPLEMENTED (single-process multi-GPU), identical design to
run_hubert.py. --devices takes one or more device strings; particles are
assigned round-robin across them, each unique device gets its own frozen
CircuitGPT replica, and --repulsion_device picks where the cross-particle
edge/node repulsion terms get combined each step (defaults to devices[0]).
See circuit_discovery/multi_particle.py's module docstring for the design
and circuit_discovery/ioi_discovery_setup.py's Particle /
per_particle_forward / combine_repulsion / per_particle_backward_step split
for exactly how each phase runs. With a single device (the default) this is
behaviorally identical to the old single-GPU pilot2b.py-style code path.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from circuit_discovery.ioi_discovery_setup import (
    build_node_incidence_for_devices,
    build_particles,
    combine_repulsion,
    finalize_and_report,
    load_gpt2_for_devices,
    node_probs_from_edge_probs,
    per_particle_backward_step,
    per_particle_forward,
    ramp_schedule,
    run_completeness_step,
)
from circuit_discovery.run import get_compute_device
from circuit_discovery.utils import fixed_order_dataloader, load_task_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--model_name", default="gpt2-small", choices=["gpt2-small", "gpt2-medium"])
    parser.add_argument("--task", default="ioi", help="dataset name under DATASET_FOLDER_PATH, e.g. 'ioi'.")
    parser.add_argument("--train_size", type=int, default=5000)
    parser.add_argument("--test_size", type=int, default=1000)
    parser.add_argument("--data_seed", type=int, default=42)

    parser.add_argument("--n_particles", type=int, default=3,
                         help="how many mutually-repelling circuits to discover simultaneously.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                         help="one seed per particle; defaults to [42, 43, 44, ...] up to n_particles.")

    parser.add_argument("--lambda_edge_max", type=float, default=0.66,
                         help="max strength of pairwise edge-probability repulsion. 0 disables it.")
    parser.add_argument("--edge_repulsion_warmup_frac", type=float, default=0.8)
    parser.add_argument("--lambda_node_max", type=float, default=10.0,
                         help="max strength of pairwise node-probability (noisy-OR) repulsion. 0 disables it.")
    parser.add_argument("--node_repulsion_warmup_frac", type=float, default=0.8)

    parser.add_argument("--n_epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_e", type=float, default=0.07)

    parser.add_argument("--edge_logit_init_mean", type=float, default=0.1)
    parser.add_argument("--edge_logit_init_std", type=float, default=0.01)
    parser.add_argument("--random_mode", default="gumbel_sigmoid", choices=["gumbel_sigmoid", "none"])
    parser.add_argument("--gs_temp_edge", type=float, default=1.0)

    parser.add_argument("--lambda_sparse_e", type=float, default=1.0)
    parser.add_argument("--min_times_lambda_sparse_e", type=float, default=0.01)
    parser.add_argument("--max_times_lambda_sparse_e", type=float, default=20.0)

    parser.add_argument("--lambda_complete_e", type=float, default=0.01)
    parser.add_argument("--completeness_start_frac", type=float, default=0.8)

    parser.add_argument("--snapshot_every", type=int, default=None,
                         help="also snapshot circuits every N epochs, not just at the end (useful for long runs).")
    parser.add_argument("--save_dir", default=None,
                         help="defaults to circuits_discovered/ioi_circuits/{model_name}/")

    parser.add_argument("--devices", nargs="+", default=None,
                         help="one or more devices (e.g. --devices cuda:0 cuda:1). Particles are "
                              "assigned round-robin across them. Defaults to a single auto-picked device.")
    parser.add_argument("--repulsion_device", default=None,
                         help="device where cross-particle edge/node repulsion is combined each step "
                              "(the multi-GPU synchronization point). Defaults to devices[0].")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    default_device = get_compute_device()
    devices = args.devices if args.devices is not None else [default_device]
    hub_device = args.repulsion_device or devices[0]

    seeds = args.seeds if args.seeds is not None else [42 + i for i in range(args.n_particles)]
    if len(seeds) != args.n_particles:
        raise ValueError(f"--seeds has {len(seeds)} entries but --n_particles={args.n_particles}.")

    save_dir = (
        Path(args.save_dir) if args.save_dir is not None
        else Path("circuits_discovered") / "ioi_circuits" / args.model_name
    )

    print(f"Loading {args.model_name}, one replica per device in {sorted(set(devices))}...")
    models_by_device = load_gpt2_for_devices(args.model_name, devices)
    data = load_task_dataset(
        args.task, batch_size=args.batch_size, train_size=args.train_size,
        test_size=args.test_size, random_seed=args.data_seed,
    )

    warmup = int(0.8 * args.n_epochs)
    discogp_config_kwargs = dict(
        model_name=args.model_name,
        prune_edges=True,
        prune_weights=False,
        n_epochs_e=args.n_epochs,
        batch_size=args.batch_size,
        lr_e=args.lr_e,
        edge_logit_init_mean=args.edge_logit_init_mean,
        edge_logit_init_std=args.edge_logit_init_std,
        random_mode=None if args.random_mode == "none" else args.random_mode,
        gs_temp_edge=args.gs_temp_edge,
        lambda_sparse_e=args.lambda_sparse_e,
        min_times_lambda_sparse_e=args.min_times_lambda_sparse_e,
        max_times_lambda_sparse_e=args.max_times_lambda_sparse_e,
        n_epoch_warmup_lambda_sparse_e=warmup,
        n_epoch_cooldown_lambda_sparse_e=args.n_epochs - warmup,
        lambda_complete_e=args.lambda_complete_e,
        completeness_start_frac=args.completeness_start_frac,
        overlap_penalty=False,
    )

    particles = build_particles(
        seeds=seeds, devices=devices, models_by_device=models_by_device,
        discogp_config_kwargs=discogp_config_kwargs,
    )
    print("Particle -> device assignment: " + ", ".join(f"seed{p.seed}->{p.device}" for p in particles))

    train_loader = fixed_order_dataloader(data.train.dataset, batch_size=args.batch_size, seed=seeds[0])
    n_epochs = args.n_epochs
    complete_start = int(args.completeness_start_frac * n_epochs)

    # One incidence matrix per unique device particles actually run on --
    # node_probs_from_edge_probs needs incidence on the SAME device as the
    # edge_probs it's multiplying against, which varies per particle now.
    incidence_by_device, node_keys = build_node_incidence_for_devices(
        particles[0].discogp.masks, devices,
    )

    snapshot_epochs = {n_epochs - 1}
    if args.snapshot_every:
        snapshot_epochs |= set(range(args.snapshot_every - 1, n_epochs, args.snapshot_every))

    pairs_n = max(1, args.n_particles * (args.n_particles - 1) // 2)

    print(
        f"Training {args.n_particles} particles jointly for {n_epochs} epochs "
        f"({len(train_loader)} steps/epoch, {len(node_keys)} nodes / {incidence_by_device[devices[0]].shape[1]} "
        f"edges in the full graph), task={args.task} on {args.model_name}, "
        f"lambda_edge_max={args.lambda_edge_max}, lambda_node_max={args.lambda_node_max}, "
        f"devices={sorted(set(devices))}, repulsion_device={hub_device}..."
    )
    t0 = time.time()

    for epoch in range(n_epochs):
        lambda_sparse_vals = [p.discogp._scheduled_lambda_sparse(mode="edge", epoch=epoch) for p in particles]
        lambda_edge_rep = ramp_schedule(epoch, n_epochs, args.edge_repulsion_warmup_frac, args.lambda_edge_max)
        lambda_node_rep = ramp_schedule(epoch, n_epochs, args.node_repulsion_warmup_frac, args.lambda_node_max)

        epoch_task_loss = [0.0] * args.n_particles
        epoch_edge_rep = 0.0
        epoch_node_rep = 0.0
        n_batches = 0

        for batch in train_loader:
            task_losses = []
            edge_probs = []
            # Each particle's forward runs on its own device -- issuing them
            # back to back here lets their CUDA kernels queue and overlap
            # across devices (see multi_particle.py's module docstring).
            for particle, lam_sparse in zip(particles, lambda_sparse_vals):
                task_loss, probs = per_particle_forward(particle, batch, lambda_sparse=lam_sparse)
                task_losses.append(task_loss)
                edge_probs.append(probs)

            node_probs = [
                node_probs_from_edge_probs(ep, incidence_by_device[str(ep.device)])
                for ep in edge_probs
            ]
            # Synchronization point: every particle's probs get moved to
            # hub_device inside combine_repulsion before the pairwise terms.
            edge_rep, node_rep = combine_repulsion(
                edge_probs, node_probs, lambda_edge=lambda_edge_rep, lambda_node=lambda_node_rep,
                hub_device=hub_device,
            )

            # task_losses live on their own particle's device; `.to(hub_device)`
            # is differentiable, so summing here still backprops correctly to
            # each particle's own edge_logits on its own device.
            joint_loss = (
                sum(l.to(device=hub_device) for l in task_losses)
                + lambda_edge_rep * edge_rep
                + lambda_node_rep * node_rep
            )
            joint_loss.backward()
            for particle in particles:
                per_particle_backward_step(particle)

            for i, l in enumerate(task_losses):
                epoch_task_loss[i] += l.item()
            epoch_edge_rep += float(edge_rep)
            epoch_node_rep += float(node_rep)
            n_batches += 1

            if epoch >= complete_start and args.lambda_complete_e > 0.0:
                for particle in particles:
                    run_completeness_step(
                        particle, batch, lambda_complete=args.lambda_complete_e,
                    )

        print(
            f"epoch {epoch:3d}  "
            f"task_loss(avg/particle)={[round(l / n_batches, 4) for l in epoch_task_loss]}  "
            f"mean_pairwise_soft_jaccard_edge={epoch_edge_rep / n_batches / pairs_n:.4f}  "
            f"mean_pairwise_soft_jaccard_node={epoch_node_rep / n_batches / pairs_n:.4f}  "
            f"lambda_edge={lambda_edge_rep:.3f}  lambda_node={lambda_node_rep:.3f}"
        )

        if epoch in snapshot_epochs:
            finalize_and_report(
                tag=f"epoch{epoch + 1}", particles=particles, data=data, save_dir=save_dir,
            )

    elapsed = time.time() - t0
    print(
        f"\nDone: {args.n_particles} IOI circuits on {args.model_name}, {n_epochs} epochs, "
        f"{elapsed:.1f}s. Saved under {save_dir}/"
    )


if __name__ == "__main__":
    main()
