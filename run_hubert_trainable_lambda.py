#!/usr/bin/env python3
"""
run_hubert_trainable_lambda.py

Variant of run_hubert.py (repo root) that treats the Frank-family t-norm
parameter (fixed via --jaccard_lambda in run_hubert.py) as a TRAINABLE
scalar instead, shared across the whole particle cohort. New file --
run_hubert.py, circuit_discovery/multi_particle.py, and
circuit_discovery/tasks/discovery_setup.py are all imported from here
UNMODIFIED; the new numerically-stable trainable-lambda machinery lives in
circuit_discovery/frank_lambda.py (also new).

Run exactly like run_hubert.py, but replace --jaccard_lambda with
--log_lam_init (u = ln(lambda) directly, NOT lambda itself -- pass 0.0 for
the old lambda=1 default):

    python run_hubert_trainable_lambda.py --task_type vowel_classification \\
        --n_particles 5 --devices cuda:0 cuda:1 --repulsion_device cuda:1 \\
        --log_lam_init 0.0 \\
        --lr_e 0.02 --lambda_edge_max 1.0 --lambda_sparse_e 2.0 --n_epochs 10 \\
        --save_dir circuits_discovered/hubert_circuits/vowel_frank_trainlam

--------------------------------------------------------------------------
Training scheme: OUTER/ALTERNATING block-coordinate descent
--------------------------------------------------------------------------
Each epoch runs in two phases, in this order:

  1. PARTICLE PHASE (identical in structure to run_hubert.py's epoch loop):
     every particle's edge_logits take their usual batch-by-batch gradient
     steps, with lambda held FIXED at whatever value the previous epoch's
     lambda phase left it at (or --log_lam_init, for epoch 0). This reuses
     the EXISTING, UNMODIFIED multi_particle.combine_repulsion /
     frank_soft_jaccard -- safe here because lambda enters this phase only
     as a plain, non-differentiated Python float (lambda_module.lam):
     Theorem 1's instability (see project writeup) is specifically about
     differentiating the raw Frank formula's `/u` node WITH RESPECT TO u;
     gradients into p/q (i.e. into edge_logits, via sigmoid) at a FIXED u
     never hit that failure mode, so nothing about differentiating this
     phase's loss w.r.t. edge_logits needs the new stable machinery.

  2. LAMBDA PHASE (run_lambda_phase, circuit_discovery/frank_lambda.py):
     AFTER that epoch's particle updates finish, every particle's
     edge_logits are held fixed (explicit torch.no_grad() + .detach() on
     every edge_probs tensor used here -- not an assumption that the
     gradient happens to vanish on its own) while log_lam takes
     --lambda_inner_iters Adam steps, each against a FRESH batch pulled
     from the train loader (cycling back through it if
     --lambda_inner_iters exceeds one epoch's worth of batches). This is
     where the new frank_soft_jaccard_stable / Taylor-patched machinery is
     actually needed, since u=log_lam is the thing being differentiated
     here.

Both phases share ONE lambda, used for edge repulsion only. Node
repulsion (--lambda_node_max, default 0 and not currently used by this
project per the design discussion) still runs through the existing
unmodified machinery in the particle phase using that same current
lambda value -- there is no separate node-specific trainable path. Node
diagnostics are still computed and still saved into the snapshot files
(save_epoch_snapshot's schema, reused unmodified from run_hubert.py,
requires them), just not surfaced in this script's own printed lines.

Regularization: lambda's own loss each inner iteration is
  lambda_edge_rep * edge_rep(u)  +  --lambda_reg_coef * u**2
an L2 pull toward u=0 (lambda=1), per the project's hyperparameter-search
finding that the optimal lambda sits near 1.0 -- without this, joint
co-descent on the repulsion term alone has a cheap, uninteresting escape
route (the Frank family is monotonic in lambda, so minimizing the
repulsion penalty by moving lambda toward an extreme is "free" in a way
that has nothing to do with the particles actually becoming more
diverse). The coefficient is a constant throughout training (not
annealed).

Diagnostics: every epoch's lambda-phase prints and logs u, lambda, the
phase's mean loss, and how many of that phase's --lambda_inner_iters
landed inside the Taylor-patched band (in_taylor_band, frank_lambda.py) --
plus a running cumulative count across the whole run, printed at the end.
Since lambda is a single shared scalar, "inside the band" is one
yes/no per inner iteration, not per particle-pair.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from circuit_discovery.frank_lambda import (
    DEFAULT_TAYLOR_BAND,
    FrankLambdaConfig,
    FrankLambdaModule,
    run_lambda_phase,
)
from circuit_discovery.run import get_compute_device
from circuit_discovery.tasks.articulatory_index import TASK_SPECS, load_articulatory_dataset
from circuit_discovery.tasks.discovery_setup import (
    build_node_incidence_for_devices,
    build_particles,
    combine_repulsion,
    evaluate_circuit_classification,
    finalize_and_report,
    flat_probs,
    load_hubert_classifiers_for_devices,
    node_probs_from_edge_probs,
    per_particle_backward_step,
    per_particle_forward,
    ramp_schedule,
    run_completeness_step,
)
from circuit_discovery.utils import fixed_order_dataloader

# Reused, unmodified, from run_hubert.py -- see that file's docstrings for
# what each does. Importing these (rather than duplicating them) is safe:
# run_hubert.py's own training run only starts under `if __name__ ==
# "__main__"`, so importing it here pulls in only its function/class
# definitions, never triggers a run.
from run_hubert import (
    build_scratch_particles,
    circuit_from_edge_probs,
    save_epoch_snapshot,
    soft_evaluate_particle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--task_type", required=True, choices=sorted(TASK_SPECS),
        help="which Articulatory Index task to discover circuits for (one task per run).",
    )
    parser.add_argument("--model_name", default="hubert-base-ls960",
                         choices=["hubert-base-ls960", "wav2vec2-base-960h", "wav2vec2-base"])
    parser.add_argument("--head_path", default=None)

    parser.add_argument("--n_particles", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)

    parser.add_argument("--lambda_edge_max", type=float, default=1.0)
    parser.add_argument("--edge_repulsion_warmup_frac", type=float, default=0.8)
    parser.add_argument("--lambda_node_max", type=float, default=0.0,
                         help="not currently used by this project; kept for parity with run_hubert.py. "
                              "If set >0, node repulsion still runs through the existing unmodified "
                              "multi_particle machinery using the SAME current (shared) lambda value.")
    parser.add_argument("--node_repulsion_warmup_frac", type=float, default=0.8)

    # --- trainable-lambda specific flags (replace run_hubert.py's fixed --jaccard_lambda) ---
    parser.add_argument("--log_lam_init", type=float, required=True,
                         help="initial value of u = ln(lambda), NOT lambda itself -- pass 0.0 for the old "
                              "lambda=1 (product t-norm) default. Passed directly (no float(log(...)) "
                              "rounding) so the exact starting point relative to --taylor_band is under "
                              "your control, e.g. for comparing an init clearly outside the band, right "
                              "at its edge, and well inside it.")
    parser.add_argument("--lambda_lr", type=float, default=1e-2,
                         help="learning rate for log_lam's own Adam optimizer.")
    parser.add_argument("--lambda_reg_coef", type=float, default=1e-3,
                         help="L2 penalty coefficient on u=log_lam (pulls toward u=0, i.e. lambda=1), "
                              "added to lambda's own loss every inner iteration. Constant, not annealed.")
    parser.add_argument("--lambda_inner_iters", type=int, default=8,
                         help="number of lambda-only Adam steps run AFTER each full epoch of particle "
                              "training (outer/alternating scheme), each against a fresh batch.")
    parser.add_argument("--taylor_band", type=float, default=DEFAULT_TAYLOR_BAND,
                         help="|u| threshold below which the closed-form linear Taylor patch is used "
                              "instead of the exact expm1/log1p formula, when differentiating w.r.t. u.")

    parser.add_argument("--repulsion_reduction", default="mean", choices=["mean", "sum"],
                         help="reduction used for BOTH the particle phase's edge/node repulsion "
                              "(existing multi_particle.combine_repulsion) and the lambda phase's own "
                              "edge-only aggregation (combine_edge_repulsion_trainable), so lambda's "
                              "loss is scaled consistently with what the particles trained under.")

    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr_e", type=float, default=0.07)

    parser.add_argument("--edge_logit_init_mean", type=float, default=10.0)
    parser.add_argument("--edge_logit_init_std", type=float, default=0.01)
    parser.add_argument("--random_mode", default="gumbel_sigmoid", choices=["gumbel_sigmoid", "none"])
    parser.add_argument("--gs_temp_edge", type=float, default=1.0)

    parser.add_argument("--lambda_sparse_e", type=float, default=1.0)
    parser.add_argument("--min_times_lambda_sparse_e", type=float, default=0.01)
    parser.add_argument("--max_times_lambda_sparse_e", type=float, default=1.0)

    parser.add_argument("--lambda_complete_e", type=float, default=0.0)
    parser.add_argument("--completeness_start_frac", type=float, default=0.0)

    parser.add_argument("--snapshot_every", type=int, default=None)
    parser.add_argument("--epoch_snapshot_every", type=int, default=1)
    parser.add_argument("--skip_snapshot_eval", action="store_true")
    parser.add_argument("--save_dir", default=None,
                         help="defaults to circuits_discovered/hubert_circuits/"
                              "{task_type}_trainlam_init{log_lam_init}_{repulsion_reduction}/")

    parser.add_argument("--devices", nargs="+", default=None)
    parser.add_argument("--repulsion_device", default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    default_device = get_compute_device()
    devices = args.devices if args.devices is not None else [default_device]
    hub_device = args.repulsion_device or devices[0]

    seeds = args.seeds if args.seeds is not None else [52 + i for i in range(args.n_particles)]
    if len(seeds) != args.n_particles:
        raise ValueError(f"--seeds has {len(seeds)} entries but --n_particles={args.n_particles}.")

    _label_extractor, num_classes, _class_names = TASK_SPECS[args.task_type]

    save_dir = (
        Path(args.save_dir) if args.save_dir is not None
        else Path("circuits_discovered") / "hubert_circuits"
        / f"{args.task_type}_trainlam_init{args.log_lam_init}_{args.repulsion_reduction}"
    )
    print(f"save_dir resolved to: {save_dir}/  (pass --save_dir explicitly to override)")

    print(
        f"Loading {args.model_name} + {args.task_type} head, one replica per device in {sorted(set(devices))}..."
    )
    models_by_device = load_hubert_classifiers_for_devices(
        args.task_type, devices, head_path=args.head_path, model_name=args.model_name,
    )
    data = load_articulatory_dataset(args.task_type, batch_size=args.batch_size)

    warmup = int(0.5 * args.n_epochs)
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
    device_counts = {d: sum(1 for p in particles if p.device == d) for d in sorted(set(devices))}
    print("Particles per device: " + ", ".join(f"{d}={n}" for d, n in device_counts.items())
          + f"  (repulsion combined on {hub_device})")

    scratch_particles = (
        build_scratch_particles(devices, models_by_device, args.model_name)
        if not args.skip_snapshot_eval else {}
    )

    # --- trainable lambda, shared across the whole cohort, lives on hub_device ---
    lambda_module = FrankLambdaModule(FrankLambdaConfig(
        log_lam_init=args.log_lam_init,
        lr=args.lambda_lr,
        reg_coef=args.lambda_reg_coef,
        band=args.taylor_band,
        inner_iters=args.lambda_inner_iters,
        device=hub_device,
    ))
    print(
        f"Trainable lambda: log_lam_init={args.log_lam_init} (lambda={lambda_module.lam:.6f}), "
        f"lr={args.lambda_lr}, reg_coef={args.lambda_reg_coef}, taylor_band={args.taylor_band}, "
        f"inner_iters={args.lambda_inner_iters} -- outer/alternating, after each epoch, particles' "
        f"edge_logits held fixed during lambda's own steps."
    )

    train_loader = fixed_order_dataloader(data.train.dataset, batch_size=args.batch_size, seed=seeds[0])
    n_epochs = args.n_epochs
    complete_start = int(args.completeness_start_frac * n_epochs)

    incidence_by_device, node_keys = build_node_incidence_for_devices(
        particles[0].discogp.masks, devices,
    )

    snapshot_epochs = {n_epochs - 1}
    if args.snapshot_every:
        snapshot_epochs |= set(range(args.snapshot_every - 1, n_epochs, args.snapshot_every))

    pairs_n = max(1, args.n_particles * (args.n_particles - 1) // 2)

    epoch_snapshot_dirs = [save_dir / f"particle{i}_epoch_snapshots" for i in range(args.n_particles)]
    epoch_snapshot_summaries: list[list[dict]] = [[] for _ in range(args.n_particles)]

    print(
        f"Training {args.n_particles} particles jointly for {n_epochs} epochs "
        f"({len(train_loader)} steps/epoch, {len(node_keys)} nodes / {incidence_by_device[devices[0]].shape[1]} "
        f"edges in the full graph), task={args.task_type} ({num_classes}-way), "
        f"lambda_edge_max={args.lambda_edge_max}, lambda_node_max={args.lambda_node_max}, "
        f"repulsion_reduction={args.repulsion_reduction}, "
        f"devices={sorted(set(devices))}, repulsion_device={hub_device}..."
    )
    if args.epoch_snapshot_every and args.skip_snapshot_eval:
        print("  (--skip_snapshot_eval: soft_test_acc/hard_test_acc will NOT be computed)")
    t0 = time.time()

    cumulative_band_hits = 0
    cumulative_lambda_iters = 0

    for epoch in range(n_epochs):
        # Lambda held fixed at this value for the ENTIRE particle phase below
        # -- only the lambda phase, at the end of this loop body, updates it.
        lam_this_epoch = lambda_module.lam

        lambda_sparse_vals = [p.discogp._scheduled_lambda_sparse(mode="edge", epoch=epoch) for p in particles]
        lambda_edge_rep = ramp_schedule(epoch, n_epochs, args.edge_repulsion_warmup_frac, args.lambda_edge_max)
        lambda_node_rep = ramp_schedule(epoch, n_epochs, args.node_repulsion_warmup_frac, args.lambda_node_max)

        epoch_task_loss = [0.0] * args.n_particles
        epoch_correct = [0] * args.n_particles
        epoch_density = [0.0] * args.n_particles
        epoch_total = 0
        epoch_edge_rep = 0.0
        epoch_node_rep = 0.0
        n_batches = 0

        # ===================== PARTICLE PHASE =====================
        # Structurally identical to run_hubert.py's own batch loop. The only
        # difference: jaccard_lambda=lam_this_epoch (a plain float, updated
        # once per epoch by the lambda phase below) instead of a fixed CLI
        # constant. combine_repulsion here is the EXISTING, UNMODIFIED
        # multi_particle.py implementation -- safe, since lambda is not
        # being differentiated in this phase (see module docstring).
        for batch in train_loader:
            task_losses = []
            edge_probs = []
            for i, (particle, lam_sparse) in enumerate(zip(particles, lambda_sparse_vals)):
                task_loss, probs, logits = per_particle_forward(particle, batch, lambda_sparse=lam_sparse)
                task_losses.append(task_loss)
                edge_probs.append(probs)
                preds = logits.detach().argmax(dim=-1)
                labels = batch["label"].to(device=logits.device)
                epoch_correct[i] += (preds == labels).sum().item()
                epoch_density[i] += (probs.detach() > 0.5).float().mean().item()
            epoch_total += batch["label"].shape[0]

            node_probs = [
                node_probs_from_edge_probs(ep, incidence_by_device[str(ep.device)])
                for ep in edge_probs
            ]
            edge_rep, node_rep = combine_repulsion(
                edge_probs, node_probs, lambda_edge=lambda_edge_rep, lambda_node=lambda_node_rep,
                hub_device=hub_device, jaccard_lambda=lam_this_epoch, reduction=args.repulsion_reduction,
            )

            joint_loss = (
                (sum(l.to(device=hub_device) for l in task_losses) / len(task_losses))
                + lambda_edge_rep * edge_rep
                + lambda_node_rep * node_rep
            )
            joint_loss.backward()
            for particle in particles:
                per_particle_backward_step(particle)

            for i, l in enumerate(task_losses):
                epoch_task_loss[i] += l.item()
            epoch_edge_rep += edge_rep.detach().item()
            epoch_node_rep += node_rep.detach().item()
            n_batches += 1

            if epoch >= complete_start and args.lambda_complete_e > 0.0:
                for particle in particles:
                    run_completeness_step(
                        particle, batch, lambda_complete=args.lambda_complete_e, num_classes=num_classes,
                    )

        mean_jaccard_edge = epoch_edge_rep / n_batches
        mean_jaccard_node = epoch_node_rep / n_batches
        if args.repulsion_reduction == "sum":
            mean_jaccard_edge /= pairs_n
            mean_jaccard_node /= pairs_n
        elif args.repulsion_reduction == "mean":
            mean_jaccard_edge *= (args.n_particles - 1) / pairs_n
            mean_jaccard_node *= (args.n_particles - 1) / pairs_n

        print(
            f"epoch {epoch:3d}  "
            f"loss={[round(l / n_batches, 4) for l in epoch_task_loss]}  "
            f"train_acc={[round(c / epoch_total, 4) for c in epoch_correct]}  "
            f"edge_density={[round(d / n_batches, 4) for d in epoch_density]}  "
            f"jaccard_edge={mean_jaccard_edge:.4f}  "
            f"lambda_edge={lambda_edge_rep:.3f}  "
            f"lam_this_epoch={lam_this_epoch:.6f}"
        )

        if args.epoch_snapshot_every and (epoch + 1) % args.epoch_snapshot_every == 0:
            for i, particle in enumerate(particles):
                snapshot_edge_probs = flat_probs(particle.discogp).detach()
                snapshot_density = (snapshot_edge_probs > 0.5).float().mean().item()

                soft_train_acc = epoch_correct[i] / epoch_total
                soft_test_acc = None
                hard_test_acc = None
                if not args.skip_snapshot_eval:
                    soft_test_acc = soft_evaluate_particle(particle, data.test)
                    hard_circuit = circuit_from_edge_probs(
                        snapshot_edge_probs, scratch_particles[particle.device],
                    )
                    hard_eval = evaluate_circuit_classification(
                        scratch_particles[particle.device].discogp.model, data.test, hard_circuit,
                    )
                    hard_test_acc = hard_eval.get("acc")

                # jaccard_lambda recorded here is lam_this_epoch -- the value
                # these particles actually trained under this epoch, NOT
                # whatever the lambda phase below updates it to for next epoch.
                save_epoch_snapshot(
                    epoch_snapshot_dirs[i], epoch=epoch + 1, seed=particle.seed, task_type=args.task_type,
                    train_acc=epoch_correct[i] / epoch_total, train_loss=epoch_task_loss[i] / n_batches,
                    edge_density=snapshot_density, jaccard_edge=mean_jaccard_edge, jaccard_node=mean_jaccard_node,
                    jaccard_lambda=lam_this_epoch, repulsion_reduction=args.repulsion_reduction,
                    edge_probs=snapshot_edge_probs, summary=epoch_snapshot_summaries[i],
                    soft_train_acc=soft_train_acc, soft_test_acc=soft_test_acc, hard_test_acc=hard_test_acc,
                )

                if not args.skip_snapshot_eval:
                    print(
                        f"  particle{i} (seed{particle.seed}, {particle.device})  "
                        f"soft_train_acc={soft_train_acc:.4f}  soft_test_acc={soft_test_acc:.4f}  "
                        f"hard_test_acc={hard_test_acc:.4f}  edge_density={snapshot_density:.4f}"
                    )

        if epoch in snapshot_epochs:
            finalize_and_report(
                tag=f"epoch{epoch + 1}", task_type=args.task_type, particles=particles, data=data, save_dir=save_dir,
            )

        # ===================== LAMBDA PHASE =====================
        # AFTER this epoch's particle updates: every particle's edge_logits
        # held fixed (torch.no_grad() + detach inside run_lambda_phase),
        # log_lam takes --lambda_inner_iters Adam steps against fresh
        # batches. Updates lambda_module in place for the NEXT epoch's
        # particle phase.
        lam_stats = run_lambda_phase(
            lambda_module, particles, train_loader,
            per_particle_forward=per_particle_forward,
            lambda_edge_coef=lambda_edge_rep,
            reduction=args.repulsion_reduction,
        )
        cumulative_band_hits += lam_stats["band_hits"]
        cumulative_lambda_iters += lam_stats["n_iters"]
        print(
            f"  lambda_phase  u={lam_stats['u']:.6f}  lambda={lam_stats['lam']:.6f}  "
            f"mean_loss={lam_stats['mean_loss']:.6f}  mean_edge_rep={lam_stats['mean_edge_rep']:.6f}  "
            f"band_hits={lam_stats['band_hits']}/{lam_stats['n_iters']}  "
            f"(cumulative={cumulative_band_hits}/{cumulative_lambda_iters})"
        )

    elapsed = time.time() - t0
    print(
        f"\nDone: {args.n_particles} {args.task_type} circuits, {n_epochs} epochs, "
        f"{elapsed:.1f}s. Saved under {save_dir}/"
    )
    print(
        f"Final lambda: u={lambda_module.u:.6f}, lambda={lambda_module.lam:.6f}. "
        f"Taylor-band occupancy across the whole run: "
        f"{cumulative_band_hits}/{cumulative_lambda_iters} lambda-update iterations "
        f"({100.0 * cumulative_band_hits / max(1, cumulative_lambda_iters):.1f}%) "
        f"landed inside |u| < {args.taylor_band}."
    )
    if args.epoch_snapshot_every:
        for i, d in enumerate(epoch_snapshot_dirs):
            print(f"Particle {i} epoch snapshots: {d}/ (scan {d}/summary.json)")


if __name__ == "__main__":
    main()