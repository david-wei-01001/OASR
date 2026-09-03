#!/usr/bin/env python3
"""
run_hubert_sequential.py

The original OASR pattern -- find-a-circuit-then-repel-from-it, sequentially
-- for HuBERT on an Articulatory Index task. This is NOT in the repo as a
runnable script (pilot1.py/pilot2.py, referenced in run_hubert.py's and
multi_particle.py's docstrings, were never committed here -- checked the
full git history, they don't exist). This reconstructs that pattern from
the same building blocks run_hubert.py and pilot_hubert.py already use:

  particle 0: trained exactly like pilot_hubert.py -- plain DiscoGP, no
              repulsion term at all (nothing to repel from yet) UNLESS
              --load_frozen points at hand-picked circuit(s) already found
              (e.g. from a prior pilot_hubert.py or run_hubert_sequential.py
              epoch snapshot), in which case it repels from those instead.
  particle k (k>=1, only relevant if --n_particles > 1 in a single
              invocation): trained the same way, EXCEPT its task_loss also
              gets
              + lambda_edge_rep * sum_j soft_jaccard(own_edge_probs, frozen_edge_probs[j])
              + lambda_node_rep * sum_j soft_jaccard(own_node_probs, frozen_node_probs[j])
              summed over every previously-discovered (now frozen, detached,
              no-longer-training) circuit j < k -- whether that circuit came
              from --load_frozen or from an earlier particle finished
              earlier in THIS invocation. Only particle k's own edge_logits
              get gradients -- the frozen probs are plain detached tensors,
              not other live particles, so there's no mutual/simultaneous
              escalation the way there is in run_hubert.py's joint version.

RECOMMENDED HAND-PICKING WORKFLOW (prune one, pick, prune the next, pick):
  1. python pilot_hubert.py --task_type vowel_classification
     -> inspect circuits_discovered/.../vowel_classification_pilot/epoch_snapshots/summary.json,
        pick the epoch with the best accuracy/density tradeoff, note its .pt path (call it A).
  2. python run_hubert_sequential.py --task_type vowel_classification \
         --n_particles 1 --seeds 53 --load_frozen A
     -> inspect THIS run's epoch_snapshots/summary.json (see below), pick an
        epoch, note its .pt path (call it B).
  3. python run_hubert_sequential.py --task_type vowel_classification \
         --n_particles 1 --seeds 54 --load_frozen A B
     -> pick a third circuit C the same way.
--load_frozen accepts any number of paths and doesn't care whether they came
from pilot_hubert.py or from a previous run_hubert_sequential.py invocation
-- both save the same {epoch, train_acc, edge_density, edge_probs} format.

Each particle trains for the full --n_epochs from scratch (its own
DiscoGP/edge_logits/optimizer/sparsity-and-repulsion ramp schedule,
restarting at epoch 0), one after another -- hence "sequential", and why
this doesn't bother with the multi-GPU dispatch run_hubert.py has: there's
only ever one particle actually training at a time, so spreading particles
across devices wouldn't overlap any work.

EPOCH SNAPSHOTS: every --epoch_snapshot_every epochs of EACH particle's own
training, this saves
    {save_dir}/particle{k}_epoch_snapshots/epoch{N:03d}.pt
containing {epoch, train_acc, loss, edge_density, jaccard_edge_vs_frozen,
jaccard_node_vs_frozen, edge_probs} -- same edge_probs format (flat_probs =
sigmoid(edge_logits), deterministic) that --load_frozen reads back in, plus
a rewritten scalars-only summary.json per particle so the whole tradeoff
curve (now 3-way: accuracy vs density vs overlap-with-prior-circuits) can be
scanned without loading tensors. This is distinct from --snapshot_every,
which still controls the heavier finalize_and_report (boolean circuit +
eval + save) calls, unchanged from before.

Run from the OASR repo root, after the same two prep steps run_hubert.py
needs:
  1. prepare_and_save_articulatory_dataset(task_type)
  2. python -m circuit_discovery.tasks.train_classification_head --task_type <task_type>

Examples:
    python run_hubert_sequential.py --task_type vowel_classification
    python run_hubert_sequential.py --task_type vowel_classification \\
        --n_particles 3 --lambda_edge_max 0.66 --lambda_node_max 10.0
    python run_hubert_sequential.py --task_type vowel_classification \\
        --n_particles 1 --seeds 53 --load_frozen circuits_discovered/.../epoch012.pt

Turning a repulsion term off: pass --lambda_edge_max 0 or --lambda_node_max 0,
same convention as run_hubert.py (ramp_schedule returns 0 for the whole run).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from circuit_discovery.run import get_compute_device
from circuit_discovery.tasks.articulatory_index import TASK_SPECS, load_articulatory_dataset
from circuit_discovery.tasks.discovery_setup import (
    build_node_incidence,
    build_particles,
    finalize_and_report,
    flat_probs,
    load_hubert_classifiers_for_devices,
    node_probs_from_edge_probs,
    per_particle_backward_step,
    per_particle_forward,
    ramp_schedule,
    run_completeness_step,
    soft_jaccard,
)
from circuit_discovery.utils import fixed_order_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--task_type", required=True, choices=sorted(TASK_SPECS))
    parser.add_argument("--model_name", default="hubert-base-ls960",
                         choices=["hubert-base-ls960", "wav2vec2-base-960h", "wav2vec2-base"])
    parser.add_argument("--head_path", default=None)

    parser.add_argument("--n_particles", type=int, default=3,
                         help="how many circuits to discover, one after another, in THIS invocation.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                         help="one seed per particle; defaults to [52, 53, 54, ...] up to n_particles.")
    parser.add_argument("--load_frozen", nargs="+", default=None,
                         help="paths to hand-picked epoch snapshot .pt files (from pilot_hubert.py or a "
                              "previous run_hubert_sequential.py run) to treat as already-frozen circuits, "
                              "in addition to any particles trained within this invocation. Lets you "
                              "hand-pick after each stage instead of always freezing the last epoch.")

    parser.add_argument("--lambda_edge_max", type=float, default=0.66,
                         help="max strength of edge repulsion vs. each frozen prior circuit. 0 disables it.")
    parser.add_argument("--edge_repulsion_warmup_frac", type=float, default=0.8)
    parser.add_argument("--lambda_node_max", type=float, default=0.0,
                         help="max strength of node repulsion vs. each frozen prior circuit. 0 disables it.")
    parser.add_argument("--node_repulsion_warmup_frac", type=float, default=0.8)

    parser.add_argument("--n_epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr_e", type=float, default=0.07)

    parser.add_argument("--edge_logit_init_mean", type=float, default=10.0)
    parser.add_argument("--edge_logit_init_std", type=float, default=0.01)
    parser.add_argument("--random_mode", default="gumbel_sigmoid", choices=["gumbel_sigmoid", "none"])
    parser.add_argument("--gs_temp_edge", type=float, default=1.0)

    parser.add_argument("--lambda_sparse_e", type=float, default=1.0)
    parser.add_argument("--min_times_lambda_sparse_e", type=float, default=0.01)
    parser.add_argument("--max_times_lambda_sparse_e", type=float, default=1.0)

    parser.add_argument("--lambda_complete_e", type=float, default=0.01)
    parser.add_argument("--completeness_start_frac", type=float, default=0.8)

    parser.add_argument("--snapshot_every", type=int, default=None,
                         help="also call finalize_and_report (boolean circuit + eval + save) every N "
                              "epochs of each particle's own training, not just at the end.")
    parser.add_argument("--epoch_snapshot_every", type=int, default=1,
                         help="save a hand-pickable {acc, density, jaccard_vs_frozen, edge_probs} snapshot "
                              "every N epochs of each particle's own training. 0 disables it.")
    parser.add_argument("--save_dir", default=None,
                         help="defaults to circuits_discovered/hubert_circuits/{task_type}_sequential/")

    parser.add_argument("--device", default=None, help="single device; defaults to auto-picked.")

    return parser.parse_args()


def save_epoch_snapshot(
    snapshot_dir: Path, *, epoch: int, seed: int, task_type: str,
    train_acc: float, train_loss: float, edge_density: float,
    jaccard_edge_vs_frozen: float, jaccard_node_vs_frozen: float,
    edge_probs: torch.Tensor, summary: list[dict],
) -> None:
    """Same format as pilot_hubert.py's save_epoch_snapshot, plus the two
    jaccard-vs-frozen fields (meaningless/0 for a particle with nothing
    frozen to repel from yet, i.e. particle 0 with no --load_frozen)."""
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "task_type": task_type, "seed": seed, "epoch": epoch,
        "train_acc": train_acc, "train_loss": train_loss, "edge_density": edge_density,
        "jaccard_edge_vs_frozen": jaccard_edge_vs_frozen, "jaccard_node_vs_frozen": jaccard_node_vs_frozen,
        "edge_probs": edge_probs.detach().cpu(),
    }
    torch.save(record, snapshot_dir / f"epoch{epoch:03d}.pt")

    summary.append({k: v for k, v in record.items() if k != "edge_probs"})
    with open(snapshot_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    args = parse_args()
    device = args.device or get_compute_device()

    seeds = args.seeds if args.seeds is not None else [52 + i for i in range(args.n_particles)]
    if len(seeds) != args.n_particles:
        raise ValueError(f"--seeds has {len(seeds)} entries but --n_particles={args.n_particles}.")

    _label_extractor, num_classes, _class_names = TASK_SPECS[args.task_type]
    save_dir = (
        Path(args.save_dir) if args.save_dir is not None
        else Path("circuits_discovered") / "hubert_circuits" / f"{args.task_type}_sequential"
    )

    print(f"Loading {args.model_name} + {args.task_type} head on {device}...")
    models_by_device = load_hubert_classifiers_for_devices(
        args.task_type, [device], head_path=args.head_path, model_name=args.model_name,
    )
    data = load_articulatory_dataset(args.task_type, batch_size=args.batch_size)
    train_loader = fixed_order_dataloader(data.train.dataset, batch_size=args.batch_size, seed=seeds[0])

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

    n_epochs = args.n_epochs
    complete_start = int(args.completeness_start_frac * n_epochs)

    # Frozen reference tensors -- plain detached tensors, never another live
    # particle, so no mutual/simultaneous escalation the way run_hubert.py's
    # joint version has. Two sources feed this list:
    #   1. --load_frozen: hand-picked snapshots from an earlier stage/run.
    #   2. particles trained within THIS invocation (n_particles > 1),
    #      auto-frozen at their own final epoch, same as before.
    frozen_edge_probs: list[torch.Tensor] = []
    frozen_node_probs: list[torch.Tensor] = []
    incidence = None  # built once, from the first particle's masks (graph structure is shared)

    if args.load_frozen:
        for path in args.load_frozen:
            snap = torch.load(path, map_location=device)
            frozen_edge_probs.append(snap["edge_probs"].to(device=device).detach())
        print(f"Loaded {len(frozen_edge_probs)} hand-picked frozen circuit(s) from --load_frozen: "
              f"{args.load_frozen}")

    t0 = time.time()

    for k, seed in enumerate(seeds):
        print(f"\n===== Particle {k} (seed={seed}) -- repelling from {len(frozen_edge_probs)} frozen circuit(s) =====")

        [particle] = build_particles(
            seeds=[seed], devices=[device], models_by_device=models_by_device,
            discogp_config_kwargs=discogp_config_kwargs,
        )

        if incidence is None:
            incidence, node_keys = build_node_incidence(particle.discogp.masks, device=device)
            print(f"graph: {len(node_keys)} nodes / {incidence.shape[1]} edges")
            # incidence wasn't available yet when --load_frozen tensors were
            # loaded above -- backfill their node_probs now, once.
            frozen_node_probs = [node_probs_from_edge_probs(ep, incidence).detach() for ep in frozen_edge_probs]

        n_frozen = max(1, len(frozen_edge_probs))  # for averaging the printed jaccard, not the loss itself
        snapshot_dir = save_dir / f"particle{k}_epoch_snapshots"
        snapshot_summary: list[dict] = []

        for epoch in range(n_epochs):
            lambda_sparse = particle.discogp._scheduled_lambda_sparse(mode="edge", epoch=epoch)
            lambda_edge_rep = ramp_schedule(epoch, n_epochs, args.edge_repulsion_warmup_frac, args.lambda_edge_max)
            lambda_node_rep = ramp_schedule(epoch, n_epochs, args.node_repulsion_warmup_frac, args.lambda_node_max)

            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            epoch_edge_rep = 0.0
            epoch_node_rep = 0.0
            n_batches = 0

            for batch in train_loader:
                task_loss, edge_probs, logits = per_particle_forward(particle, batch, lambda_sparse=lambda_sparse)

                if frozen_edge_probs:
                    edge_rep = sum(soft_jaccard(edge_probs, fp) for fp in frozen_edge_probs)
                    node_probs = node_probs_from_edge_probs(edge_probs, incidence)
                    node_rep = sum(soft_jaccard(node_probs, fp) for fp in frozen_node_probs)
                else:
                    edge_rep = torch.zeros((), device=device)
                    node_rep = torch.zeros((), device=device)

                loss = task_loss + lambda_edge_rep * edge_rep + lambda_node_rep * node_rep
                loss.backward()
                per_particle_backward_step(particle)

                epoch_loss += task_loss.item()
                preds = logits.detach().argmax(dim=-1)
                labels = batch["label"].to(device=logits.device)
                epoch_correct += (preds == labels).sum().item()
                epoch_total += labels.shape[0]
                epoch_edge_rep += float(edge_rep)
                epoch_node_rep += float(node_rep)
                n_batches += 1

                if epoch >= complete_start and args.lambda_complete_e > 0.0:
                    run_completeness_step(
                        particle, batch, lambda_complete=args.lambda_complete_e, num_classes=num_classes,
                    )

            # Once per epoch, deterministic -- single source of truth for this
            # epoch's density AND the tensor saved for hand-picking / the next
            # stage's repulsion.
            epoch_edge_probs = flat_probs(particle.discogp).detach()
            edge_density = (epoch_edge_probs > 0.5).float().mean().item()
            mean_loss = epoch_loss / n_batches
            mean_acc = epoch_correct / epoch_total
            mean_jaccard_edge = epoch_edge_rep / n_batches / n_frozen
            mean_jaccard_node = epoch_node_rep / n_batches / n_frozen

            print(
                f"  epoch {epoch:3d}  loss={mean_loss:.4f}  "
                f"train_acc={mean_acc:.4f}  edge_density={edge_density:.4f}  "
                f"jaccard_edge_vs_frozen={mean_jaccard_edge:.4f}  "
                f"jaccard_node_vs_frozen={mean_jaccard_node:.4f}  "
                f"lambda_sparse={lambda_sparse:.3f}  lambda_edge={lambda_edge_rep:.3f}  lambda_node={lambda_node_rep:.3f}"
            )

            if args.epoch_snapshot_every and (epoch + 1) % args.epoch_snapshot_every == 0:
                save_epoch_snapshot(
                    snapshot_dir, epoch=epoch + 1, seed=seed, task_type=args.task_type,
                    train_acc=mean_acc, train_loss=mean_loss, edge_density=edge_density,
                    jaccard_edge_vs_frozen=mean_jaccard_edge, jaccard_node_vs_frozen=mean_jaccard_node,
                    edge_probs=epoch_edge_probs, summary=snapshot_summary,
                )

            if args.snapshot_every and (epoch + 1) % args.snapshot_every == 0:
                finalize_and_report(
                    tag=f"particle{k}_epoch{epoch + 1}", task_type=args.task_type,
                    particles=[particle], data=data, save_dir=save_dir,
                )

        finalize_and_report(
            tag=f"particle{k}_final", task_type=args.task_type, particles=[particle], data=data, save_dir=save_dir,
        )
        if args.epoch_snapshot_every:
            print(f"Per-epoch snapshots for hand-picking: {snapshot_dir}/ "
                  f"(scan {snapshot_dir}/summary.json)")

        # Freeze this particle's final epoch for any FURTHER particles within
        # THIS SAME invocation (n_particles > 1). If you're hand-picking
        # between stages instead, ignore this and just point the next
        # invocation's --load_frozen at whichever epoch{N:03d}.pt you chose
        # from snapshot_dir above, rather than always the final one.
        final_edge_probs = flat_probs(particle.discogp).detach()
        frozen_edge_probs.append(final_edge_probs)
        frozen_node_probs.append(node_probs_from_edge_probs(final_edge_probs, incidence).detach())

    elapsed = time.time() - t0
    print(f"\nDone: {args.n_particles} {args.task_type} circuits (sequential), {elapsed:.1f}s. Saved under {save_dir}/")


if __name__ == "__main__":
    main()