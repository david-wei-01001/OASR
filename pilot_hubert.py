#!/usr/bin/env python3
"""
pilot_hubert.py

Reproduce pilot1.py's purpose (a plain, no-repulsion, single-seed baseline)
for HuBERT on an Articulatory Index task, instead of GPT2/IOI.

IMPORTANT: this does NOT call DiscoGP.discover_circuit() directly the way
pilot1.py does. That method's internal _train_mode() calls
`self.model(batch["input_ids"], runtime_masks=runtime_masks)` without
`lengths=`, which for CircuitHubertClassifier silently falls back to
*unmasked* mean pooling -- the exact padding-dilution bug fixed earlier.
Instead this reuses run_hubert.py's own per_particle_forward /
finalize_and_report machinery (which does pass `lengths=batch["length"]`)
with n_particles pinned to 1, so repulsion is structurally a no-op
(itertools.combinations over one particle has zero pairs) rather than
something you have to remember to disable via a flag.

This is 90% "python run_hubert.py --task_type ... --n_particles 1" with the
n_particles bookkeeping (Jaccard prints, lambda ramps, multi-device
assignment) stripped out for a cleaner diagnostic read on whether a single
un-repelled circuit can fit the task at all.

Run from the OASR repo root, after the same two prep steps run_hubert.py
needs:
  1. prepare_and_save_articulatory_dataset(task_type)
  2. python -m circuit_discovery.tasks.train_classification_head --task_type <task_type>

Example:
    python pilot_hubert.py --task_type vowel_classification
    python pilot_hubert.py --task_type vowel_classification --seed 42 --n_epochs 40
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from circuit_discovery.run import get_compute_device
from circuit_discovery.tasks.articulatory_index import TASK_SPECS, load_articulatory_dataset
from circuit_discovery.tasks.discovery_setup import (
    build_particles,
    finalize_and_report,
    load_hubert_classifiers_for_devices,
    per_particle_backward_step,
    per_particle_forward,
    run_completeness_step,
)
from circuit_discovery.utils import fixed_order_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task_type", required=True, choices=sorted(TASK_SPECS))
    parser.add_argument("--model_name", default="hubert-base-ls960",
                         choices=["hubert-base-ls960", "wav2vec2-base-960h", "wav2vec2-base"])
    parser.add_argument("--head_path", default=None,
                         help="defaults to circuit_discovery/trained_heads/{task_type}_head.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=4)
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
    parser.add_argument("--save_dir", default=None,
                         help="defaults to circuits_discovered/hubert_circuits/{task_type}_pilot/")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_compute_device()
    print(device)

    _label_extractor, num_classes, _class_names = TASK_SPECS[args.task_type]
    save_dir = (
        Path(args.save_dir) if args.save_dir is not None
        else Path("circuits_discovered") / "hubert_circuits" / f"{args.task_type}_pilot"
    )

    models_by_device = load_hubert_classifiers_for_devices(
        args.task_type, [device], head_path=args.head_path, model_name=args.model_name,
    )
    data = load_articulatory_dataset(args.task_type, batch_size=args.batch_size)

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
        overlap_penalty=False,  # <-- key: no repulsion, pure single-seed baseline
    )

    # n_particles=1 -> nothing downstream needs repulsion wiring at all:
    # there's only ever one particle to loop over.
    [particle] = build_particles(
        seeds=[args.seed], devices=[device], models_by_device=models_by_device,
        discogp_config_kwargs=discogp_config_kwargs,
    )

    train_loader = fixed_order_dataloader(data.train.dataset, batch_size=args.batch_size, seed=args.seed)
    n_epochs = args.n_epochs
    complete_start = int(args.completeness_start_frac * n_epochs)

    print(
        f"Training 1 particle (seed={args.seed}) for {n_epochs} epochs "
        f"({len(train_loader)} steps/epoch), task={args.task_type} ({num_classes}-way), "
        f"no repulsion, device={device}..."
    )
    t0 = time.time()

    for epoch in range(n_epochs):
        lambda_sparse = particle.discogp._scheduled_lambda_sparse(mode="edge", epoch=epoch)

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        n_batches = 0

        for batch in train_loader:
            task_loss, _probs, logits = per_particle_forward(particle, batch, lambda_sparse=lambda_sparse)
            task_loss.backward()
            per_particle_backward_step(particle)

            epoch_loss += task_loss.item()
            preds = logits.detach().argmax(dim=-1)
            labels = batch["label"].to(device=logits.device)
            epoch_correct += (preds == labels).sum().item()
            epoch_total += labels.shape[0]
            n_batches += 1

            if epoch >= complete_start and args.lambda_complete_e > 0.0:
                run_completeness_step(
                    particle, batch, lambda_complete=args.lambda_complete_e, num_classes=num_classes,
                )

        print(
            f"epoch {epoch:3d}  loss={epoch_loss / n_batches:.4f}  "
            f"train_acc={epoch_correct / epoch_total:.4f}  lambda_sparse={lambda_sparse:.4f}"
        )

    elapsed = time.time() - t0
    finalize_and_report(
        tag="final", task_type=args.task_type, particles=[particle], data=data, save_dir=save_dir,
    )
    print(f"\nDone: 1 {args.task_type} circuit (seed={args.seed}), {n_epochs} epochs, "
          f"{elapsed:.1f}s. Saved under {save_dir}/")


if __name__ == "__main__":
    main()
