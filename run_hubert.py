#!/usr/bin/env python3
"""
run_hubert.py

CLI entrypoint for OASR-style joint multi-particle circuit discovery --
edge repulsion AND node repulsion applied simultaneously to N particles
trained together in one loop, not the sequential find-a-circuit-then-repel-
from-it pattern -- against a CircuitHubert backbone plus a frozen, already-
trained classification head, on one Articulatory Index task per run.

Run from the OASR repo root, after:
  1. Building the task's dataset:
       from circuit_discovery.tasks.articulatory_index import prepare_and_save_articulatory_dataset
       prepare_and_save_articulatory_dataset("vowel_classification", data_dir=DATA_DIR)
       prepare_and_save_articulatory_dataset("consonant_classification", data_dir=DATA_DIR)
  2. Training and saving that task's head:
       python -m circuit_discovery.tasks.train_classification_head --task_type vowel_classification
       python -m circuit_discovery.tasks.train_classification_head --task_type consonant_classification

Examples:
    python run_hubert.py --task_type vowel_classification
    python run_hubert.py --task_type consonant_classification \\
        --n_particles 3 --lambda_edge_max 0.66 --lambda_node_max 10.0

    # spread 6 particles round-robin over two GPUs, combine repulsion on cuda:0
    python run_hubert.py --task_type vowel_classification \\
        --n_particles 6 --devices cuda:0 cuda:1 --repulsion_device cuda:0

    # 5 particles unevenly split (2 on cuda:0, 3 on cuda:1) via a repeated
    # device entry, with the repulsion combine on the less-loaded GPU
    python run_hubert.py --task_type vowel_classification \\
        --n_particles 5 --devices cuda:0 cuda:1 cuda:1 --repulsion_device cuda:0

Turning a repulsion term off: pass --lambda_edge_max 0 or --lambda_node_max 0.
The ramp schedule returns 0 for the whole run in that case -- there's no
separate --disable_* flag, one fewer thing to keep in sync.

--jaccard_lambda selects WHICH fuzzy-set formula every pairwise repulsion
term (edge and node alike) is computed with, via the Frank family of
t-norms/t-conorms -- a one-parameter family whose named landmarks are:
    0.0   -> min/max (Zadeh's original fuzzy min/max)
    1.0   -> product / probabilistic-sum (DEFAULT -- the ORIGINAL
             soft_jaccard formula, unchanged from every prior run)
    inf   -> Lukasiewicz
This is a different axis from --lambda_edge_max/--lambda_node_max: those
scale how MUCH the repulsion term counts in the loss; --jaccard_lambda
changes what "overlap" even MEANS in that term's formula. See
circuit_discovery/multi_particle.py's frank_soft_jaccard() for the math.

Run the two tasks as two independent invocations (as designed): this script
discovers N mutually-repelling circuits for ONE task at a time. It does not
run vowel and consonant particles against each other.

Parallelism: IMPLEMENTED (single-process multi-GPU). --devices takes one or
more device strings; particles are assigned round-robin across them, each
unique device gets its own frozen model replica, and --repulsion_device
picks where the cross-particle edge/node repulsion terms get combined each
step (defaults to devices[0]). Round-robin is purely mechanical (index
modulo len(devices)) -- it has NO notion of how full a GPU is. To skew the
split (e.g. because one GPU can only fit 3 particles), repeat a device
entry in --devices (e.g. `--devices cuda:0 cuda:1 cuda:1` puts 2 particles
on cuda:0 and 3 on cuda:1 for 5 total particles) and point
--repulsion_device at whichever GPU ends up least loaded, since combining
the pairwise repulsion terms adds extra memory pressure on top of whatever
particles already live there. See
circuit_discovery/multi_particle.py's module docstring for the design and
circuit_discovery/tasks/discovery_setup.py's Particle /
per_particle_forward / combine_repulsion / per_particle_backward_step split
for exactly how each phase runs. With a single device (the default) this is
behaviorally identical to the old single-GPU code path.

Each epoch also prints edge_density: per particle, the fraction of edges
currently kept (sigmoid(edge_logit) > 0.5 -- same threshold convention as
DiscoGP's own hard=(logits>0).float() straight-through mask) out of the
full graph's edge count. This is the SAME quantity pilot_hubert.py and
run_hubert_sequential.py already log, added here so the joint run can be
read the same way: is a particle's accuracy dropping because the task is
genuinely failing, or because its circuit is shrinking out from under it?
Distinct from the soft/relaxed density used inside the sparsity loss itself.
NOTE: density alone can be a poor diagnostic -- two particles can have
near-identical density (in aggregate, and pairwise with each other) while
one has quietly lost a handful of task-critical edges and the other hasn't.
See soft_train_acc / soft_test_acc / hard_test_acc below for a diagnostic
that actually distinguishes those cases.

ACCURACY REPORTING (three numbers, not one): every --epoch_snapshot_every
epochs, for EACH particle this now reports:
  - soft_train_acc: accuracy from the same stochastic gumbel-sigmoid
    sampled forward pass used for training, evaluated on the training data
    seen this epoch. (This is the same number the main per-epoch line's
    train_acc list already showed -- just relabeled "soft" here for
    clarity, and repeated per-particle in this block.)
  - soft_test_acc: the SAME stochastic sampled-mask forward, run on the
    held-out test set instead of train data (see soft_evaluate_particle).
    Directly comparable to soft_train_acc -- same sampling method, only the
    data differs -- so a soft_train/soft_test gap tells you about
    generalization, not about discretization.
  - hard_test_acc: binarize each edge at sigmoid(logit) > 0.5 (same
    threshold used everywhere else in this codebase), rebuild a boolean
    Circuit via a throwaway scratch particle (never the live training
    particle -- see circuit_from_edge_probs), and evaluate that on the
    test set exactly the way finalize_and_report / compare_circuits.py do.
Why all three: if soft_test_acc ALSO collapses toward chance alongside
soft_train_acc, the circuit is genuinely broken (repulsion likely knocked
out task-critical edges even though aggregate density looks fine -- the
Frank/Lukasiewicz repulsion term is not smooth, and can concentrate almost
all of its gradient on exactly the highest-overlap, i.e. most task-critical,
edges rather than spreading pressure evenly across the whole circuit,
especially at jaccard_lambda=inf). If soft_test_acc stays healthy but only
hard_test_acc collapses, it's a discretization artifact: probabilities
haven't fully separated from 0.5 yet (early in the --edge_repulsion_warmup_frac
/ sparsity ramp), and the >0.5 threshold happens to disconnect something
that the soft/relaxed circuit was still using fine.

Parallelism note for the eval passes: soft_evaluate_particle and the hard
circuit rebuild add extra test-set forward passes per particle per snapshot
epoch. Pass --skip_snapshot_eval to disable both and fall back to only the
density-based diagnostics if this is too slow for your setup.

EPOCH SNAPSHOTS: every --epoch_snapshot_every epochs, ALL --n_particles
circuits from that epoch are each saved to
    {save_dir}/particle{i}_epoch_snapshots/epoch{N:03d}.pt
containing {epoch, train_acc, loss, edge_density, jaccard_edge,
jaccard_node, edge_probs, soft_train_acc, soft_test_acc, hard_test_acc} --
the same base format pilot_hubert.py and run_hubert_sequential.py already
write (edge_probs = flat_probs(p) = sigmoid(edge_logits), deterministic,
computed once per particle per epoch after the batch loop -- not a
mid-epoch batch sample), plus the three accuracy numbers above, so these
files are directly usable by compare_circuits.py or as --load_frozen inputs
to run_hubert_sequential.py, no reconstruction needed. jaccard_edge/jaccard_node
here are this epoch's OVERALL (all-pairs) repulsion values, same numbers
already in the printed line -- identical across every particle's snapshot
for a given epoch, since that's a property of the whole cohort, not any one
particle; kept per-file anyway so each snapshot is self-contained. A
scalars-only summary.json per particle (no tensors) is rewritten every
epoch for quick scanning.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from circuit_discovery.circuit import Circuit
from circuit_discovery.run import get_compute_device
from circuit_discovery.tasks.articulatory_index import TASK_SPECS, load_articulatory_dataset
from circuit_discovery.tasks.discovery_setup import (
    Particle,
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
    parser.add_argument("--head_path", default=None,
                         help="defaults to circuit_discovery/trained_heads/{task_type}_head.pt")

    parser.add_argument("--n_particles", type=int, default=2,
                         help="how many mutually-repelling circuits to discover simultaneously.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                         help="one seed per particle; defaults to [52, 53, 54, ...] up to n_particles.")

    parser.add_argument("--lambda_edge_max", type=float, default=1.0,
                         help="max strength of pairwise edge-probability repulsion. 0 disables it.")
    parser.add_argument("--edge_repulsion_warmup_frac", type=float, default=0.8)
    parser.add_argument("--lambda_node_max", type=float, default=0.0,
                         help="max strength of pairwise node-probability (noisy-OR) repulsion. 0 disables it.")
    parser.add_argument("--node_repulsion_warmup_frac", type=float, default=0.8)

    parser.add_argument("--jaccard_lambda", type=float, default=1.0,
                         help="Frank family t-norm/t-conorm parameter for EVERY pairwise repulsion term "
                              "(edge and node alike). 0.0 = min/max (Zadeh fuzzy min/max), 1.0 = product / "
                              "probabilistic-sum (the ORIGINAL soft_jaccard, default -- identical behavior "
                              "to every run before this flag existed), inf = Lukasiewicz, any other positive "
                              "float = the general Frank interpolation between those three. Unrelated to "
                              "--lambda_edge_max/--lambda_node_max, which scale the WHOLE repulsion term's "
                              "weight in the loss -- this instead changes which fuzzy-set formula that term "
                              "is computed with. Pass 'inf' on the command line for math.inf.")
    parser.add_argument("--repulsion_reduction", default="mean", choices=["mean", "sum"],
                         help="THE FIX for an unnormalized-loss bug: the repulsion term used to be an "
                              "unnormalized SUM over all C(n_particles,2) pairs, so at a fixed "
                              "--lambda_edge_max/--lambda_node_max, each particle's repulsion gradient grew "
                              "LINEARLY in (n_particles-1) while its fidelity gradient did not -- meaning "
                              "the same numeric lambda was silently a much stronger relative penalty at "
                              "n_particles=12 than at n_particles=3. 'mean' (DEFAULT) averages over all "
                              "pairs instead, so lambda_edge_max/lambda_node_max mean the same thing "
                              "regardless of n_particles -- use this for any real run. 'sum' reproduces the "
                              "OLD (buggy) behavior; keep it only to run an explicit before/after ablation "
                              "showing this scaling artifact explains previously-observed large-n or "
                              "large-jaccard_lambda collapses, i.e. run the same sweep at 'sum' vs 'mean' "
                              "and check whether the collapse disappears under 'mean'. lambda_edge_max "
                              "values calibrated under old 'sum'-mode runs will need to be recalibrated "
                              "under 'mean' -- they are not comparable numbers.")

    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr_e", type=float, default=0.02)

    parser.add_argument("--edge_logit_init_mean", type=float, default=10.0)
    parser.add_argument("--edge_logit_init_std", type=float, default=0.01)
    parser.add_argument("--random_mode", default="gumbel_sigmoid", choices=["gumbel_sigmoid", "none"])
    parser.add_argument("--gs_temp_edge", type=float, default=1.0)

    parser.add_argument("--lambda_sparse_e", type=float, default=1.0)
    parser.add_argument("--min_times_lambda_sparse_e", type=float, default=0.01)
    parser.add_argument("--max_times_lambda_sparse_e", type=float, default=1.0)

    parser.add_argument("--lambda_complete_e", type=float, default=0.0)
    parser.add_argument("--completeness_start_frac", type=float, default=0.0)

    parser.add_argument("--snapshot_every", type=int, default=None,
                         help="also call finalize_and_report (boolean circuit + eval + save) every N "
                              "epochs, not just at the end (useful for long runs).")
    parser.add_argument("--epoch_snapshot_every", type=int, default=1,
                         help="save a hand-pickable {acc, density, jaccard, edge_probs} snapshot for "
                              "EVERY particle, every N epochs. 0 disables it. Also controls the cadence "
                              "of the soft_train_acc/soft_test_acc/hard_test_acc report (see "
                              "--skip_snapshot_eval to disable just the accuracy computation).")
    parser.add_argument("--skip_snapshot_eval", action="store_true",
                         help="skip the soft_test_acc / hard_test_acc test-set passes at snapshot time "
                              "(they add one extra full test-set forward per particle, twice, per "
                              "snapshot epoch). Density and soft_train_acc are still reported either way.")
    parser.add_argument("--save_dir", default=None,
                         help="defaults to circuits_discovered/hubert_circuits/{task_type}_lam{jaccard_lambda}/ "
                              "-- jaccard_lambda is baked into the default specifically so sweeping "
                              "--jaccard_lambda across runs can never silently overwrite a previous "
                              "lambda's epoch snapshots at the same epoch numbers. Pass an explicit "
                              "--save_dir to opt out of that.")

    parser.add_argument("--devices", nargs="+", default=None,
                         help="one or more devices (e.g. --devices cuda:0 cuda:1). Particles are "
                              "assigned round-robin across them -- purely mechanical (index modulo "
                              "len(devices)), no awareness of GPU memory headroom. Repeat a device "
                              "entry to skew the split toward it, e.g. --devices cuda:0 cuda:1 cuda:1 "
                              "for a 2-vs-3 split of 5 particles. Defaults to a single auto-picked device.")
    parser.add_argument("--repulsion_device", default=None,
                         help="device where cross-particle edge/node repulsion is combined each step "
                              "(the multi-GPU synchronization point) -- this adds memory pressure on top "
                              "of whatever particles already live there, so if your split is uneven, "
                              "usually best pointed at the LESS-loaded device. Defaults to devices[0].")

    return parser.parse_args()


def circuit_from_edge_probs(edge_probs: torch.Tensor, scratch_particle: Particle) -> Circuit:
    """Binarize this epoch's edge_probs (sigmoid(logit) > 0.5, same threshold
    convention used everywhere else in this codebase) and write the decision
    into `scratch_particle`'s edge_logits -- a particle built once purely as
    scratch space (see build_scratch_particles), NEVER one of the live
    training particles, so this can never perturb an in-progress run. Then
    run the exact same two calls finalize_and_report / compare_circuits.py
    both use to turn that into an evaluable boolean Circuit. Mirrors
    compare_circuits.py's circuit_from_edge_probs exactly (duplicated here
    rather than imported, since that file is a standalone script, not a
    library module)."""
    p = scratch_particle.discogp
    edge_probs = edge_probs.to(device=scratch_particle.device)
    idx = 0
    with torch.no_grad():
        for group in p.masks.edge_logits:
            n = group.numel()
            chunk = edge_probs[idx:idx + n].reshape(group.shape)
            idx += n
            group.data = torch.where(chunk > 0.5, torch.full_like(chunk, 30.0), torch.full_like(chunk, -30.0))
    if idx != edge_probs.numel():
        raise ValueError(
            f"edge_probs has {edge_probs.numel()} entries but this model's edge-logit groups "
            f"total {idx} -- mismatched model_name/task_type between the snapshot and this run?"
        )
    circuit = p.masks.boolean_circuit(use_edges=True, use_weights=False)
    return p.model.finalize_circuit(circuit)


def build_scratch_particles(
    devices: list[str], models_by_device: dict, model_name: str,
) -> dict[str, Particle]:
    """One throwaway particle per unique device, built once before training
    starts, used ONLY as reusable scratch space for circuit_from_edge_probs
    (its edge_logits get overwritten every time it's used -- never read
    beforehand, so reuse across particles/epochs is safe). Kept completely
    separate from the actual training `particles` list so there is no way
    an eval-time binarization can leak into a live particle's state."""
    return {
        device: build_particles(
            seeds=[999], devices=[device], models_by_device=models_by_device,
            discogp_config_kwargs=dict(model_name=model_name, prune_edges=True, prune_weights=False),
        )[0]
        for device in sorted(set(devices))
    }


def soft_evaluate_particle(particle: Particle, test_loader) -> float:
    """'Soft' test accuracy: the SAME stochastic gumbel-sigmoid sampled-mask
    forward pass per_particle_forward uses for training, just run over the
    held-out test set under no_grad instead of a train batch. Directly
    comparable to the per-epoch soft_train_acc, since both come from
    identical sampling -- only the data differs. This is what actually
    tells you whether the circuit ITSELF is broken (soft_test_acc collapses
    too) versus whether only the hard >0.5 threshold is the problem
    (soft_test_acc stays fine, hard_test_acc doesn't -- see module
    docstring). lambda_sparse is irrelevant here since only `logits` (not
    the loss) is used."""
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            _, _, logits = per_particle_forward(particle, batch, lambda_sparse=0.0)
            preds = logits.argmax(dim=-1)
            labels = batch["label"].to(device=logits.device)
            correct += (preds == labels).sum().item()
            total += labels.shape[0]
    return correct / total if total else float("nan")


def save_epoch_snapshot(
    snapshot_dir: Path, *, epoch: int, seed: int, task_type: str,
    train_acc: float, train_loss: float, edge_density: float,
    jaccard_edge: float, 
    jaccard_node: float, 
    jaccard_lambda: float, repulsion_reduction: str,
    edge_probs: torch.Tensor, summary: list[dict],
    soft_train_acc: float | None = None, soft_test_acc: float | None = None,
    hard_test_acc: float | None = None,
) -> None:
    """Same {epoch, train_acc, edge_density, edge_probs, ...} format as
    pilot_hubert.py / run_hubert_sequential.py's save_epoch_snapshot, plus
    this run's overall jaccard_edge/jaccard_node for that epoch (same value
    across every particle's file for a given epoch -- it's a cohort-level
    number -- kept per-file so each snapshot stands alone), the Frank-family
    jaccard_lambda that produced them, repulsion_reduction ("mean", the fix,
    or "sum", the old unnormalized-by-pair-count behavior), AND (new)
    soft_train_acc / soft_test_acc / hard_test_acc -- so a saved circuit is
    fully self-describing about which loss variant discovered it and
    exactly how "healthy" vs "collapsed" it was, without having to track
    that separately against the run's command line or re-derive it later.
    jaccard_edge/jaccard_node here are always the TRUE per-pair average
    regardless of which reduction mode drove the loss (see the
    reduction-aware division right before this is called). The three
    accuracy fields are None whenever --skip_snapshot_eval is passed (or,
    for soft_train_acc, this is just always epoch_correct[i]/epoch_total,
    already available -- kept optional only for signature symmetry)."""
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "task_type": task_type, "seed": seed, "epoch": epoch,
        "train_acc": train_acc, "train_loss": train_loss, "edge_density": edge_density,
        "jaccard_edge": jaccard_edge, 
        "jaccard_node": jaccard_node,
         "jaccard_lambda": jaccard_lambda,
        "repulsion_reduction": repulsion_reduction,
        "soft_train_acc": soft_train_acc, "soft_test_acc": soft_test_acc, "hard_test_acc": hard_test_acc,
        "edge_probs": edge_probs.detach().cpu(),
    }
    torch.save(record, snapshot_dir / f"epoch{epoch:03d}.pt")

    summary.append({k: v for k, v in record.items() if k != "edge_probs"})
    with open(snapshot_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


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
        / f"{args.task_type}_lam{args.jaccard_lambda}_{args.repulsion_reduction}"
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

    # Scratch particles, one per unique device -- ONLY used to rebuild a
    # binarized boolean Circuit from a live particle's edge_probs for
    # hard_test_acc. Never touched by the training loop itself.
    scratch_particles = (
        build_scratch_particles(devices, models_by_device, args.model_name)
        if not args.skip_snapshot_eval else {}
    )

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

    # One epoch-snapshot dir + running summary per particle.
    epoch_snapshot_dirs = [save_dir / f"particle{i}_epoch_snapshots" for i in range(args.n_particles)]
    epoch_snapshot_summaries: list[list[dict]] = [[] for _ in range(args.n_particles)]

    print(
        f"Training {args.n_particles} particles jointly for {n_epochs} epochs "
        f"({len(train_loader)} steps/epoch, {len(node_keys)} nodes / {incidence_by_device[devices[0]].shape[1]} "
        f"edges in the full graph), task={args.task_type} ({num_classes}-way), "
        # f"lambda_edge_max={args.lambda_edge_max}, lambda_node_max={args.lambda_node_max}, "
        f"lambda_edge_max={args.lambda_edge_max}, "
        f"jaccard_lambda={args.jaccard_lambda} (Frank family: 0=min/max, 1=product, inf=Lukasiewicz), "
        f"repulsion_reduction={args.repulsion_reduction}, "
        f"devices={sorted(set(devices))}, repulsion_device={hub_device}..."
    )
    if args.epoch_snapshot_every:
        print(f"Epoch snapshots (all {args.n_particles} particles) every {args.epoch_snapshot_every} "
              f"epoch(s) -> {save_dir}/particle{{i}}_epoch_snapshots/")
        if args.skip_snapshot_eval:
            print("  (--skip_snapshot_eval: soft_test_acc/hard_test_acc will NOT be computed)")
    t0 = time.time()

    for epoch in range(n_epochs):
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

        for batch in train_loader:
            task_losses = []
            edge_probs = []
            # Each particle's forward runs on its own device -- issuing them
            # back to back here lets their CUDA kernels queue and overlap
            # across devices (see multi_particle.py's module docstring).
            for i, (particle, lam_sparse) in enumerate(zip(particles, lambda_sparse_vals)):
                task_loss, probs, logits = per_particle_forward(particle, batch, lambda_sparse=lam_sparse)
                task_losses.append(task_loss)
                edge_probs.append(probs)
                preds = logits.detach().argmax(dim=-1)
                labels = batch["label"].to(device=logits.device)
                epoch_correct[i] += (preds == labels).sum().item()
                # Same convention as pilot_hubert.py / run_hubert_sequential.py:
                # probs is flat_probs(p) = sigmoid(edge_logits), deterministic
                # (no gumbel sampling in it). Threshold at 0.5 == logit>0, same
                # as DiscoGP's own straight-through hard mask.
                epoch_density[i] += (probs.detach() > 0.5).float().mean().item()
            epoch_total += batch["label"].shape[0]

            node_probs = [
                node_probs_from_edge_probs(ep, incidence_by_device[str(ep.device)])
                for ep in edge_probs
            ]
            # Synchronization point: every particle's probs get moved to
            # hub_device inside combine_repulsion before the pairwise terms.
            edge_rep, node_rep = combine_repulsion(
                edge_probs, node_probs, lambda_edge=lambda_edge_rep, lambda_node=lambda_node_rep,
                hub_device=hub_device, jaccard_lambda=args.jaccard_lambda, reduction=args.repulsion_reduction,
            )
            # edge_rep = combine_repulsion(
            #     edge_probs, lambda_edge=lambda_edge_rep,
            #     hub_device=hub_device, jaccard_lambda=args.jaccard_lambda, reduction=args.repulsion_reduction,
            # )

            # task_losses live on their own particle's device; `.to(hub_device)`
            # is differentiable, so summing here still backprops correctly to
            # each particle's own edge_logits on its own device.
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
            # combine_repulsion's "mean" branch now divides by (n_particles-1),
            # not C(n_particles,2) -- rescale back to the true per-pair average
            # for display, matching the "sum" branch's correction above.
            mean_jaccard_edge *= (args.n_particles - 1) / pairs_n
            mean_jaccard_node *= (args.n_particles - 1) / pairs_n
        
        print(
            f"epoch {epoch:3d}  "
            f"loss={[round(l / n_batches, 4) for l in epoch_task_loss]}  "
            f"train_acc={[round(c / epoch_total, 4) for c in epoch_correct]}  "
            f"edge_density={[round(d / n_batches, 4) for d in epoch_density]}  "
            f"jaccard_edge={mean_jaccard_edge:.4f}  "
            f"jaccard_node={mean_jaccard_node:.4f}  "
            f"lambda_edge={lambda_edge_rep:.3f}  lambda_node={lambda_node_rep:.3f}"
            f"lambda_edge={lambda_edge_rep:.3f}"
        )

        if args.epoch_snapshot_every and (epoch + 1) % args.epoch_snapshot_every == 0:
            for i, particle in enumerate(particles):
                # Once per particle per epoch, deterministic -- single source of
                # truth for both this epoch's printed density AND the tensor
                # saved for hand-picking / reuse as a --load_frozen reference.
                # (Recomputed here rather than reusing the last batch's `probs`
                # so the saved value doesn't depend on which batch happened to
                # be last in the epoch.)
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

                save_epoch_snapshot(
                    epoch_snapshot_dirs[i], epoch=epoch + 1, seed=particle.seed, task_type=args.task_type,
                    train_acc=epoch_correct[i] / epoch_total, train_loss=epoch_task_loss[i] / n_batches,
                    edge_density=snapshot_density, jaccard_edge=mean_jaccard_edge, jaccard_node=mean_jaccard_node,
                    jaccard_lambda=args.jaccard_lambda, repulsion_reduction=args.repulsion_reduction,
                    edge_probs=snapshot_edge_probs, summary=epoch_snapshot_summaries[i],
                    soft_train_acc=soft_train_acc, soft_test_acc=soft_test_acc, hard_test_acc=hard_test_acc,
                )
                # save_epoch_snapshot(
                #     epoch_snapshot_dirs[i], epoch=epoch + 1, seed=particle.seed, task_type=args.task_type,
                #     train_acc=epoch_correct[i] / epoch_total, train_loss=epoch_task_loss[i] / n_batches,
                #     edge_density=snapshot_density, jaccard_edge=mean_jaccard_edge,
                #     jaccard_lambda=args.jaccard_lambda, repulsion_reduction=args.repulsion_reduction,
                #     edge_probs=snapshot_edge_probs, summary=epoch_snapshot_summaries[i],
                #     soft_train_acc=soft_train_acc, soft_test_acc=soft_test_acc, hard_test_acc=hard_test_acc,
                # )

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

    elapsed = time.time() - t0
    print(
        f"\nDone: {args.n_particles} {args.task_type} circuits, {n_epochs} epochs, "
        f"{elapsed:.1f}s. Saved under {save_dir}/"
    )
    if args.epoch_snapshot_every:
        for i, d in enumerate(epoch_snapshot_dirs):
            print(f"Particle {i} epoch snapshots: {d}/ (scan {d}/summary.json)")


if __name__ == "__main__":
    main()