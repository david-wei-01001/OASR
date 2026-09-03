#!/usr/bin/env python3
"""
compare_circuits.py

Load any mix of saved circuits -- boolean Circuit .pt files written by
finalize_and_report (from run_hubert.py's joint runs, or
run_hubert_sequential.py's own particleK_final saves), AND/OR raw
{epoch, edge_probs} snapshot .pt files written by pilot_hubert.py /
run_hubert_sequential.py's --epoch_snapshot_every (i.e. your hand-picked
epochs) -- and compute one directly-comparable report across all of them:
per-circuit test accuracy + edge/node counts, every pairwise overlap
(Jaccard etc, via circuit_discovery.circuit.overlap_stats -- the exact same
function finalize_and_report already uses, so numbers here are consistent
with anything you've already seen printed), and n-way mutual/union stats
-- both across everything given, and separately within any named groups you
mark (e.g. "sequential" vs "simultaneous") so the two pipelines' diversity
can be read off directly instead of eyeballed across separate log files.

Both saved formats are handled transparently -- this inspects each file's
keys and only reconstructs a Circuit from raw edge_probs when there isn't
already one saved:

  - {"circuit": Circuit, "evaluation": {...}, ...}   -> used as-is.
  - {"edge_probs": Tensor, "epoch": ..., ...}         -> a Circuit is
      rebuilt by writing that epoch's kept/not-kept decision (same
      threshold convention used everywhere else: sigmoid(logit) > 0.5)
      into a throwaway particle's edge_logits, then calling the same
      p.masks.boolean_circuit() / p.model.finalize_circuit() finalize_and_report
      itself uses -- so a hand-picked epoch is scored exactly the way a
      final circuit would be, not approximated.

Test accuracy for the raw-edge_probs case is computed fresh here (the
epoch snapshot only stored TRAIN accuracy, batch-by-batch during training --
this evaluates the finalized boolean circuit on the held-out test set,
matching what finalize_and_report reports for the joint/simultaneous
circuits, so the two numbers in your comparison table mean the same thing).

Usage:
    python compare_circuits.py --task_type vowel_classification \\
        --circuit sequential:circuits_discovered/.../sequential/particle0_epoch_snapshots/epoch012.pt \\
        --circuit sequential:circuits_discovered/.../sequential/particle0_epoch_snapshots/epoch018.pt \\
        --circuit sequential:circuits_discovered/.../sequential/particle0_epoch_snapshots/epoch027.pt \\
        --circuit simultaneous:circuits_discovered/.../vowel_classification_epoch40_seed_52.pt \\
        --circuit simultaneous:circuits_discovered/.../vowel_classification_epoch40_seed_53.pt \\
        --circuit simultaneous:circuits_discovered/.../vowel_classification_epoch40_seed_54.pt

Each --circuit is "group:path" -- group is an arbitrary label (used for the
within-group n-way stats at the end); use any string, repeat freely.
Writes a full JSON report (every pairwise stat, not just what's printed) to
--out (default: circuit_comparison_report.json).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from circuit_discovery.circuit import Circuit, intersection, overlap_stats, union
from circuit_discovery.run import get_compute_device
from circuit_discovery.tasks.articulatory_index import TASK_SPECS, load_articulatory_dataset
from circuit_discovery.tasks.discovery_setup import (
    build_particles,
    evaluate_circuit_classification,
    load_hubert_classifiers_for_devices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task_type", required=True, choices=sorted(TASK_SPECS))
    parser.add_argument("--model_name", default="hubert-base-ls960",
                         choices=["hubert-base-ls960", "wav2vec2-base-960h", "wav2vec2-base"])
    parser.add_argument("--head_path", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--circuit", dest="circuits", action="append", required=True, metavar="GROUP:PATH",
        help="one saved circuit to load, as 'group:path' (e.g. 'sequential:...epoch012.pt'). "
             "Repeat for every circuit you want in the comparison.",
    )
    parser.add_argument("--out", default="circuit_comparison_report.json")
    return parser.parse_args()


def circuit_from_edge_probs(edge_probs: torch.Tensor, particle) -> Circuit:
    """Write this epoch's kept/not-kept decision into `particle`'s edge_logits
    (reused as scratch space -- boolean_circuit()/finalize_circuit() below
    return an independent snapshot, not a live view, so it's safe to
    overwrite and reuse `particle` for the next circuit in the loop), then
    run the exact same two calls finalize_and_report uses."""
    p = particle.discogp
    edge_probs = edge_probs.to(device=particle.device)
    idx = 0
    with torch.no_grad():
        for group in p.masks.edge_logits:
            n = group.numel()
            chunk = edge_probs[idx:idx + n].reshape(group.shape)
            idx += n
            # push to sigmoid(logit) ~= 1.0 / ~= 0.0 -- same "kept" decision
            # as the >0.5 threshold used everywhere else, expressed as a
            # logit boolean_circuit() can read off.
            group.data = torch.where(chunk > 0.5, torch.full_like(chunk, 30.0), torch.full_like(chunk, -30.0))
    if idx != edge_probs.numel():
        raise ValueError(
            f"edge_probs has {edge_probs.numel()} entries but this model's edge-logit groups "
            f"total {idx} -- was this snapshot saved against a different model_name/task_type?"
        )
    circuit = p.masks.boolean_circuit(use_edges=True, use_weights=False)
    circuit = p.model.finalize_circuit(circuit)
    return circuit


def load_named_circuit(path: str, particle, data) -> dict[str, Any]:
    obj = torch.load(path, map_location=particle.device)
    stem = Path(path).stem

    if "circuit" in obj:
        circuit = obj["circuit"]
        evaluation = obj.get("evaluation") or evaluate_circuit_classification(particle.discogp.model, data.test, circuit)
        source = "saved_circuit"
        train_acc = None
    elif "edge_probs" in obj:
        circuit = circuit_from_edge_probs(obj["edge_probs"], particle)
        evaluation = evaluate_circuit_classification(particle.discogp.model, data.test, circuit)
        source = "edge_probs_snapshot"
        train_acc = obj.get("train_acc")
    else:
        raise ValueError(f"Unrecognized snapshot format at {path}: keys={list(obj.keys())}")

    return {
        "name": stem,
        "path": path,
        "seed": obj.get("seed"),
        "epoch": obj.get("epoch"),
        "source": source,
        "train_acc": train_acc,
        "test_acc": evaluation.get("acc"),
        "kept_edges": circuit.num_kept_edges(),
        "kept_nodes": circuit.num_kept_nodes(),
        "edge_density": circuit.edge_density(),
        "circuit": circuit,
    }


def n_way_stats(entries: list[dict[str, Any]]) -> dict[str, Any]:
    if len(entries) < 2:
        return {}
    everything = entries[0]["circuit"]
    mutual = entries[0]["circuit"]
    for e in entries[1:]:
        everything = union(everything, e["circuit"])
        mutual = intersection(mutual, e["circuit"])
    denom = everything.num_kept_edges()
    return {
        "n": len(entries),
        "names": [e["name"] for e in entries],
        "mutual_edges": mutual.num_kept_edges(),
        "union_edges": denom,
        "edge_jaccard": mutual.num_kept_edges() / denom if denom else None,
        "mutual_nodes": mutual.num_kept_nodes(),
        "union_nodes": everything.num_kept_nodes(),
        "node_jaccard": mutual.num_kept_nodes() / everything.num_kept_nodes() if everything.num_kept_nodes() else None,
    }


def main() -> None:
    args = parse_args()
    device = get_compute_device()

    parsed = []
    for spec in args.circuits:
        if ":" not in spec:
            raise ValueError(f"--circuit expects 'group:path', got: {spec!r}")
        group, path = spec.split(":", 1)
        parsed.append((group, path))

    models_by_device = load_hubert_classifiers_for_devices(
        args.task_type, [device], head_path=args.head_path, model_name=args.model_name,
    )
    data = load_articulatory_dataset(args.task_type, batch_size=args.batch_size)
    # Scratch particle: only its .masks structure and frozen .model are used,
    # for BOTH re-hydrating edge_probs snapshots into Circuits AND (via
    # evaluate_circuit_classification) evaluating every circuit's test acc,
    # loaded ones included. Seed is irrelevant -- masks get fully overwritten
    # (or simply unused) per circuit.
    [particle] = build_particles(
        seeds=[999], devices=[device], models_by_device=models_by_device,
        discogp_config_kwargs=dict(model_name=args.model_name, prune_edges=True, prune_weights=False),
    )

    entries = []
    groups: dict[str, list[dict[str, Any]]] = {}
    print(f"Loading and evaluating {len(parsed)} circuit(s) against the {args.task_type} test set...\n")
    for group, path in parsed:
        entry = load_named_circuit(path, particle, data)
        entry["group"] = group
        entries.append(entry)
        groups.setdefault(group, []).append(entry)
        print(
            f"[{group}] {entry['name']}  test_acc={entry['test_acc']}  "
            f"train_acc={entry['train_acc']}  kept_edges={entry['kept_edges']}  "
            f"kept_nodes={entry['kept_nodes']}  edge_density={entry['edge_density']:.4f}  "
            f"(source={entry['source']})"
        )

    print("\n===== pairwise overlap =====")
    pairwise = []
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            a, b = entries[i], entries[j]
            stats = overlap_stats(a["circuit"], b["circuit"])
            pairwise.append({"a": a["name"], "b": b["name"], "a_group": a["group"], "b_group": b["group"], **stats})
            print(f"\n{a['name']} ({a['group']}) vs {b['name']} ({b['group']}):")
            for k, v in stats.items():
                print(f"  {k}: {v}")

    print("\n===== n-way stats: ALL circuits =====")
    all_nway = n_way_stats(entries)
    for k, v in all_nway.items():
        print(f"  {k}: {v}")

    per_group_nway = {}
    for group, group_entries in groups.items():
        print(f"\n===== n-way stats: group '{group}' only =====")
        stats = n_way_stats(group_entries)
        per_group_nway[group] = stats
        for k, v in stats.items():
            print(f"  {k}: {v}")

    report = {
        "task_type": args.task_type,
        "circuits": [{k: v for k, v in e.items() if k != "circuit"} for e in entries],
        "pairwise": pairwise,
        "n_way_all": all_nway,
        "n_way_per_group": per_group_nway,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report written to {args.out}")


if __name__ == "__main__":
    main()