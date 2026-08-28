"""
circuit_discovery/ioi_discovery_setup.py

Reusable pieces for running OASR-style joint multi-particle circuit discovery
-- edge repulsion AND node repulsion, applied simultaneously to N particles
being trained together, not the sequential find-one-then-repel-from-it
pattern of pilot1.py/pilot2.py -- against a CircuitGPT backbone on the IOI
task.

This is the direct GPT2/IOI analogue of circuit_discovery/tasks/discovery_setup.py
(HuBERT). The two modules deliberately mirror each other function-for-function
(Particle, build_particles, per_particle_forward, per_particle_backward_step,
run_completeness_step, finalize_and_report) and both delegate the
architecture-agnostic repulsion math and multi-GPU dispatch to
circuit_discovery/multi_particle.py, so a fix or improvement to the
repulsion/device-dispatch logic benefits both tasks at once.

Differences from discovery_setup.py, all inherited directly from IOI's own
shape rather than invented here:
  - No separately-trained classification head to load: metrics.discogp_fidelity_loss
    / discogp_completeness_loss operate directly on next-token logits against
    a "target good" / "target bad" pair, using the existing IOI dataset
    pipeline (utils.load_task_dataset) rather than a custom Articulatory
    Index loader.
  - evaluate_good_bad_accuracy (not evaluate_classification_accuracy) is the
    IOI-side eval metric.

This is a library module, not a script. run_gpt2_ioi.py (repo root) is the
CLI entrypoint that imports from here -- see that file's docstring for the
multi-GPU design, shared with run_hubert.py.

Nothing here modifies circuit.py, algorithms/*.py, models/*, or configs.yaml.
metrics.py's evaluate_good_bad_accuracy had a multi-device-safety fix (moving
each batch to the model's own device before its forward pass, since batches
are pinned to one fixed device at dataset-load time); nothing else there
changed. The existing pilotN.py notebooks/scripts, if any, are untouched --
this module supersedes their *logic* for joint multi-particle discovery but
doesn't modify them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .algorithms.discogp import DiscoGP, DiscoGPConfig
from .circuit import Circuit, intersection, overlap_stats, union
from .metrics import discogp_completeness_loss, discogp_fidelity_loss, evaluate_good_bad_accuracy
from .models import load_circuit_model
from .multi_particle import (
    assign_devices,
    build_node_incidence,
    build_node_incidence_for_devices,
    combine_repulsion,
    flat_probs,
    move_batch_to_device,
    node_probs_from_edge_probs,
    ramp_schedule,
    soft_jaccard,
)
from .utils import set_seed

logger = logging.getLogger(__name__)

__all__ = [
    "evaluate_circuit_ioi",
    "load_gpt2_for_devices",
    # re-exported from multi_particle, matching discovery_setup.py's surface
    "flat_probs",
    "soft_jaccard",
    "build_node_incidence",
    "build_node_incidence_for_devices",
    "node_probs_from_edge_probs",
    "ramp_schedule",
    "assign_devices",
    "move_batch_to_device",
    "Particle",
    "build_particles",
    "per_particle_forward",
    "combine_repulsion",
    "per_particle_backward_step",
    "run_completeness_step",
    "finalize_and_report",
]

# --------------------------------------------------------------------------------------
# model loading
# --------------------------------------------------------------------------------------


def evaluate_circuit_ioi(model: Any, dataloader: Any, circuit: Circuit) -> dict[str, Any]:
    """Same (model, dataloader, circuit) shape as run.py's evaluate_circuit /
    discovery_setup.py's evaluate_circuit_classification."""
    return evaluate_good_bad_accuracy(model=model, dataloader=dataloader, circuit=circuit)


def load_gpt2_for_devices(
    model_name: str,
    devices: list[str],
) -> dict[str, Any]:
    """One frozen CircuitGPT replica per unique device, for multi-GPU
    dispatch. CircuitGPT freezes its own parameters at load time (no
    separately-trained head to worry about, unlike CircuitHubertClassifier),
    so -- exactly as on the HuBERT side -- replicating it across devices is a
    memory cost only, never a training-correctness concern."""
    return {
        device: load_circuit_model(model_name, device=device)
        for device in dict.fromkeys(devices)  # de-duplicate, keep first-seen order
    }

# --------------------------------------------------------------------------------------
# particle bookkeeping
# --------------------------------------------------------------------------------------
# Mirrors discovery_setup.py's Particle / build_particles exactly, just with
# CircuitGPT instead of CircuitHubertClassifier as the frozen model type.
# See multi_particle.py's module docstring for the multi-GPU design.


@dataclass
class Particle:
    seed: int
    device: str
    model: Any  # CircuitGPT
    discogp: DiscoGP
    optimizer: torch.optim.Optimizer = field(init=False)

    def __post_init__(self) -> None:
        self.optimizer = self.discogp._optimizer("edge")


def build_particles(
    *,
    seeds: list[int],
    devices: list[str],
    models_by_device: dict[str, Any],
    discogp_config_kwargs: dict[str, Any],
) -> list[Particle]:
    """Each particle is assigned a device round-robin over `devices` and
    uses that device's frozen CircuitGPT replica from `models_by_device`
    (see load_gpt2_for_devices). Every particle still gets its own DiscoGP /
    DiscoGPMasks -- its own edge_logits nn.Parameters and its own optimizer
    -- only the frozen backbone is shared *within* a device."""
    particle_devices = assign_devices(len(seeds), devices)
    missing = {d for d in particle_devices if d not in models_by_device}
    if missing:
        raise ValueError(
            f"models_by_device is missing replicas for device(s) {sorted(missing)}; "
            f"pass every entry of `devices` through load_gpt2_for_devices."
        )

    particles: list[Particle] = []
    for seed, device in zip(seeds, particle_devices):
        set_seed(seed)
        model = models_by_device[device]
        cfg = DiscoGPConfig(**discogp_config_kwargs)
        discogp = DiscoGP(model=model, config=cfg, device=device)
        particles.append(Particle(seed=seed, device=device, model=model, discogp=discogp))
    return particles

# --------------------------------------------------------------------------------------
# one joint training step
# --------------------------------------------------------------------------------------


def per_particle_forward(
    particle: Particle,
    batch: dict[str, Any],
    *,
    lambda_sparse: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (task_loss, edge_probs) for one particle on one batch, both on
    particle.device. IOI batches (input_ids, seq_lens, target good, target
    bad) are pinned to one fixed device at dataset-load time, so every call
    moves its own copy onto particle.device first -- discogp_fidelity_loss
    indexes logits with seq_lens/target good/target bad via advanced
    indexing, which requires every one of those tensors on logits.device."""
    p = particle.discogp
    batch = move_batch_to_device(batch, particle.device)
    sparsity = p._sparsity_loss("edge")
    runtime_masks = p._sampled_runtime_masks_for_mode("edge")
    logits = p.model(batch["input_ids"], runtime_masks=runtime_masks)
    fidelity = discogp_fidelity_loss(batch, logits)
    task_loss = fidelity + lambda_sparse * sparsity
    return task_loss, flat_probs(p)


def per_particle_backward_step(particle: Particle) -> None:
    particle.optimizer.step()
    particle.optimizer.zero_grad(set_to_none=True)


def run_completeness_step(
    particle: Particle,
    batch: dict[str, Any],
    *,
    lambda_complete: float,
) -> None:
    p = particle.discogp
    batch = move_batch_to_device(batch, particle.device)
    reverse_masks = p._sampled_runtime_masks_for_mode("edge", reverse=True)
    reverse_logits = p.model(batch["input_ids"], runtime_masks=reverse_masks)
    completeness = lambda_complete * discogp_completeness_loss(batch, reverse_logits)
    completeness.backward()
    particle.optimizer.step()
    particle.optimizer.zero_grad(set_to_none=True)

# --------------------------------------------------------------------------------------
# snapshotting / reporting
# --------------------------------------------------------------------------------------


def finalize_and_report(
    *,
    tag: str,
    particles: list[Particle],
    data: Any,
    save_dir: Path,
) -> dict[str, Circuit]:
    circuits: dict[str, Circuit] = {}
    print(f"\n########## Snapshot: {tag} (ioi) ##########")
    save_dir.mkdir(parents=True, exist_ok=True)

    for particle in particles:
        p = particle.discogp
        circuit = p.masks.boolean_circuit(use_edges=True, use_weights=False)
        circuit = p.model.finalize_circuit(circuit)
        # evaluate_circuit_ioi -> evaluate_good_bad_accuracy now moves its
        # own batches to p.model's device internally, so this is multi-GPU
        # safe even though `data.test` is one shared dataloader.
        evaluation = evaluate_circuit_ioi(p.model, data.test, circuit)
        name = f"{tag}_seed_{particle.seed}"
        circuits[name] = circuit
        torch.save(
            {"task": "ioi", "seed": particle.seed, "circuit": circuit, "evaluation": evaluation},
            save_dir / f"ioi_{name}.pt",
        )
        print(
            f"{name} (device={particle.device}): acc={evaluation.get('acc')}, "
            f"kept_edges={circuit.num_kept_edges()}, "
            f"edge_density={circuit.edge_density():.4f}"
        )

    names = list(circuits)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            print(f"\n{a} vs {b}:")
            for k, v in overlap_stats(circuits[a], circuits[b]).items():
                print(f"  {k}: {v}")

    if len(names) >= 2:
        everything = circuits[names[0]]
        mutual = circuits[names[0]]
        for name in names[1:]:
            everything = union(everything, circuits[name])
            mutual = intersection(mutual, circuits[name])
        denom = everything.num_kept_edges()
        print(
            f"\n{len(names)}-way mutual edges: {mutual.num_kept_edges()}, "
            f"union edges: {denom}, "
            f"{len(names)}-way edge Jaccard: {mutual.num_kept_edges() / denom if denom else None}"
        )

    return circuits
