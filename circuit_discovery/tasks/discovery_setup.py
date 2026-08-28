"""
circuit_discovery/tasks/discovery_setup.py

Reusable pieces for running OASR-style joint multi-particle circuit discovery
-- edge repulsion AND node repulsion, applied simultaneously to N particles
being trained together, not the sequential find-one-then-repel-from-it
pattern -- against a CircuitHubert backbone plus a frozen, already-trained
classification head (see train_classification_head.py), on the Articulatory
Index tasks (consonant_classification / vowel_classification / ...).

This is a library module, not a script. run_hubert.py (repo root) is the CLI
entrypoint that imports from here. Kept separate for two reasons:
  1. the loss/repulsion math and the particle bookkeeping are reusable and
     testable outside argument parsing.
  2. multi-GPU/multi-process dispatch has one place to change -- see
     circuit_discovery/multi_particle.py, which this module now delegates
     the device-agnostic repulsion/dispatch math to.

Multi-GPU: IMPLEMENTED (single-process; see multi_particle.py's module
docstring for the design). Each particle is assigned a device round-robin
over `--devices`; each unique device gets its own frozen model replica
(backbone+head are frozen, so replicating them costs memory, not
correctness); combine_repulsion is the synchronization point where every
particle's edge/node probabilities get moved to one `--repulsion_device`
before the pairwise soft-Jaccard terms.

Nothing here modifies circuit.py, algorithms/*.py, models/*, run.py, or
configs.yaml. metrics.py had one bug fixed (see below) plus a
multi-device-safety fix to evaluate_classification_accuracy; everything else
there is untouched. Existing GPT2/IOI infrastructure and the pilotN.py
scripts are untouched -- see circuit_discovery/ioi_discovery_setup.py for
the GPT2/IOI analogue of this file, which shares multi_particle.py with it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ..algorithms.discogp import DiscoGP, DiscoGPConfig
from ..circuit import Circuit, intersection, overlap_stats, union
from ..metrics import evaluate_classification_accuracy
from ..models import load_circuit_model
from ..multi_particle import (
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
from ..utils import DEVICE, set_seed
from .articulatory_index import TASK_SPECS, load_articulatory_dataset
from .hubert_classifier import CircuitHubertClassifier
from .train_classification_head import DEFAULT_HEAD_CHECKPOINT_DIR

logger = logging.getLogger(__name__)

__all__ = [
    "discogp_fidelity_loss_classification",
    "discogp_completeness_loss_classification",
    "evaluate_circuit_classification",
    "default_head_path",
    "load_hubert_classifier_for_task",
    "load_hubert_classifiers_for_devices",
    # re-exported from multi_particle for anyone importing them from here,
    # matching this module's pre-refactor public surface
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
# classification-task losses
# --------------------------------------------------------------------------------------
# metrics.discogp_fidelity_loss / discogp_completeness_loss hardcode the IOI
# "target good"/"target bad"/"seq_lens" batch shape and index into a
# [batch, seq, vocab] logits tensor -- unusable for CircuitHubertClassifier's
# [batch, num_classes] output. These are the direct N-way analogues, same
# shape of intent as the originals (fidelity: predict the truth; completeness:
# the complement circuit should be uninformative), just generalized.


def discogp_fidelity_loss_classification(batch: dict[str, Any], logits: torch.Tensor) -> torch.Tensor:
    """The circuit's logits should predict the true label."""
    return F.cross_entropy(logits, batch["label"].to(device=logits.device))


def discogp_completeness_loss_classification(
    batch: dict[str, Any],
    logits: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """
    Mirrors discogp_completeness_loss's shape exactly: the complement
    (reverse-masked) circuit should be uninformative, so push it toward a
    uniform soft target over all classes instead of the 0.5/0.5 good/bad
    target used for the 2-way IOI case.
    """
    target = torch.full_like(logits, 1.0 / num_classes)
    return F.cross_entropy(logits, target)


def evaluate_circuit_classification(model: Any, dataloader: Any, circuit: Circuit) -> dict[str, Any]:
    """Same (model, dataloader, circuit) shape as run.py's evaluate_circuit."""
    return evaluate_classification_accuracy(model=model, dataloader=dataloader, circuit=circuit)

# --------------------------------------------------------------------------------------
# model / head loading
# --------------------------------------------------------------------------------------


def default_head_path(task_type: str, head_dir: str | Path | None = None) -> Path:
    head_dir = Path(head_dir) if head_dir is not None else DEFAULT_HEAD_CHECKPOINT_DIR
    return head_dir / f"{task_type}_head.pt"


def load_hubert_classifier_for_task(
    task_type: str,
    *,
    head_path: str | Path | None = None,
    model_name: str = "hubert-base-ls960",
    device: str | None = None,
    backbone: Any = None,  # pass an already-loaded CircuitHubert to reuse it across tasks/particles
) -> CircuitHubertClassifier:
    if task_type not in TASK_SPECS:
        raise ValueError(f"unknown task_type {task_type!r}; expected one of {sorted(TASK_SPECS)}.")

    device = device or DEVICE
    head_path = Path(head_path) if head_path is not None else default_head_path(task_type)
    if not head_path.exists():
        raise FileNotFoundError(
            f"no trained head at {head_path} -- run "
            f"`python -m circuit_discovery.tasks.train_classification_head "
            f"--task_type {task_type}` first."
        )

    backbone = backbone if backbone is not None else load_circuit_model(model_name, device=device)
    classifier = CircuitHubertClassifier.load_head(head_path, backbone, freeze=True, device=device)
    classifier.eval()
    return classifier


def load_hubert_classifiers_for_devices(
    task_type: str,
    devices: list[str],
    *,
    head_path: str | Path | None = None,
    model_name: str = "hubert-base-ls960",
) -> dict[str, CircuitHubertClassifier]:
    """One frozen backbone+head replica per unique device, for multi-GPU
    dispatch. Backbone and head are frozen (no gradients ever flow into
    them, see CircuitHubertClassifier.freeze_all) -- replicating them across
    devices is a memory cost only, never a training-correctness concern.
    Each replica is loaded independently via load_hubert_classifier_for_task
    (rather than e.g. deep-copying one CPU instance) since that's the
    simplest way to get correct device placement for every buffer/parameter
    without hunting down anything the deep-copy path might miss."""
    return {
        device: load_hubert_classifier_for_task(
            task_type, head_path=head_path, model_name=model_name, device=device,
        )
        for device in dict.fromkeys(devices)  # de-duplicate, keep first-seen order
    }

# --------------------------------------------------------------------------------------
# particle bookkeeping
# --------------------------------------------------------------------------------------
# Multi-GPU dispatch: IMPLEMENTED. See circuit_discovery/multi_particle.py's
# module docstring for the full design; short version:
#
#   per_particle_forward     -- runs entirely on particle.device. Independent
#                                across particles -- CUDA ops on different
#                                devices queue and overlap without any
#                                explicit parallelism construct needed.
#   combine_repulsion        -- the synchronization point (in
#                                multi_particle.py): every particle's
#                                edge/node probs get moved to one
#                                `hub_device` first.
#   per_particle_backward_step -- independent again, one optimizer step per
#                                particle, on that particle's own device.


@dataclass
class Particle:
    seed: int
    device: str
    model: CircuitHubertClassifier
    discogp: DiscoGP
    optimizer: torch.optim.Optimizer = field(init=False)

    def __post_init__(self) -> None:
        self.optimizer = self.discogp._optimizer("edge")


def build_particles(
    *,
    seeds: list[int],
    devices: list[str],
    models_by_device: dict[str, CircuitHubertClassifier],
    discogp_config_kwargs: dict[str, Any],
) -> list[Particle]:
    """
    Each particle is assigned a device round-robin over `devices` (see
    multi_particle.assign_devices) and uses that device's frozen model
    replica from `models_by_device` (see load_hubert_classifiers_for_devices).
    Every particle still gets its own DiscoGP / DiscoGPMasks -- i.e. its own
    edge_logits nn.Parameters and its own optimizer -- only the frozen
    backbone+head is shared *within* a device across the particles placed
    there.
    """
    particle_devices = assign_devices(len(seeds), devices)
    missing = {d for d in particle_devices if d not in models_by_device}
    if missing:
        raise ValueError(
            f"models_by_device is missing replicas for device(s) {sorted(missing)}; "
            f"pass every entry of `devices` through load_hubert_classifiers_for_devices."
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
    particle.device. Batches arrive pinned to one fixed device (dataset-load
    time), so every call moves its own copy onto particle.device first --
    this is also where the old `device` NameError bug lived (it referenced
    an undefined bare name instead of particle.device)."""
    p = particle.discogp
    batch = move_batch_to_device(batch, particle.device)
    sparsity = p._sparsity_loss("edge")
    runtime_masks = p._sampled_runtime_masks_for_mode("edge")
    logits = p.model(batch["input_ids"], runtime_masks=runtime_masks, lengths=batch["length"])
    fidelity = discogp_fidelity_loss_classification(batch, logits)
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
    num_classes: int,
) -> None:
    p = particle.discogp
    batch = move_batch_to_device(batch, particle.device)
    reverse_masks = p._sampled_runtime_masks_for_mode("edge", reverse=True)
    reverse_logits = p.model(batch["input_ids"], runtime_masks=reverse_masks, lengths=batch["length"])
    completeness = lambda_complete * discogp_completeness_loss_classification(batch, reverse_logits, num_classes)
    completeness.backward()
    particle.optimizer.step()
    particle.optimizer.zero_grad(set_to_none=True)

# --------------------------------------------------------------------------------------
# snapshotting / reporting
# --------------------------------------------------------------------------------------


def finalize_and_report(
    *,
    tag: str,
    task_type: str,
    particles: list[Particle],
    data: Any,
    save_dir: Path,
) -> dict[str, Circuit]:
    circuits: dict[str, Circuit] = {}
    print(f"\n########## Snapshot: {tag} ({task_type}) ##########")
    save_dir.mkdir(parents=True, exist_ok=True)

    for particle in particles:
        p = particle.discogp
        circuit = p.masks.boolean_circuit(use_edges=True, use_weights=False)
        circuit = p.model.finalize_circuit(circuit)
        # evaluate_circuit_classification -> evaluate_classification_accuracy now
        # moves its own batches to p.model's device internally, so this is
        # multi-GPU safe even though `data.test` is one shared dataloader.
        evaluation = evaluate_circuit_classification(p.model, data.test, circuit)
        name = f"{tag}_seed_{particle.seed}"
        circuits[name] = circuit
        torch.save(
            {"task_type": task_type, "seed": particle.seed, "circuit": circuit, "evaluation": evaluation},
            save_dir / f"{task_type}_{name}.pt",
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
