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
  2. future multi-GPU/multi-process dispatch has one place to change --
     see the note above `Particle` below -- rather than being tangled into
     a CLI script.

Nothing here modifies circuit.py, algorithms/*.py, models/*, run.py,
metrics.py, or configs.yaml. Existing GPT2/IOI infrastructure, and the
existing pilotN.py scripts, are untouched.
"""

from __future__ import annotations

import itertools
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
    "flat_probs",
    "soft_jaccard",
    "build_node_incidence",
    "node_probs_from_edge_probs",
    "ramp_schedule",
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

# --------------------------------------------------------------------------------------
# repulsion math
# --------------------------------------------------------------------------------------
# Architecture-agnostic: only touches masks.edge_logits / masks.edge_logit_keys,
# which DiscoGPMasks populates via CircuitModel.edge_logit_group_specs -- this
# is identical code to pilot2b.py's, unchanged, because CircuitHubertClassifier
# delegates edge_logit_group_specs straight to the CircuitHubert backbone.


def flat_probs(runner: DiscoGP) -> torch.Tensor:
    return torch.cat([torch.sigmoid(p).reshape(-1) for p in runner.masks.edge_logits])


def soft_jaccard(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    inter = (p * q).sum()
    union_ = (p + q - p * q).sum()
    return inter / (union_ + eps)


def build_node_incidence(masks: Any, device: str) -> tuple[torch.Tensor, list]:
    """(n_nodes, n_edges) 0/1 matrix; row i, col j = 1 if edge j touches node i.
    Built once from mask structure (identical across particles, since every
    particle is instantiated from the same frozen model + base circuit)."""
    all_edge_keys = [key for keys in masks.edge_logit_keys for key in keys]
    node_keys = sorted({n for pair in all_edge_keys for n in pair})
    node_index = {n: i for i, n in enumerate(node_keys)}
    incidence = torch.zeros(len(node_keys), len(all_edge_keys), device=device)
    for e_idx, (dst_key, src_key) in enumerate(all_edge_keys):
        incidence[node_index[dst_key], e_idx] = 1.0
        incidence[node_index[src_key], e_idx] = 1.0
    return incidence, node_keys


def node_probs_from_edge_probs(edge_probs: torch.Tensor, incidence: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Noisy-OR / probabilistic union: p_node = 1 - prod(1 - p_e) over
    incident edges, computed in log-space for stability."""
    log1m = torch.log1p(-edge_probs.clamp(max=1 - eps))
    node_log1m_sum = incidence @ log1m
    return 1 - torch.exp(node_log1m_sum)


def ramp_schedule(epoch: int, n_epochs: int, warmup_frac: float, lambda_max: float) -> float:
    """Linear ramp 0 -> lambda_max over the first warmup_frac of training,
    then held constant. lambda_max=0.0 is how a repulsion term gets turned
    off entirely -- there's deliberately no separate on/off flag."""
    if lambda_max == 0.0:
        return 0.0
    warmup_epochs = max(1, int(warmup_frac * n_epochs))
    if epoch >= warmup_epochs:
        return lambda_max
    return lambda_max * (epoch / warmup_epochs)

# --------------------------------------------------------------------------------------
# particle bookkeeping
# --------------------------------------------------------------------------------------
# Not parallel yet -- this is the seam left for it.
#
# Training one joint step decomposes into three phases, kept as separate
# functions specifically so a future dispatcher can change *how* each phase
# runs without touching the loss/repulsion math itself:
#
#   per_particle_forward     -- independent per particle, embarrassingly
#                                parallel (one particle's forward+loss
#                                doesn't depend on any other's).
#   combine_repulsion        -- needs every particle's edge/node probs at
#                                once. This is the natural synchronization /
#                                all-gather boundary for a future
#                                multi-process or multi-GPU implementation.
#   per_particle_backward_step -- independent again, one optimizer step per
#                                particle.
#
# `Particle.device` already exists so a per-particle device assignment is
# threaded through today, even though every particle actually runs on the
# same device for now (see build_particles). Two things worth knowing before
# actually parallelizing this, so they don't surprise whoever picks it up:
#   1. Single-process multi-GPU (each particle's parameters/buffers placed on
#      a different `cuda:i`, still one Python process) is a smaller step than
#      it looks -- PyTorch already dispatches ops to whatever device a
#      tensor lives on, so this doesn't strictly require multiprocessing.
#      combine_repulsion's soft_jaccard calls would just need every
#      edge_probs[i]/node_probs[i] moved to one common device first, since
#      elementwise ops across tensors on different devices error.
#   2. Batches are currently pinned to `utils.DEVICE` at dataset-load time
#      (`load_articulatory_dataset` -> `.with_format("torch", ...,
#      device=DEVICE)`), not per-`--devices` entry -- true per-GPU particles
#      will need their own `.to(particle.device)` copy of each batch, or a
#      per-device dataloader.


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
    model: CircuitHubertClassifier,
    discogp_config_kwargs: dict[str, Any],
) -> list[Particle]:
    """
    All particles currently share one `model` instance/device: backbone and
    head are frozen, so this is safe -- only each particle's own
    masks.edge_logits differ. `devices` is accepted and validated here so a
    future per-device model-replica change is localized to this function.
    """
    if len(set(devices)) > 1:
        logger.warning(
            "multiple devices requested (%s) but multi-device dispatch isn't "
            "implemented yet -- running every particle on %s for now.",
            devices, devices[0],
        )

    particles: list[Particle] = []
    for seed in seeds:
        set_seed(seed)
        cfg = DiscoGPConfig(**discogp_config_kwargs)
        discogp = DiscoGP(model=model, config=cfg, device=devices[0])
        particles.append(Particle(seed=seed, device=devices[0], model=model, discogp=discogp))
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
    """Returns (task_loss, edge_probs) for one particle on one batch."""
    p = particle.discogp
    sparsity = p._sparsity_loss("edge")
    runtime_masks = p._sampled_runtime_masks_for_mode("edge")
    logits = p.model(batch["input_ids"], runtime_masks=runtime_masks, lengths=batch["length"].to(device=device))
    fidelity = discogp_fidelity_loss_classification(batch, logits)
    task_loss = fidelity + lambda_sparse * sparsity
    return task_loss, flat_probs(p)


def combine_repulsion(
    edge_probs: list[torch.Tensor],
    node_probs: list[torch.Tensor],
    *,
    lambda_edge: float,
    lambda_node: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum of pairwise soft-Jaccard over all particle pairs, for edges and
    for nodes. Returns zero tensors (not skipped) when a lambda is 0, so
    callers can log the raw repulsion value even with that term switched off."""
    pairs = list(itertools.combinations(range(len(edge_probs)), 2))
    device = edge_probs[0].device

    edge_rep = sum((soft_jaccard(edge_probs[i], edge_probs[j]) for i, j in pairs), torch.zeros((), device=device))
    node_rep = sum((soft_jaccard(node_probs[i], node_probs[j]) for i, j in pairs), torch.zeros((), device=device))
    return edge_rep, node_rep


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
    reverse_masks = p._sampled_runtime_masks_for_mode("edge", reverse=True)
    reverse_logits = p.model(batch["input_ids"], runtime_masks=reverse_masks, lengths=batch["length"].to(device=device))
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
        evaluation = evaluate_circuit_classification(p.model, data.test, circuit)
        name = f"{tag}_seed_{particle.seed}"
        circuits[name] = circuit
        torch.save(
            {"task_type": task_type, "seed": particle.seed, "circuit": circuit, "evaluation": evaluation},
            save_dir / f"{task_type}_{name}.pt",
        )
        print(
            f"{name}: acc={evaluation.get('acc')}, "
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
