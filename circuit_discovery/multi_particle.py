"""
circuit_discovery/multi_particle.py

Architecture-agnostic pieces shared by every OASR-style joint multi-particle
circuit discovery script (HuBERT's discovery_setup.py, GPT2/IOI's
ioi_discovery_setup.py, and anything added later): the edge/node repulsion
math, and the single-process multi-GPU dispatch primitives.

None of this file touches model internals -- everything here only cares
about (a) plain tensors of edge/node probabilities and (b) torch.device
placement, so the exact same functions serve HuBERT's classification task
and GPT2's IOI task without change.

--------------------------------------------------------------------------
Multi-GPU design (implemented in this file; single-process, no torchrun /
multiprocessing needed)
--------------------------------------------------------------------------
Each particle's frozen model replica + its own DiscoGPMasks/edge_logits live
entirely on one device (`Particle.device`). A joint training step still
decomposes into the same three phases as the single-GPU version:

    per_particle_forward       -- runs entirely on particle.device.
                                   Embarrassingly parallel across particles:
                                   nothing here reads another particle's
                                   tensors, so on distinct devices these
                                   forward passes overlap for free -- CUDA
                                   ops are queued asynchronously per device,
                                   so issuing particle 0's forward on cuda:0
                                   and particle 1's forward on cuda:1 back
                                   to back does not block particle 0's GPU
                                   work on particle 1's.
    combine_repulsion          -- THE synchronization point. Every
                                   particle's edge_probs/node_probs get
                                   moved (via a differentiable `.to()`) to
                                   one `hub_device` before the pairwise
                                   soft-Jaccard terms are computed, because
                                   elementwise ops across tensors on
                                   different devices raise in PyTorch.
                                   `.to()` between devices is an autograd-
                                   tracked op, so calling `.backward()` on
                                   the resulting joint loss still correctly
                                   routes gradients back through the device
                                   copy to each particle's own edge_logits
                                   on its own device -- no manual gradient
                                   shuffling required.
    per_particle_backward_step -- each particle's optimizer only ever
                                   touches parameters on that particle's own
                                   device, so this stays independent.

What this does NOT implement: true concurrent execution (e.g. via
multiprocessing / torch.distributed, one process per GPU). That would let
particles' *Python-side* bookkeeping (building the sampled runtime masks,
etc.) run concurrently too, not just their CUDA kernels. For this repo's
scale (a handful of particles, a HuBERT-base or GPT2-small backbone) the
async-CUDA-queueing behavior described above already captures most of the
benefit of spreading particles across GPUs, at a fraction of the complexity
of a true multi-process rewrite. If particle count or model size grows
enough that Python-side overhead (not GPU compute) becomes the bottleneck,
that's the point to revisit and go multi-process.
"""

from __future__ import annotations

import itertools
from typing import Any

import torch

__all__ = [
    "flat_probs",
    "soft_jaccard",
    "build_node_incidence",
    "build_node_incidence_for_devices",
    "node_probs_from_edge_probs",
    "ramp_schedule",
    "assign_devices",
    "move_batch_to_device",
    "combine_repulsion",
]

# --------------------------------------------------------------------------------------
# repulsion math (unchanged from pilot2b.py / the original discovery_setup.py port --
# only touches plain tensors, never model internals)
# --------------------------------------------------------------------------------------


def flat_probs(runner: Any) -> torch.Tensor:
    """runner: a DiscoGP instance. Flattens sigmoid(edge_logits) across all
    logit groups into one 1-D tensor, on that DiscoGP's own device."""
    return torch.cat([torch.sigmoid(p).reshape(-1) for p in runner.masks.edge_logits])


def soft_jaccard(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """p and q must already be on the same device -- callers (combine_repulsion)
    are responsible for that; this function does not move tensors itself so
    it stays a pure, trivially-testable elementwise op."""
    inter = (p * q).sum()
    union_ = (p + q - p * q).sum()
    return inter / (union_ + eps)


def build_node_incidence(masks: Any, device: str) -> tuple[torch.Tensor, list]:
    """(n_nodes, n_edges) 0/1 matrix; row i, col j = 1 if edge j touches node i.
    Single-device version -- use build_node_incidence_for_devices when
    particles are spread across more than one device."""
    all_edge_keys = [key for keys in masks.edge_logit_keys for key in keys]
    node_keys = sorted({n for pair in all_edge_keys for n in pair})
    node_index = {n: i for i, n in enumerate(node_keys)}
    incidence = torch.zeros(len(node_keys), len(all_edge_keys), device=device)
    for e_idx, (dst_key, src_key) in enumerate(all_edge_keys):
        incidence[node_index[dst_key], e_idx] = 1.0
        incidence[node_index[src_key], e_idx] = 1.0
    return incidence, node_keys


def build_node_incidence_for_devices(
    masks: Any, devices: list[str],
) -> tuple[dict[str, torch.Tensor], list]:
    """Build the incidence matrix once (on CPU), then place one copy on every
    unique device a particle might run on. Each particle needs an
    incidence matrix on its OWN device -- node_probs_from_edge_probs does
    `incidence @ log1m`, which errors if incidence and the edge probs it's
    being multiplied against live on different devices."""
    incidence_cpu, node_keys = build_node_incidence(masks, device="cpu")
    incidence_by_device = {device: incidence_cpu.to(device) for device in dict.fromkeys(devices)}
    return incidence_by_device, node_keys


def node_probs_from_edge_probs(edge_probs: torch.Tensor, incidence: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Noisy-OR / probabilistic union: p_node = 1 - prod(1 - p_e) over
    incident edges, computed in log-space for stability. `incidence` must
    already be on edge_probs.device (see build_node_incidence_for_devices)."""
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
# device dispatch
# --------------------------------------------------------------------------------------


def assign_devices(n_particles: int, devices: list[str]) -> list[str]:
    """Round-robin assignment of particles to devices, e.g. 5 particles over
    ["cuda:0", "cuda:1"] -> [cuda:0, cuda:1, cuda:0, cuda:1, cuda:0]. With one
    device this trivially puts every particle on it, matching the old
    single-GPU behavior exactly."""
    if not devices:
        raise ValueError("`devices` must be a non-empty list.")
    return [devices[i % len(devices)] for i in range(n_particles)]


def move_batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    """Batches are pinned to one fixed device at dataset-load time
    (`.with_format("torch", ..., device=...)`), not per-particle -- so every
    particle needs its own-device copy of each batch before its forward
    pass. Non-tensor batch entries (there usually aren't any, but just in
    case) pass through unchanged."""
    return {
        key: (value.to(device=device) if isinstance(value, torch.Tensor) else value)
        for key, value in batch.items()
    }


def combine_repulsion(
    edge_probs: list[torch.Tensor],
    node_probs: list[torch.Tensor],
    *,
    lambda_edge: float,
    lambda_node: float,
    hub_device: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum of pairwise soft-Jaccard over all particle pairs, for edges and
    for nodes. This IS the multi-GPU synchronization point (see module
    docstring): every edge_probs[i]/node_probs[i] is moved to `hub_device`
    (default: edge_probs[0]'s device, i.e. single-GPU behavior when every
    particle already lives on the same device) before the pairwise ops,
    since soft_jaccard requires both arguments on one device. Those `.to()`
    moves are differentiable, so `joint_loss.backward()` downstream still
    correctly propagates gradients back to each particle's own edge_logits
    on its own device.

    Returns zero tensors (not skipped) when a lambda is 0, so callers can
    log the raw repulsion value even with that term switched off."""
    hub_device = hub_device or str(edge_probs[0].device)
    edge_probs = [p.to(device=hub_device) for p in edge_probs]
    node_probs = [p.to(device=hub_device) for p in node_probs]

    pairs = list(itertools.combinations(range(len(edge_probs)), 2))
    edge_rep = sum(
        (soft_jaccard(edge_probs[i], edge_probs[j]) for i, j in pairs),
        torch.zeros((), device=hub_device),
    )
    node_rep = sum(
        (soft_jaccard(node_probs[i], node_probs[j]) for i, j in pairs),
        torch.zeros((), device=hub_device),
    )
    return edge_rep, node_rep
