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
import math
from typing import Any

import torch

__all__ = [
    "flat_probs",
    "soft_jaccard",
    "frank_soft_jaccard",
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


import math
from typing import Union

import torch


def soft_jaccard(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """p and q must already be on the same device -- callers (combine_repulsion)
    are responsible for that; this function does not move tensors itself so
    it stays a pure, trivially-testable elementwise op.

    Unchanged, kept for backward compatibility with any existing caller/import
    (e.g. run_hubert_sequential.py's --load_frozen repulsion). This is exactly
    frank_soft_jaccard(p, q, lam=1.0) below -- the Frank family's product
    t-norm / probabilistic-sum t-conorm -- just without the lam-dispatch
    overhead when a caller never needs anything but the default."""
    inter = (p * q).sum()
    union_ = (p + q - p * q).sum()
    return inter / (union_ + eps)


def _frank_t_general(p: torch.Tensor, q: torch.Tensor, u: Union[float, torch.Tensor]) -> torch.Tensor:
    """T_lam(p,q) via the general Frank formula, parameterized by u = ln(lam).
    `u` may be a plain Python float (fixed-lambda sweep case) OR a 0-d torch
    tensor with requires_grad=True (trainable case) -- ordinary Python
    arithmetic (`u * p`, etc.) on a tensor is tracked by autograd exactly
    like an explicit torch op, so this one implementation serves both.

    Uses expm1 throughout instead of naive `lam**x - 1` / `lam - 1`, so this
    stays numerically accurate arbitrarily close to u=0 (lam=1) -- expm1(up),
    expm1(uq), and expm1(u) are all O(u) and computed WITHOUT cancellation,
    so their ratio stays accurate down to very small |u|. This means (unlike
    the old naive-power formula) NO Taylor-series patch is needed for a
    neighborhood of u=0 -- only the exact point u=0 is unsafe (division by
    exact zero); frank_soft_jaccard below special-cases that separately with
    the exact limit t = p*q, never calling this function there.
    """
    lam_minus_1 = torch.expm1(u) if isinstance(u, torch.Tensor) else math.expm1(u)
    lam_pow_p_minus_1 = torch.expm1(u * p)
    lam_pow_q_minus_1 = torch.expm1(u * q)
    t_arg = lam_pow_p_minus_1 * lam_pow_q_minus_1 / lam_minus_1
    t = torch.log1p(t_arg) / u
    return t.clamp(0.0, 1.0)  # float-precision insurance only; analytically already in [0,1]


def frank_soft_jaccard(
    p: torch.Tensor, q: torch.Tensor, lam: Union[float, torch.Tensor] = 1.0, eps: float = 1e-8,
) -> torch.Tensor:
    """Generalized soft Jaccard: same aggregate sum(intersection)/sum(union)
    pattern as soft_jaccard above (a fuzzy/weighted Jaccard, i.e. Ruzicka
    similarity -- NOT an average of per-element Jaccards), but built from the
    Frank family of t-norms/t-conorms instead of always using the algebraic
    product/probabilistic-sum pair. `lam` (traditionally called the Frank
    family's own lambda parameter -- unrelated to lambda_edge/lambda_node,
    the OUTER coefficients that scale this whole term in the loss) selects
    which pair:

        lam == 0.0        -> min/max (Zadeh's original fuzzy min/max),
                              exact closed form, not a finite-lambda
                              approximation.
        lam == 1.0         -> algebraic product / probabilistic sum,
                              i.e. exactly soft_jaccard() above (same
                              formula, bit-for-bit).
        math.isinf(lam)    -> Lukasiewicz t-norm/t-conorm, exact closed
                              form.
        any other lam > 0  -> the general Frank formula (via expm1/log1p,
                              _frank_t_general above), continuously
                              interpolating between the three landmarks
                              above (lam->0 recovers min/max, lam->1
                              recovers product, lam->inf recovers
                              Lukasiewicz). Only the exact point lam==1.0
                              is special-cased separately (division by
                              exact zero in u=ln(lam)); expm1 keeps the
                              general formula numerically accurate for
                              every other lam, including values extremely
                              close to (but not exactly at) 1.

    `lam` may be a plain Python float (fixed sweep, the old behavior,
    unchanged) OR a 0-d torch.Tensor with requires_grad=True, if you want
    the Frank-family parameter itself to be trainable. Branching below on
    its concrete value (to pick which of the four cases above applies)
    uses `.item()` purely for control flow -- the actual arithmetic in
    whichever branch is taken always runs on the original `lam`/`u` object,
    so gradients still flow into a trainable `lam` correctly. Only the
    lam==0 and math.isinf(lam) cases are incompatible with a trainable lam
    (those are hard discrete choices of t-norm family member, not points
    the general formula could ever gradient-descend to on its own) --
    if you're training lam, keep it away from exactly those two boundary
    values, e.g. by construction (parametrize lam = exp(theta) for
    unconstrained theta, which can approach but never reach exactly 0,
    and simply don't expect it to land on math.inf).

    S is obtained from T via s = p + q - t rather than a second
    T(1-p, 1-q)-style evaluation. This is NOT an approximation:
    p + q - T(p,q) = S(p,q) is an EXACT identity for every member of the
    Frank family -- it is in fact the defining functional equation the
    family is named after (Frank 1979: T and x+y-T(x,y) simultaneously
    associative). Verified here at all three named landmarks too (e.g.
    p+q-min(p,q) = max(p,q) exactly; p+q-max(0,p+q-1) = min(1,p+q)
    exactly), not just in the general case. Halves the transcendental-
    function work relative to the old De-Morgan-duality formulation, for
    no loss of accuracy.

    p and q must already be on the same device, same convention as
    soft_jaccard above.
    """
    lam_val = float(lam.item()) if isinstance(lam, torch.Tensor) else float(lam)
    if lam_val < 0.0:
        raise ValueError(f"frank_soft_jaccard: lam must be 0, a positive float, or inf; got {lam_val}")

    if lam_val == 0.0:
        t = torch.minimum(p, q)
    elif math.isinf(lam_val):
        t = torch.clamp(p + q - 1.0, min=0.0)
    elif lam_val == 1.0:
        # u = ln(lam) = 0 exactly here -- _frank_t_general would divide by
        # exact zero (expm1(0) == 0). Exact limit, not an approximation --
        # matches soft_jaccard()'s formula bit-for-bit.
        t = p * q
    else:
        u = torch.log(lam) if isinstance(lam, torch.Tensor) else math.log(lam_val)
        t = _frank_t_general(p, q, u)

    # Exact Frank identity (see docstring) -- replaces the old second
    # log1p(...) call over T(1-p, 1-q) with a single subtraction, and
    # applies uniformly across every branch above.
    s = (p + q - t).clamp(0.0, 1.0)  # float-precision insurance only; analytically already in [0,1]

    inter = t.sum()
    union_ = s.sum()
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
    jaccard_lambda: float = 1.0,
    reduction: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine pairwise soft-Jaccard over all particle pairs, for edges and
    for nodes. This IS the multi-GPU synchronization point (see module
    docstring): every edge_probs[i]/node_probs[i] is moved to `hub_device`
    (default: edge_probs[0]'s device, i.e. single-GPU behavior when every
    particle already lives on the same device) before the pairwise ops,
    since soft_jaccard requires both arguments on one device. Those `.to()`
    moves are differentiable, so `joint_loss.backward()` downstream still
    correctly propagates gradients back to each particle's own edge_logits
    on its own device.

    `reduction` controls how the C(n,2) pairwise terms are combined before
    being scaled by lambda_edge/lambda_node:
      "mean" (DEFAULT, the FIX) -- average over all pairs. Each particle's
          gradient contribution from this term stays O(1) in n_particles
          regardless of how many particles are being discovered jointly,
          so lambda_edge_max/lambda_node_max mean the same thing (roughly
          "average pairwise overlap pressure per particle") at n=3 as at
          n=12. This is what you want for any real run.
      "sum" -- the ORIGINAL (buggy) behavior: an unnormalized sum over all
          C(n,2) pairs. Each particle sees (n-1) pairwise terms, so its
          repulsion gradient grows LINEARLY in n at a fixed lambda, while
          its fidelity gradient does not -- meaning the same lambda_edge_max
          numeric value is silently a much stronger relative penalty at
          n=12 than at n=3. Kept ONLY so it can be selected explicitly as
          the "before" condition in an ablation showing this scaling
          artifact is the actual mechanism behind large-n collapse (i.e.
          run the same n-particle sweep at reduction="sum" vs "mean" and
          check whether collapse-at-large-n disappears under "mean" --
          if so, that confirms the mechanism rather than just describing
          a symptom).

    `jaccard_lambda` (default 1.0, i.e. plain product/probabilistic-sum --
    IDENTICAL behavior to every run before this parameter existed) selects
    which Frank-family t-norm/t-conorm pair frank_soft_jaccard uses for
    every pairwise term: 0.0 = min/max, 1.0 = product (the original
    soft_jaccard, unchanged default), inf = Lukasiewicz, anything else =
    the general Frank interpolation. This is the Frank family's OWN
    parameter -- unrelated to lambda_edge/lambda_node, which scale this
    whole repulsion term's weight in the joint loss, and unrelated to
    `reduction`, which controls how the per-pair terms are aggregated
    across particles.

    Returns zero tensors (not skipped) when a lambda is 0, so callers can
    log the raw repulsion value even with that term switched off."""
    if reduction not in ("mean", "sum"):
        raise ValueError(f"combine_repulsion: reduction must be 'mean' or 'sum', got {reduction!r}")

    hub_device = hub_device or str(edge_probs[0].device)
    edge_probs = [p.to(device=hub_device) for p in edge_probs]
    node_probs = [p.to(device=hub_device) for p in node_probs]

    pairs = list(itertools.combinations(range(len(edge_probs)), 2))
    n_pairs = max(1, len(pairs))  # guard n_particles=1 (zero pairs): denominator harmless, numerator already 0
    edge_rep = sum(
        (frank_soft_jaccard(edge_probs[i], edge_probs[j], lam=jaccard_lambda) for i, j in pairs),
        torch.zeros((), device=hub_device),
    )
    node_rep = sum(
        (frank_soft_jaccard(node_probs[i], node_probs[j], lam=jaccard_lambda) for i, j in pairs),
        torch.zeros((), device=hub_device),
    )
    if reduction == "mean":
        # edge_rep = edge_rep / n_pairs
        edge_rep = edge_rep / max(1, len(edge_probs) - 1)
        # node_rep = node_rep / n_pairs
        node_rep = node_rep / max(1, len(node_probs) - 1)
    return edge_rep, node_rep
    # return edge_rep