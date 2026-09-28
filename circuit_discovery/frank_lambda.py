"""
circuit_discovery/frank_lambda.py

Numerically-stable machinery for treating the Frank-family t-norm parameter
(`jaccard_lambda` everywhere else in this codebase) as a TRAINABLE scalar,
shared across the whole particle cohort, rather than a fixed CLI float.

New file, imported by run_hubert_trainable_lambda.py (repo root). Nothing
in circuit_discovery/multi_particle.py, circuit_discovery/tasks/
discovery_setup.py, or run_hubert.py is modified -- this module only adds
new functions, reusing (never editing) the existing frank_soft_jaccard /
combine_repulsion / Particle / per_particle_forward machinery those files
already provide.

--------------------------------------------------------------------------
Why this needs its own math, separate from multi_particle.frank_soft_jaccard
--------------------------------------------------------------------------
Write u = ln(lambda). T(u), the Frank t-norm, is a 0/0 indeterminate form
at u=0 as literally written, but is provably a REMOVABLE singularity: T
extends to a real-analytic function with T(0) = p*q (Frank 1979; Klement,
Mesiar & Pap 2000, Sec. 2.1). multi_particle.py's frank_soft_jaccard
already evaluates T's VALUE stably near u=0 via expm1/log1p -- that part
needed no further work here.

What it does NOT handle safely is differentiating T with respect to u
itself. Reverse-mode autodiff through the literal `log1p(...)/u` division
node computes dT/du via the quotient rule as two terms that both diverge
like O(1/u) and must cancel down to the true O(1) derivative -- a
textbook catastrophic-cancellation pattern, distinct from (and NOT fixed
by) the expm1/log1p reformulation of the value. Empirically this makes
naive AD's gradient wrong by >100% already at |u| ~ 1e-4 in float32, and
the gradient is essentially unusable anywhere in bfloat16. See the
project writeup ("Numerical Stability of the Trainable Frank-Family
Parameter") for the full theorem and proof.

This is only a live concern once `lambda` is a trainable parameter that
can sit arbitrarily close to u=0 (lambda=1) for arbitrarily long -- a
fixed-lambda run (run_hubert.py's --jaccard_lambda, always a plain
Python float, never differentiated) never hits this at all, which is why
multi_particle.py's version doesn't need it.

--------------------------------------------------------------------------
The fix: a degree-1 Taylor-patched estimator, forced to float32
--------------------------------------------------------------------------
Inside a band |u| < TAYLOR_BAND, use the closed-form linear patch
  T_1(u) = p*q + (u/2)*p*q*(p+q-1-p*q)
which matches T's value and derivative at u=0 exactly (by direct Taylor
expansion of the raw Frank formula) and, being an explicit polynomial in
u with no division by u anywhere, has NO analogue of the cancellation
above -- d(T_1)/du is the u-independent constant (p*q/2)*(p+q-1-p*q),
computed once, exactly, by autograd, regardless of how close to u=0
training sits. Outside the band, use the ordinary expm1/log1p formula
(safe there; the cancellation problem is specifically a NEAR-u=0
phenomenon). TAYLOR_BAND defaults to 1e-4, matching this project's
float32 threshold (theoretically delta* = Theta(sqrt(eps)) ~ 3e-4 for
float32; see writeup).

This module ALWAYS computes in float32 regardless of the caller's
ambient dtype (even under a bf16-autocast training loop elsewhere in the
project), upcasting on entry and restoring the original dtype on exit --
bf16 was found empirically to have no usable precision regime for this
gradient anywhere near u=0 (chain-accumulated rounding across ~10
elementary ops, not just local cancellation), so this is a hard
requirement, not a tunable choice.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from itertools import cycle, islice
from typing import Any, Union

import torch

__all__ = [
    "DEFAULT_TAYLOR_BAND",
    "frank_t_stable",
    "frank_soft_jaccard_stable",
    "in_taylor_band",
    "FrankLambdaConfig",
    "FrankLambdaModule",
    "combine_edge_repulsion_trainable",
    "run_lambda_phase",
]

# float32 rule-of-thumb band width: delta* = Theta(sqrt(eps)) for the
# degree-1 (n=1) patch, eps_f32 ~ 1.19e-7, sqrt(eps_f32) ~ 3.4e-4. 1e-4 is
# comfortably inside the safe/accurate side of that crossover (see the
# empirical float32 error table in the writeup: relative gradient error
# is still ~1e-4 at |u|=1e-3 and only blows past ~4% by |u|=1e-4), while
# still being a "you are basically at lambda=1" regime in absolute terms.
DEFAULT_TAYLOR_BAND = 1e-4


def _as_float32(x: Union[float, torch.Tensor], device: torch.device | str | None = None) -> torch.Tensor:
    """Upcast a plain float or a tensor of any dtype to a float32 tensor,
    WITHOUT breaking the autograd graph if x already requires grad (i.e.
    if x is the trainable log_lam parameter itself, `.to(torch.float32)`
    on a tensor that's already float32 is a no-op that still returns the
    same tensor object -- not a detach -- so gradients keep flowing)."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype=torch.float32, device=device) if device is not None else x.to(dtype=torch.float32)
    return torch.tensor(float(x), dtype=torch.float32, device=device)


def _frank_t_general_f32(p: torch.Tensor, q: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Exact Frank t-norm value via expm1/log1p, forced float32. Safe for
    the VALUE at any u != 0 (including very close to 0); NOT safe to
    differentiate w.r.t. u near u=0 -- callers must route small |u|
    through the Taylor patch instead (see frank_t_stable). u == 0.0
    exactly is not handled here (division by exact zero); the patch
    covers u=0 too, so frank_t_stable never calls this function there."""
    lam_minus_1 = torch.expm1(u)
    lam_pow_p_minus_1 = torch.expm1(u * p)
    lam_pow_q_minus_1 = torch.expm1(u * q)
    t_arg = lam_pow_p_minus_1 * lam_pow_q_minus_1 / lam_minus_1
    t = torch.log1p(t_arg) / u
    return t.clamp(0.0, 1.0)


def _frank_t_linear_patch_f32(p: torch.Tensor, q: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Degree-1 (n=1) Taylor patch, forced float32. No division by u
    anywhere -- affine in u -- so both the value AND autograd's gradient
    w.r.t. u are exact-to-this-order and free of the cancellation that
    afflicts _frank_t_general_f32's gradient near u=0. See module
    docstring / project writeup for the derivation."""
    pq = p * q
    return pq + u * (pq / 2.0) * (p + q - 1.0 - pq)


def frank_t_stable(
    p: torch.Tensor, q: torch.Tensor, u: Union[float, torch.Tensor],
    band: float = DEFAULT_TAYLOR_BAND,
) -> torch.Tensor:
    """T_lambda(p, q) with u = ln(lambda), safe to differentiate w.r.t. u
    (as well as p, q) anywhere, including arbitrarily close to u=0. Always
    computed in float32 regardless of input dtypes; p and q are upcast
    (and moved to u's device, if u is a tensor) but NOT detached, so
    gradients into p/q (e.g. edge_logits, via sigmoid) still flow
    normally when this is used in a context where that's wanted.

    `u` may be a plain Python float (matches multi_particle.py's
    convention for a fixed, non-trainable lambda) or a 0-d torch.Tensor
    with requires_grad=True (the trainable case this module exists for).
    """
    device = u.device if isinstance(u, torch.Tensor) else None
    u32 = _as_float32(u, device=device)
    p32 = _as_float32(p, device=u32.device)
    q32 = _as_float32(q, device=u32.device)

    if isinstance(u, torch.Tensor):
        small = u32.abs() < band
        t_patch = _frank_t_linear_patch_f32(p32, q32, u32)
        # Only evaluate the exact formula where it's actually needed --
        # torch.where still evaluates BOTH branches eagerly (no lazy
        # short-circuit), so _frank_t_general_f32 runs on the full tensor
        # regardless; guard u=0 exactly (would divide by exact zero) by
        # clamping u away from 0 for that branch's OWN computation only
        # -- its result there is discarded by torch.where anyway since
        # `small` is True at u=0 (0 < band always, band > 0).
        u_safe_for_exact = torch.where(u32.abs() < 1e-30, torch.full_like(u32, 1e-30), u32)
        t_exact = _frank_t_general_f32(p32, q32, u_safe_for_exact)
        t = torch.where(small, t_patch, t_exact)
    else:
        # plain float, non-trainable path: cheap branch on the concrete
        # value, matching multi_particle.py's own dispatch style.
        if abs(float(u)) < band:
            t = _frank_t_linear_patch_f32(p32, q32, u32)
        else:
            t = _frank_t_general_f32(p32, q32, u32)

    return t.clamp(0.0, 1.0)


def frank_soft_jaccard_stable(
    p: torch.Tensor, q: torch.Tensor, u: Union[float, torch.Tensor],
    band: float = DEFAULT_TAYLOR_BAND, eps: float = 1e-8,
) -> torch.Tensor:
    """Same Ruzicka/soft-Jaccard aggregate (sum(T)/sum(S)) as
    multi_particle.frank_soft_jaccard, using frank_t_stable for T and the
    exact Frank identity S = p + q - T (see multi_particle.py's docstring
    for why this identity is exact, not an approximation, across the
    whole family) for S. p and q must already be on the same device as
    each other; u may live on any device (frank_t_stable moves p, q to
    u's device when u is a tensor)."""
    t = frank_t_stable(p, q, u, band=band)
    p32 = _as_float32(p, device=t.device)
    q32 = _as_float32(q, device=t.device)
    s = (p32 + q32 - t).clamp(0.0, 1.0)
    inter = t.sum()
    union_ = s.sum()
    return inter / (union_ + eps)


def in_taylor_band(u: Union[float, torch.Tensor], band: float = DEFAULT_TAYLOR_BAND) -> bool:
    """Whether a given u currently falls inside the Taylor-patched region.
    Since lambda is ONE shared scalar across the whole cohort (see
    FrankLambdaModule), this is a single yes/no per lambda value -- not
    something that varies per particle-pair -- which is what makes the
    band-occupancy diagnostic in run_lambda_phase cheap to track (one
    check per lambda-update iteration, not one per pair)."""
    u_val = float(u.item()) if isinstance(u, torch.Tensor) else float(u)
    return abs(u_val) < band


@dataclass
class FrankLambdaConfig:
    """Hyperparameters for the trainable Frank-family lambda. `log_lam_init`
    is u itself (NOT lambda) -- pass u directly (e.g. 0.0 for lambda=1,
    the previous fixed default) so there is no float(log(...)) rounding
    step between what you type on the command line and what the
    parameter is actually initialized to."""
    log_lam_init: float
    lr: float = 1e-2
    reg_coef: float = 1e-3
    band: float = DEFAULT_TAYLOR_BAND
    inner_iters: int = 8
    device: str = "cpu"


class FrankLambdaModule:
    """Owns the single trainable scalar u = log(lambda), shared across the
    whole particle cohort, plus its own Adam optimizer. lambda = exp(u)
    is used only via `.lam`/`.u` read accessors and inside
    combine_edge_repulsion_trainable -- there is deliberately no `.step()`
    that also touches any Particle's edge_logits; see run_lambda_phase
    for how particle gradients are kept out of this module's backward
    pass (explicit no_grad + detach on every edge_probs tensor used here,
    not an assumption that the gradient happens to vanish on its own).
    """

    def __init__(self, config: FrankLambdaConfig) -> None:
        self.config = config
        self.log_lam = torch.nn.Parameter(
            torch.tensor(config.log_lam_init, dtype=torch.float32, device=config.device)
        )
        self.optimizer = torch.optim.Adam([self.log_lam], lr=config.lr)

    @property
    def u(self) -> float:
        return float(self.log_lam.detach().item())

    @property
    def lam(self) -> float:
        return float(torch.exp(self.log_lam.detach()).item())

    def in_band(self) -> bool:
        return in_taylor_band(self.log_lam, band=self.config.band)

    def regularization(self) -> torch.Tensor:
        """coef * u^2 -- an L2 pull toward u=0 (lambda=1), per the
        hyperparameter-search finding that the optimal lambda sits near
        1.0. Kept as a plain constant-coefficient penalty each step, not
        annealed -- with joint co-descent already providing one moving
        target (the particles' own edge_probs, which keep changing
        epoch to epoch) there is no clear reason to also move the
        regularization strength, and a constant coefficient keeps this
        one part of the objective easy to reason about."""
        return self.config.reg_coef * self.log_lam**2


def combine_edge_repulsion_trainable(
    edge_probs: list[torch.Tensor],
    log_lam: torch.Tensor,
    *,
    band: float,
    hub_device: str,
    reduction: str = "mean",
) -> torch.Tensor:
    """Edge-only pairwise Frank-Jaccard repulsion, mirroring
    multi_particle.combine_repulsion's pairwise loop, but (a) using
    frank_soft_jaccard_stable so u=log_lam can be safely differentiated,
    and (b) taking every edge_probs tensor as given -- callers are
    responsible for detaching them first (see run_lambda_phase) since
    this function itself does not detach, to stay a pure, reusable op.

    `reduction="mean"` divides by (n_particles - 1), matching this
    project's existing combine_repulsion convention (not C(n_particles,2)
    -- see project discussion for why that convention was chosen), so
    lambda's own loss is scaled consistently with what the particles
    experienced under the same epoch's edge_rep term.

    Node repulsion is intentionally NOT included here: this project is
    not currently using node-level repulsion in practice (lambda_node_max
    defaults to 0), so the trainable-lambda path only covers the edge
    term. If node repulsion is ever turned on, it uses the existing,
    unmodified, fixed-lambda multi_particle.frank_soft_jaccard machinery,
    completely decoupled from this trainable u.
    """
    edge_probs = [p.to(device=hub_device) for p in edge_probs]
    pairs = list(itertools.combinations(range(len(edge_probs)), 2))
    rep = sum(
        (frank_soft_jaccard_stable(edge_probs[i], edge_probs[j], log_lam, band=band) for i, j in pairs),
        torch.zeros((), device=hub_device, dtype=torch.float32),
    )
    if reduction == "mean":
        rep = rep / max(1, len(edge_probs) - 1)
    elif reduction != "sum":
        raise ValueError(f"combine_edge_repulsion_trainable: reduction must be 'mean' or 'sum', got {reduction!r}")
    return rep


def run_lambda_phase(
    lambda_module: FrankLambdaModule,
    particles: list[Any],
    train_loader: Any,
    *,
    per_particle_forward: Any,
    lambda_edge_coef: float,
    reduction: str = "mean",
) -> dict[str, Any]:
    """Outer/alternating lambda-training phase: run AFTER a full epoch of
    particle updates, with every particle's edge_logits held fixed. For
    `lambda_module.config.inner_iters` iterations, pulls a FRESH batch
    each time (cycling back through train_loader if inner_iters exceeds
    its length), runs each particle's forward pass under torch.no_grad()
    (so no graph is ever built through the frozen backbone or the
    particles' edge_logits -- this is the explicit mechanism, not an
    assumption, by which lambda's gradient never reaches any particle's
    parameters), detaches the resulting edge_probs, computes lambda's own
    loss (lambda_edge_coef * edge_rep(u) + L2 regularization toward
    u=0), and takes one Adam step on log_lam alone.

    `lambda_edge_coef` should be the SAME lambda_edge_rep value (from
    ramp_schedule) the particles' own joint_loss used during the epoch
    that just finished, so lambda's loss is scaled consistently with
    what the particles actually experienced.

    Returns a dict of per-phase diagnostics for logging/snapshotting:
    {mean_loss, mean_edge_rep, band_hits, n_iters, u, lam}.
    """
    n_iters = lambda_module.config.inner_iters
    hub_device = str(lambda_module.log_lam.device)
    band = lambda_module.config.band

    losses: list[float] = []
    edge_reps: list[float] = []
    band_hits = 0

    for batch in islice(cycle(train_loader), n_iters):
        edge_probs = []
        with torch.no_grad():
            for particle in particles:
                _, probs, _ = per_particle_forward(particle, batch, lambda_sparse=0.0)
                edge_probs.append(probs.detach())

        edge_rep = combine_edge_repulsion_trainable(
            edge_probs, lambda_module.log_lam, band=band, hub_device=hub_device, reduction=reduction,
        )
        loss = lambda_edge_coef * edge_rep + lambda_module.regularization()

        lambda_module.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        lambda_module.optimizer.step()

        losses.append(loss.item())
        edge_reps.append(edge_rep.item())
        band_hits += int(lambda_module.in_band())

    return {
        "mean_loss": sum(losses) / len(losses) if losses else None,
        "mean_edge_rep": sum(edge_reps) / len(edge_reps) if edge_reps else None,
        "band_hits": band_hits,
        "n_iters": n_iters,
        "u": lambda_module.u,
        "lam": lambda_module.lam,
    }