#!/usr/bin/env python3
"""
run_gpt2_trainable_lambda.py

Trainable-lambda variant of run_gpt2_ioi.py: the Frank-family t-norm parameter
(fixed via --jaccard_lambda there) becomes a TRAINABLE scalar, shared across
the whole particle cohort. The GPT-2 counterpart of
run_hubert_trainable_lambda.py, sharing its numerics
(circuit_discovery/frank_lambda.py, imported unmodified).

Like run_gpt2_ioi.py it is task-generic: --task ioi | ioi_abba | ioi_abab |
blimp | code | anything registered in circuit_discovery/lm_tasks.py.

Replace --jaccard_lambda with --log_lam_init (u = ln(lambda) directly, NOT
lambda -- pass 0.0 for lambda=1, the product t-norm):

    python run_gpt2_trainable_lambda.py --task ioi --n_particles 5  --devices cuda:0 cuda:1 --repulsion_device cuda:1 \
        --log_lam_init 0.0 --lambda_lr 1e-2 --lambda_reg_coef 1e-3 --lambda_inner_iters 8 --lambda_reg_coef 1.0

    python run_gpt2_trainable_lambda.py --task blimp --n_particles 5  --devices cuda:0 cuda:1 --repulsion_device cuda:1 \
        --log_lam_init 0.0 --lambda_lr 1e-2 --lambda_reg_coef 1e-3 --lambda_inner_iters 8 --lambda_reg_coef 1.0

Training scheme (identical to run_hubert_trainable_lambda.py): OUTER/ALTERNATING.
Each epoch = (1) a PARTICLE PHASE with lambda held fixed at its current value,
then (2) a LAMBDA PHASE where every particle's edge_logits are frozen and
log_lam takes --lambda_inner_iters Adam steps against
    lambda_edge_rep * edge_rep(u) + --lambda_reg_coef * u**2
(an L2 pull toward u=0). Full discussion: run_hubert_trainable_lambda.py's and
frank_lambda.py's docstrings.

Two things worth knowing when choosing hyperparameters:
  - Reach. Adam moves u by at most about --lambda_lr per step, so over a run u
    can move at most ~ lambda_lr * n_epochs * lambda_inner_iters. GPT-2's
    default is 40 epochs (HuBERT's is 10), so the same lambda_lr/inner_iters
    reaches further here.
  - The Frank family is monotone in lambda (overlap shrinks as lambda grows),
    so descending on repulsion alone drives lambda outward; --lambda_reg_coef
    (and lambda_edge_max) are what set where it settles. The lambda-phase loss
    is scaled by the ramped lambda_edge, so lambda barely moves until the
    repulsion ramp has started.

GPT-2-specific efficiency: the lambda phase needs only each particle's
edge_probs = sigmoid(edge_logits), which don't depend on the batch, so no
GPT-2 forward passes are run during it (see lm_discovery_setup.probs_only_forward).
"""

from __future__ import annotations

import argparse

from circuit_discovery.frank_lambda import DEFAULT_TAYLOR_BAND
from circuit_discovery.lm_joint_discovery import TrainableLambda, add_common_args, run_joint_discovery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(parser)
    parser.add_argument("--log_lam_init", type=float, required=True,
                         help="initial value of u = ln(lambda), NOT lambda itself -- 0.0 means lambda=1 "
                              "(product t-norm). Passed straight through (no float(log(...)) rounding) so "
                              "the starting point relative to --taylor_band is under your control.")
    parser.add_argument("--lambda_lr", type=float, default=1e-2,
                         help="learning rate for log_lam's own Adam optimizer.")
    parser.add_argument("--lambda_reg_coef", type=float, default=1e-3,
                         help="L2 penalty coefficient on u=log_lam (pulls toward lambda=1). Constant.")
    parser.add_argument("--lambda_inner_iters", type=int, default=8,
                         help="lambda-only Adam steps run AFTER each epoch of particle training.")
    parser.add_argument("--taylor_band", type=float, default=DEFAULT_TAYLOR_BAND,
                         help="|u| below which the closed-form linear Taylor patch replaces the exact "
                              "expm1/log1p formula when differentiating w.r.t. u.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    controller = TrainableLambda(
        log_lam_init=args.log_lam_init, lr=args.lambda_lr, reg_coef=args.lambda_reg_coef,
        inner_iters=args.lambda_inner_iters, band=args.taylor_band,
    )
    run_joint_discovery(args, controller)


if __name__ == "__main__":
    main()
