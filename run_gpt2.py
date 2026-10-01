#!/usr/bin/env python3
"""
run_gpt2.py

CLI entrypoint for OASR-style joint multi-particle circuit discovery on a
CircuitGPT (gpt2-small / gpt2-medium) backbone, with a FIXED Frank-family
t-norm parameter (--jaccard_lambda). The GPT-2 counterpart of run_hubert.py.

Despite the historical filename this is NOT IOI-specific: --task selects any
dataset with the good/bad schema (bundled: ioi, ioi_abba, ioi_abab, blimp,
code) and needs no per-task code -- see circuit_discovery/lm_tasks.py. Only
the filename is kept so existing commands keep working.

All the logic lives in circuit_discovery/lm_joint_discovery.py (shared with
run_gpt2_trainable_lambda.py); this file only parses the fixed-lambda flag
and picks the FixedLambda controller.

Run from the OASR repo root, after `unzip circuit_discovery/datasets.zip -d circuit_discovery`.
Unlike HuBERT there is no head to train first.

Examples:
    python run_gpt2.py --n_particles 5 --jaccard_lambda 1.0 --devices cuda:0 cuda:1 --repulsion_device cuda:1
    python run_gpt2.py --task blimp --n_particles 5 --jaccard_lambda 1.0 --devices cuda:0 cuda:1 --repulsion_device cuda:1
    python run_gpt2.py --task code --n_particles 5 --jaccard_lambda 1.0 --devices cuda:0 cuda:1 --repulsion_device cuda:1
    
    python run_gpt2.py --task blimp --n_particles 5 --jaccard_lambda 0.3
    python run_gpt2.py --task code --jaccard_lambda inf

    # sweep the Frank lambda, like execute.sh does for HuBERT
    for LAM in 0.0 0.1 0.3 1.0 3.0 10.0 inf; do
        python run_gpt2.py --task ioi --n_particles 5 --jaccard_lambda $LAM
    done

    # spread 6 particles over two GPUs, combine repulsion on cuda:0
    python run_gpt2.py --n_particles 6 --devices cuda:0 cuda:1 --repulsion_device cuda:0

--jaccard_lambda selects WHICH fuzzy-set formula every pairwise repulsion term
(edge and node) uses, via the Frank family of t-norms:
    0.0 -> min/max     1.0 -> product / probabilistic-sum (DEFAULT, the
    original soft_jaccard)     inf -> Lukasiewicz
It is a different axis from --lambda_edge_max/--lambda_node_max, which scale
how MUCH the repulsion counts. See multi_particle.frank_soft_jaccard.

Changes vs the previous version of this file (it predated run_hubert.py's
latest features):
  new     --jaccard_lambda, --repulsion_reduction, --task_loss_reduction,
          --sparsity_warmup_frac, --epoch_snapshot_every, --skip_snapshot_eval
  new     per-epoch train accuracy, edge density, soft/hard test accuracy and
          per-particle epoch snapshots (same schema as run_hubert.py's)
  changed joint task loss is now the MEAN over particles (run_hubert.py's
          convention); pass --task_loss_reduction sum for the old behaviour
  changed default --save_dir is circuits_discovered/lm_circuits/{model}/{task}_lam{lam}_{reduction}/
  kept    every pre-existing flag and its default value
"""

from __future__ import annotations

import argparse

from circuit_discovery.lm_joint_discovery import FixedLambda, add_common_args, run_joint_discovery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(parser)
    parser.add_argument(
        "--jaccard_lambda", type=float, default=1.0,
        help="Frank family t-norm/t-conorm parameter for EVERY pairwise repulsion term (edge and node). "
             "0.0 = min/max, 1.0 = product/probabilistic-sum (default; the original soft_jaccard), "
             "inf = Lukasiewicz, any other positive float = general Frank interpolation. Unrelated to "
             "--lambda_edge_max/--lambda_node_max. Pass 'inf' for math.inf.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_joint_discovery(args, FixedLambda(args.jaccard_lambda))


if __name__ == "__main__":
    main()
