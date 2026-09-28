source /mnt/data4/u/davidwei/OASR/myenv/bin/activate
export CUDA_VISIBLE_DEVICES=5
python pilot_hubert.py --task_type vowel_classification --edge_logit_init_mean 10.0

export CUDA_VISIBLE_DEVICES=0,1
for LAM in 0.0 0.1 0.3 3.0 10.0 inf; do
  python run_hubert.py --task_type vowel_classification \
      --n_particles 6 --devices cuda:0 cuda:1 \
      --jaccard_lambda $LAM \
      --save_dir circuits_discovered/hubert_circuits/vowel_frank_lam${LAM}
done

python run_hubert.py --task_type vowel_classification \
      --n_particles 5 --devices cuda:0 cuda:1 \
      --repulsion_device cuda:1 \
      --jaccard_lambda 1.0 \
      --lr_e 0.02 \
      --save_dir circuits_discovered/hubert_circuits/vowel_frank_lam1.0 \
      --lambda_edge_max 1.0 \
      --lambda_sparse_e 2.0 \
      --n_epochs 10

python run_hubert.py --task_type vowel_classification \
      --n_particles 5 --devices cuda:0 cuda:1 \
      --repulsion_device cuda:1 \
      --jaccard_lambda inf \
      --lr_e 0.02 \
      --save_dir circuits_discovered/hubert_circuits/vowel_frank_laminf \
      --lambda_edge_max 1.0 \
      --lambda_sparse_e 2.0 \
      --n_epochs 10

python run_hubert.py --task_type vowel_classification \
      --n_particles 5 --devices cuda:0 cuda:1 \
      --repulsion_device cuda:1 \
      --jaccard_lambda 1.0 \
      --lr_e 0.02 \
      --save_dir circuits_discovered/hubert_circuits/vowel_frank_lam1.0 \
      --lambda_edge_max 1.0 \
      --lambda_sparse_e 2.0 \
      --n_epochs 10

python run_hubert.py --task_type consonant_classification \
      --n_particles 5 --devices cuda:0 cuda:1 \
      --repulsion_device cuda:1 \
      --jaccard_lambda 1.0 \
      --lr_e 0.01 \
      --save_dir circuits_discovered/hubert_circuits/consonant_frank_lam1.0 \
      --lambda_edge_max 6.0 \
      --lambda_sparse_e 10.0 \
      --n_epochs 10

python run_hubert.py --task_type consonant_classification \
      --n_particles 5 --devices cuda:0 cuda:1 \
      --repulsion_device cuda:1 \
      --jaccard_lambda 0.3 \
      --lr_e 0.01 \
      --save_dir circuits_discovered/hubert_circuits/consonant_frank_lam0.3 \
      --lambda_edge_max 6.0 \
      --lambda_sparse_e 10.0 \
      --n_epochs 10

python analyze_jaccard_gap.py \
    --report joint:circuit_comparison_report.json \
    --out jaccard_gap_analysis

python3 compare_circuits.py --task_type vowel_classification \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.3/particle0_epoch_snapshots/epoch006.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.3/particle1_epoch_snapshots/epoch006.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.3/particle2_epoch_snapshots/epoch006.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.3/particle3_epoch_snapshots/epoch006.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.3/particle4_epoch_snapshots/epoch006.pt \
    --out report_lam0.3.json

python3 compare_circuits.py --task_type vowel_classification \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_laminf/particle0_epoch_snapshots/epoch006.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_laminf/particle1_epoch_snapshots/epoch006.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_laminf/particle2_epoch_snapshots/epoch006.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_laminf/particle3_epoch_snapshots/epoch006.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_laminf/particle4_epoch_snapshots/epoch006.pt \
      --out report_laminf.json

python3 compare_circuits.py --task_type vowel_classification \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam10.0/particle0_epoch_snapshots/epoch006.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam10.0/particle1_epoch_snapshots/epoch006.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam10.0/particle2_epoch_snapshots/epoch006.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam10.0/particle3_epoch_snapshots/epoch006.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam10.0/particle4_epoch_snapshots/epoch006.pt \
    --out report_lam10.0.json

python3 compare_circuits.py --task_type vowel_classification \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam3.0/particle0_epoch_snapshots/epoch006.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam3.0/particle1_epoch_snapshots/epoch006.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam3.0/particle2_epoch_snapshots/epoch006.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam3.0/particle3_epoch_snapshots/epoch006.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam3.0/particle4_epoch_snapshots/epoch006.pt \
    --out report_lam3.0.json

python3 compare_circuits.py --task_type vowel_classification \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.1/particle0_epoch_snapshots/epoch006.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.1/particle1_epoch_snapshots/epoch006.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.1/particle2_epoch_snapshots/epoch006.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.1/particle3_epoch_snapshots/epoch006.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.1/particle4_epoch_snapshots/epoch006.pt \
    --out report_lam0.1.json

python3 compare_circuits.py --task_type vowel_classification \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.0/particle0_epoch_snapshots/epoch006.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.0/particle1_epoch_snapshots/epoch006.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.0/particle2_epoch_snapshots/epoch006.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.0/particle3_epoch_snapshots/epoch006.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.0/particle4_epoch_snapshots/epoch006.pt \
    --out report_lam0.0.json

python3 compare_circuits.py --task_type vowel_classification \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam1.0/particle0_epoch_snapshots/epoch006.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam1.0/particle1_epoch_snapshots/epoch006.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam1.0/particle2_epoch_snapshots/epoch006.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam1.0/particle3_epoch_snapshots/epoch006.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam1.0/particle4_epoch_snapshots/epoch006.pt \
    --out report_lam1.0.json

python summarize_lambda_sweep.py \
    --report 0.0:report_lam0.0.json \
    --report 0.1:report_lam0.1.json \
    --report 0.3:report_lam0.3.json \
    --report 1.0:report_lam1.0.json \
    --report 3.0:report_lam3.0.json \
    --report 10.0:report_lam10.0.json \
    --report inf:report_laminf.json \
    --baseline 1.0 \
    --out lambda_sweep_summary

python curvature_risk_analysis.py \
    --pair lam0.1:circuits_discovered/hubert_circuits/vowel_frank_lam0.1/particle0_epoch_snapshots/epoch004.pt:circuits_discovered/hubert_circuits/vowel_frank_lam0.1/particle1_epoch_snapshots/epoch004.pt \
    --pair lam0.1:circuits_discovered/hubert_circuits/vowel_frank_lam0.9/particle0_epoch_snapshots/epoch004.pt:circuits_discovered/hubert_circuits/vowel_frank_lam0.9/particle2_epoch_snapshots/epoch004.pt \
    --pair lam0.3:circuits_discovered/hubert_circuits/vowel_frank_lam1.0/particle0_epoch_snapshots/epoch004.pt:circuits_discovered/hubert_circuits/vowel_frank_lam1.0/particle1_epoch_snapshots/epoch004.pt \
    --pair lam3.0:circuits_discovered/hubert_circuits/vowel_frank_lam10.0/particle0_epoch_snapshots/epoch004.pt:circuits_discovered/hubert_circuits/vowel_frank_lam10.0/particle1_epoch_snapshots/epoch004.pt \
    --candidate_lambdas 0.1 0.3 1.0 3.0 10.0 \
    --out curvature_risk


python run_hubert_trainable_lambda.py --task_type vowel_classification \
      --n_particles 5 --devices cuda:0 cuda:1 --repulsion_device cuda:1 \
      --log_lam_init=1e-2 --lambda_lr 1e-3 --lambda_inner_iters 10 --taylor_band 1e-4 \
      --lr_e 0.02 --lambda_edge_max 1.0 --lambda_sparse_e 2.0 --n_epochs 10 \
      --save_dir circuits_discovered/hubert_circuits/vowel_frank_trainlam_init0

python run_hubert_trainable_lambda.py --task_type consonant_classification \
      --n_particles 5 --devices cuda:0 cuda:1 --repulsion_device cuda:1 \
      --log_lam_init=0.0 --lambda_lr 1e-5 --lambda_inner_iters 10 --taylor_band 1e-4 \
      --lr_e 0.01 --lambda_edge_max 6.0 --lambda_sparse_e 12.0 --n_epochs 10 \
      --save_dir circuits_discovered/hubert_circuits/consonant_frank_trainlam_init0


(consonant hyper: ep 7)
(consonant trained: ep 7)
(vowel hyper: ep 6)
(vowel trained: ep 6)
