source /mnt/data4/u/davidwei/OASR/myenv/bin/activate
export CUDA_VISIBLE_DEVICES=7
python pilot_hubert.py --task_type vowel_classification --edge_logit_init_mean 10.0

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
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
      --jaccard_lambda 0.1 \
      --lr_e 0.01 \
      --save_dir circuits_discovered/hubert_circuits/consonant_frank_lam0.1 \
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

python3 compare_circuits.py --task_type consonant_classification \
    --circuit simultaneous:circuits_discovered/hubert_circuits/consonant_frank_lam0.1/particle0_epoch_snapshots/epoch007.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/consonant_frank_lam0.1/particle1_epoch_snapshots/epoch007.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/consonant_frank_lam0.1/particle2_epoch_snapshots/epoch007.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/consonant_frank_lam0.1/particle3_epoch_snapshots/epoch007.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/consonant_frank_lam0.1/particle4_epoch_snapshots/epoch007.pt \
    --out consonant_report_lam0.1.json

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

python summarize_lambda_sweep.py \
    --report 0.0:consonant_report_lam0.0.json \
    --report 0.1:consonant_report_lam0.1.json \
    --report 0.3:consonant_report_lam0.3.json \
    --report 1.0:consonant_report_lam1.0.json \
    --report 3.0:consonant_report_lam3.0.json \
    --report 10.0:consonant_report_lam10.0.json \
    --report inf:consonant_report_laminf.json \
    --baseline 1.0 \
    --out consonant_lambda_sweep_summary

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

python run_hubert_trainable_lambda.py --task_type vowel_classification \
      --n_particles 15 --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 --repulsion_device cuda:4 \
      --log_lam_init=1e-2 --lambda_lr 1e-3 --lambda_inner_iters 10 --taylor_band 1e-4 \
      --lr_e 0.02 --lambda_edge_max 1.0 --lambda_sparse_e 2.0 --n_epochs 10 \
      --save_dir circuits_discovered/hubert_circuits/vowel_frank_trainlam_circ

python run_hubert_trainable_lambda.py --task_type consonant_classification \
      --n_particles 5 --devices cuda:0 cuda:1 --repulsion_device cuda:1 \
      --log_lam_init=0.0 --lambda_lr 1e-5 --lambda_inner_iters 10 --taylor_band 1e-4 \
      --lr_e 0.01 --lambda_edge_max 6.0 --lambda_sparse_e 12.0 --n_epochs 10 \
      --save_dir circuits_discovered/hubert_circuits/consonant_frank_trainlam_init0


(consonant hyper: ep 7)
(consonant trained: ep 7)
(vowel hyper: ep 6)
(vowel trained: ep 6)

python run_gpt2.py --n_particles 5 --jaccard_lambda 1.0 --devices cuda:0 cuda:1 --repulsion_device cuda:1

python run_gpt2.py --task blimp --n_particles 5 --jaccard_lambda 1.0 --devices cuda:0 --repulsion_device cuda:0


python run_gpt2_trainable_lambda.py --task ioi --n_particles 5 --devices cuda:0 cuda:1 --repulsion_device cuda:1 \
        --log_lam_init 0.0 --lambda_lr 1e-2 --lambda_reg_coef 1e-3 --lambda_inner_iters 8  --lambda_reg_coef 1.0

python run_gpt2_trainable_lambda.py --task blimp --n_particles 10 --devices cuda:0 cuda:1 --repulsion_device cuda:1 \
        --log_lam_init 0.0 --lambda_lr 1e-2 --lambda_reg_coef 1e-3 --lambda_inner_iters 8  --lambda_reg_coef 1.0 --save_dir circuits_discovered/lm_circuits/gpt2-small/blimp_trainlam_init0.0_n10

python run_gpt2_trainable_lambda.py --task ioi --n_particles 10 --devices cuda:0 cuda:1 --repulsion_device cuda:1 \
        --log_lam_init 0.0 --lambda_lr 1e-2 --lambda_reg_coef 1e-3 --lambda_inner_iters 8  --lambda_reg_coef 1.0 --save_dir circuits_discovered/lm_circuits/gpt2-small/ioi_trainlam_init0.0_n10

python compare_lm_circuits.py --epoch 40 \
    --run_dir circuits_discovered/lm_circuits/gpt2-small/blimp_trainlam_init0.0_n10 \
    --finalize --out report_n10.json