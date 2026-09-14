source /mnt/data4/u/davidwei/OASR/myenv/bin/activate
export CUDA_VISIBLE_DEVICES=6
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
      --jaccard_lambda inf \
      --save_dir circuits_discovered/hubert_circuits/vowel_frank_laminf

python run_hubert.py --task_type vowel_classification \
      --n_particles 5 --devices cuda:0 cuda:1 \
      --repulsion_device cuda:1 \
      --jaccard_lambda 0.0 \
      --save_dir circuits_discovered/hubert_circuits/vowel_frank_lam0.0

python run_hubert.py --task_type vowel_classification \
      --n_particles 5 --devices cuda:0 cuda:1 \
      --repulsion_device cuda:1 \
      --jaccard_lambda 1.0 \
      --lr_e 0.02 \
      --save_dir circuits_discovered/hubert_circuits/vowel_frank_lam1.0 \
      --lambda_sparse_e 3.0


python analyze_jaccard_gap.py \
    --report joint:circuit_comparison_report.json \
    --out jaccard_gap_analysis

python3 compare_circuits.py --task_type vowel_classification \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.3/particle0_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.3/particle1_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.3/particle2_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.3/particle3_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.3/particle4_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.3/particle5_epoch_snapshots/epoch004.pt \
    --out report_lam0.3.json

python3 compare_circuits.py --task_type vowel_classification \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_laminf/particle0_epoch_snapshots/epoch002.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_laminf/particle1_epoch_snapshots/epoch002.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_laminf/particle2_epoch_snapshots/epoch002.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_laminf/particle3_epoch_snapshots/epoch002.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_laminf/particle4_epoch_snapshots/epoch002.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_laminf/particle5_epoch_snapshots/epoch002.pt \
      --out report_laminf.json

python3 compare_circuits.py --task_type vowel_classification \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam10.0/particle0_epoch_snapshots/epoch002.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam10.0/particle1_epoch_snapshots/epoch002.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam10.0/particle2_epoch_snapshots/epoch002.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam10.0/particle3_epoch_snapshots/epoch002.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam10.0/particle4_epoch_snapshots/epoch002.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam10.0/particle5_epoch_snapshots/epoch002.pt \
    --out report_lam10.0.json

python3 compare_circuits.py --task_type vowel_classification \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam3.0/particle0_epoch_snapshots/epoch002.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam3.0/particle1_epoch_snapshots/epoch002.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam3.0/particle2_epoch_snapshots/epoch002.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam3.0/particle3_epoch_snapshots/epoch002.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam3.0/particle4_epoch_snapshots/epoch002.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam3.0/particle5_epoch_snapshots/epoch002.pt \
    --out report_lam3.0.json

python3 compare_circuits.py --task_type vowel_classification \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.1/particle0_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.1/particle1_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.1/particle2_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.1/particle3_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.1/particle4_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.1/particle5_epoch_snapshots/epoch004.pt \
    --out report_lam0.1.json

python3 compare_circuits.py --task_type vowel_classification \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.0/particle0_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.0/particle1_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.0/particle2_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.0/particle3_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.0/particle4_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.0/particle5_epoch_snapshots/epoch004.pt \
    --out report_lam0.0.json

python3 compare_circuits.py --task_type vowel_classification \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam1.0/particle0_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam1.0/particle1_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam1.0/particle2_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam1.0/particle3_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam1.0/particle4_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam1.0/particle5_epoch_snapshots/epoch004.pt \
    --out report_lam1.0.json

python3 compare_circuits.py --task_type vowel_classification \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam3.0/particle0_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam3.0/particle1_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam3.0/particle2_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam3.0/particle3_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam3.0/particle4_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam3.0/particle5_epoch_snapshots/epoch004.pt \
    --out report_lam3.0.json

python3 compare_circuits.py --task_type vowel_classification \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam10.0/particle0_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam10.0/particle1_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam10.0/particle2_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam10.0/particle3_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam10.0/particle4_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam10.0/particle5_epoch_snapshots/epoch004.pt \
    --out report_lam10.0.json

python3 compare_circuits.py --task_type vowel_classification \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_laminf/particle0_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_laminf/particle1_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_laminf/particle2_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_laminf/particle3_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_laminf/particle4_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_laminf/particle5_epoch_snapshots/epoch004.pt \
    --out report_laminf.json

python3 compare_circuits.py --task_type vowel_classification \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.9/particle0_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.9/particle1_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.9/particle2_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.9/particle3_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.9/particle4_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.9/particle5_epoch_snapshots/epoch004.pt \
    --out report_lam0.9.json

python3 compare_circuits.py --task_type vowel_classification \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.7/particle0_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.7/particle1_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.7/particle2_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.7/particle3_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.7/particle4_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.7/particle5_epoch_snapshots/epoch004.pt \
    --out report_lam0.7.json

python3 compare_circuits.py --task_type vowel_classification \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.5/particle0_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.5/particle1_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.5/particle2_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.5/particle3_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.5/particle4_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam0.5/particle5_epoch_snapshots/epoch004.pt \
    --out report_lam0.5.json

python3 compare_circuits.py --task_type vowel_classification \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam2.0/particle0_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam2.0/particle1_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam2.0/particle2_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam2.0/particle3_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam2.0/particle4_epoch_snapshots/epoch004.pt \
    --circuit simultaneous:circuits_discovered/hubert_circuits/vowel_frank_lam2.0/particle5_epoch_snapshots/epoch004.pt \
    --out report_lam2.0.json

3.0 - epoch 4
0.3 - epoch 3
0.1 - epoch 3

python summarize_lambda_sweep.py \
    --report 0.0:report_lam0.0.json \
    --report 0.1:report_lam0.1.json \
    --report 0.3:report_lam0.3.json \
    --report 0.5:report_lam0.5.json \
    --report 0.7:report_lam0.7.json \
    --report 0.9:report_lam0.9.json \
    --report 1.0:report_lam1.0.json \
    --report 2.0:report_lam2.0.json \
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