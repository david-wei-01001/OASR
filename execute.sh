source /mnt/data4/u/davidwei/OASR/myenv/bin/activate
export CUDA_VISIBLE_DEVICES=4
python pilot_hubert.py --task_type vowel_classification --edge_logit_init_mean 10.0