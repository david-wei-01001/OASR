```bash
pip install torchcodec --index-url https://download.pytorch.org/whl/cpu

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p ~/miniconda3

~/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main

~/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

~/miniconda3/bin/conda create -y -n ffmpeg-libs -c conda-forge ffmpeg

echo 'export LD_LIBRARY_PATH=~/miniconda3/envs/ffmpeg-libs/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

ls -la ~/miniconda3/envs/ffmpeg-libs/lib/ | grep libavutil

python3 -c "
from torchcodec.decoders import AudioDecoder
import glob
f = glob.glob('/w/435/cse/noise_wer_correlations/disco_data/articulatory_index/AI_LSCP/isolated_sounds/wav/*.wav')[0]
d = AudioDecoder(f)
samples = d.get_all_samples()
print('decoded OK:', samples.data.shape, samples.sample_rate)
"

python3 -c '
from circuit_discovery.tasks.articulatory_index import prepare_and_save_articulatory_dataset

for task in ("consonant_classification", "vowel_classification"):
    prepare_and_save_articulatory_dataset(task, data_dir="DATA_DIR")
'
python -m circuit_discovery.tasks.train_classification_head \
    --task_type consonant_classification --n_epochs 20

python -m circuit_discovery.tasks.train_classification_head \
    --task_type vowel_classification --n_epochs 20
```
