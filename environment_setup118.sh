#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV=${1:-"paspa"}
if [ -n "$CONDA_ENV" ]; then
    # This is required to activate conda environment
    eval "$(conda shell.bash hook)"

    conda create -n "$CONDA_ENV" python=3.12.0 -y
    conda activate "$CONDA_ENV"
    # This is optional if you prefer to use built-in nvcc
    conda install -c nvidia cuda-toolkit=11.8 -y
else
    echo "Skipping conda environment creation. Make sure you have the correct environment activated."
fi

pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install accelerate
pip install diffusers transformers==4.50.0
pip install hydra-core 
pip install timm pytz termcolor ipdb
pip install lmdb textgrid pandas librosa smplx
pip install matplotlib fasttext
pip install bitsandbytes
pip install trimesh
pip install vocos
pip install wget
pip install resampy
pip install torchlibrosa pyrender opencv-python
pip install speechbrain
pip install h5py
pip install positional_encodings
pip install flash-linear-attention liger-kernel
# if you need wget
# wget https://github.com/MiroPsota/torch_packages_builder/releases/download/pytorch3d-0.7.8%2B5043d15/pytorch3d-0.7.8+5043d15pt2.5.1cu118-cp312-cp312-linux_x86_64.whl
# pip install pytorch3d-0.7.8+5043d15pt2.5.1cu118-cp312-cp312-linux_x86_64.whl

pip install nvitop
pip install -e .

hf download H-Liu1997/emage_audio --local-dir ./tools/EMAGE/pretrained_vqvae
hf download H-Liu1997/emage_evaltools --local-dir ./tools/EMAGE/emage_evaltools
hf download charactr/vocos-mel-24khz --local-dir ./tools/vocos
hf download patrickvonplaten/wavlm-libri-clean-100h-large --local-dir ./tools/wavlm-libri-clean-100h-large
