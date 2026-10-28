#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v conda >/dev/null 2>&1; then
  for base in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3"; do
    if [[ -f "$base/etc/profile.d/conda.sh" ]]; then
      source "$base/etc/profile.d/conda.sh"
      break
    fi
  done
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "Conda is required. Install Miniconda/Anaconda first." >&2
  exit 1
fi
eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx robotics-exp1; then
  conda create -y -n robotics-exp1 python=3.10 pip
fi
conda activate robotics-exp1

python -m pip install 'numpy==1.26.4' 'opencv-python==4.11.0.86' \
  'pybullet==3.2.7' 'matplotlib==3.10.9' 'omegaconf==2.3.1' \
  'pyrender==0.1.45' 'hydra-core==1.3.7' 'scipy==1.15.3' \
  'PyOpenGL==3.1.0' 'networkx==2.8.8' 'trimesh==5.1.1' \
  'pyglet==2.1.19' 'pillow==12.3.0' 'lxml==6.1.3' \
  'pycollada==0.9.3' 'imageio==2.38.0' 'imageio-ffmpeg==0.6.0'
python -m pip install 'urdfpy==0.0.22' --no-deps
if [[ "${1:-gpu}" == "cpu" ]]; then
  python -m pip install 'torch==2.11.0+cpu' 'torchvision==0.26.0+cpu' \
    --index-url https://download.pytorch.org/whl/cpu
else
  python -m pip install 'torch==2.11.0+cu128' 'torchvision==0.26.0+cu128' \
    --index-url https://download.pytorch.org/whl/cu128
fi
python -m pip install 'timm==1.0.30'

if [[ ! -d vendor/tacto/.git ]]; then
  mkdir -p vendor
  git clone https://github.com/facebookresearch/tacto.git vendor/tacto
fi
git -C vendor/tacto fetch origin a21d0c4626d74a546d94859226c2fea348babb6a
git -C vendor/tacto checkout --detach a21d0c4626d74a546d94859226c2fea348babb6a
python -m pip install -e vendor/tacto --no-deps
python -c 'import torch, pybullet, tacto; print("Environment ready:", torch.__version__, "CUDA:", torch.cuda.is_available())'
