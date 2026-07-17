#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="${HOME}/miniconda3"
ENV_NAME="vad-study"

sudo apt-get update
sudo apt-get install -y \
  build-essential ca-certificates curl git libgl1 libglib2.0-0 ninja-build wget

if [[ ! -x "${CONDA_ROOT}/bin/conda" ]]; then
  installer="$(mktemp --suffix=.sh)"
  wget -O "${installer}" \
    https://repo.anaconda.com/miniconda/Miniconda3-py38_24.3.0-0-Linux-x86_64.sh
  bash "${installer}" -b -p "${CONDA_ROOT}"
  rm -f "${installer}"
fi

# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -n "${ENV_NAME}" python=3.8 -y
fi
conda activate "${ENV_NAME}"

python -m pip install --upgrade 'pip<25' 'setuptools<70' wheel ninja
python -m pip install \
  torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
  --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install \
  mmcv-full==1.7.2 \
  -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html
python -m pip install \
  mmdet==2.28.2 mmsegmentation==0.30.0 \
  numpy==1.23.5 numba==0.56.4 networkx==2.8.8 \
  nuscenes-devkit==1.1.11 shapely==1.8.5.post1 \
  scikit-image==0.19.3 scipy==1.9.3 \
  matplotlib==3.7.5 opencv-python==4.8.1.78 \
  timm==0.6.13 tensorboard==2.13.0 \
  lyft-dataset-sdk==0.0.8 plyfile==0.7.4 trimesh==3.9.35

if [[ ! -x /usr/local/cuda-11.7/bin/nvcc ]]; then
  keyring="$(mktemp --suffix=.deb)"
  wget -O "${keyring}" \
    https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i "${keyring}"
  rm -f "${keyring}"
  sudo apt-get update
  sudo apt-get install -y cuda-nvcc-11-7 cuda-cudart-dev-11-7
fi

export CUDA_HOME=/usr/local/cuda-11.7
export PATH="${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST='8.6+PTX'
export MAX_JOBS="${MAX_JOBS:-8}"

python "${PROJECT_ROOT}/scripts/patch_mmdet3d_compat.py" \
  "${PROJECT_ROOT}/_deps/mmdetection3d-0.17.1"
python -m pip install -v --no-build-isolation -e \
  "${PROJECT_ROOT}/_deps/mmdetection3d-0.17.1"

python "${PROJECT_ROOT}/scripts/smoke_test_env.py"

cat <<EOF

Environment ready.
Activate it in a new shell with:
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate ${ENV_NAME}
  export CUDA_HOME=/usr/local/cuda-11.7
  export TORCH_CUDA_ARCH_LIST='8.6+PTX'
EOF
