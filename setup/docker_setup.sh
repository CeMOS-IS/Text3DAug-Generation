#!/bin/bash

# GPU supported PyTorch3D
pip install git+https://github.com/facebookresearch/pytorch3d.git

# One-2-3-45 Weights
cd ./packages/One-2-3-45/ && python3 download_ckpt.py
cd ..
cd ..

# One-2-3-45 GPU supported packages
export TORCH_CUDA_ARCH_LIST="7.0;7.2;8.0;8.6;9.0+PTX"
export IABN_FORCE_CUDA=1
pip install inplace_abn
FORCE_CUDA=1 pip install --no-cache-dir git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0

# bug
pip uninstall ninja --yes && pip install Ninja

# Keep container running
exec /bin/bash
