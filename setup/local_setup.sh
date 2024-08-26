#!/bin/bash

python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install --no-cache-dir git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
pip install git+https://github.com/facebookresearch/pytorch3d.git
pip install inplace_abn

pip install -e packages/point-e
pip install -e packages/shap-e
pip install -e packages/cap3d

pip install -r setup/requirements.txt