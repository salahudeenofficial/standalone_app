#!/bin/bash
set -e  # stop on error
# === System dependencies ===
sudo apt-get update
sudo apt-get install -y \
    libegl1 \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    libegl1-mesa \
    libglu1-mesa \
    libosmesa6


conda create -n chatgarment python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chatgarment
pip install --upgrade pip
pip install -e ".[train]"

git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.6.3          
pip install build
python -m build --wheel --no-isolation
pip install ./dist/flash_attn-2.6.3-cp310-cp310-linux_x86_64.whl
cd ..


cd GarmentCodeRC
pip install numpy matplotlib PyQt5
cd NvidiaWrap-GarmentCodeRC
export CUDA_PATH=/usr/local/cuda
chmod +x ./tools/packman/packman
python build_lib.py
pip install -e .
cd ..
pip install pygarment

cat <<EOF > system.json
{
  "output": "./Logs/",
  "datasets_path": "",
  "datasets_sim": "",
  "sim_configs_path": "./assets/Sim_props",
  "bodies_default_path": "./assets/bodies",
  "body_samples_path": ""
}
EOF

cd ..
mkdir -p ./checkpoints/try_7b_lr1e_4_v3_garmentcontrol_4h100_v4_final
cd ./checkpoints/try_7b_lr1e_4_v3_garmentcontrol_4h100_v4_final
pip install gdown
gdown --id 1Qb8tSJghEDOmcZaOpU66IX3KByIJ57AU
cd ../../

ln -s /workspace/ChatGarment/GarmentCodeRC/assets assets
