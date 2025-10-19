#!/bin/bash
sudo apt install -y software-properties-common
sudo apt update 
sudo apt install -y python3.10 python3.10-venv python3.10-dev

python3.10 -m venv venv
source ./venv/bin/activate
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
bash download_models.sh

