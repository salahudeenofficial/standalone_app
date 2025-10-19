#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

echo "🔧 Installing system dependencies..."

sudo add-apt-repository ppa:deadsnakes/ppa -y
echo "📦 Updating package list..."
sudo apt update

echo "🐍 Installing Python 3.10 and development packages..."
sudo apt install -y python3.10 python3.10-venv python3.10-dev

echo "🧱 Creating Python virtual environment..."
python3.10 -m venv venv

echo "✅ Activating virtual environment..."
source ./venv/bin/activate

echo "🔥 Installing PyTorch with CUDA 13.0 support..."
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130

echo "📦 Installing project requirements..."
pip install -r requirements.txt

echo "⬇️ Downloading models..."
bash download_models.sh

echo "✅ Setup complete!"
