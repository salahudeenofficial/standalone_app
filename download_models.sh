#!/bin/bash

# Standalone App Model Downloader - Updated for Individual Component Loading
# Usage: ./download_models.sh

echo "Creating model directories..."
mkdir -p models/diffusion_models models/text_encoders models/vaes models/loras

echo "Downloading individual model components for standalone app..."

# UNET Model
echo "Downloading UNET model..."
wget -c --timeout=0 --tries=0 --retry-connrefused --no-check-certificate --progress=bar:force:noscroll \
  -O ./models/diffusion_models/wan_2.1_diffusion_model.safetensors \
  "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_vace_14B_fp16.safetensors"

# CLIP Model  
echo "Downloading CLIP model..."
wget -c --timeout=0 --tries=0 --retry-connrefused --no-check-certificate --progress=bar:force:noscroll \
  -O ./models/text_encoders/wan_clip_model.safetensors \
  "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp16.safetensors"

# VAE Model
echo "Downloading VAE model..."
wget -c --timeout=0 --tries=0 --retry-connrefused --no-check-certificate --progress=bar:force:noscroll \
  -O ./models/vaes/wan_vae.safetensors \
  "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors"

# LoRA Model
echo "Downloading LoRA model..."
wget -c --timeout=0 --tries=0 --retry-connrefused --no-check-certificate --progress=bar:force:noscroll \
  -O ./models/loras/Wan21_CausVid_14B_T2V_lora_rank32.safetensors \
  "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan21_CausVid_14B_T2V_lora_rank32.safetensors"

echo "All models downloaded!"
echo ""
echo "Note: Using individual component loading approach for better compatibility." 