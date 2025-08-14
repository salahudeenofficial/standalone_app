#!/bin/bash

# Standalone App Model Downloader - Updated for ComfyUI Native Loading
# Usage: ./download_models.sh

echo "Creating model directories..."
mkdir -p models/checkpoints models/loras

echo "Downloading models for standalone app using ComfyUI native loading..."

# Main Checkpoint (contains UNET, CLIP, and VAE)
echo "Downloading main checkpoint (UNET + CLIP + VAE)..."
wget -c --timeout=0 --tries=0 --retry-connrefused --no-check-certificate --progress=bar:force:noscroll \
  -O ./models/checkpoints/wan_2.1_complete.safetensors \
  "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/wan_2.1_complete.safetensors"

# Alternative: If the complete checkpoint is too large, download individual components
# and we'll modify the pipeline to use them separately
echo "Downloading individual components as backup..."
mkdir -p models/diffusion_models models/text_encoders models/vaes

# UNET Model
wget -c --timeout=0 --tries=0 --retry-connrefused --no-check-certificate --progress=bar:force:noscroll \
  -O ./models/diffusion_models/wan_2.1_diffusion_model.safetensors \
  "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_vace_14B_fp16.safetensors"

# CLIP Model  
wget -c --timeout=0 --tries=0 --retry-connrefused --no-check-certificate --progress=bar:force:noscroll \
  -O ./models/text_encoders/wan_clip_model.safetensors \
  "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp16.safetensors"

# VAE Model
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
echo "Note: The pipeline now uses ComfyUI's native model loading system."
echo "If the complete checkpoint download fails, the pipeline will fall back to individual components." 