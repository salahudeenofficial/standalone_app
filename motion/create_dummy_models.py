#!/usr/bin/env python3
"""
Create dummy model files for testing on Vast AI instance
"""

import torch
import os
import sys

def create_dummy_model_files():
    """Create dummy model files for testing"""
    print("🧪 Creating dummy model files for testing...")
    
    # Create directories
    os.makedirs("models/diffusion_models", exist_ok=True)
    os.makedirs("models/text_encoders", exist_ok=True)
    os.makedirs("models/vaes", exist_ok=True)
    
    # Create dummy UNet model (WAN2.1 style)
    print("📦 Creating dummy UNet model...")
    dummy_unet = {
        'model.diffusion_model.input_blocks.0.0.weight': torch.randn(320, 16, 3, 3, 3),
        'model.diffusion_model.input_blocks.0.0.bias': torch.randn(320),
        'model.diffusion_model.middle_block.0.weight': torch.randn(320, 320, 3, 3, 3),
        'model.diffusion_model.middle_block.0.bias': torch.randn(320),
        'model.diffusion_model.output_blocks.0.0.weight': torch.randn(320, 320, 3, 3, 3),
        'model.diffusion_model.output_blocks.0.0.bias': torch.randn(320),
        'model.diffusion_model.out.0.weight': torch.randn(16, 320, 3, 3, 3),
        'model.diffusion_model.out.0.bias': torch.randn(16),
    }
    
    # Add more dummy parameters to make it realistic
    for i in range(10):
        dummy_unet[f'model.diffusion_model.input_blocks.{i}.0.weight'] = torch.randn(320, 320, 3, 3, 3)
        dummy_unet[f'model.diffusion_model.input_blocks.{i}.0.bias'] = torch.randn(320)
    
    torch.save(dummy_unet, "models/diffusion_models/wan_2.1_diffusion_model.safetensors")
    print("✅ Dummy UNet model created")
    
    # Create dummy CLIP model
    print("📦 Creating dummy CLIP model...")
    dummy_clip = {
        'text_model.embeddings.token_embedding.weight': torch.randn(49408, 4096),
        'text_model.embeddings.position_embedding.weight': torch.randn(77, 4096),
        'text_model.encoder.layers.0.self_attn.q_proj.weight': torch.randn(4096, 4096),
        'text_model.encoder.layers.0.self_attn.k_proj.weight': torch.randn(4096, 4096),
        'text_model.encoder.layers.0.self_attn.v_proj.weight': torch.randn(4096, 4096),
        'text_model.encoder.layers.0.self_attn.out_proj.weight': torch.randn(4096, 4096),
        'text_model.encoder.layers.0.layer_norm1.weight': torch.randn(4096),
        'text_model.encoder.layers.0.layer_norm1.bias': torch.randn(4096),
        'text_model.encoder.layers.0.mlp.fc1.weight': torch.randn(16384, 4096),
        'text_model.encoder.layers.0.mlp.fc1.bias': torch.randn(16384),
        'text_model.encoder.layers.0.mlp.fc2.weight': torch.randn(4096, 16384),
        'text_model.encoder.layers.0.mlp.fc2.bias': torch.randn(4096),
        'text_model.encoder.layers.0.layer_norm2.weight': torch.randn(4096),
        'text_model.encoder.layers.0.layer_norm2.bias': torch.randn(4096),
    }
    
    torch.save(dummy_clip, "models/text_encoders/wan_clip_model.safetensors")
    print("✅ Dummy CLIP model created")
    
    # Create dummy VAE model
    print("📦 Creating dummy VAE model...")
    dummy_vae = {
        'encoder.conv_in.weight': torch.randn(128, 3, 3, 3),
        'encoder.conv_in.bias': torch.randn(128),
        'encoder.conv_out.weight': torch.randn(16, 128, 3, 3),
        'encoder.conv_out.bias': torch.randn(16),
        'decoder.conv_in.weight': torch.randn(128, 16, 3, 3),
        'decoder.conv_in.bias': torch.randn(128),
        'decoder.conv_out.weight': torch.randn(3, 128, 3, 3),
        'decoder.conv_out.bias': torch.randn(3),
    }
    
    torch.save(dummy_vae, "models/vaes/wan_vae.safetensors")
    print("✅ Dummy VAE model created")
    
    print("\n🎉 All dummy model files created!")
    print("📁 Model files:")
    print("   models/diffusion_models/wan_2.1_diffusion_model.safetensors")
    print("   models/text_encoders/wan_clip_model.safetensors")
    print("   models/vaes/wan_vae.safetensors")
    print("\n🚀 You can now test the pipeline with these dummy models!")

if __name__ == "__main__":
    create_dummy_model_files()
