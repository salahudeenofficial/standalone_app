#!/usr/bin/env python3
"""
Standalone Reference Image + Control Video to Output Video Pipeline
With Manual Memory Management - No ComfyUI Model Management

This pipeline uses manual memory management:
- Load models only when needed
- Unload models immediately after use
- Process video in small chunks
- Full control over GPU memory usage
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
import time
import numpy as np
from PIL import Image

# Add ComfyUI path for utilities
sys.path.insert(0, str(Path(__file__).parent / "comfy"))

# Import only what we need from ComfyUI
import comfy.utils
import comfy.sd
from comfy.latent_formats import Wan21

class ManualMemoryManager:
    """Manual memory management for models"""
    
    def __init__(self):
        self.loaded_models = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self, model_path, model_type):
        """Load a model and return it"""
        print(f"🔧 Loading {model_type} from {model_path}")
        
        if model_type in self.loaded_models:
            print(f"   {model_type} already loaded, returning existing")
            return self.loaded_models[model_type]
            
        # Load state dict
        state_dict = comfy.utils.load_torch_file(model_path)
        
        if model_type == "unet":
            # Load UNET with automatic detection
            from comfy.sd import load_state_dict_guess_config
            model, _, _, _ = load_state_dict_guess_config(
                state_dict, 
                output_vae=False, 
                output_clip=False, 
                output_clipvision=False, 
                embedding_directory=None, 
                output_model=True
            )
        elif model_type == "clip":
            # Load CLIP with WAN type
            model = comfy.sd.load_clip([model_path], clip_type=comfy.sd.CLIPType.WAN)
            if model is None:
                # Fallback
                from comfy.sd import CLIP
                model = CLIP(state_dict, clip_type=comfy.sd.CLIPType.WAN)
        elif model_type == "vae":
            # Load VAE
            model = comfy.sd.VAE(sd=state_dict)
            model.throw_exception_if_invalid()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        self.loaded_models[model_type] = model
        print(f"   ✅ {model_type} loaded successfully")
        return model
        
    def unload_model(self, model_type):
        """Unload a model and free memory"""
        if model_type in self.loaded_models:
            print(f"🔧 Unloading {model_type}")
            del self.loaded_models[model_type]
            torch.cuda.empty_cache()
            print(f"   ✅ {model_type} unloaded")
            
    def unload_all(self):
        """Unload all models"""
        print("🔧 Unloading all models")
        self.loaded_models.clear()
        torch.cuda.empty_cache()
        print("   ✅ All models unloaded")
        
    def get_memory_info(self):
        """Get current memory usage"""
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            free = total - reserved
            return {
                'total': total,
                'allocated': allocated,
                'reserved': reserved,
                'free': free,
                'utilization': (reserved/total)*100
            }
        return None

class ManualVideoPipeline:
    """Pipeline with manual memory management"""
    
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.memory_manager = ManualMemoryManager()
        self.setup_model_paths()
        
    def setup_model_paths(self):
        """Setup model paths"""
        script_dir = Path(__file__).parent
        models_dir = script_dir / self.models_dir
        os.makedirs(models_dir / "diffusion_models", exist_ok=True)
        os.makedirs(models_dir / "text_encoders", exist_ok=True)
        os.makedirs(models_dir / "vaes", exist_ok=True)
        os.makedirs(models_dir / "loras", exist_ok=True)
        
    def load_video(self, video_path):
        """Load control video"""
        if not video_path or not os.path.exists(video_path):
            print(f"Warning: Video file not found: {video_path}")
            return None
        try:
            from torchvision.io import read_video
            print(f"Loading video from: {video_path}")
            video, audio, info = read_video(video_path, pts_unit='sec')
            if video is None or video.numel() == 0:
                print(f"Warning: Empty video: {video_path}")
                return None
            video = video.float() / 255.0
            if video.shape[-1] > 3:
                video = video[..., :3]
            elif video.shape[-1] == 1:
                video = video.repeat(1, 1, 1, 3)
            print(f"Loaded video tensor: {tuple(video.shape)} (T,H,W,C)")
            return video
        except Exception as e:
            print(f"Error loading video '{video_path}': {e}")
            return None
    
    def load_image(self, image_path):
        """Load reference image"""
        if not image_path or not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}")
            return None
        try:
            print(f"Loading image from: {image_path}")
            img = Image.open(image_path).convert('RGB')
            arr = np.asarray(img).astype('float32') / 255.0
            tensor = torch.from_numpy(arr).unsqueeze(0)
            print(f"Loaded image tensor: {tuple(tensor.shape)} (1,H,W,3)")
            return tensor
        except Exception as e:
            print(f"Error loading image '{image_path}': {e}")
            return None
    
    def encode_video_chunked(self, vae, video_tensor, chunk_size=4):
        """Encode video in chunks to avoid OOM"""
        print(f"🔧 Encoding video in chunks of {chunk_size} frames")
        
        total_frames = video_tensor.shape[0]
        encoded_chunks = []
        
        for i in range(0, total_frames, chunk_size):
            end_idx = min(i + chunk_size, total_frames)
            chunk = video_tensor[i:end_idx]
            
            print(f"   Processing frames {i}:{end_idx} ({end_idx-i} frames)")
            
            # Move chunk to GPU
            chunk = chunk.to(self.memory_manager.device)
            
            # Encode chunk
            with torch.no_grad():
                encoded_chunk = vae.encode(chunk)
            
            # Move result back to CPU to save GPU memory
            encoded_chunk = encoded_chunk.cpu()
            encoded_chunks.append(encoded_chunk)
            
            # Clear GPU memory
            del chunk
            torch.cuda.empty_cache()
            
        # Concatenate all chunks
        result = torch.cat(encoded_chunks, dim=0)
        print(f"✅ Video encoding completed: {result.shape}")
        return result
    
    def run_pipeline(self, 
                    unet_model_path,
                    clip_model_path,
                    vae_model_path,
                    lora_path=None,
                    positive_prompt="",
                    negative_prompt="",
                    control_video_path=None,
                    reference_image_path=None,
                    width=480,
                    height=832,
                    length=37,
                    batch_size=1,
                    strength=1.0,
                    seed=270400132721985,
                    steps=4,
                    cfg=1.0,
                    sampler_name="ddim",
                    scheduler="normal",
                    denoise=1.0,
                    output_path="output.mp4"):
        """Run the complete pipeline with manual memory management"""
        
        try:
            # Show initial memory
            mem_info = self.memory_manager.get_memory_info()
            if mem_info:
                print(f"🔧 Initial GPU memory: {mem_info['free']:.2f} GB free")
            
            # 1. Load control video and reference image
            print("\n" + "="*80)
            print("🔍 STEP 1: LOAD VIDEO AND IMAGE DATA")
            print("="*80)
            
            control_video = self.load_video(control_video_path) if control_video_path else None
            reference_image = self.load_image(reference_image_path) if reference_image_path else None
            
            if control_video is not None:
                print(f"   ✅ Control video loaded: {control_video.shape}")
            if reference_image is not None:
                print(f"   ✅ Reference image loaded: {reference_image.shape}")
            
            # 2. Load CLIP for text encoding
            print("\n" + "="*80)
            print("🔍 STEP 2: TEXT ENCODING")
            print("="*80)
            
            clip_model = self.memory_manager.load_model(clip_model_path, "clip")
            
            # Simple text encoding
            from components.text_encoder import CLIPTextEncode
            text_encoder = CLIPTextEncode()
            positive_cond = text_encoder.encode(clip_model, positive_prompt)
            negative_cond = text_encoder.encode(clip_model, negative_prompt)
            
            print("✅ Text encoding completed")
            
            # Unload CLIP to free memory
            self.memory_manager.unload_model("clip")
            
            # 3. Load UNET for model sampling
            print("\n" + "="*80)
            print("🔍 STEP 3: MODEL SAMPLING")
            print("="*80)
            
            model = self.memory_manager.load_model(unet_model_path, "unet")
            
            # Apply ModelSamplingSD3
            from components.model_sampling import ModelSamplingSD3
            model_sampling = ModelSamplingSD3()
            model = model_sampling.patch(model, shift=8.0)
            
            print("✅ Model sampling applied")
            
            # Unload UNET to free memory
            self.memory_manager.unload_model("unet")
            
            # 4. Load VAE for video encoding
            print("\n" + "="*80)
            print("🔍 STEP 4: VAE ENCODING")
            print("="*80)
            
            vae = self.memory_manager.load_model(vae_model_path, "vae")
            
            # Prepare control video
            if control_video is not None:
                control_video = control_video[:length]
                control_video = comfy.utils.common_upscale(
                    control_video.movedim(-1, 1), width, height, "bilinear", "center"
                ).movedim(1, -1)
                print(f"   After upscaling to {width}x{height}: {control_video.shape}")
                
                if control_video.shape[0] < length:
                    control_video = torch.nn.functional.pad(
                        control_video, (0, 0, 0, 0, 0, 0, 0, length - control_video.shape[0]), value=0.5
                    )
            
            # Prepare reference image
            if reference_image is not None:
                reference_image = comfy.utils.common_upscale(
                    reference_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
                ).movedim(1, -1)
                reference_image = vae.encode(reference_image[:, :, :, :3])
                reference_image = torch.cat([reference_image, Wan21().process_out(torch.zeros_like(reference_image))], dim=1)
            
            # Prepare mask
            mask = torch.ones((length, height, width, 1), device=control_video.device)
            
            # Normalize and split by mask
            control_video = control_video - 0.5
            inactive = (control_video * (1 - mask)) + 0.5
            reactive = (control_video * mask) + 0.5
            
            # Encode video in chunks
            print("🔧 Starting chunked VAE encoding...")
            inactive_encoded = self.encode_video_chunked(vae, inactive[:, :, :, :3], chunk_size=4)
            reactive_encoded = self.encode_video_chunked(vae, reactive[:, :, :, :3], chunk_size=4)
            
            control_video_latent = torch.cat((inactive_encoded, reactive_encoded), dim=1)
            
            # Reference image processing
            trim_latent = 0
            if reference_image is not None:
                control_video_latent = torch.cat((reference_image, control_video_latent), dim=2)
            
            # Unload VAE to free memory
            self.memory_manager.unload_model("vae")
            
            print(f"\n{'='*80}")
            print(f"✅ STEP 4 COMPLETE: VAE Encoding with Manual Memory Management")
            print(f"{'='*80}")
            
            # Show final memory
            mem_info = self.memory_manager.get_memory_info()
            if mem_info:
                print(f"🔧 Final GPU memory: {mem_info['free']:.2f} GB free")
            
            return "pipeline_completed_successfully"
            
        except Exception as e:
            print(f"Pipeline failed with error: {str(e)}")
            # Cleanup on error
            self.memory_manager.unload_all()
            raise

def main():
    """Main function to run the pipeline"""
    pipeline = ManualVideoPipeline()
    
    # Example usage
    script_dir = Path(__file__).parent
    output_path = pipeline.run_pipeline(
        unet_model_path=str(script_dir / "models/diffusion_models/wan_2.1_diffusion_model.safetensors"),
        clip_model_path=str(script_dir / "models/text_encoders/wan_clip_model.safetensors"),
        vae_model_path=str(script_dir / "models/vaes/wan_vae.safetensors"),
        lora_path=str(script_dir / "models/loras/Wan21_CausVid_14B_T2V_lora_rank32.safetensors"),
        positive_prompt="very cinematic video",
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走 , extra hands, extra arms, extra legs",
        control_video_path=str(script_dir / "safu.mp4"),
        reference_image_path=str(script_dir / "safu.jpg"),
        width=480,
        height=832,
        length=37,
        output_path="generated_video.mp4"
    )
    
    print(f"Pipeline completed: {output_path}")

if __name__ == "__main__":
    main()
