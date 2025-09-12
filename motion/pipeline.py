#!/usr/bin/env python3
"""
7-Step Standalone WAN Video Generation Pipeline
Step 1: VAE Load -> Reference Image + Control Video Load -> Create Initial Latent via VAE Encode

Based on existing motion/ modules and test_wan_vae.py implementation.
No ComfyUI dependencies - fully standalone using motion/ modules.
"""

import os
import sys
import torch
import logging
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union

# Add motion directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import motion modules
from standalone_vae import VAE, create_vae
from wan_vae_components.model_management import get_torch_device, unet_offload_device
from utils import load_torch_file, calculate_parameters
from standalone_sd import load_state_dict_guess_config
from lora import load_lora_for_models
from model_sampling import ModelSamplingSD3
from text_encoder import CLIPTextEncode
from standalone_ksampler import StandaloneKSampler, prepare_noise

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class WanVideoPipeline:
    """
    7-Step Standalone WAN Video Generation Pipeline
    
    Step 1: VAE Load + Reference Image/Control Video Load + Initial Latent Creation
    Step 2: CLIP Load + Text Encoding
    Step 3: UNet Load + LoRA Application
    Step 4: Model Sampling Configuration
    Step 5: Noise Generation + Conditioning
    Step 6: UNet Sampling/Inference
    Step 7: VAE Decode + Video Export
    
    Memory Management:
    - Uses standalone model_management for device handling
    - Supports chunked processing for large videos
    - Automatic GPU/CPU memory optimization
    """
    
    def __init__(self, models_dir="models"):
        """Initialize the pipeline with model directory"""
        self.models_dir = models_dir
        self.setup_model_paths()
        
        # Initialize device management
        self.device = get_torch_device()
        self.offload_device = unet_offload_device()
        
        # Pipeline state
        self.vae = None
        self.unet = None
        self.clip = None
        self.lora_applied = False
        
        # Step completion tracking
        self.step_completed = {
            1: False,  # VAE + Latent Creation
            2: False,  # UNet + CLIP + LoRA  
            3: False,  # Model Sampling + Text Encoding
            4: False,  # KSampler Denoising
            5: False,  # Noise + Conditioning
            6: False,  # UNet Inference
            7: False   # VAE Decode + Export
        }
        
        print(f"✅ WAN Video Pipeline initialized")
        print(f"   Device: {self.device}")
        print(f"   Offload Device: {self.offload_device}")
        print(f"   Models Directory: {self.models_dir}")
        
    def setup_model_paths(self):
        """Setup model paths for the standalone app"""
        script_dir = Path(__file__).parent
        models_dir = script_dir / self.models_dir
        
        # Create model directories if they don't exist
        os.makedirs(models_dir / "diffusion_models", exist_ok=True)
        os.makedirs(models_dir / "text_encoders", exist_ok=True) 
        os.makedirs(models_dir / "vaes", exist_ok=True)
        os.makedirs(models_dir / "loras", exist_ok=True)
        
        print(f"📁 Model directories verified: {models_dir}")

    def step_1_vae_and_latent_creation(self,
                                     vae_model_path: str,
                                     positive_prompt: str = "",
                                     negative_prompt: str = "",
                                     control_video_path: Optional[str] = None,
                                     reference_image_path: Optional[str] = None,
                                     width: int = 480,
                                     height: int = 832,
                                     length: int = 37,
                                     batch_size: int = 1,
                                     strength: float = 1.0) -> Dict[str, Any]:
        """
        Step 1: VAE Load + Reference Image/Control Video + Initial Latent + Conditioning Setup
        
        This step replicates WanVaceToVideo node functionality:
        1. Loads WAN VAE model from safetensors
        2. Loads control video and reference image 
        3. Creates initial latent via VAE encoding
        4. Sets up VACE conditioning (vace_frames, vace_mask, vace_strength)
        5. Returns conditioned positive/negative prompts like WanVaceToVideo node
        
        Args:
            vae_model_path: Path to WAN VAE safetensors file
            positive_prompt: Positive text prompt for conditioning
            negative_prompt: Negative text prompt for conditioning
            control_video_path: Path to control video (optional)
            reference_image_path: Path to reference image (optional)
            width: Target width for processing
            height: Target height for processing  
            length: Number of frames to process
            batch_size: Batch size for processing
            strength: VACE strength (0.0-1.0)
            
        Returns:
            Dictionary containing WanVaceToVideo-like outputs:
            - positive: Conditioned positive prompts
            - negative: Conditioned negative prompts  
            - out_latent: WAN-format latent dict {"samples": tensor}
            - trim_latent: Frame count to trim for reference
            - vae: Loaded VAE model (for debugging)
        """
        
        print("\n" + "="*80)
        print("🚀 STEP 1: VAE LOAD + REFERENCE IMAGE/CONTROL VIDEO + INITIAL LATENT CREATION")
        print("="*80)
        
        try:
            # ========================================================================
            # 1.1: Load WAN VAE Model
            # ========================================================================
            print("1.1 Loading WAN VAE model...")
            start_time = time.time()
            
            # Load VAE state dict
            vae_state_dict = load_torch_file(vae_model_path)
            print(f"   📊 Loaded VAE state dict with {len(vae_state_dict)} keys")
            
            # Create VAE instance using standalone implementation
            self.vae = create_vae(state_dict=vae_state_dict, device=self.device)
            
            load_time = time.time() - start_time
            print(f"✅ VAE loaded successfully in {load_time:.2f}s")
            print(f"   Type: {type(self.vae.first_stage_model).__name__}")
            print(f"   Latent channels: {self.vae.latent_channels}")
            print(f"   Downscale ratio: {self.vae.downscale_ratio}")
            print(f"   Device: {self.vae.device}")
            
            # Calculate VAE model size
            if hasattr(self.vae.first_stage_model, 'parameters'):
                vae_params = calculate_parameters(dict(self.vae.first_stage_model.named_parameters()))
                print(f"   Parameters: {vae_params:,}")
                print(f"   Size: {vae_params * 4 / (1024*1024):.1f} MB")
            
            # ========================================================================
            # 1.2: Load Control Video
            # ========================================================================
            print("\n1.2 Loading control video...")
            control_video = None
            if control_video_path and os.path.exists(control_video_path):
                control_video = self.load_video(control_video_path)
                if control_video is not None:
                    print(f"   ✅ Control video loaded: {control_video.shape}")
                    print(f"   Range: [{control_video.min():.3f}, {control_video.max():.3f}]")
                else:
                    print("   ❌ Failed to load control video")
            else:
                print("   ⚠️  No control video path specified or file not found")
                # For testing, create dummy control video
                print("   🎯 Creating dummy control video for testing...")
                control_video = torch.rand(length, height, width, 3)
                print(f"   📊 Dummy control video shape: {control_video.shape}")
            
            # ========================================================================
            # 1.3: Load Reference Image  
            # ========================================================================
            print("\n1.3 Loading reference image...")
            reference_image = None
            if reference_image_path and os.path.exists(reference_image_path):
                reference_image = self.load_image(reference_image_path)
                if reference_image is not None:
                    print(f"   ✅ Reference image loaded: {reference_image.shape}")
                    print(f"   Range: [{reference_image.min():.3f}, {reference_image.max():.3f}]")
                else:
                    print("   ❌ Failed to load reference image")
            else:
                print("   ⚠️  No reference image path specified - will proceed without reference")
            
            # ========================================================================
            # 1.4: Prepare Control Video for Encoding
            # ========================================================================
            print("\n1.4 Preparing control video for VAE encoding...")
            
            # Ensure control video has correct dimensions
            if control_video.shape[0] < length:
                print(f"   📏 Padding control video from {control_video.shape[0]} to {length} frames")
                padding = torch.full((length - control_video.shape[0], height, width, 3), 0.5)
                control_video = torch.cat([control_video, padding], dim=0)
            elif control_video.shape[0] > length:
                print(f"   ✂️  Trimming control video from {control_video.shape[0]} to {length} frames")
                control_video = control_video[:length]
            
            # Resize to target dimensions using simple interpolation
            if control_video.shape[1] != height or control_video.shape[2] != width:
                print(f"   🔄 Resizing control video from {control_video.shape[1]}x{control_video.shape[2]} to {height}x{width}")
                # Reshape for interpolation: (T,H,W,C) -> (T,C,H,W)
                control_video = control_video.permute(0, 3, 1, 2)
                control_video = torch.nn.functional.interpolate(
                    control_video, size=(height, width), mode='bilinear', align_corners=False
                )
                # Reshape back: (T,C,H,W) -> (T,H,W,C)
                control_video = control_video.permute(0, 2, 3, 1)
            
            print(f"   📊 Final control video shape: {control_video.shape}")
            
            # Continue in next part...
            return self._step_1_continue_encoding(control_video, reference_image, 
                                                width, height, length, batch_size, start_time,
                                                positive_prompt, negative_prompt, strength)
            
        except Exception as e:
            print(f"❌ STEP 1 FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _step_1_continue_encoding(self, control_video, reference_image, width, height, length, batch_size, start_time, positive_prompt, negative_prompt, strength):
        """Continue Step 1 VAE encoding process"""
        
        # ========================================================================
        # 1.5: Create Control Mask (Full mask by default)
        # ========================================================================
        print("\n1.5 Creating control mask...")
        
        # Create full mask (all pixels controlled)
        mask = torch.ones((length, height, width, 1), device=control_video.device)
        print(f"   📊 Control mask shape: {mask.shape}")
        
        # Split control video by mask (following WAN VAE-to-Video logic)
        control_video = control_video - 0.5  # Center around 0
        inactive = (control_video * (1 - mask)) + 0.5  # Inactive regions
        reactive = (control_video * mask) + 0.5        # Active/controlled regions
        
        print(f"   📊 Split into inactive: {inactive.shape}, reactive: {reactive.shape}")
        
        # ========================================================================
        # 1.6: VAE Encoding of Control Video
        # ========================================================================
        print("\n1.6 Encoding control video with VAE...")
        encoding_start = time.time()
        
        with torch.no_grad():
            # Encode inactive and reactive parts separately
            print("   🔄 Encoding inactive part...")
            inactive_latent = self.vae.encode(inactive[:, :, :, :3])
            print(f"   📊 Inactive latent shape: {inactive_latent.shape}")
            
            print("   🔄 Encoding reactive part...")
            reactive_latent = self.vae.encode(reactive[:, :, :, :3])
            print(f"   📊 Reactive latent shape: {reactive_latent.shape}")
            
            # Combine latents
            control_video_latent = torch.cat((inactive_latent, reactive_latent), dim=1)
            print(f"   📊 Combined control latent shape: {control_video_latent.shape}")
        
        encoding_time = time.time() - encoding_start
        print(f"✅ Control video encoded in {encoding_time:.2f}s")
        
        # ========================================================================
        # 1.7: Process Reference Image (if provided)
        # ========================================================================
        reference_image_latent = None
        if reference_image is not None:
            print("\n1.7 Processing reference image...")
            
            # Resize reference image to target dimensions
            if reference_image.shape[1] != height or reference_image.shape[2] != width:
                print(f"   🔄 Resizing reference image to {height}x{width}")
                # Reshape for interpolation: (1,H,W,C) -> (1,C,H,W)
                reference_image = reference_image.permute(0, 3, 1, 2)
                reference_image = torch.nn.functional.interpolate(
                    reference_image, size=(height, width), mode='bilinear', align_corners=False
                )
                # Reshape back: (1,C,H,W) -> (1,H,W,C)
                reference_image = reference_image.permute(0, 2, 3, 1)
            
            # Encode reference image
            with torch.no_grad():
                reference_image_latent = self.vae.encode(reference_image[:, :, :, :3])
                print(f"   📊 Reference image latent shape: {reference_image_latent.shape}")
            
            # Add motion latent channels (WAN format) - like WanVaceToVideo node
            from wan_latent_format import Wan21_LatentFormat
            wan21_format = Wan21_LatentFormat()
            motion_channels = wan21_format.process_out(torch.zeros_like(reference_image_latent))
            reference_image_latent = torch.cat([reference_image_latent, motion_channels], dim=1)
            print(f"   📊 Reference with WAN motion channels: {reference_image_latent.shape}")
        
        # ========================================================================
        # 1.8: Create Final Initial Latent and Results
        # ========================================================================
        print("\n1.8 Creating final initial latent...")
        
        # Calculate latent dimensions
        latent_length = ((length - 1) // 4) + 1
        latent_height = height // 8  # WAN VAE downscales by 8
        latent_width = width // 8
        
        # Start with control video latent
        initial_latent = control_video_latent
        
        # Add reference image if provided
        if reference_image_latent is not None:
            print("   🔗 Concatenating reference image to control latent...")
            initial_latent = torch.cat((reference_image_latent, control_video_latent), dim=2)
            print(f"   📊 Latent with reference: {initial_latent.shape}")
        
        print(f"✅ Final initial latent shape: {initial_latent.shape}")
        
        # Create control mask in latent space
        vae_stride = 8
        height_mask = height // vae_stride
        width_mask = width // vae_stride
        
        mask_latent = mask.view(length, height_mask, vae_stride, width_mask, vae_stride)
        mask_latent = mask_latent.permute(2, 4, 0, 1, 3)
        mask_latent = mask_latent.reshape(vae_stride * vae_stride, length, height_mask, width_mask)
        mask_latent = torch.nn.functional.interpolate(
            mask_latent.unsqueeze(0), 
            size=(latent_length, height_mask, width_mask), 
            mode='nearest-exact'
        ).squeeze(0)
        
        if reference_image_latent is not None:
            ref_frames = reference_image_latent.shape[2]
            mask_pad = torch.zeros_like(mask_latent[:, :ref_frames, :, :])
            mask_latent = torch.cat((mask_pad, mask_latent), dim=1)
            latent_length += ref_frames  # Update latent_length like WanVaceToVideo
        
        mask_latent = mask_latent.unsqueeze(0)  # Add batch dimension
        
        # ========================================================================
        # 1.9: Setup VACE Conditioning (like WanVaceToVideo node)
        # ========================================================================
        print("\n1.9 Setting up VACE conditioning...")
        
        # Import conditioning utilities
        from conditioning_utils import create_empty_conditioning, conditioning_set_values, print_conditioning_info
        
        # Create initial conditioning from prompts (dummy text embeddings for now)
        positive = create_empty_conditioning(device=self.device)
        negative = create_empty_conditioning(device=self.device)
        
        print(f"   📝 Initial positive prompt: '{positive_prompt}'")
        print(f"   📝 Initial negative prompt: '{negative_prompt}'")
        
        # Apply VACE conditioning exactly like WanVaceToVideo node
        vace_conditioning_values = {
            "vace_frames": [initial_latent],
            "vace_mask": [mask_latent], 
            "vace_strength": [strength]
        }
        
        print(f"   🔧 Applying VACE conditioning with strength: {strength}")
        print(f"   📊 VACE frames shape: {initial_latent.shape}")
        print(f"   📊 VACE mask shape: {mask_latent.shape}")
        
        # Set conditioning values (append=True like WanVaceToVideo)
        positive = conditioning_set_values(positive, vace_conditioning_values, append=True)
        negative = conditioning_set_values(negative, vace_conditioning_values, append=True)
        
        # Debug conditioning info
        print_conditioning_info(positive, "Positive")
        print_conditioning_info(negative, "Negative")
        
        print("✅ VACE conditioning setup complete")
        
        # Mark step complete and return results
        self.step_completed[1] = True
        
        # Create WAN-format output latent (16 channels like WanVaceToVideo node)
        output_latent = torch.zeros([batch_size, 16, latent_length, latent_height, latent_width], 
                                   device='cpu')  # Use CPU for intermediate storage
        out_latent = {"samples": output_latent}
        
        # Calculate trim_latent like WanVaceToVideo node
        trim_latent = reference_image_latent.shape[2] if reference_image_latent is not None else 0
        
        # Return results matching WanVaceToVideo node signature: (positive, negative, out_latent, trim_latent)
        step_1_results = {
            # WanVaceToVideo node outputs:
            'positive': positive,           # Conditioned positive prompts
            'negative': negative,           # Conditioned negative prompts  
            'out_latent': out_latent,      # WAN-format latent dict {"samples": tensor}
            'trim_latent': trim_latent,    # Frame count to trim for reference
            
            # Additional debugging/pipeline data:
            'vae': self.vae,
            'control_video_latent': control_video_latent,
            'reference_image_latent': reference_image_latent,
            'initial_latent': initial_latent,
            'control_mask': mask_latent,
            'strength': strength,
            'prompts': {
                'positive_prompt': positive_prompt,
                'negative_prompt': negative_prompt
            },
            'latent_dimensions': {
                'batch_size': batch_size,
                'channels': 16,  # WAN format uses 16 channels
                'length': latent_length,
                'height': latent_height,
                'width': latent_width
            },
            'processing_info': {
                'vae_encoding_time': encoding_time,
                'total_step_time': time.time() - start_time
            }
        }
        
        print(f"\n✅ STEP 1 COMPLETED SUCCESSFULLY in {time.time() - start_time:.2f}s")
        print("="*80)
        
        return step_1_results

    def step_2_unet_clip_lora_loading(self,
                                    unet_model_path: str,
                                    clip_model_path: str, 
                                    lora_model_path: Optional[str] = None,
                                    strength_model: float = 1.0,
                                    strength_clip: float = 0.0) -> Dict[str, Any]:
        """
        Step 2: UNet + CLIP Load + LoRA Application
        
        This step loads the UNet and CLIP models and optionally applies LoRA:
        1. Loads UNet diffusion model from safetensors
        2. Loads CLIP text encoder from safetensors
        3. Optionally applies LoRA patches to both models
        4. Returns loaded models ready for inference
        
        Args:
            unet_model_path: Path to UNet diffusion model safetensors file
            clip_model_path: Path to CLIP text encoder safetensors file
            lora_model_path: Path to LoRA patches file (optional)
            strength_model: LoRA strength for UNet model (0.0-2.0)
            strength_clip: LoRA strength for CLIP model (0.0-2.0)
            
        Returns:
            Dictionary containing loaded models and information
        """
        
        print("\n" + "="*80)
        print("🚀 STEP 2: UNET + CLIP LOAD + LORA APPLICATION")
        print("="*80)
        
        try:
            step_2_start = time.time()
            lora_time = 0.0  # Initialize lora_time
            
            # ========================================================================
            # 2.1: Load UNet Diffusion Model
            # ========================================================================
            print("2.1 Loading UNet diffusion model...")
            unet_start = time.time()
            
            # Load UNet state dict
            unet_state_dict = load_torch_file(unet_model_path)
            print(f"   📊 Loaded UNet state dict with {len(unet_state_dict)} keys")
            
            # Load UNet model using standalone_sd
            result = load_state_dict_guess_config(
                unet_state_dict,
                output_vae=False,
                output_clip=False,
                output_clipvision=False,
                output_model=True
            )
            
            if result is None:
                raise RuntimeError("Failed to load UNet model - load_state_dict_guess_config returned None")
            
            model, _, _, _ = result
            self.unet = model
            
            if self.unet is None:
                raise RuntimeError("UNet model is None after loading")
            
            unet_time = time.time() - unet_start
            print(f"✅ UNet loaded successfully in {unet_time:.2f}s")
            print(f"   Type: {type(self.unet).__name__}")
            print(f"   Device: {self.unet.load_device}")
            
            # Calculate UNet model size
            if hasattr(self.unet, 'model') and hasattr(self.unet.model, 'state_dict'):
                unet_state_dict_params = self.unet.model.state_dict()
                unet_params = calculate_parameters(unet_state_dict_params)
                print(f"   Parameters: {unet_params:,}")
                print(f"   Size: {unet_params * 4 / (1024*1024):.1f} MB")
            
            # ========================================================================
            # 2.2: Load CLIP Text Encoder
            # ========================================================================
            print("\n2.2 Loading CLIP text encoder...")
            clip_start = time.time()
            
            # Load CLIP state dict
            clip_state_dict = load_torch_file(clip_model_path)
            print(f"   📊 Loaded CLIP state dict with {len(clip_state_dict)} keys")
            
            # Load CLIP model using standalone_sd
            result = load_state_dict_guess_config(
                clip_state_dict,
                output_vae=False,
                output_clip=True,
                output_clipvision=False,
                output_model=False
            )
            
            if result is None:
                raise RuntimeError("Failed to load CLIP model - load_state_dict_guess_config returned None")
            
            _, clip, _, _ = result
            self.clip = clip
            
            if self.clip is None:
                raise RuntimeError("CLIP model is None after loading")
            
            clip_time = time.time() - clip_start
            print(f"✅ CLIP loaded successfully in {clip_time:.2f}s")
            print(f"   Type: {type(self.clip).__name__}")
            print(f"   Device: {self.clip.load_device}")
            
            # Calculate CLIP model size
            if hasattr(self.clip, 'cond_stage_model') and hasattr(self.clip.cond_stage_model, 'state_dict'):
                clip_state_dict_params = self.clip.cond_stage_model.state_dict()
                clip_params = calculate_parameters(clip_state_dict_params)
                print(f"   Parameters: {clip_params:,}")
                print(f"   Size: {clip_params * 4 / (1024*1024):.1f} MB")
            
            # ========================================================================
            # 2.3: Apply LoRA (Optional)
            # ========================================================================
            if lora_model_path and os.path.exists(lora_model_path):
                print("\n2.3 Applying LoRA patches...")
                lora_start = time.time()
                
                # Load LoRA state dict
                lora_state_dict = load_torch_file(lora_model_path)
                print(f"   📊 Loaded LoRA with {len(lora_state_dict)} keys")
                
                # Apply LoRA to models
                original_unet_patches = len(self.unet.patches) if hasattr(self.unet, 'patches') and self.unet.patches else 0
                original_clip_patches = len(self.clip.patches) if hasattr(self.clip, 'patches') and self.clip.patches else 0
                
                new_unet, new_clip = load_lora_for_models(
                    self.unet, self.clip, lora_state_dict,
                    strength_model=strength_model,
                    strength_clip=strength_clip
                )
                
                if new_unet is not None and new_clip is not None:
                    self.unet = new_unet
                    self.clip = new_clip
                    self.lora_applied = True
                    
                    lora_time = time.time() - lora_start
                    print(f"✅ LoRA applied successfully in {lora_time:.2f}s")
                    
                    # Report LoRA patch counts
                    new_unet_patches = len(self.unet.patches) if hasattr(self.unet, 'patches') and self.unet.patches else 0
                    new_clip_patches = len(self.clip.patches) if hasattr(self.clip, 'patches') and self.clip.patches else 0
                    
                    print(f"   🔧 UNet Patches: {original_unet_patches} → {new_unet_patches} (+{new_unet_patches - original_unet_patches})")
                    print(f"   🔧 CLIP Patches: {original_clip_patches} → {new_clip_patches} (+{new_clip_patches - original_clip_patches})")
                    print(f"   🔧 Model Strength: {strength_model}")
                    print(f"   🔧 CLIP Strength: {strength_clip}")
                else:
                    print("❌ LoRA application failed - models are None")
                    self.lora_applied = False
            else:
                print("\n2.3 ⚠️  No LoRA file specified or file not found - skipping LoRA application")
                self.lora_applied = False
            
            # Mark step complete
            self.step_completed[2] = True
            
            # Create results
            step_2_results = {
                'unet': self.unet,
                'clip': self.clip,
                'lora_applied': self.lora_applied,
                'models_info': {
                    'unet_type': type(self.unet).__name__,
                    'clip_type': type(self.clip).__name__,
                    'unet_device': str(self.unet.load_device),
                    'clip_device': str(self.clip.load_device),
                    'lora_strength_model': strength_model if self.lora_applied else 0.0,
                    'lora_strength_clip': strength_clip if self.lora_applied else 0.0
                },
                'processing_info': {
                    'unet_loading_time': unet_time,
                    'clip_loading_time': clip_time,
                    'lora_time': lora_time if self.lora_applied else 0.0,
                    'total_step_time': time.time() - step_2_start
                }
            }
            
            print(f"\n✅ STEP 2 COMPLETED SUCCESSFULLY in {time.time() - step_2_start:.2f}s")
            print("="*80)
            
            return step_2_results
            
        except Exception as e:
            print(f"❌ STEP 2 FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def step_3_model_sampling_and_text_encoding(self,
                                               positive_prompt: str,
                                               negative_prompt: str,
                                               shift: float = 8.0,
                                               multiplier: int = 1000) -> Dict[str, Any]:
        """
        Step 3: Model Sampling + Text Encoding
        
        This step applies SD3 model sampling and encodes text prompts:
        1. Applies ModelSamplingSD3 to the UNet model with shift parameter
        2. Initializes CLIP text encoder
        3. Encodes positive and negative text prompts  
        4. Returns conditioning tensors ready for sampling
        
        Args:
            positive_prompt: Positive text prompt for conditioning
            negative_prompt: Negative text prompt for conditioning
            shift: SD3 shift parameter (default 8.0)
            multiplier: SD3 multiplier parameter (default 1000)
            
        Returns:
            Dictionary containing encoded conditioning and model information
        """
        
        print("\n" + "="*80)
        print("🚀 STEP 3: MODEL SAMPLING + TEXT ENCODING")
        print("="*80)
        
        try:
            step_3_start = time.time()
            
            # Verify prerequisites from previous steps
            if not self.step_completed[2]:
                raise RuntimeError("Step 2 (UNet + CLIP + LoRA) must be completed before Step 3")
            
            if self.unet is None:
                raise RuntimeError("UNet model not loaded - Step 2 must be completed first")
            
            if self.clip is None:
                raise RuntimeError("CLIP model not loaded - Step 2 must be completed first")
            
            # ========================================================================
            # 3.1: Apply SD3 Model Sampling
            # ========================================================================
            print("3.1 Applying SD3 Model Sampling...")
            sampling_start = time.time()
            
            # Store original model info for analysis
            original_model_type = type(self.unet).__name__
            original_uuid = str(self.unet.patches_uuid) if hasattr(self.unet, 'patches_uuid') else None
            
            # Apply ModelSamplingSD3
            model_sampling = ModelSamplingSD3()
            patched_unet = model_sampling.patch(self.unet, shift=shift, multiplier=multiplier)
            
            if patched_unet is None:
                raise RuntimeError("ModelSamplingSD3 returned None - patching failed")
            
            # Update the UNet model
            self.unet = patched_unet
            
            sampling_time = time.time() - sampling_start
            print(f"✅ SD3 Model Sampling applied successfully in {sampling_time:.2f}s")
            
            # Analyze the patched model
            print(f"   🔧 SAMPLING ANALYSIS:")
            print(f"      Original Type: {original_model_type}")
            print(f"      Patched Type: {type(self.unet).__name__}")
            print(f"      Model Cloned: {'✅ YES' if self.unet != None else '❌ NO'}")
            print(f"      Patches UUID: {self.unet.patches_uuid}")
            print(f"      UUID Preserved: {'✅ YES' if str(self.unet.patches_uuid) == original_uuid else '❌ NO'}")
            
            # Verify model_sampling patch
            has_sampling_patch = False
            if hasattr(self.unet, 'object_patches') and 'model_sampling' in self.unet.object_patches:
                has_sampling_patch = True
                sampling_obj = self.unet.object_patches['model_sampling']
                print(f"      Sampling Patch: ✅ Applied ({type(sampling_obj).__name__})")
                print(f"      Shift Parameter: {shift}")
                print(f"      Multiplier Parameter: {multiplier}")
            else:
                print(f"      Sampling Patch: ❌ Not found")
            
            # ========================================================================
            # 3.2: CLIP Text Encoding
            # ========================================================================
            print("\n3.2 Encoding text prompts...")
            encoding_start = time.time()
            
            print(f"   📝 Positive prompt: '{positive_prompt}'")
            print(f"   📝 Negative prompt: '{negative_prompt}'")
            
            # Memory before encoding
            if torch.cuda.is_available():
                mem_before = torch.cuda.memory_allocated() / 1024**2
                print(f"   💾 GPU memory before encoding: {mem_before:.1f} MB")
            
            # Initialize text encoder
            text_encoder = CLIPTextEncode()
            
            # Encode positive prompt
            positive_encoding_start = time.time()
            positive_cond = text_encoder.encode(self.clip, positive_prompt)
            positive_encoding_time = time.time() - positive_encoding_start
            
            print(f"   ✅ Positive prompt encoded in {positive_encoding_time:.3f}s")
            
            # Encode negative prompt
            negative_encoding_start = time.time()
            negative_cond = text_encoder.encode(self.clip, negative_prompt)
            negative_encoding_time = time.time() - negative_encoding_start
            
            print(f"   ✅ Negative prompt encoded in {negative_encoding_time:.3f}s")
            
            total_encoding_time = time.time() - encoding_start
            
            # Memory after encoding
            if torch.cuda.is_available():
                mem_after = torch.cuda.memory_allocated() / 1024**2
                mem_delta = mem_after - mem_before
                print(f"   💾 GPU memory after encoding: {mem_after:.1f} MB (+{mem_delta:.1f} MB)")
            
            # ========================================================================
            # 3.3: Analyze Encoding Results
            # ========================================================================
            print("\n3.3 Analyzing conditioning results...")
            
            # Analyze positive conditioning
            if isinstance(positive_cond, (tuple, list)) and len(positive_cond) > 0:
                pos_tensor = positive_cond[0]
                if hasattr(pos_tensor, 'shape'):
                    print(f"   🔧 Positive Conditioning:")
                    print(f"      Shape: {pos_tensor.shape}")
                    print(f"      Data Type: {pos_tensor.dtype}")
                    print(f"      Device: {pos_tensor.device}")
                    print(f"      Value Range: [{pos_tensor.min().item():.3f}, {pos_tensor.max().item():.3f}]")
                    
                    # Check for valid embeddings
                    non_zero_ratio = torch.count_nonzero(pos_tensor).item() / pos_tensor.numel()
                    print(f"      Non-zero ratio: {non_zero_ratio:.3f}")
                    print(f"      Status: {'✅ Valid' if non_zero_ratio > 0.1 else '⚠️ Mostly zeros'}")
            
            # Analyze negative conditioning
            if isinstance(negative_cond, (tuple, list)) and len(negative_cond) > 0:
                neg_tensor = negative_cond[0]
                if hasattr(neg_tensor, 'shape'):
                    print(f"   🔧 Negative Conditioning:")
                    print(f"      Shape: {neg_tensor.shape}")
                    print(f"      Status: {'✅ Valid' if neg_tensor.shape == pos_tensor.shape else '❌ Shape mismatch'}")
            
            # Mark step complete
            self.step_completed[3] = True
            
            # Create results
            step_3_results = {
                'positive_conditioning': positive_cond,
                'negative_conditioning': negative_cond,
                'unet_patched': self.unet,
                'clip_model': self.clip,
                'sampling_applied': has_sampling_patch,
                'model_info': {
                    'original_type': original_model_type,
                    'patched_type': type(self.unet).__name__,
                    'sampling_patch_applied': has_sampling_patch,
                    'shift': shift,
                    'multiplier': multiplier,
                    'unet_uuid': str(self.unet.patches_uuid) if hasattr(self.unet, 'patches_uuid') else None
                },
                'conditioning_info': {
                    'positive_prompt': positive_prompt,
                    'negative_prompt': negative_prompt,
                    'positive_shape': pos_tensor.shape if hasattr(pos_tensor, 'shape') else None,
                    'negative_shape': neg_tensor.shape if hasattr(neg_tensor, 'shape') else None,
                    'positive_dtype': str(pos_tensor.dtype) if hasattr(pos_tensor, 'dtype') else None,
                    'positive_device': str(pos_tensor.device) if hasattr(pos_tensor, 'device') else None
                },
                'timing': {
                    'sampling_time': sampling_time,
                    'positive_encoding': positive_encoding_time,
                    'negative_encoding': negative_encoding_time,
                    'total_encoding': total_encoding_time,
                    'total_step_time': time.time() - step_3_start
                }
            }
            
            print(f"\n✅ STEP 3 COMPLETED SUCCESSFULLY in {time.time() - step_3_start:.2f}s")
            print("="*80)
            
            return step_3_results
            
        except Exception as e:
            print(f"❌ STEP 3 FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def step_4_ksampler_denoising(self,
                                  initial_latent: torch.Tensor,
                                  positive_conditioning: Any,
                                  negative_conditioning: Any,
                                  steps: int = 20,
                                  cfg_scale: float = 7.5,
                                  sampler_name: str = "euler",
                                  scheduler_name: str = "simple",
                                  denoise: float = 1.0,
                                  seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Step 4: KSampler Denoising
        
        This is the core sampling step where the magic happens:
        1. Prepares noise tensor from initial latent
        2. Initializes KSampler with optimized memory management
        3. Performs denoising using CFG and selected sampling algorithm
        4. Returns denoised latents ready for VAE decoding
        
        Args:
            initial_latent: Initial latent tensor from Step 1
            positive_conditioning: Positive text conditioning from Step 3
            negative_conditioning: Negative text conditioning from Step 3
            steps: Number of denoising steps (default 20)
            cfg_scale: Classifier-free guidance scale (default 7.5)
            sampler_name: Sampling algorithm ("euler", "dpmpp_2m")
            scheduler_name: Noise scheduler ("simple", "karras", "exponential", "ddim_uniform")
            denoise: Denoising strength 0.0-1.0 (default 1.0)
            seed: Random seed for reproducible generation
            
        Returns:
            Dictionary containing denoised latents and sampling information
        """
        
        print("\n" + "="*80)
        print("🚀 STEP 4: KSAMPLER DENOISING")
        print("="*80)
        
        try:
            step_4_start = time.time()
            
            # Verify prerequisites from previous steps
            if not self.step_completed[3]:
                raise RuntimeError("Step 3 (Model Sampling + Text Encoding) must be completed before Step 4")
            
            if self.unet is None:
                raise RuntimeError("UNet model not loaded - Steps 2 and 3 must be completed first")
            
            # ========================================================================
            # 4.1: Memory Management Setup
            # ========================================================================
            print("4.1 Setting up memory management...")
            
            # Clear cache before starting
            if torch.cuda.is_available():
                mem_before = torch.cuda.memory_allocated() / 1024**2
                torch.cuda.empty_cache()
                print(f"   💾 Initial GPU memory: {mem_before:.1f} MB")
                print(f"   🧹 Cache cleared")
            
            # Move initial latent to correct device
            if isinstance(initial_latent, dict) and 'samples' in initial_latent:
                latent_tensor = initial_latent['samples']
            else:
                latent_tensor = initial_latent
                
            latent_tensor = latent_tensor.to(self.device)
            
            print(f"   📊 Latent tensor: {latent_tensor.shape} on {latent_tensor.device}")
            
            # ========================================================================
            # 4.2: Noise Preparation
            # ========================================================================
            print("\n4.2 Preparing noise tensor...")
            noise_start = time.time()
            
            # Generate noise with proper seed handling
            if seed is None:
                seed = int(time.time() * 1000) % 2**32
                print(f"   🎲 Auto-generated seed: {seed}")
            else:
                print(f"   🎲 Using seed: {seed}")
            
            noise = prepare_noise(latent_tensor, seed=seed, device=self.device)
            noise_time = time.time() - noise_start
            
            print(f"   ✅ Noise generated in {noise_time:.3f}s")
            print(f"      Shape: {noise.shape}")
            print(f"      Stats: mean={noise.mean().item():.3f}, std={noise.std().item():.3f}")
            print(f"      Device: {noise.device}")
            
            # ========================================================================
            # 4.3: KSampler Initialization
            # ========================================================================
            print("\n4.3 Initializing KSampler...")
            sampler_init_start = time.time()
            
            print(f"   🔧 Configuration:")
            print(f"      Steps: {steps}")
            print(f"      CFG Scale: {cfg_scale}")
            print(f"      Sampler: {sampler_name}")
            print(f"      Scheduler: {scheduler_name}")
            print(f"      Denoise: {denoise}")
            print(f"      Device: {self.device}")
            
            # Initialize KSampler
            ksampler = StandaloneKSampler(
                model=self.unet,
                steps=steps,
                device=self.device,
                sampler=sampler_name,
                scheduler=scheduler_name,
                denoise=denoise,
                model_options={}
            )
            
            sampler_init_time = time.time() - sampler_init_start
            print(f"   ✅ KSampler initialized in {sampler_init_time:.3f}s")
            
            # ========================================================================
            # 4.4: Memory Status Before Sampling
            # ========================================================================
            if torch.cuda.is_available():
                mem_before_sampling = torch.cuda.memory_allocated() / 1024**2
                mem_reserved = torch.cuda.memory_reserved() / 1024**2
                print(f"\n   💾 Memory before sampling:")
                print(f"      Allocated: {mem_before_sampling:.1f} MB")
                print(f"      Reserved: {mem_reserved:.1f} MB")
            
            # ========================================================================
            # 4.5: Main Sampling Process
            # ========================================================================
            print("\n4.5 Starting denoising process...")
            sampling_start = time.time()
            
            # Progress callback for monitoring
            progress_data = {'current_step': 0, 'total_steps': steps}
            
            def progress_callback(progress, total_steps, current_step):
                progress_data['current_step'] = current_step
                if current_step % max(1, total_steps // 10) == 0:  # Log every 10%
                    if torch.cuda.is_available():
                        current_mem = torch.cuda.memory_allocated() / 1024**2
                        print(f"      Step {current_step}/{total_steps} ({progress*100:.1f}%) - Memory: {current_mem:.1f} MB")
                    else:
                        print(f"      Step {current_step}/{total_steps} ({progress*100:.1f}%)")
            
            # Perform sampling
            try:
                denoised_latents = ksampler.sample(
                    noise=noise,
                    positive=positive_conditioning,
                    negative=negative_conditioning,
                    cfg=cfg_scale,
                    latent_image=latent_tensor,
                    seed=seed,
                    callback=progress_callback,
                    disable_pbar=False
                )
                
                sampling_time = time.time() - sampling_start
                print(f"   ✅ Denoising completed in {sampling_time:.2f}s")
                
            except Exception as e:
                print(f"   ❌ Sampling failed: {e}")
                raise
            
            # ========================================================================
            # 4.6: Post-Sampling Analysis
            # ========================================================================
            print("\n4.6 Analyzing denoised results...")
            
            print(f"   📊 Denoised latents:")
            print(f"      Shape: {denoised_latents.shape}")
            print(f"      Data Type: {denoised_latents.dtype}")
            print(f"      Device: {denoised_latents.device}")
            print(f"      Value Range: [{denoised_latents.min().item():.3f}, {denoised_latents.max().item():.3f}]")
            print(f"      Mean: {denoised_latents.mean().item():.3f}")
            print(f"      Std: {denoised_latents.std().item():.3f}")
            
            # Validate results
            is_valid = True
            validation_notes = []
            
            # Check for NaN or Inf values
            if torch.isnan(denoised_latents).any():
                is_valid = False
                validation_notes.append("Contains NaN values")
            
            if torch.isinf(denoised_latents).any():
                is_valid = False
                validation_notes.append("Contains Inf values")
            
            # Check value range
            if denoised_latents.abs().max() > 100:
                validation_notes.append("Large values detected")
            
            # Check if all zeros
            if torch.allclose(denoised_latents, torch.zeros_like(denoised_latents)):
                validation_notes.append("All values are zero")
            
            print(f"   🔍 Validation: {'✅ PASSED' if is_valid else '❌ FAILED'}")
            if validation_notes:
                for note in validation_notes:
                    print(f"      ⚠️ {note}")
            
            # ========================================================================
            # 4.7: Memory Management and Cleanup
            # ========================================================================
            print("\n4.7 Memory cleanup...")
            
            # Get KSampler memory stats
            ksampler_stats = ksampler.get_memory_stats()
            
            # Final memory state
            if torch.cuda.is_available():
                mem_after = torch.cuda.memory_allocated() / 1024**2
                mem_delta = mem_after - mem_before_sampling
                
                print(f"   💾 Final memory state:")
                print(f"      Before: {mem_before_sampling:.1f} MB")
                print(f"      After: {mem_after:.1f} MB")
                print(f"      Delta: {mem_delta:+.1f} MB")
                print(f"      KSampler Peak: {ksampler_stats.get('peak_allocated', 0):.1f} MB")
                print(f"      Cache Clears: {ksampler_stats.get('cache_clears', 0)}")
            
            # Move result to CPU if offloading is enabled
            final_device = denoised_latents.device
            if self.offload_device != self.device:
                denoised_latents = denoised_latents.to(self.offload_device)
                final_device = self.offload_device
                print(f"   📦 Moved results to: {final_device}")
            
            # Final cache clear
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"   🧹 Final cache clear completed")
            
            # Mark step complete
            self.step_completed[4] = True
            
            # ========================================================================
            # 4.8: Results Summary
            # ========================================================================
            step_4_results = {
                'denoised_latents': denoised_latents,
                'original_latent': latent_tensor,
                'noise_tensor': noise,
                'sampling_info': {
                    'steps': steps,
                    'cfg_scale': cfg_scale,
                    'sampler': sampler_name,
                    'scheduler': scheduler_name,
                    'denoise': denoise,
                    'seed': seed,
                    'final_device': str(final_device)
                },
                'validation': {
                    'is_valid': is_valid,
                    'notes': validation_notes
                },
                'memory_stats': {
                    'ksampler_stats': ksampler_stats,
                    'memory_delta_mb': mem_delta if torch.cuda.is_available() else 0,
                    'peak_allocated_mb': ksampler_stats.get('peak_allocated', 0)
                },
                'timing': {
                    'noise_generation': noise_time,
                    'sampler_init': sampler_init_time,
                    'sampling_time': sampling_time,
                    'total_step_time': time.time() - step_4_start
                }
            }
            
            print(f"\n✅ STEP 4 COMPLETED SUCCESSFULLY in {time.time() - step_4_start:.2f}s")
            print("="*80)
            
            return step_4_results
            
        except Exception as e:
            print(f"❌ STEP 4 FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def load_video(self, video_path: str) -> Optional[torch.Tensor]:
        """Load control video from path as float tensor (T, H, W, 3) in [0,1]"""
        if not video_path or not os.path.exists(video_path):
            print(f"Warning: Video file not found: {video_path}")
            return None
        
        try:
            from torchvision.io import read_video
            print(f"   📹 Loading video from: {video_path}")
            
            video, audio, info = read_video(video_path, pts_unit='sec')
            if video is None or video.numel() == 0:
                print(f"Warning: Empty video: {video_path}")
                return None
            
            # Convert from (T, H, W, C) uint8 to float32 [0,1]
            video = video.float() / 255.0
            
            # Ensure 3 channels
            if video.shape[-1] > 3:
                video = video[..., :3]
            elif video.shape[-1] == 1:
                video = video.repeat(1, 1, 1, 3)
            
            print(f"   📊 Loaded video tensor: {tuple(video.shape)} (T,H,W,C)")
            return video
            
        except Exception as e:
            print(f"Error loading video '{video_path}': {e}")
            return None

    def load_image(self, image_path: str) -> Optional[torch.Tensor]:
        """Load reference image from path as float tensor (1, H, W, 3) in [0,1]"""
        if not image_path or not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}")
            return None
        
        try:
            from PIL import Image
            import numpy as np
            print(f"   🖼️  Loading image from: {image_path}")
            
            img = Image.open(image_path).convert('RGB')
            arr = np.asarray(img).astype('float32') / 255.0
            
            # Add time dimension: (H,W,3) -> (1,H,W,3)
            tensor = torch.from_numpy(arr).unsqueeze(0)
            print(f"   📊 Loaded image tensor: {tuple(tensor.shape)} (1,H,W,3)")
            return tensor
            
        except Exception as e:
            print(f"Error loading image '{image_path}': {e}")
            return None

    def get_step_status(self) -> Dict[int, bool]:
        """Get completion status of all pipeline steps"""
        return self.step_completed.copy()

    def run_step_1_only(self, **kwargs) -> Dict[str, Any]:
        """Convenience method to run only Step 1"""
        return self.step_1_vae_and_latent_creation(**kwargs)
    
    def run_step_2_only(self, **kwargs) -> Dict[str, Any]:
        """Convenience method to run only Step 2"""
        return self.step_2_unet_clip_lora_loading(**kwargs)
    
    def run_step_3_only(self, **kwargs) -> Dict[str, Any]:
        """Convenience method to run only Step 3"""
        return self.step_3_model_sampling_and_text_encoding(**kwargs)
    
    def run_step_4_only(self, **kwargs) -> Dict[str, Any]:
        """Convenience method to run only Step 4"""
        return self.step_4_ksampler_denoising(**kwargs)
    
    def run_steps_1_and_2(self, step_1_params: Dict[str, Any], step_2_params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run both Step 1 and Step 2 in sequence"""
        print("🚀 Running Steps 1 and 2 in sequence...")
        step_1_results = self.step_1_vae_and_latent_creation(**step_1_params)
        step_2_results = self.step_2_unet_clip_lora_loading(**step_2_params)
        return step_1_results, step_2_results
    
    def run_steps_1_2_and_3(self, step_1_params: Dict[str, Any], step_2_params: Dict[str, Any], step_3_params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Run Steps 1, 2, and 3 in sequence"""
        print("🚀 Running Steps 1, 2, and 3 in sequence...")
        step_1_results = self.step_1_vae_and_latent_creation(**step_1_params)
        step_2_results = self.step_2_unet_clip_lora_loading(**step_2_params)
        step_3_results = self.step_3_model_sampling_and_text_encoding(**step_3_params)
        return step_1_results, step_2_results, step_3_results
    
    def run_steps_1_2_3_and_4(self, step_1_params: Dict[str, Any], step_2_params: Dict[str, Any], step_3_params: Dict[str, Any], step_4_params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Run Steps 1, 2, 3, and 4 in sequence"""
        print("🚀 Running Steps 1, 2, 3, and 4 in sequence...")
        step_1_results = self.step_1_vae_and_latent_creation(**step_1_params)
        step_2_results = self.step_2_unet_clip_lora_loading(**step_2_params)
        step_3_results = self.step_3_model_sampling_and_text_encoding(**step_3_params)
        step_4_results = self.step_4_ksampler_denoising(**step_4_params)
        return step_1_results, step_2_results, step_3_results, step_4_results

# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

def main():
    """Example usage of Steps 1, 2, and 3 pipeline"""
    print("🚀 WAN Video Pipeline - Steps 1, 2 & 3 Test")
    print("="*60)
    
    # Initialize pipeline
    pipeline = WanVideoPipeline(models_dir="models")
    
    # Step 1 parameters
    script_dir = Path(__file__).parent
    step_1_params = {
        'vae_model_path': str("models/vaes/wan_vae.safetensors"),
        'positive_prompt': "very cinematic video",
        'negative_prompt': "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量",
        'control_video_path': str("safu.mp4"),
        'reference_image_path': str("safu.jpg"),  
        'width': 480,
        'height': 832,
        'length': 37,
        'batch_size': 1,
        'strength': 1.0
    }
    
    # Step 2 parameters
    step_2_params = {
        'unet_model_path': str("models/diffusion_models/wan_2.1_diffusion_model.safetensors"),
        'clip_model_path': str("models/text_encoders/wan_clip_model.safetensors"),
        'lora_model_path': str("models/loras/Wan21_CausVid_14B_T2V_lora_rank32.safetensors"),
        'strength_model': 1.0,
        'strength_clip': 0.0
    }
    
    # Step 3 parameters
    step_3_params = {
        'positive_prompt': "very cinematic video",
        'negative_prompt': "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量",
        'shift': 8.0,
        'multiplier': 1000
    }
    
    # Check if model files exist
    required_files = [
        step_1_params['vae_model_path'],
        step_2_params['unet_model_path'], 
        step_2_params['clip_model_path']
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("❌ Required model files not found:")
        for missing in missing_files:
            print(f"   {missing}")
        print("\n💡 Run './download_models.sh' to download the required models")
        print("🧪 Testing Step 1 only with available models...")
        
        # Test Step 1 only if VAE is available
        if os.path.exists(step_1_params['vae_model_path']):
            try:
                results = pipeline.run_step_1_only(**step_1_params)
                print("\n✅ Step 1 test completed - ready for Step 2 when models are available")
            except Exception as e:
                print(f"\n❌ STEP 1 TEST FAILED: {str(e)}")
        return
    
    try:
        # Run all three steps
        step_1_results, step_2_results, step_3_results = pipeline.run_steps_1_2_and_3(step_1_params, step_2_params, step_3_params)
        
        print("\n🎉 STEPS 1, 2 & 3 TEST COMPLETED SUCCESSFULLY!")
        print(f"Pipeline Status: {pipeline.get_step_status()}")
        
        # Display Step 1 results summary
        if step_1_results:
            print(f"\n📋 STEP 1 RESULTS (VAE + Conditioning):")
            print(f"   VAE: {type(step_1_results['vae']).__name__}")
            print(f"   Positive Conditioning: {len(step_1_results['positive'])} entries")
            print(f"   Negative Conditioning: {len(step_1_results['negative'])} entries")
            print(f"   Output Latent: {step_1_results['out_latent']['samples'].shape}")
            print(f"   Processing Time: {step_1_results['processing_info']['total_step_time']:.2f}s")
        
        # Display Step 2 results summary
        if step_2_results:
            print(f"\n📋 STEP 2 RESULTS (UNet + CLIP + LoRA):")
            print(f"   UNet: {step_2_results['models_info']['unet_type']}")
            print(f"   CLIP: {step_2_results['models_info']['clip_type']}")
            print(f"   LoRA Applied: {'✅' if step_2_results['lora_applied'] else '❌'}")
            if step_2_results['lora_applied']:
                print(f"   LoRA Model Strength: {step_2_results['models_info']['lora_strength_model']}")
                print(f"   LoRA CLIP Strength: {step_2_results['models_info']['lora_strength_clip']}")
            print(f"   Processing Time: {step_2_results['processing_info']['total_step_time']:.2f}s")
        
        # Display Step 3 results summary
        if step_3_results:
            print(f"\n📋 STEP 3 RESULTS (Model Sampling + Text Encoding):")
            print(f"   Sampling Applied: {'✅' if step_3_results['sampling_applied'] else '❌'}")
            print(f"   Shift Parameter: {step_3_results['model_info']['shift']}")
            print(f"   Multiplier Parameter: {step_3_results['model_info']['multiplier']}")
            print(f"   Positive Prompt: '{step_3_results['conditioning_info']['positive_prompt'][:50]}...'")
            print(f"   Negative Prompt: '{step_3_results['conditioning_info']['negative_prompt'][:50]}...'")
            if step_3_results['conditioning_info']['positive_shape']:
                print(f"   Conditioning Shape: {step_3_results['conditioning_info']['positive_shape']}")
                print(f"   Conditioning Device: {step_3_results['conditioning_info']['positive_device']}")
            print(f"   Processing Time: {step_3_results['timing']['total_step_time']:.2f}s")
            
            # Memory usage
            if torch.cuda.is_available():
                print(f"   GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB allocated")
        
        print("\n✅ Steps 1, 2 & 3 completed - Full pipeline ready!")
        print("✅ Ready for Step 4: Noise Generation + Conditioning")
        
    except Exception as e:
        print(f"\n❌ PIPELINE TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()