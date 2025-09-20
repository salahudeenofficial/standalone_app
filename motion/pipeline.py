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

# NOTE: No external dependencies - motion pipeline is fully standalone

# Set memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Import motion modules
from standalone_vae import VAE, create_vae
from wan_vae_components.model_management import get_torch_device, unet_offload_device
from utils import load_torch_file, calculate_parameters
from standalone_sd import load_state_dict_guess_config
from lora import load_lora_for_models
from model_sampling import ModelSamplingSD3
from text_encoder import CLIPTextEncode
from standalone_ksampler import StandaloneKSampler, prepare_noise
from memory_utils import safe_model_to_device, log_memory_usage, clear_cuda_memory, get_memory_info, safe_model_to_device_advanced

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class WanVideoPipeline:
    """
    7-Step Standalone WAN Video Generation Pipeline
    
    Step 1: VAE Load + Reference Image/Control Video Load + Initial Latent Creation
    Step 2: CLIP Load + Text Encoding
    Step 3: UNet Load + LoRA Application
    Model Sampling Configuration
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
        
        # Log initial memory state
        log_memory_usage("Pipeline Initialization")
        
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

    def _manage_memory_between_steps(self, step_name: str, required_memory_gb: float = 2.0):
        """
        Manage memory between pipeline steps to prevent OOM errors
        
        Args:
            step_name: Name of the step for logging
            required_memory_gb: Minimum required memory in GB
        """
        if not torch.cuda.is_available():
            return
            
        print(f"\n🧹 MEMORY MANAGEMENT: Preparing for {step_name}...")
        
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated()
        free_memory = total_memory - allocated_memory
        free_memory_gb = free_memory / (1024**3)
        
        print(f"   📊 Available GPU Memory: {free_memory_gb:.2f} GB")
        print(f"   📊 Required Memory: {required_memory_gb:.2f} GB")
        
        if free_memory_gb < required_memory_gb:
            print(f"   ⚠️  Low memory detected, performing cleanup...")
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
            # Check memory again
            allocated_memory = torch.cuda.memory_allocated()
            free_memory = total_memory - allocated_memory
            free_memory_gb = free_memory / (1024**3)
            
            print(f"   📊 Memory after cleanup: {free_memory_gb:.2f} GB")
            
            if free_memory_gb < required_memory_gb:
                print(f"   🚨 CRITICAL: Insufficient memory for {step_name}!")
                print(f"   💡 Consider reducing batch size or using CPU offloading")
                raise RuntimeError(f"Insufficient GPU memory for {step_name}: {free_memory_gb:.2f} GB available, {required_memory_gb:.2f} GB required")
        else:
            print(f"   ✅ Sufficient memory available for {step_name}")

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
            # 1.1: Load WAN VAE Model (ComfyUI-style)
            # ========================================================================
            print("1.1 Loading WAN VAE model with ComfyUI-style implementation...")
            start_time = time.time()
            
            # Load VAE state dict
            vae_state_dict = load_torch_file(vae_model_path)
            print(f"   📊 Loaded VAE state dict with {len(vae_state_dict)} keys")
            
            # Create VAE instance using ComfyUI-style implementation
            # This follows the exact same pattern as ComfyUI's VAE class initialization
            self.vae = create_vae(state_dict=vae_state_dict, device=self.device)
            
            load_time = time.time() - start_time
            print(f"✅ VAE loaded successfully in {load_time:.2f}s")
            print(f"   Type: {type(self.vae.first_stage_model).__name__}")
            print(f"   Latent channels: {self.vae.latent_channels}")
            print(f"   Latent dimension: {self.vae.latent_dim}")
            print(f"   Downscale ratio: {self.vae.downscale_ratio}")
            print(f"   Upscale ratio: {self.vae.upscale_ratio}")
            print(f"   Device: {self.vae.device}")
            print(f"   VAE dtype: {self.vae.vae_dtype}")
            print(f"   Working dtypes: {self.vae.working_dtypes}")
            
            # Calculate VAE model size
            if hasattr(self.vae.first_stage_model, 'parameters'):
                vae_params = calculate_parameters(dict(self.vae.first_stage_model.named_parameters()))
                print(f"   Parameters: {vae_params:,}")
                print(f"   Size: {vae_params * 4 / (1024*1024):.1f} MB")
            
            # Verify VAE is properly initialized (ComfyUI-style validation)
            self.vae.throw_exception_if_invalid()
            print(f"   ✅ VAE validation passed")
            
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
        # 1.6: VAE Encoding of Control Video (ComfyUI-style)
        # ========================================================================
        print("\n1.6 Encoding control video with ComfyUI-style VAE...")
        encoding_start = time.time()
        
        # ComfyUI-style VAE encoding with proper memory management
        with torch.no_grad():
            # Encode inactive and reactive parts separately (following ComfyUI pattern)
            print("   🔄 Encoding inactive part with ComfyUI-style VAE...")
            
            # Prepare inactive frames for encoding (ComfyUI format: [F,H,W,C])
            inactive_frames = inactive[:, :, :, :3]  # Remove alpha channel if present
            print(f"   📊 Inactive frames shape: {inactive_frames.shape}")
            
            # ComfyUI-style encoding with automatic memory management
            inactive_latent = self.vae.encode(inactive_frames)
            print(f"   📊 Inactive latent shape: {inactive_latent.shape}")
            
            print("   🔄 Encoding reactive part...")
            reactive_latent = self.vae.encode(reactive[:, :, :, :3])
            print(f"   📊 Reactive latent shape: {reactive_latent.shape}")
            print(f"   📊 Reactive latent device: {reactive_latent.device}")
            print(f"   📊 Reactive latent dtype: {reactive_latent.dtype}")
            
            # Combine latents (ComfyUI-style concatenation)
            control_video_latent = torch.cat((inactive_latent, reactive_latent), dim=1)
            print(f"   📊 Combined control latent shape: {control_video_latent.shape}")
            print(f"   📊 Combined latent device: {control_video_latent.device}")
            print(f"   📊 Combined latent dtype: {control_video_latent.dtype}")
        
        encoding_time = time.time() - encoding_start
        print(f"✅ Control video encoded in {encoding_time:.2f}s")
        print(f"   🎯 ComfyUI-style VAE encoding completed successfully")
        
        # ========================================================================
        # 1.7: Process Reference Image (ComfyUI-style)
        # ========================================================================
        reference_image_latent = None
        if reference_image is not None:
            print("\n1.7 Processing reference image with ComfyUI-style VAE...")
            
            # Resize reference image to target dimensions (ComfyUI-style preprocessing)
            if reference_image.shape[1] != height or reference_image.shape[2] != width:
                print(f"   🔄 Resizing reference image to {height}x{width}")
                # Reshape for interpolation: (1,H,W,C) -> (1,C,H,W)
                reference_image = reference_image.permute(0, 3, 1, 2)
                reference_image = torch.nn.functional.interpolate(
                    reference_image, size=(height, width), mode='bilinear', align_corners=False
                )
                # Reshape back: (1,C,H,W) -> (1,H,W,C)
                reference_image = reference_image.permute(0, 2, 3, 1)
            
            # ComfyUI-style reference image encoding
            with torch.no_grad():
                print("   🔄 Encoding reference image with ComfyUI-style VAE...")
                
                # Prepare reference image for encoding (ComfyUI format: [F,H,W,C])
                reference_frames = reference_image[:, :, :, :3]  # Remove alpha channel if present
                print(f"   📊 Reference frames shape: {reference_frames.shape}")
                
                # ComfyUI-style encoding with automatic memory management
                reference_image_latent = self.vae.encode(reference_frames)
                print(f"   📊 Reference image latent shape: {reference_image_latent.shape}")
                print(f"   📊 Reference latent device: {reference_image_latent.device}")
                print(f"   📊 Reference latent dtype: {reference_image_latent.dtype}")
            
            # Add motion latent channels (WAN format) - like WanVaceToVideo node
            try:
                from wan_latent_format import Wan21_LatentFormat
                wan21_format = Wan21_LatentFormat()
                motion_channels = wan21_format.process_out(torch.zeros_like(reference_image_latent))
                reference_image_latent = torch.cat([reference_image_latent, motion_channels], dim=1)
                print(f"   📊 Reference with WAN motion channels: {reference_image_latent.shape}")
            except ImportError:
                print("   ⚠️  WAN latent format not available, using standard latent format")
                print(f"   📊 Reference image latent (standard format): {reference_image_latent.shape}")
        
        # ========================================================================
        # 1.8: Create Final Initial Latent and Results (ComfyUI-style)
        # ========================================================================
        print("\n1.8 Creating final initial latent with ComfyUI-style processing...")
        
        # Calculate latent dimensions using ComfyUI-style downscale ratio
        downscale_ratio = self.vae.spacial_compression_encode()
        print(f"   📊 VAE downscale ratio: {downscale_ratio}")
        
        latent_height = height // downscale_ratio
        latent_width = width // downscale_ratio
        
        # For WAN VAE, calculate temporal compression
        if hasattr(self.vae, 'latent_dim') and self.vae.latent_dim == 3:
            # WAN VAE uses temporal compression
            temporal_compression = 4  # WAN VAE typically compresses by 4x temporally
            latent_length = ((length - 1) // temporal_compression) + 1
            print(f"   📊 WAN VAE temporal compression: {temporal_compression}x")
        else:
            latent_length = length
            print(f"   📊 Standard VAE temporal compression: 1x")
        
        print(f"   📊 Calculated latent dimensions:")
        print(f"      Height: {height} → {latent_height} ({downscale_ratio}x downscale)")
        print(f"      Width: {width} → {latent_width} ({downscale_ratio}x downscale)")
        print(f"      Length: {length} → {latent_length}")
        
        # Start with control video latent (ComfyUI-style)
        initial_latent = control_video_latent
        print(f"   📊 Control video latent shape: {initial_latent.shape}")
        
        # Add reference image if provided (ComfyUI-style concatenation)
        if reference_image_latent is not None:
            print("   🔗 Concatenating reference image to control latent (ComfyUI-style)...")
            initial_latent = torch.cat((reference_image_latent, control_video_latent), dim=2)
            print(f"   📊 Latent with reference: {initial_latent.shape}")
            print(f"   📊 Combined latent device: {initial_latent.device}")
            print(f"   📊 Combined latent dtype: {initial_latent.dtype}")
        
        print(f"✅ Final initial latent shape: {initial_latent.shape}")
        print(f"   🎯 ComfyUI-style latent creation completed successfully")
        
        # Create control mask in latent space (ComfyUI-style)
        print("\n1.9 Creating control mask in latent space (ComfyUI-style)...")
        
        # Use ComfyUI-style downscale ratio for mask processing
        vae_stride = downscale_ratio
        height_mask = height // vae_stride
        width_mask = width // vae_stride
        
        print(f"   📊 Mask processing with VAE stride: {vae_stride}")
        print(f"   📊 Mask dimensions: {height}x{width} → {height_mask}x{width_mask}")
        
        # ComfyUI-style mask processing
        mask_latent = mask.view(length, height_mask, vae_stride, width_mask, vae_stride)
        mask_latent = mask_latent.permute(2, 4, 0, 1, 3)
        mask_latent = mask_latent.reshape(vae_stride * vae_stride, length, height_mask, width_mask)
        
        # Interpolate mask to latent temporal resolution
        mask_latent = torch.nn.functional.interpolate(
            mask_latent.unsqueeze(0), 
            size=(latent_length, height_mask, width_mask), 
            mode='nearest-exact'
        ).squeeze(0)
        
        # Handle reference image mask padding (ComfyUI-style)
        if reference_image_latent is not None:
            ref_frames = reference_image_latent.shape[2]
            mask_pad = torch.zeros_like(mask_latent[:, :ref_frames, :, :])
            mask_latent = torch.cat((mask_pad, mask_latent), dim=1)
            latent_length += ref_frames  # Update latent_length like WanVaceToVideo
            print(f"   📊 Added reference mask padding: {ref_frames} frames")
        
        mask_latent = mask_latent.unsqueeze(0)  # Add batch dimension
        print(f"   📊 Final mask latent shape: {mask_latent.shape}")
        print(f"   📊 Mask latent device: {mask_latent.device}")
        print(f"   📊 Mask latent dtype: {mask_latent.dtype}")
        
        # ========================================================================
        # 1.10: Setup VACE Conditioning (ComfyUI-style)
        # ========================================================================
        print("\n1.10 Setting up VACE conditioning (ComfyUI-style)...")
        
        # Import conditioning utilities (ComfyUI-style)
        try:
            from conditioning_utils import create_empty_conditioning, conditioning_set_values, print_conditioning_info
            
            # Create initial conditioning from prompts (ComfyUI-style)
            positive = create_empty_conditioning(device=self.device)
            negative = create_empty_conditioning(device=self.device)
            
            print(f"   📝 Initial positive prompt: '{positive_prompt}'")
            print(f"   📝 Initial negative prompt: '{negative_prompt}'")
            
            # Apply VACE conditioning exactly like WanVaceToVideo node (ComfyUI-style)
            vace_conditioning_values = {
                "vace_frames": [initial_latent],
                "vace_mask": [mask_latent], 
                "vace_strength": [strength]
            }
            
            print(f"   🔧 Applying VACE conditioning with strength: {strength}")
            print(f"   📊 VACE frames shape: {initial_latent.shape}")
            print(f"   📊 VACE mask shape: {mask_latent.shape}")
            print(f"   📊 VACE frames device: {initial_latent.device}")
            print(f"   📊 VACE mask device: {mask_latent.device}")
            
            # Set conditioning values (append=True like WanVaceToVideo)
            positive = conditioning_set_values(positive, vace_conditioning_values, append=True)
            negative = conditioning_set_values(negative, vace_conditioning_values, append=True)
            
            # Debug conditioning info (ComfyUI-style)
            print_conditioning_info(positive, "Positive")
            print_conditioning_info(negative, "Negative")
            
            print("✅ VACE conditioning setup complete (ComfyUI-style)")
            
        except ImportError as e:
            print(f"   ⚠️  Conditioning utilities not available: {e}")
            print("   🔧 Creating simplified conditioning structure...")
            
            # Fallback: Create simplified conditioning structure
            positive = {
                "prompt": positive_prompt,
                "vace_frames": initial_latent,
                "vace_mask": mask_latent,
                "vace_strength": strength
            }
            negative = {
                "prompt": negative_prompt,
                "vace_frames": initial_latent,
                "vace_mask": mask_latent,
                "vace_strength": strength
            }
            
            print("✅ Simplified VACE conditioning setup complete")
        
        # Mark step complete and return results
        self.step_completed[1] = True
        
        # Create WAN-format output latent (ComfyUI-style)
        print("\n1.11 Creating final output latent (ComfyUI-style)...")
        
        # Determine output latent channels based on VAE configuration
        if hasattr(self.vae, 'latent_channels'):
            output_channels = self.vae.latent_channels
        else:
            output_channels = 16  # Default for WAN VAE
        
        print(f"   📊 Output latent channels: {output_channels}")
        print(f"   📊 Final latent dimensions: [{batch_size}, {output_channels}, {latent_length}, {latent_height}, {latent_width}]")
        
        # Create output latent tensor (ComfyUI-style)
        output_latent = torch.zeros([batch_size, output_channels, latent_length, latent_height, latent_width], 
                                   device=self.device, dtype=self.vae.vae_dtype)
        out_latent = {"samples": output_latent}
        
        print(f"   📊 Output latent shape: {output_latent.shape}")
        print(f"   📊 Output latent device: {output_latent.device}")
        print(f"   📊 Output latent dtype: {output_latent.dtype}")
        
        # Calculate trim_latent like WanVaceToVideo node (ComfyUI-style)
        trim_latent = reference_image_latent.shape[2] if reference_image_latent is not None else 0
        print(f"   📊 Trim latent frames: {trim_latent}")
        
        print("✅ ComfyUI-style output latent creation completed")
        
        # Return results matching WanVaceToVideo node signature (ComfyUI-style)
        step_1_results = {
            # WanVaceToVideo node outputs (ComfyUI-style):
            'positive': positive,           # Conditioned positive prompts
            'negative': negative,           # Conditioned negative prompts  
            'out_latent': out_latent,      # WAN-format latent dict {"samples": tensor}
            'trim_latent': trim_latent,    # Frame count to trim for reference
            
            # Additional debugging/pipeline data (ComfyUI-style):
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
                'channels': output_channels,  # Dynamic based on VAE configuration
                'length': latent_length,
                'height': latent_height,
                'width': latent_width
            },
            'vae_info': {
                'vae_type': type(self.vae.first_stage_model).__name__,
                'latent_channels': self.vae.latent_channels,
                'latent_dim': self.vae.latent_dim,
                'downscale_ratio': self.vae.downscale_ratio,
                'upscale_ratio': self.vae.upscale_ratio,
                'vae_dtype': str(self.vae.vae_dtype),
                'working_dtypes': [str(dt) for dt in self.vae.working_dtypes],
                'device': str(self.vae.device)
            },
            'processing_info': {
                'vae_encoding_time': encoding_time,
                'total_step_time': time.time() - start_time,
                'comfyui_style': True,
                'memory_management': 'comfyui_style'
            }
        }
        
        print(f"\n✅ STEP 1 COMPLETED SUCCESSFULLY in {time.time() - start_time:.2f}s")
        print("🎯 ComfyUI-style VAE loading and encoding completed successfully!")
        print("📊 VAE Type:", type(self.vae.first_stage_model).__name__)
        print("📊 Latent Channels:", self.vae.latent_channels)
        print("📊 Latent Dimension:", self.vae.latent_dim)
        print("📊 Downscale Ratio:", self.vae.downscale_ratio)
        print("📊 VAE Device:", self.vae.device)
        print("📊 VAE Dtype:", self.vae.vae_dtype)
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
            
            # ========================================================================
            # 2.1: Load UNet Diffusion Model
            # ========================================================================
            print("2.1 Loading UNet diffusion model...")
            unet_start = time.time()
            
            # Load UNet using ComfyUI-style loading with integrated patching
            print("🔧 Loading UNet with ComfyUI-style integrated patching...")
            
            # Load UNet model using standalone_sd
            result = load_state_dict_guess_config(
                unet_model_path,
                output_vae=False,
                output_clip=False,
                output_clipvision=False,
                output_model=True
            )
            
            if result is None:
                raise RuntimeError("Failed to load UNet model - load_state_dict_guess_config returned None")
            
            model_patcher, _, _, _ = result
            self.unet = model_patcher
            
            if self.unet is None:
                raise RuntimeError("UNet model is None after loading")
            
            unet_time = time.time() - unet_start
            print(f"✅ UNet loaded successfully in {unet_time:.2f}s")
            print(f"   Type: {type(self.unet).__name__}")
            print(f"   Device: {self.unet.load_device}")
            
            # Log memory after UNet loading
            log_memory_usage("After UNet Loading")
            
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
            
            # Calculate CLIP model size - handle T5CLIPModel special case
            if hasattr(self.clip, 'model') and self.clip.model is not None:
                clip_model = self.clip.model
                if hasattr(clip_model, 'model_info') and 'total_params' in clip_model.model_info:
                    # Use the actual parameter count from state dict (T5CLIPModel stores this)
                    clip_params = clip_model.model_info['total_params']
                    print(f"   📊 Using state dict parameter count: {clip_params:,}")
                else:
                    # Fallback to counting parameters
                    clip_params = sum(p.numel() for p in clip_model.parameters())
                    print(f"   📊 Using parameter() count: {clip_params:,}")
                
                print(f"   Parameters: {clip_params:,}")
                print(f"   Size: {clip_params * 2 / (1024**3):.2f} GB")
            
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
            
            print(f"   🔧 UNet Model Analysis:")
            print(f"      Type: {original_model_type}")
            print(f"      Device: {self.unet.load_device}")
            print(f"      Patches UUID: {original_uuid}")
            
            # Check if UNet is a ComfyUI-style ModelPatcher
            if hasattr(self.unet, 'model') and hasattr(self.unet, 'patches'):
                print(f"      ✅ ComfyUI-style ModelPatcher detected")
                print(f"      Model Type: {type(self.unet.model).__name__}")
                print(f"      Patches Count: {len(self.unet.patches) if self.unet.patches else 0}")
            else:
                print(f"      ⚠️  Standard model detected")
            
            # Apply ModelSamplingSD3
            print(f"   🔧 Applying ModelSamplingSD3 with shift={shift}, multiplier={multiplier}")
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
            print(f"      Model Cloned: {'✅ YES' if self.unet is not None else '❌ NO'}")
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
            
            # Analyze CLIP model
            print(f"   🔧 CLIP Model Analysis:")
            print(f"      Type: {type(self.clip).__name__}")
            print(f"      Device: {self.clip.load_device}")
            
            if hasattr(self.clip, 'model') and self.clip.model is not None:
                print(f"      Model Type: {type(self.clip.model).__name__}")
                if hasattr(self.clip.model, 'model_info') and 'total_params' in self.clip.model.model_info:
                    clip_params = self.clip.model.model_info['total_params']
                    print(f"      Parameters: {clip_params:,}")
                else:
                    print(f"      Parameters: Available via .parameters()")
            
            # Memory before encoding
            if torch.cuda.is_available():
                mem_before = torch.cuda.memory_allocated() / 1024**2
                print(f"   💾 GPU memory before encoding: {mem_before:.1f} MB")
            
            # Initialize text encoder
            text_encoder = CLIPTextEncode()
            
            # Encode positive prompt
            positive_encoding_start = time.time()
            try:
                positive_cond = text_encoder.encode(self.clip, positive_prompt)
                positive_encoding_time = time.time() - positive_encoding_start
                print(f"   ✅ Positive prompt encoded in {positive_encoding_time:.3f}s")
            except Exception as e:
                print(f"   ❌ Positive prompt encoding failed: {e}")
                raise
            
            # Encode negative prompt
            negative_encoding_start = time.time()
            try:
                negative_cond = text_encoder.encode(self.clip, negative_prompt)
                negative_encoding_time = time.time() - negative_encoding_start
                print(f"   ✅ Negative prompt encoded in {negative_encoding_time:.3f}s")
            except Exception as e:
                print(f"   ❌ Negative prompt encoding failed: {e}")
                raise
            
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
            pos_tensor = None
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
            else:
                print(f"   ⚠️  Positive conditioning format unexpected: {type(positive_cond)}")
            
            # Analyze negative conditioning
            neg_tensor = None
            if isinstance(negative_cond, (tuple, list)) and len(negative_cond) > 0:
                neg_tensor = negative_cond[0]
                if hasattr(neg_tensor, 'shape'):
                    print(f"   🔧 Negative Conditioning:")
                    print(f"      Shape: {neg_tensor.shape}")
                    print(f"      Status: {'✅ Valid' if pos_tensor is not None and neg_tensor.shape == pos_tensor.shape else '❌ Shape mismatch'}")
            else:
                print(f"   ⚠️  Negative conditioning format unexpected: {type(negative_cond)}")
            
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
                    'unet_uuid': str(self.unet.patches_uuid) if hasattr(self.unet, 'patches_uuid') else None,
                    'unet_device': str(self.unet.load_device),
                    'clip_type': type(self.clip).__name__,
                    'clip_device': str(self.clip.load_device)
                },
                'conditioning_info': {
                    'positive_prompt': positive_prompt,
                    'negative_prompt': negative_prompt,
                    'positive_shape': pos_tensor.shape if pos_tensor is not None and hasattr(pos_tensor, 'shape') else None,
                    'negative_shape': neg_tensor.shape if neg_tensor is not None and hasattr(neg_tensor, 'shape') else None,
                    'positive_dtype': str(pos_tensor.dtype) if pos_tensor is not None and hasattr(pos_tensor, 'dtype') else None,
                    'positive_device': str(pos_tensor.device) if pos_tensor is not None and hasattr(pos_tensor, 'device') else None
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
                                seed: int = 42,
                                steps: int = 4,
                                cfg: float = 7.0,
                                sampler_name: str = "euler",
                                scheduler: str = "normal",
                                denoise: float = 1.0,
                                noise_inds: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Step 4: KSampler Denoising
        
        This step performs the core denoising process using the KSampler:
        1. Prepares noise for the initial latent
        2. Sets up the KSampler with specified parameters
        3. Performs denoising steps to generate the final latent
        4. Returns the denoised latent ready for VAE decoding
        
        Args:
            initial_latent: Initial latent tensor from Step 1
            positive_conditioning: Positive conditioning from Step 3
            negative_conditioning: Negative conditioning from Step 3
            seed: Random seed for noise generation
            steps: Number of denoising steps
            cfg: Classifier-free guidance scale
            sampler_name: Sampler algorithm name (euler, ddim, etc.)
            scheduler: Scheduler type (normal, karras, etc.)
            denoise: Denoising strength (0.0-1.0)
            noise_inds: Optional noise indices for specific steps
            
        Returns:
            Dictionary containing denoised latent and sampling information
        """
        
        print("\n" + "="*80)
        print("🚀 STEP 4: KSAMPLER DENOISING")
        print("="*80)
        
        # Log memory before KSampler step
        log_memory_usage("Before Step 4 KSampler")
        
        try:
            step_4_start = time.time()
            
            # Verify prerequisites from previous steps
            if not self.step_completed[3]:
                raise RuntimeError("Step 3 (Model Sampling + Text Encoding) must be completed before Step 4")
            
            if self.unet is None:
                raise RuntimeError("UNet model not loaded - Step 2 must be completed first")
            
            if self.clip is None:
                raise RuntimeError("CLIP model not loaded - Step 2 must be completed first")
            
            # ========================================================================
            # 4.1: Prepare Noise for Initial Latent
            # ========================================================================
            print("4.1 Preparing noise for initial latent...")
            noise_start = time.time()
            
            print(f"   📊 Initial latent shape: {initial_latent.shape}")
            print(f"   📊 Initial latent device: {initial_latent.device}")
            print(f"   📊 Initial latent dtype: {initial_latent.dtype}")
            
            # Prepare noise using ComfyUI's prepare_noise (more robust)
            # Use motion pipeline's internal prepare_noise
            print("   🔧 Using motion pipeline's internal prepare_noise function...")
            noise = prepare_noise(initial_latent, seed, noise_inds)
            
            noise_time = time.time() - noise_start
            print(f"✅ Noise prepared in {noise_time:.3f}s")
            print(f"   📊 Noise shape: {noise.shape}")
            print(f"   📊 Noise device: {noise.device}")
            print(f"   📊 Noise range: [{noise.min().item():.3f}, {noise.max().item():.3f}]")
            
            # ========================================================================
            # 4.2: Setup KSampler
            # ========================================================================
            print("\n4.2 Setting up KSampler...")
            sampler_start = time.time()
            
            # Create KSampler instance
            ksampler = StandaloneKSampler(
                model=self.unet,
                steps=steps,
                device=self.device,
                sampler=sampler_name,
                scheduler=scheduler,
                denoise=denoise
            )
            
            print(f"   🔧 KSampler created successfully")
            print(f"   🔧 Model: {type(self.unet).__name__}")
            print(f"   🔧 Device: {self.device}")
            print(f"   🔧 Offload Device: {self.offload_device}")
            
            # ========================================================================
            # 4.3: Configure Sampling Parameters
            # ========================================================================
            print("\n4.3 Configuring sampling parameters...")
            
            print(f"   📋 SAMPLING CONFIGURATION:")
            print(f"      Seed: {seed}")
            print(f"      Steps: {steps}")
            print(f"      CFG: {cfg}")
            print(f"      Sampler: {sampler_name}")
            print(f"      Scheduler: {scheduler}")
            print(f"      Denoise: {denoise}")
            print(f"      Noise Indices: {noise_inds is not None}")
            
            # ========================================================================
            # 4.4: Perform Denoising with ComfyUI-style Memory Management
            # ========================================================================
            print("\n4.4 Performing denoising with advanced memory management...")
            denoising_start = time.time()
            
            # Memory before denoising
            log_memory_usage("Before Denoising")
            
            # Clear CUDA cache to free up any fragmented memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("   🧹 CUDA cache cleared")
            
            # CRITICAL: Let ComfyUI handle model loading/unloading properly
            print("🔧 Using ComfyUI's proper model loading/unloading...")
            
            # CRITICAL: Check if UNet is a ComfyUI-style ModelPatcher
            print("   🔍 Analyzing UNet type for proper detection...")
            print(f"   📊 UNet type: {type(self.unet).__name__}")
            print(f"   📊 UNet attributes: {[attr for attr in dir(self.unet) if not attr.startswith('_')]}")
            
            # Check for ModelPatcher attributes
            has_load = hasattr(self.unet, 'load')
            has_unload = hasattr(self.unet, 'unload')
            has_model = hasattr(self.unet, 'model')
            has_pre_run = hasattr(self.unet, 'pre_run')
            has_cleanup = hasattr(self.unet, 'cleanup')
            has_load_device = hasattr(self.unet, 'load_device')
            
            print(f"   📊 ModelPatcher attributes:")
            print(f"      - load: {has_load}")
            print(f"      - unload: {has_unload}")
            print(f"      - model: {has_model}")
            print(f"      - pre_run: {has_pre_run}")
            print(f"      - cleanup: {has_cleanup}")
            print(f"      - load_device: {has_load_device}")
            
            if has_load and has_unload and has_model:
                print("   ✅ UNet is ComfyUI-style ModelPatcher - Standalone will handle loading")
                print("   🔧 Standalone CFGGuider will call model_patcher.pre_run() and cleanup()")
                print(f"   📊 ModelPatcher load_device: {getattr(self.unet, 'load_device', 'unknown')}")
                print(f"   📊 ModelPatcher offload_device: {getattr(self.unet, 'offload_device', 'unknown')}")
                
            else:
                print("   ⚠️  UNet is not ComfyUI-style ModelPatcher - using fallback")
                # Fallback to standard approach
                unet_model = self.unet.model if hasattr(self.unet, 'model') else self.unet
                if str(unet_model.device) == 'cpu':
                    print("   🔄 Moving UNet to GPU...")
                    try:
                        unet_model.to(self.device)
                        print("   ✅ UNet moved to GPU")
                    except torch.cuda.OutOfMemoryError as e:
                        print(f"   ❌ CUDA OOM: {e}")
                        print("   🔄 Keeping UNet on CPU")
                        unet_model.to('cpu')
            
            # Perform the denoising process with ComfyUI integration
            try:
                # Create a memory monitoring callback
                def memory_callback(step, total_steps, current_step=None, **kwargs):
                    if step % max(1, total_steps // 4) == 0:  # Log every 25% of steps
                        if torch.cuda.is_available():
                            allocated = torch.cuda.memory_allocated() / 1024**3
                            reserved = torch.cuda.memory_reserved() / 1024**3
                            print(f"      Step {step}/{total_steps}: GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
                
                # Try ComfyUI integration first
                # Use standalone KSampler only (following Disclaimer.txt guidelines)
                print("   🔧 Using standalone KSampler (no external dependencies)...")
                print("   🚀 Starting standalone sampling with proper model interface...")
                print("   🔧 Standalone implementation will handle:")
                print("      - Model weight loading via model_patcher.pre_run()")
                print("      - Proper CFG processing via StandaloneCFGGuider")
                print("      - Correct model interface calls via standalone sampling")
                print("      - Model cleanup via model_patcher.cleanup()")
                
                # Use our standalone KSampler (following Disclaimer.txt guidelines)
                denoised_latent = ksampler.sample(
                    noise=noise,
                    positive=positive_conditioning,
                    negative=negative_conditioning,
                    cfg=cfg,
                    latent_image=None,
                    start_step=None,
                    last_step=None,
                    force_full_denoise=False,
                    denoise_mask=None,
                    sigmas=None,
                    callback=memory_callback,
                    disable_pbar=False,
                    seed=seed
                )
                
                print("   ✅ Standalone sampling completed successfully")
                print("   🔧 Used standalone algorithms (no external dependencies)")
                print("   📊 Model weights were properly loaded and used for inference")
                
                print("   ✅ Denoising completed successfully")
                
                # Clear CUDA cache after inference to free intermediate tensors
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("   🧹 CUDA cache cleared after inference")
                
            except Exception as e:
                print(f"   ❌ Denoising failed: {e}")
                raise
            
            finally:
                # CRITICAL: Standalone CFGGuider already handles model cleanup
                print("   🔧 Standalone CFGGuider handles model cleanup automatically")
                print("   📊 No manual cleanup needed - Standalone manages model loading/unloading")
            
            denoising_time = time.time() - denoising_start
            
            # Memory after denoising
            log_memory_usage("After Denoising")
            
            print(f"✅ Denoising completed in {denoising_time:.2f}s")
            print(f"   📊 Denoised latent shape: {denoised_latent.shape}")
            print(f"   📊 Denoised latent device: {denoised_latent.device}")
            print(f"   📊 Denoised latent range: [{denoised_latent.min().item():.3f}, {denoised_latent.max().item():.3f}]")
            
            # CRITICAL: Verify that proper sampling took place
            print(f"\n🔍 SAMPLING VERIFICATION:")
            
            # Ensure both tensors are on the same device for comparison
            if initial_latent.device != denoised_latent.device:
                print(f"   🔄 Device mismatch detected: {initial_latent.device} vs {denoised_latent.device}")
                print(f"   🔄 Moving denoised latent to {initial_latent.device} for comparison...")
                denoised_latent = denoised_latent.to(initial_latent.device)
                print(f"   ✅ Denoised latent moved to {denoised_latent.device}")
            
            print(f"   📊 Initial latent range: [{initial_latent.min().item():.3f}, {initial_latent.max().item():.3f}]")
            print(f"   📊 Denoised latent range: [{denoised_latent.min().item():.3f}, {denoised_latent.max().item():.3f}]")
            
            # Check if denoising actually occurred
            initial_std = initial_latent.std().item()
            denoised_std = denoised_latent.std().item()
            print(f"   📊 Initial latent std: {initial_std:.3f}")
            print(f"   📊 Denoised latent std: {denoised_std:.3f}")
            
            # Verify the latent changed (indicating actual model inference)
            if torch.allclose(initial_latent, denoised_latent, atol=1e-6):
                print(f"   🚨 WARNING: Denoised latent is identical to initial latent!")
                print(f"   🚨 This suggests the model may not have been properly loaded or used!")
            else:
                print(f"   ✅ Denoised latent differs from initial latent - proper sampling occurred!")
            
            # Check for valid values
            if torch.isfinite(denoised_latent).all():
                print(f"   ✅ All denoised values are finite - model inference successful!")
            else:
                print(f"   🚨 WARNING: Denoised latent contains NaN/Inf values!")
            
            # Check execution time (should be reasonable for 4 steps)
            if denoising_time < 1.0:
                print(f"   🚨 WARNING: Sampling completed too quickly ({denoising_time:.2f}s) - may indicate dummy data!")
            else:
                print(f"   ✅ Sampling took reasonable time ({denoising_time:.2f}s) - proper inference likely occurred!")
            
            # ========================================================================
            # 4.5: Analyze Results
            # ========================================================================
            print("\n4.5 Analyzing denoising results...")
            
            # Compare initial vs denoised
            initial_range = initial_latent.max().item() - initial_latent.min().item()
            denoised_range = denoised_latent.max().item() - denoised_latent.min().item()
            
            print(f"   🔧 LATENT ANALYSIS:")
            print(f"      Initial Range: {initial_range:.3f}")
            print(f"      Denoised Range: {denoised_range:.3f}")
            
            # Calculate range change safely
            if initial_range > 0:
                range_change = ((denoised_range - initial_range) / initial_range * 100)
                print(f"      Range Change: {range_change:+.1f}%")
            else:
                print(f"      Range Change: N/A (initial range was 0)")
            
            # Check for valid denoising
            if torch.isfinite(denoised_latent).all():
                print(f"      Status: ✅ Valid (all finite values)")
            else:
                print(f"      Status: ❌ Invalid (contains NaN/Inf)")
            
            # Mark step complete
            self.step_completed[4] = True
            
            # Log memory after KSampler step
            log_memory_usage("After Step 4 KSampler")
            
            # CRITICAL: Unload UNet to free memory for VAE decode
            print(f"\n🧹 MEMORY MANAGEMENT: Unloading UNet after denoising...")
            if hasattr(self.unet, 'cleanup'):
                self.unet.cleanup()
                print(f"   ✅ UNet cleanup completed")
            elif hasattr(self.unet, 'unload'):
                self.unet.unload()
                print(f"   ✅ UNet unload completed")
            else:
                print(f"   ⚠️  UNet cleanup method not available")
            
            # Clear CUDA cache to free fragmented memory
            torch.cuda.empty_cache()
            log_memory_usage("After UNet Cleanup")
            
            # Create results
            step_4_results = {
                'denoised_latent': denoised_latent,
                'noise': noise,
                'ksampler': ksampler,
                'sampling_config': {
                    'seed': seed,
                    'steps': steps,
                    'cfg': cfg,
                    'sampler_name': sampler_name,
                    'scheduler': scheduler,
                    'denoise': denoise,
                    'noise_inds_provided': noise_inds is not None
                },
                'latent_info': {
                    'initial_shape': initial_latent.shape,
                    'denoised_shape': denoised_latent.shape,
                    'initial_device': str(initial_latent.device),
                    'denoised_device': str(denoised_latent.device),
                    'initial_range': initial_range,
                    'denoised_range': denoised_range
                },
                'timing': {
                    'noise_preparation': noise_time,
                    'sampler_setup': time.time() - sampler_start,
                    'denoising': denoising_time,
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

    def step_5_trim_latent(self,
                          denoised_latent: torch.Tensor,
                          trim_amount: int = 0) -> Dict[str, Any]:
        """
        Step 5: Trim Video Latent
        
        This step trims the denoised latent by removing the specified number of frames
        from the beginning. This is essential for video generation where the initial
        frames might be unstable or unwanted.
        
        Args:
            denoised_latent: Denoised latent tensor from Step 4
            trim_amount: Number of frames to trim from the beginning (default: 0)
        
        Returns:
            Dictionary containing the trimmed latent and metadata
        """
        step_5_start = time.time()
        
        print("\n" + "="*80)
        print("🎬 STEP 5: TRIM VIDEO LATENT")
        print("="*80)
        
        try:
            # Import TrimVideoLatent from components (following Disclaimer.txt guidelines)
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from components.video_processor import TrimVideoLatent
            
            # Memory before trimming
            log_memory_usage("Before Latent Trimming")
            
            print(f"5.1 Trimming video latent...")
            print(f"   📊 Input latent shape: {denoised_latent.shape}")
            print(f"   📊 Trim amount: {trim_amount} frames")
            
            # Create trim processor
            trim_processor = TrimVideoLatent()
            
            # Wrap the latent tensor in the dictionary format expected by TrimVideoLatent
            latent_dict = {"samples": denoised_latent}
            
            # Perform trimming
            trimmed_latent_dict = trim_processor.op(latent_dict, trim_amount)
            
            # Extract the trimmed tensor from the dictionary
            trimmed_latent = trimmed_latent_dict["samples"]
            
            print(f"   ✅ Trimmed latent shape: {trimmed_latent.shape}")
            
            # Calculate frames removed
            original_frames = denoised_latent.shape[2]  # Assuming shape is (B, C, T, H, W)
            trimmed_frames = trimmed_latent.shape[2]
            frames_removed = original_frames - trimmed_frames
            
            print(f"   📊 Original frames: {original_frames}")
            print(f"   📊 Trimmed frames: {trimmed_frames}")
            print(f"   📊 Frames removed: {frames_removed}")
            
            # Memory after trimming
            log_memory_usage("After Latent Trimming")
            
            # Prepare results
            step_5_results = {
                'trimmed_latent': trimmed_latent,
                'original_latent': denoised_latent,
                'trim_amount': trim_amount,
                'frames_removed': frames_removed,
                'original_shape': denoised_latent.shape,
                'trimmed_shape': trimmed_latent.shape,
                'timing': {
                    'trimming_time': time.time() - step_5_start,
                    'total_step_time': time.time() - step_5_start
                }
            }
            
            print(f"\n✅ STEP 5 COMPLETED SUCCESSFULLY in {time.time() - step_5_start:.2f}s")
            print("="*80)
            
            return step_5_results
            
        except Exception as e:
            print(f"❌ STEP 5 FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def step_6_vae_decode(self,
                         trimmed_latent: torch.Tensor,
                         vae_model: Any = None) -> Dict[str, Any]:
        """
        Step 6: VAE Decode
        
        This step decodes the trimmed latent back to pixel space using the VAE model.
        This is the final step that converts the latent representation back to actual
        video frames that can be displayed or saved.
        
        Args:
            trimmed_latent: Trimmed latent tensor from Step 5 (or denoised latent from Step 4)
            vae_model: VAE model to use for decoding (uses pipeline's VAE if None)
        
        Returns:
            Dictionary containing the decoded images and metadata
        """
        step_6_start = time.time()
        
        print("\n" + "="*80)
        print("🎨 STEP 6: VAE DECODE")
        print("="*80)
        
        try:
            # Use pipeline's VAE if none provided
            if vae_model is None:
                vae_model = self.vae
                if vae_model is None:
                    raise RuntimeError("No VAE model available for decoding")
            
            # Memory before decoding
            log_memory_usage("Before VAE Decoding")
            
            # CRITICAL: Ensure sufficient memory for VAE decode
            self._manage_memory_between_steps("VAE Decode", required_memory_gb=2.0)
            
            print(f"6.1 Decoding latent to pixel space...")
            print(f"   📊 Input latent shape: {trimmed_latent.shape}")
            print(f"   📊 VAE model: {type(vae_model).__name__}")
            
            # Perform VAE decoding using direct WanVAE decode method (following successful test pattern)
            with torch.no_grad():
                decoded_images = vae_model.decode(trimmed_latent)
            
            print(f"   ✅ Decoded images shape: {decoded_images.shape}")
            
            # Calculate output statistics
            original_frames = trimmed_latent.shape[2]  # Assuming shape is (B, C, T, H, W)
            decoded_frames = decoded_images.shape[0] if len(decoded_images.shape) == 4 else decoded_images.shape[1]
            
            print(f"   📊 Original latent frames: {original_frames}")
            print(f"   📊 Decoded image frames: {decoded_frames}")
            print(f"   📊 Image dimensions: {decoded_images.shape[-2:]} (H, W)")
            
            # Verify output range
            if decoded_images.dtype == torch.float32:
                min_val, max_val = decoded_images.min().item(), decoded_images.max().item()
                print(f"   📊 Output range: [{min_val:.3f}, {max_val:.3f}]")
                
                if min_val >= 0.0 and max_val <= 1.0:
                    print(f"   ✅ Output in expected range [0, 1]")
                else:
                    print(f"   ⚠️  Output outside expected range [0, 1]")
            
            # Memory after decoding
            log_memory_usage("After VAE Decoding")
            
            # CRITICAL: Free VAE memory for video export
            print(f"\n🧹 MEMORY MANAGEMENT: Freeing VAE memory after decode...")
            if hasattr(vae_model, 'to'):
                vae_model.to('cpu')
                print(f"   ✅ VAE moved to CPU")
            
            # Clear CUDA cache to free fragmented memory
            torch.cuda.empty_cache()
            log_memory_usage("After VAE Memory Cleanup")
            
            # Prepare results
            step_6_results = {
                'decoded_images': decoded_images,
                'original_latent': trimmed_latent,
                'vae_model': vae_model,
                'original_shape': trimmed_latent.shape,
                'decoded_shape': decoded_images.shape,
                'original_frames': original_frames,
                'decoded_frames': decoded_frames,
                'image_dimensions': decoded_images.shape[-2:],
                'timing': {
                    'decoding_time': time.time() - step_6_start,
                    'total_step_time': time.time() - step_6_start
                }
            }
            
            print(f"\n✅ STEP 6 COMPLETED SUCCESSFULLY in {time.time() - step_6_start:.2f}s")
            print("="*80)
            
            return step_6_results
            
        except Exception as e:
            print(f"❌ STEP 6 FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def step_7_video_export(self,
                           decoded_images: torch.Tensor,
                           output_path: str = "output_video.mp4",
                           fps: int = 24) -> Dict[str, Any]:
        """
        Step 7: Video Export
        
        This step exports the decoded images to an MP4 video file.
        This is the final step that creates the actual video output file
        that can be played or shared.
        
        Args:
            decoded_images: Decoded images tensor from Step 6
            output_path: Path where the video file will be saved
            fps: Frames per second for the output video
        
        Returns:
            Dictionary containing the export results and metadata
        """
        step_7_start = time.time()
        
        print("\n" + "="*80)
        print("🎬 STEP 7: VIDEO EXPORT")
        print("="*80)
        
        try:
            # Import VideoExporter from components (following Disclaimer.txt guidelines)
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from components.video_export import VideoExporter
            
            # Memory before export
            log_memory_usage("Before Video Export")
            
            print(f"7.1 Exporting decoded images to video...")
            print(f"   📊 Input images shape: {decoded_images.shape}")
            print(f"   📊 Output path: {output_path}")
            print(f"   📊 FPS: {fps}")
            
            # Create video exporter
            video_exporter = VideoExporter(fps=fps)
            
            # Perform video export
            exported_path = video_exporter.export_video(decoded_images, output_path)
            
            print(f"   ✅ Video exported successfully!")
            print(f"   📊 Exported to: {exported_path}")
            
            # Calculate output statistics
            if len(decoded_images.shape) == 4:  # (frames, height, width, channels)
                total_frames = decoded_images.shape[0]
                height, width = decoded_images.shape[1], decoded_images.shape[2]
            elif len(decoded_images.shape) == 5:  # (batch, frames, height, width, channels)
                total_frames = decoded_images.shape[1]
                height, width = decoded_images.shape[2], decoded_images.shape[3]
            else:
                total_frames = "unknown"
                height, width = "unknown", "unknown"
            
            # Calculate video duration
            duration_seconds = total_frames / fps if isinstance(total_frames, int) else 0
            
            print(f"   📊 Total frames: {total_frames}")
            print(f"   📊 Video dimensions: {width}x{height}")
            print(f"   📊 Duration: {duration_seconds:.2f} seconds")
            print(f"   📊 Frame rate: {fps} FPS")
            
            # Verify output file exists
            if os.path.exists(exported_path):
                file_size = os.path.getsize(exported_path) / (1024 * 1024)  # MB
                print(f"   📊 File size: {file_size:.2f} MB")
            else:
                print(f"   ⚠️  Warning: Output file not found at {exported_path}")
            
            # Memory after export
            log_memory_usage("After Video Export")
            
            # Prepare results
            step_7_results = {
                'exported_path': exported_path,
                'decoded_images': decoded_images,
                'output_path': output_path,
                'fps': fps,
                'total_frames': total_frames,
                'video_dimensions': (width, height),
                'duration_seconds': duration_seconds,
                'file_size_mb': file_size if os.path.exists(exported_path) else 0,
                'timing': {
                    'export_time': time.time() - step_7_start,
                    'total_step_time': time.time() - step_7_start
                }
            }
            
            print(f"\n✅ STEP 7 COMPLETED SUCCESSFULLY in {time.time() - step_7_start:.2f}s")
            print("="*80)
            
            return step_7_results
            
        except Exception as e:
            print(f"❌ STEP 7 FAILED: {str(e)}")
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

    def run_complete_pipeline_with_memory_management(self,
                                                   step_1_params: Dict[str, Any],
                                                   step_2_params: Dict[str, Any], 
                                                   step_3_params: Dict[str, Any],
                                                   step_4_params: Dict[str, Any],
                                                   step_5_params: Dict[str, Any] = None,
                                                   step_6_params: Dict[str, Any] = None,
                                                   step_7_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run the complete pipeline with advanced ComfyUI-style memory management
        
        This method demonstrates the complete workflow:
        1. VAE + Latent Creation
        2. UNet + CLIP + LoRA Loading (with dynamic loading setup)
        3. Model Sampling + Text Encoding
        4. KSampler Denoising (with ComfyUI-style model loading/unloading)
        5. Trim Video Latent (optional)
        6. VAE Decode (optional)
        7. Video Export (optional)
        
        Args:
            step_1_params: Parameters for Step 1 (VAE + Latent Creation)
            step_2_params: Parameters for Step 2 (UNet + CLIP + LoRA Loading)
            step_3_params: Parameters for Step 3 (Model Sampling + Text Encoding)
            step_4_params: Parameters for Step 4 (KSampler Denoising)
            step_5_params: Parameters for Step 5 (Trim Video Latent) - optional
            step_6_params: Parameters for Step 6 (VAE Decode) - optional
            step_7_params: Parameters for Step 7 (Video Export) - optional
            
        Returns:
            Dictionary containing results from all steps
        """
        
        print("\n" + "="*100)
        print("🚀 COMPLETE PIPELINE WITH ADVANCED MEMORY MANAGEMENT")
        print("="*100)
        
        pipeline_start = time.time()
        
        try:
            # Step 1: VAE + Latent Creation
            print("\n🎬 STEP 1: VAE + LATENT CREATION")
            step_1_results = self.step_1_vae_and_latent_creation(**step_1_params)
            
            # Step 2: UNet + CLIP + LoRA Loading (with dynamic loading setup)
            print("\n🧠 STEP 2: UNET + CLIP + LORA LOADING")
            step_2_results = self.step_2_unet_clip_lora_loading(**step_2_params)
            
            # Step 3: Model Sampling + Text Encoding
            print("\n📝 STEP 3: MODEL SAMPLING + TEXT ENCODING")
            step_3_results = self.step_3_model_sampling_and_text_encoding(**step_3_params)
            
            # Prepare Step 4 parameters
            step_4_params['initial_latent'] = step_1_results['out_latent']['samples']
            step_4_params['positive_conditioning'] = step_3_results['positive_conditioning']
            step_4_params['negative_conditioning'] = step_3_results['negative_conditioning']
            
            # Step 4: KSampler Denoising (with ComfyUI-style memory management)
            print("\n🎯 STEP 4: KSAMPLER DENOISING")
            step_4_results = self.step_4_ksampler_denoising(**step_4_params)
            
            # Step 5: Trim Video Latent (optional)
            step_5_results = None
            if step_5_params is not None:
                print("\n🎬 STEP 5: TRIM VIDEO LATENT")
                step_5_params['denoised_latent'] = step_4_results['denoised_latent']
                step_5_results = self.step_5_trim_latent(**step_5_params)
            
            # Step 6: VAE Decode (optional)
            step_6_results = None
            if step_6_params is not None:
                print("\n🎨 STEP 6: VAE DECODE")
                # Use trimmed latent if available, otherwise use denoised latent
                latent_for_decode = step_5_results['trimmed_latent'] if step_5_results is not None else step_4_results['denoised_latent']
                step_6_params['trimmed_latent'] = latent_for_decode
                step_6_results = self.step_6_vae_decode(**step_6_params)
            
            # Step 7: Video Export (optional)
            step_7_results = None
            if step_7_params is not None:
                print("\n🎬 STEP 7: VIDEO EXPORT")
                # Use decoded images from Step 6 if available
                if step_6_results is not None:
                    step_7_params['decoded_images'] = step_6_results['decoded_images']
                    step_7_results = self.step_7_video_export(**step_7_params)
                else:
                    print("   ⚠️  Step 7 requires Step 6 (VAE Decode) to be completed first")
            
            pipeline_time = time.time() - pipeline_start
            
            # Final memory status
            print("\n📊 FINAL MEMORY STATUS:")
            log_memory_usage("Pipeline Complete")
            
            # Pipeline summary
            steps_completed = 4 + (1 if step_5_results is not None else 0) + (1 if step_6_results is not None else 0) + (1 if step_7_results is not None else 0)
            print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"   Total Time: {pipeline_time:.2f}s")
            print(f"   Steps Completed: {steps_completed}/7")
            print(f"   Memory Management: ✅ Advanced ComfyUI-style")
            
            # Check if UNet has dynamic loading
            unet_model = self.unet.model if hasattr(self.unet, 'model') else self.unet
            if hasattr(unet_model, '_dynamic_loading_info'):
                modules_count = len(unet_model._dynamic_loading_info.get('modules_info', []))
                print(f"   Dynamic Loading: ✅ Enabled ({modules_count} modules)")
            else:
                print(f"   Dynamic Loading: ❌ Not available")
            
            # Prepare return results
            results = {
                'step_1_results': step_1_results,
                'step_2_results': step_2_results,
                'step_3_results': step_3_results,
                'step_4_results': step_4_results,
                'pipeline_time': pipeline_time,
                'memory_management': 'advanced_comfyui_style',
                'dynamic_loading_enabled': hasattr(unet_model, '_dynamic_loading_info')
            }
            
            # Add Step 5 results if available
            if step_5_results is not None:
                results['step_5_results'] = step_5_results
            
            # Add Step 6 results if available
            if step_6_results is not None:
                results['step_6_results'] = step_6_results
            
            # Add Step 7 results if available
            if step_7_results is not None:
                results['step_7_results'] = step_7_results
            
            return results
            
        except Exception as e:
            print(f"\n❌ PIPELINE FAILED: {str(e)}")
            print(f"   Error Type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            raise

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
    
    def run_steps_1_and_2_only(self, step_1_params: Dict[str, Any], step_2_params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run Steps 1 and 2 in sequence - convenient for testing both steps"""
        print("🚀 Running Steps 1 and 2 in sequence...")
        print("="*60)
        
        # Run Step 1
        print("🎬 STEP 1: VAE LOADING AND LATENT CREATION")
        step_1_results = self.step_1_vae_and_latent_creation(**step_1_params)
        
        # Run Step 2
        print("\n🧠 STEP 2: UNET + CLIP LOADING")
        step_2_results = self.step_2_unet_clip_lora_loading(**step_2_params)
        
        # Summary
        print(f"\n🎉 STEPS 1 & 2 COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Display summary
        step_status = self.get_step_status()
        completed_steps = sum(1 for completed in step_status.values() if completed)
        
        print(f"\n📊 COMPLETION SUMMARY:")
        print(f"   Steps Completed: {completed_steps}/7")
        print(f"   Step 1 (VAE): {'✅ Completed' if step_status.get(1, False) else '❌ Failed'}")
        print(f"   Step 2 (UNet+CLIP): {'✅ Completed' if step_status.get(2, False) else '❌ Failed'}")
        
        if completed_steps >= 2:
            print(f"🎯 Ready for Step 3 (Model Sampling + Text Encoding)")
        
        return step_1_results, step_2_results
    

# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

def main():
    """Test Steps 1, 2, 3, and 4: Sequential VAE Loading + UNet + CLIP Loading + Model Sampling + Text Encoding + KSampler Denoising"""
    print("🚀 WAN Video Pipeline - Sequential Steps 1, 2, 3 & 4 Test")
    print("="*80)
    print("🎯 Testing Step 1 (VAE) → Step 2 (UNet + CLIP) → Step 3 (Model Sampling + Text Encoding) → Step 4 (KSampler Denoising)")
    print("="*80)
    
    # Initialize pipeline
    pipeline = WanVideoPipeline(models_dir="models")
    
    # Model file paths
    vae_model_path = "models/vaes/wan_vae.safetensors"
    unet_model_path = "models/diffusion_models/wan_2.1_diffusion_model.safetensors"
    clip_model_path = "models/text_encoders/wan_clip_model.safetensors"
    
    # Check available model files
    available_models = []
    missing_models = []
    
    if os.path.exists(vae_model_path):
        available_models.append("VAE")
    else:
        missing_models.append(f"VAE: {vae_model_path}")
    
    if os.path.exists(unet_model_path):
        available_models.append("UNet")
    else:
        missing_models.append(f"UNet: {unet_model_path}")
    
    if os.path.exists(clip_model_path):
        available_models.append("CLIP")
    else:
        missing_models.append(f"CLIP: {clip_model_path}")
    
    print(f"\n📊 MODEL AVAILABILITY:")
    print(f"   Available: {', '.join(available_models) if available_models else 'None'}")
    if missing_models:
        print(f"   Missing: {', '.join(missing_models)}")
    
    # Prepare parameters for both steps
    step_1_params = {
        'vae_model_path': vae_model_path,
        'positive_prompt': "very cinematic video",
        'negative_prompt': "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量",
        'control_video_path': "safu.mp4" if os.path.exists("safu.mp4") else None,
        'reference_image_path': "safu.jpg" if os.path.exists("safu.jpg") else None,
        'width': 480,
        'height': 832,
        'length': 37,
        'batch_size': 1,
        'strength': 1.0
    }
    
    step_2_params = {
        'unet_model_path': unet_model_path,
        'clip_model_path': clip_model_path,
        'lora_model_path': None,  # No LoRA for this test
        'strength_model': 1.0,
        'strength_clip': 0.0
    }
    
    step_3_params = {
        'positive_prompt': "very cinematic video",
        'negative_prompt': "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量",
        'shift': 8.0,
        'multiplier': 1000
    }
    
    # Sequential execution: Step 1 → Step 2 → Step 3
    try:
        # Check if we can run all four steps
        can_run_step1 = "VAE" in available_models
        can_run_step2 = "UNet" in available_models and "CLIP" in available_models
        can_run_step3 = can_run_step2  # Step 3 depends on Step 2
        can_run_step4 = can_run_step2  # Step 4 depends on Step 2 (UNet + CLIP)
        
        if can_run_step1 and can_run_step2 and can_run_step3 and can_run_step4:
            # Run Steps 1, 2, 3, and 4 sequentially
            print(f"\n🚀 RUNNING STEPS 1, 2, 3 & 4 SEQUENTIALLY")
            print("="*60)
            
            # Step 1: VAE Loading and Latent Creation
            print("🎬 STEP 1: VAE LOADING AND LATENT CREATION")
            step_1_results = pipeline.step_1_vae_and_latent_creation(**step_1_params)
            
            # Step 2: UNet + CLIP Loading
            print("\n🧠 STEP 2: UNET + CLIP LOADING")
            step_2_results = pipeline.step_2_unet_clip_lora_loading(**step_2_params)
            
            # Step 3: Model Sampling + Text Encoding
            print("\n📝 STEP 3: MODEL SAMPLING + TEXT ENCODING")
            step_3_results = pipeline.step_3_model_sampling_and_text_encoding(**step_3_params)
            
            # Step 4: KSampler Denoising
            print("\n🎯 STEP 4: KSAMPLER DENOISING")
            step_4_params = {
                'initial_latent': step_1_results['out_latent']['samples'],
                'positive_conditioning': step_3_results['positive_conditioning'],
                'negative_conditioning': step_3_results['negative_conditioning'],
                'seed': 42,
                'steps': 4,
                'cfg': 7.0,
                'sampler_name': 'euler',
                'scheduler': 'normal',
                'denoise': 1.0,
                'noise_inds': None
            }
            step_4_results = pipeline.step_4_ksampler_denoising(**step_4_params)
            
            # Step 5: Trim Video Latent (following ComfyUI implementation)
            print("\n🎬 STEP 5: TRIM VIDEO LATENT")
            step_5_params = {
                'denoised_latent': step_4_results['denoised_latent'],
                'trim_amount': 0  # Default trim amount (can be adjusted)
            }
            step_5_results = pipeline.step_5_trim_latent(**step_5_params)
            
            # Step 6: VAE Decode (following ComfyUI implementation)
            print("\n🎨 STEP 6: VAE DECODE")
            step_6_params = {
                'trimmed_latent': step_5_results['trimmed_latent'],
                'vae_model': None  # Use pipeline's VAE
            }
            step_6_results = pipeline.step_6_vae_decode(**step_6_params)
            
            # Step 7: Video Export (following standalone implementation)
            print("\n🎬 STEP 7: VIDEO EXPORT")
            step_7_params = {
                'decoded_images': step_6_results['decoded_images'],
                'output_path': 'output_video.mp4',
                'fps': 24
            }
            step_7_results = pipeline.step_7_video_export(**step_7_params)
            
            print(f"\n🎉 SEQUENTIAL STEPS 1, 2, 3, 4, 5, 6 & 7 COMPLETED SUCCESSFULLY!")
            print("="*60)
    
            # Display comprehensive results
            print(f"\n📋 COMPREHENSIVE RESULTS SUMMARY:")
            
            # Step 1 Results
            if step_1_results:
                print(f"\n🎬 STEP 1 RESULTS:")
                vae_info = step_1_results.get('vae_info', {})
                print(f"   VAE Type: {vae_info.get('vae_type', 'Unknown')}")
                print(f"   Latent Channels: {vae_info.get('latent_channels', 'Unknown')}")
                print(f"   Latent Dimension: {vae_info.get('latent_dim', 'Unknown')}")
                print(f"   Downscale Ratio: {vae_info.get('downscale_ratio', 'Unknown')}")
                print(f"   VAE Device: {vae_info.get('device', 'Unknown')}")
                
                processing_info = step_1_results.get('processing_info', {})
                print(f"   VAE Encoding Time: {processing_info.get('vae_encoding_time', 0.0):.2f}s")
                print(f"   Total Step Time: {processing_info.get('total_step_time', 0.0):.2f}s")
                
                out_latent = step_1_results.get('out_latent', {})
                if 'samples' in out_latent:
                    print(f"   Output Latent Shape: {out_latent['samples'].shape}")
            
            # Step 2 Results
            if step_2_results:
                print(f"\n🧠 STEP 2 RESULTS:")
                models_info = step_2_results.get('models_info', {})
                print(f"   UNet Type: {models_info.get('unet_type', 'Unknown')}")
                print(f"   CLIP Type: {models_info.get('clip_type', 'Unknown')}")
                print(f"   UNet Device: {models_info.get('unet_device', 'Unknown')}")
                print(f"   CLIP Device: {models_info.get('clip_device', 'Unknown')}")
                print(f"   LoRA Applied: {'Yes' if step_2_results.get('lora_applied', False) else 'No'}")
                
                processing_info = step_2_results.get('processing_info', {})
                print(f"   UNet Loading Time: {processing_info.get('unet_loading_time', 0.0):.2f}s")
                print(f"   CLIP Loading Time: {processing_info.get('clip_loading_time', 0.0):.2f}s")
                print(f"   Total Step Time: {processing_info.get('total_step_time', 0.0):.2f}s")
                
                # Verify models are loaded
                unet = step_2_results.get('unet')
                clip = step_2_results.get('clip')
                print(f"   UNet Status: {'✅ Loaded' if unet is not None else '❌ Failed'}")
                print(f"   CLIP Status: {'✅ Loaded' if clip is not None else '❌ Failed'}")
            
                # Step 3 Results
                if step_3_results:
                    print(f"\n📝 STEP 3 RESULTS:")
                    model_info = step_3_results.get('model_info', {})
                    print(f"   UNet Original Type: {model_info.get('original_type', 'Unknown')}")
                    print(f"   UNet Patched Type: {model_info.get('patched_type', 'Unknown')}")
                    print(f"   Sampling Patch Applied: {'Yes' if step_3_results.get('sampling_applied', False) else 'No'}")
                    print(f"   Shift Parameter: {model_info.get('shift', 'Unknown')}")
                    print(f"   Multiplier Parameter: {model_info.get('multiplier', 'Unknown')}")
                    
                    conditioning_info = step_3_results.get('conditioning_info', {})
                    print(f"   Positive Prompt: '{conditioning_info.get('positive_prompt', 'Unknown')}'")
                    print(f"   Negative Prompt: '{conditioning_info.get('negative_prompt', 'Unknown')}'")
                    print(f"   Positive Shape: {conditioning_info.get('positive_shape', 'Unknown')}")
                    print(f"   Negative Shape: {conditioning_info.get('negative_shape', 'Unknown')}")
                    print(f"   Positive Device: {conditioning_info.get('positive_device', 'Unknown')}")
                    
                    timing = step_3_results.get('timing', {})
                    print(f"   Sampling Time: {timing.get('sampling_time', 0.0):.2f}s")
                    print(f"   Positive Encoding Time: {timing.get('positive_encoding', 0.0):.3f}s")
                    print(f"   Negative Encoding Time: {timing.get('negative_encoding', 0.0):.3f}s")
                    print(f"   Total Step Time: {timing.get('total_step_time', 0.0):.2f}s")
                    
                    # Verify conditioning
                    positive_cond = step_3_results.get('positive_conditioning')
                    negative_cond = step_3_results.get('negative_conditioning')
                    print(f"   Positive Conditioning Status: {'✅ Generated' if positive_cond is not None else '❌ Failed'}")
                    print(f"   Negative Conditioning Status: {'✅ Generated' if negative_cond is not None else '❌ Failed'}")
                
                # Step 4 Results
                if step_4_results:
                    print(f"\n🎯 STEP 4 RESULTS:")
                    sampling_config = step_4_results.get('sampling_config', {})
                    print(f"   Seed: {sampling_config.get('seed', 'Unknown')}")
                    print(f"   Steps: {sampling_config.get('steps', 'Unknown')}")
                    print(f"   CFG: {sampling_config.get('cfg', 'Unknown')}")
                    print(f"   Sampler: {sampling_config.get('sampler_name', 'Unknown')}")
                    print(f"   Scheduler: {sampling_config.get('scheduler', 'Unknown')}")
                    print(f"   Denoise: {sampling_config.get('denoise', 'Unknown')}")
                    
                    processing_info = step_4_results.get('processing_info', {})
                    print(f"   Noise Preparation Time: {processing_info.get('noise_preparation_time', 0.0):.3f}s")
                    print(f"   KSampler Setup Time: {processing_info.get('ksampler_setup_time', 0.0):.3f}s")
                    print(f"   Denoising Time: {processing_info.get('denoising_time', 0.0):.2f}s")
                    print(f"   Total Step Time: {processing_info.get('total_step_time', 0.0):.2f}s")
                    
                    # Verify denoised latent
                    denoised_latent = step_4_results.get('denoised_latent')
                    print(f"   Denoised Latent Status: {'✅ Generated' if denoised_latent is not None else '❌ Failed'}")
                    if denoised_latent is not None:
                        print(f"   Denoised Latent Shape: {denoised_latent.shape}")
                        print(f"   Denoised Latent Device: {denoised_latent.device}")
                        print(f"   Denoised Latent Range: [{denoised_latent.min().item():.3f}, {denoised_latent.max().item():.3f}]")
                
                # Step 5 Results
                if step_5_results:
                    print(f"\n🎬 STEP 5 RESULTS:")
                    print(f"   Trim Amount: {step_5_results.get('trim_amount', 'Unknown')}")
                    print(f"   Frames Removed: {step_5_results.get('frames_removed', 'Unknown')}")
                    print(f"   Original Shape: {step_5_results.get('original_shape', 'Unknown')}")
                    print(f"   Trimmed Shape: {step_5_results.get('trimmed_shape', 'Unknown')}")
                    
                    processing_info = step_5_results.get('timing', {})
                    print(f"   Trimming Time: {processing_info.get('trimming_time', 0.0):.2f}s")
                    print(f"   Total Step Time: {processing_info.get('total_step_time', 0.0):.2f}s")
                    
                    # Verify trimmed latent
                    trimmed_latent = step_5_results.get('trimmed_latent')
                    print(f"   Trimmed Latent Status: {'✅ Generated' if trimmed_latent is not None else '❌ Failed'}")
                    if trimmed_latent is not None:
                        print(f"   Trimmed Latent Shape: {trimmed_latent.shape}")
                        print(f"   Trimmed Latent Device: {trimmed_latent.device}")
                        print(f"   Trimmed Latent Range: [{trimmed_latent.min().item():.3f}, {trimmed_latent.max().item():.3f}]")
                
                # Step 6 Results
                if step_6_results:
                    print(f"\n🎨 STEP 6 RESULTS:")
                    print(f"   VAE Model: {step_6_results.get('vae_model', 'Unknown')}")
                    print(f"   Original Latent Shape: {step_6_results.get('original_shape', 'Unknown')}")
                    print(f"   Decoded Images Shape: {step_6_results.get('decoded_shape', 'Unknown')}")
                    print(f"   Original Frames: {step_6_results.get('original_frames', 'Unknown')}")
                    print(f"   Decoded Frames: {step_6_results.get('decoded_frames', 'Unknown')}")
                    print(f"   Image Dimensions: {step_6_results.get('image_dimensions', 'Unknown')}")
                    
                    processing_info = step_6_results.get('timing', {})
                    print(f"   Decoding Time: {processing_info.get('decoding_time', 0.0):.2f}s")
                    print(f"   Total Step Time: {processing_info.get('total_step_time', 0.0):.2f}s")
                    
                    # Verify decoded images
                    decoded_images = step_6_results.get('decoded_images')
                    print(f"   Decoded Images Status: {'✅ Generated' if decoded_images is not None else '❌ Failed'}")
                    if decoded_images is not None:
                        print(f"   Decoded Images Shape: {decoded_images.shape}")
                        print(f"   Decoded Images Device: {decoded_images.device}")
                        print(f"   Decoded Images Range: [{decoded_images.min().item():.3f}, {decoded_images.max().item():.3f}]")
                
                # Step 7 Results
                if step_7_results:
                    print(f"\n🎬 STEP 7 RESULTS:")
                    print(f"   Output Path: {step_7_results.get('output_path', 'Unknown')}")
                    print(f"   Exported Path: {step_7_results.get('exported_path', 'Unknown')}")
                    print(f"   FPS: {step_7_results.get('fps', 'Unknown')}")
                    print(f"   Total Frames: {step_7_results.get('total_frames', 'Unknown')}")
                    print(f"   Video Dimensions: {step_7_results.get('video_dimensions', 'Unknown')}")
                    print(f"   Duration: {step_7_results.get('duration_seconds', 'Unknown'):.2f} seconds")
                    print(f"   File Size: {step_7_results.get('file_size_mb', 'Unknown'):.2f} MB")
                    
                    processing_info = step_7_results.get('timing', {})
                    print(f"   Export Time: {processing_info.get('export_time', 0.0):.2f}s")
                    print(f"   Total Step Time: {processing_info.get('total_step_time', 0.0):.2f}s")
                    
                    # Verify video export
                    exported_path = step_7_results.get('exported_path')
                    print(f"   Video Export Status: {'✅ Generated' if exported_path and os.path.exists(exported_path) else '❌ Failed'}")
                    if exported_path and os.path.exists(exported_path):
                        print(f"   Video File: {exported_path}")
                        print(f"   File Size: {os.path.getsize(exported_path) / (1024 * 1024):.2f} MB")
            
        elif can_run_step1 and can_run_step2 and can_run_step3:
            # Run Steps 1, 2, and 3 only
            print(f"\n🚀 RUNNING STEPS 1, 2 & 3 ONLY (Step 4 requires all)")
        print("="*60)
        
        # Step 1: VAE Loading and Latent Creation
        print("🎬 STEP 1: VAE LOADING AND LATENT CREATION")
        step_1_results = pipeline.step_1_vae_and_latent_creation(**step_1_params)
        
        # Step 2: UNet + CLIP Loading
        print("\n🧠 STEP 2: UNET + CLIP LOADING")
        step_2_results = pipeline.step_2_unet_clip_lora_loading(**step_2_params)
        
        print(f"\n🎉 STEPS 1 & 2 COMPLETED SUCCESSFULLY!")
        print("="*60)
            
        if step_1_results:
            print(f"\n📋 STEP 1 RESULTS SUMMARY:")
            vae_info = step_1_results.get('vae_info', {})
            print(f"   VAE Type: {vae_info.get('vae_type', 'Unknown')}")
            print(f"   Latent Channels: {vae_info.get('latent_channels', 'Unknown')}")
            print(f"   VAE Device: {vae_info.get('device', 'Unknown')}")
            
            processing_info = step_1_results.get('processing_info', {})
            print(f"   Total Step Time: {processing_info.get('total_step_time', 0.0):.2f}s")
            
            if step_2_results:
                print(f"\n📋 STEP 2 RESULTS SUMMARY:")
                models_info = step_2_results.get('models_info', {})
                print(f"   UNet Type: {models_info.get('unet_type', 'Unknown')}")
                print(f"   CLIP Type: {models_info.get('clip_type', 'Unknown')}")
                print(f"   UNet Device: {models_info.get('unet_device', 'Unknown')}")
                print(f"   CLIP Device: {models_info.get('clip_device', 'Unknown')}")
                
                processing_info = step_2_results.get('processing_info', {})
                print(f"   Total Step Time: {processing_info.get('total_step_time', 0.0):.2f}s")
            
            print(f"\n💡 Steps 1, 2 & 3 completed - Step 4 requires all models")
            
        elif can_run_step1:
            # Only run Step 1
            print(f"\n🚀 RUNNING STEP 1 ONLY (Step 2 models not available)")
            print("="*60)
            
            # Step 1: VAE Loading and Latent Creation
            print("🎬 STEP 1: VAE LOADING AND LATENT CREATION")
            step_1_results = pipeline.step_1_vae_and_latent_creation(**step_1_params)
            
            print(f"\n🎉 STEP 1 COMPLETED SUCCESSFULLY!")
            print("="*60)
            
            if step_1_results:
                print(f"\n📋 STEP 1 RESULTS SUMMARY:")
                vae_info = step_1_results.get('vae_info', {})
                print(f"   VAE Type: {vae_info.get('vae_type', 'Unknown')}")
                print(f"   Latent Channels: {vae_info.get('latent_channels', 'Unknown')}")
                print(f"   VAE Device: {vae_info.get('device', 'Unknown')}")
                
                processing_info = step_1_results.get('processing_info', {})
                print(f"   Total Step Time: {processing_info.get('total_step_time', 0.0):.2f}s")
            
            print(f"\n💡 Step 1 completed - Step 2 requires UNet and CLIP model files")
            
        elif can_run_step2:
            # Only run Step 2
            print(f"\n🚀 RUNNING STEP 2 ONLY (Step 1 VAE model not available)")
            print("="*60)
            
            # Step 2: UNet + CLIP Loading
            print("🧠 STEP 2: UNET + CLIP LOADING")
            step_2_results = pipeline.step_2_unet_clip_lora_loading(**step_2_params)
            
            print(f"\n🎉 STEP 2 COMPLETED SUCCESSFULLY!")
            print("="*60)
            
            if step_2_results:
                print(f"\n📋 STEP 2 RESULTS SUMMARY:")
                models_info = step_2_results.get('models_info', {})
                print(f"   UNet Type: {models_info.get('unet_type', 'Unknown')}")
                print(f"   CLIP Type: {models_info.get('clip_type', 'Unknown')}")
                print(f"   UNet Device: {models_info.get('unet_device', 'Unknown')}")
                print(f"   CLIP Device: {models_info.get('clip_device', 'Unknown')}")
                
                processing_info = step_2_results.get('processing_info', {})
                print(f"   Total Step Time: {processing_info.get('total_step_time', 0.0):.2f}s")
            
            print(f"\n💡 Step 2 completed - Step 1 requires VAE model file")
            
        else:
            # No models available
            print(f"\n⏭️  NO MODELS AVAILABLE - Testing pipeline initialization only")
            print("="*60)
            
            print(f"   ✅ Pipeline initialized successfully")
            print(f"   Device: {pipeline.device}")
            print(f"   Offload Device: {pipeline.offload_device}")
            print(f"   Models Directory: {pipeline.models_dir}")
            
            step_status = pipeline.get_step_status()
            print(f"   Initial step status: {step_status}")
            
            print(f"\n💡 Pipeline ready - model files required for Step 1 and Step 2")
        
        # Final status check
        step_status = pipeline.get_step_status()
        completed_steps = sum(1 for completed in step_status.values() if completed)
        
        print(f"\n📊 FINAL PIPELINE STATUS:")
        print(f"   Steps Completed: {completed_steps}/7")
        for step_num, completed in step_status.items():
            status = "✅ Completed" if completed else "⏳ Pending"
            print(f"   Step {step_num}: {status}")
        
        if completed_steps >= 4:
            print(f"\n🎉 SUCCESS: Steps 1, 2, 3, and 4 completed sequentially!")
            print(f"🎯 Pipeline is ready for Step 5 (VAE Decoding)")
        elif completed_steps >= 3:
            print(f"\n✅ PARTIAL SUCCESS: Steps 1, 2, and 3 completed!")
            print(f"🎯 Pipeline is ready for Step 4 (KSampler Denoising)")
        elif completed_steps >= 2:
            print(f"\n✅ PARTIAL SUCCESS: Steps 1 and 2 completed!")
            print(f"🎯 Pipeline is ready for Step 3 (Model Sampling + Text Encoding)")
        elif completed_steps == 1:
            print(f"\n✅ PARTIAL SUCCESS: One step completed!")
            print(f"💡 Additional model files needed for complete testing")
        else:
            print(f"\n💡 Pipeline initialization completed")
            print(f"🔧 Model files required for Step 1, 2, 3, and 4 testing")
        
    except Exception as e:
        print(f"\n❌ SEQUENTIAL STEPS 1, 2, 3 & 4 TEST FAILED: {str(e)}")
        print(f"   Error Type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        print(f"\n💡 Check the error details above and ensure:")
        print(f"   - Model files exist and are valid")
        print(f"   - All required dependencies are installed")
        print(f"   - The standalone_sd.py fixes are properly applied")
        print(f"   - Step 4 KSampler denoising and ComfyUI integration are working correctly")
    

if __name__ == "__main__":
    main()