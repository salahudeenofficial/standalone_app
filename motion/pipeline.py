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

def analyze_ksampler_inputs(positive_conditioning, negative_conditioning, initial_latent):
    """
    Analyze K-Sampler inputs in detail
    
    Args:
        positive_conditioning: Positive conditioning with VACE
        negative_conditioning: Negative conditioning with VACE  
        initial_latent: Initial latent tensor or dict with 'samples'
    """
    print("\n🎯 K-SAMPLER INPUT ANALYSIS:")
    print("   📋 Analyzing inputs that will be used by K-Sampler:")
    print()
    
    total_tensors = 0
    total_memory = 0
    
    # Analyze positive conditioning
    print("   📋 Positive Conditioning (K-Sampler input):")
    pos_tensors, pos_memory = _analyze_conditioning(positive_conditioning, "positive")
    total_tensors += pos_tensors
    total_memory += pos_memory
    print()
    
    # Analyze negative conditioning  
    print("   📋 Negative Conditioning (K-Sampler input):")
    neg_tensors, neg_memory = _analyze_conditioning(negative_conditioning, "negative")
    total_tensors += neg_tensors
    total_memory += neg_memory
    print()
    
    # Analyze latent image
    print("   📋 Latent Image (K-Sampler input):")
    lat_tensors, lat_memory = _analyze_latent_image(initial_latent)
    total_tensors += lat_tensors
    total_memory += lat_memory
    print()
    
    print(f"   ✅ K-Sampler inputs analyzed successfully")

def _analyze_conditioning(conditioning, name):
    """Analyze conditioning structure in detail"""
    tensor_count = 0
    total_memory = 0
    
    if isinstance(conditioning, list):
        print(f"      Type: list")
        print(f"      List length: {len(conditioning)}")
        
        for i, item in enumerate(conditioning):
            if hasattr(item, 'shape'):
                # This is a tensor
                tensor_count += 1
                memory = _get_tensor_memory(item)
                total_memory += memory
                
                print(f"          Tensor {tensor_count}:")
                _print_tensor_info(item, memory, indent="            ")
                
            elif isinstance(item, dict):
                # This is a dictionary, analyze each key
                for key, value in item.items():
                    print(f"          Key '{key}': {type(value).__name__}")
                    if hasattr(value, 'shape'):
                        # Direct tensor
                        tensor_count += 1
                        memory = _get_tensor_memory(value)
                        total_memory += memory
                        
                        print(f"            -> Tensor with shape {value.shape}")
                        print(f"                Tensor {tensor_count}:")
                        _print_tensor_info(value, memory, indent="                  ")
                        
                    elif isinstance(value, list):
                        print(f"            -> Contains {len(value)} items")
                        for j, list_item in enumerate(value):
                            if hasattr(list_item, 'shape'):
                                tensor_count += 1
                                memory = _get_tensor_memory(list_item)
                                total_memory += memory
                                
                                print(f"              Item {j}: Tensor with shape {list_item.shape}")
                                print(f"                Tensor {tensor_count}:")
                                _print_tensor_info(list_item, memory, indent="                  ")
                            elif isinstance(list_item, (int, float, str)):
                                print(f"              Item {j}: {type(list_item).__name__} = {list_item}")
                                if j >= 2:  # Limit output for long lists
                                    print(f"              ... (and {len(value) - j - 1} more items)")
                                    break
                    elif value is None:
                        print(f"            -> None")
                    else:
                        print(f"            -> {type(value).__name__}")
            else:
                print(f"          Item {i}: {type(item).__name__}")
    
    elif isinstance(conditioning, dict):
        print(f"      Type: dict")
        print(f"      Dict keys: {list(conditioning.keys())}")
        # Similar analysis for dict case...
    
    else:
        print(f"      Type: {type(conditioning).__name__}")
    
    print(f"      Total tensors found: {tensor_count}")
    print(f"      Total memory: {total_memory / (1024**2):.2f} MB")
    
    if tensor_count == 3:  # Expected: text + VACE frames + VACE mask
        print(f"      ✅ VERIFIED: 3 tensors found (text + VACE frames + VACE mask)")
    else:
        print(f"      ⚠️  Expected 3 tensors, found {tensor_count}")
    
    return tensor_count, total_memory

def _analyze_latent_image(latent_image):
    """Analyze latent image structure"""
    tensor_count = 0
    total_memory = 0
    
    if isinstance(latent_image, dict):
        print(f"      Type: dict")
        print(f"      Dict keys: {list(latent_image.keys())}")
        
        if 'samples' in latent_image:
            samples = latent_image['samples']
            print(f"      Samples type: {type(samples).__name__}")
            
            if hasattr(samples, 'shape'):
                tensor_count = 1
                memory = _get_tensor_memory(samples)
                total_memory = memory
                
                _print_tensor_info(samples, memory, indent="      ")
    
    elif hasattr(latent_image, 'shape'):
        print(f"      Type: Tensor")
        tensor_count = 1
        memory = _get_tensor_memory(latent_image)
        total_memory = memory
        
        _print_tensor_info(latent_image, memory, indent="      ")
    
    else:
        print(f"      Type: {type(latent_image).__name__}")
    
    return tensor_count, total_memory

def _print_tensor_info(tensor, memory, indent=""):
    """Print detailed tensor information"""
    print(f"{indent}Shape: {tensor.shape}")
    print(f"{indent}Dtype: {tensor.dtype}")
    print(f"{indent}Device: {tensor.device}")
    print(f"{indent}Memory: {memory / (1024**2):.2f} MB")
    
    # Get value range and mean
    min_val = tensor.min().item()
    max_val = tensor.max().item()
    mean_val = tensor.mean().item()
    print(f"{indent}Value Range: [{min_val:.4f}, {max_val:.4f}], Mean: {mean_val:.4f}")
    
    # Get first 5 values (flattened)
    flat_tensor = tensor.flatten()
    first_values = [f"{flat_tensor[i].item():.4f}" for i in range(min(5, len(flat_tensor)))]
    print(f"{indent}First 5 values: {first_values}")

def _get_tensor_memory(tensor):
    """Calculate tensor memory usage in bytes"""
    if hasattr(tensor, 'numel') and hasattr(tensor, 'element_size'):
        return tensor.numel() * tensor.element_size()
    else:
        return 0


def remove_debug_code():
    """
    Instructions to remove debug code:
    1. Delete this function
    2. Delete all calls to _print_tensor_debug_info()
    3. Delete the _print_tensor_debug_info() function definition
    4. Search for "DEBUG:" comments and remove those sections
    """
    pass

def common_upscale(samples, width, height, upscale_method, crop):
    """
    ComfyUI-compatible common_upscale function
    Borrowed from ComfyUI comfy/utils.py to ensure identical processing
    """
    orig_shape = tuple(samples.shape)
    if len(orig_shape) > 4:
        samples = samples.reshape(samples.shape[0], samples.shape[1], -1, samples.shape[-2], samples.shape[-1])
        samples = samples.movedim(2, 1)
        samples = samples.reshape(-1, orig_shape[1], orig_shape[-2], orig_shape[-1])
    
    if crop == "center":
        old_width = samples.shape[-1]
        old_height = samples.shape[-2]
        old_aspect = old_width / old_height
        new_aspect = width / height
        x = 0
        y = 0
        if old_aspect > new_aspect:
            x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
        elif old_aspect < new_aspect:
            y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
        s = samples.narrow(-2, y, old_height - y * 2).narrow(-1, x, old_width - x * 2)
    else:
        s = samples

    # Use torch.nn.functional.interpolate for bilinear (ComfyUI default)
    out = torch.nn.functional.interpolate(s, size=(height, width), mode=upscale_method)

    if len(orig_shape) == 4:
        return out

    out = out.reshape((orig_shape[0], -1, orig_shape[1]) + (height, width))
    return out.movedim(2, 1).reshape(orig_shape[:-2] + (height, width))

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
            # Load VAE state dict
            vae_state_dict = load_torch_file(vae_model_path)
            
            # GPU Monitoring: Check GPU state before VAE creation
            self._log_gpu_state("BEFORE VAE CREATION")
            
            # Create VAE instance using ComfyUI-style implementation
            self.vae = create_vae(state_dict=vae_state_dict, device=self.device)
            
            # GPU Monitoring: Check GPU state after VAE creation
            self._log_gpu_state("AFTER VAE CREATION")
            
            # Verify VAE is properly initialized
            self.vae.throw_exception_if_invalid()
            
            # Force VAE to GPU if needed
            if torch.cuda.is_available() and self.device.type == "cuda":
                print(f"🔧 FORCING VAE TO GPU: {self.device}")
                self.vae.first_stage_model.to(self.device)
                print(f"✅ VAE moved to GPU: {self.device}")
            
            # Device verification: Ensure VAE model is on GPU
            self._verify_vae_device()
            
            # Load control video
            control_video = None
            if control_video_path and os.path.exists(control_video_path):
                control_video = self.load_video(control_video_path)
            else:
                # Use real video file for testing (safu.mp4)
                real_video_path = "safu.mp4"
                if os.path.exists(real_video_path):
                    print(f"🎬 Using real video file: {real_video_path}")
                    control_video = self.load_video(real_video_path)
                else:
                    print(f"⚠️  Real video file not found: {real_video_path}")
                    print(f"   Creating dummy control video for testing")
                    control_video = torch.rand(length, height, width, 3)
            
            # Load reference image
            reference_image = None
            if reference_image_path and os.path.exists(reference_image_path):
                reference_image = self.load_image(reference_image_path)
            else:
                # Use real reference image for testing (safu.jpg)
                real_image_path = "safu.jpg"
                if os.path.exists(real_image_path):
                    print(f"🖼️  Using real reference image: {real_image_path}")
                    reference_image = self.load_image(real_image_path)
                else:
                    print(f"⚠️  Real reference image not found: {real_image_path}")
                    print(f"   No reference image will be used")
            
            # Process control video using ComfyUI-compatible method (exact match to WanVaceToVideo)
            if control_video is not None:
                # Use ComfyUI's common_upscale with movedim (exact match to WanVaceToVideo)
                control_video = common_upscale(
                    control_video[:length].movedim(-1, 1), 
                    width, height, "bilinear", "center"
                ).movedim(1, -1)
                
                # Use ComfyUI's padding method (exact match to WanVaceToVideo)
                if control_video.shape[0] < length:
                    control_video = torch.nn.functional.pad(
                        control_video, (0, 0, 0, 0, 0, 0, 0, length - control_video.shape[0]), 
                        value=0.5
                    )
            else:
                control_video = torch.ones((length, height, width, 3)) * 0.5
            
            # Continue with encoding
            return self._step_1_continue_encoding(control_video, reference_image, 
                                                width, height, length, batch_size, time.time(),
                                                positive_prompt, negative_prompt, strength)
            
        except Exception as e:
            print(f"❌ STEP 1 FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _step_1_continue_encoding(self, control_video, reference_image, width, height, length, batch_size, start_time, positive_prompt, negative_prompt, strength):
        """Continue Step 1 VAE encoding process"""
        
        # Create control mask using ComfyUI-compatible method (exact match to WanVaceToVideo)
        mask = torch.ones((length, height, width, 1), device=control_video.device)
        
        # CRITICAL FIX: Use proper control video processing (exact match to ComfyUI WanVaceToVideo)
        # ComfyUI WanVaceToVideo does: control_video = control_video - 0.5, then splits by mask
        control_video_centered = control_video - 0.5  # Center around 0
        inactive = (control_video_centered * (1 - mask)) + 0.5  # Inactive regions
        reactive = (control_video_centered * mask) + 0.5        # Active/controlled regions
        
        print(f"🔍 CONTROL VIDEO PROCESSING DEBUG:")
        print(f"   Original control_video range: [{control_video.min().item():.6f}, {control_video.max().item():.6f}]")
        print(f"   Original control_video mean: {control_video.mean().item():.6f}")
        print(f"   Original control_video std: {control_video.std().item():.6f}")
        print(f"   Centered control_video range: [{control_video_centered.min().item():.6f}, {control_video_centered.max().item():.6f}]")
        print(f"   Centered control_video mean: {control_video_centered.mean().item():.6f}")
        print(f"   Centered control_video std: {control_video_centered.std().item():.6f}")
        print(f"   Inactive tensor range: [{inactive.min().item():.6f}, {inactive.max().item():.6f}]")
        print(f"   Inactive tensor mean: {inactive.mean().item():.6f}")
        print(f"   Inactive tensor std: {inactive.std().item():.6f}")
        print(f"   Reactive tensor range: [{reactive.min().item():.6f}, {reactive.max().item():.6f}]")
        print(f"   Reactive tensor mean: {reactive.mean().item():.6f}")
        print(f"   Reactive tensor std: {reactive.std().item():.6f}")
        print()
        
        # VAE encoding of control video (exact match to ComfyUI - pass same range)
        with torch.no_grad():
            # GPU Monitoring: Check initial GPU state
            self._log_gpu_state("BEFORE VAE ENCODING")
            
            # DEBUG: Print tensor info before VAE encoding
            print(f"🔍 CONTROL VIDEO TENSORS BEFORE VAE ENCODING:")
            print(f"   Inactive tensor:")
            print(f"     Shape: {inactive[:, :, :, :3].shape}")
            print(f"     Dtype: {inactive[:, :, :, :3].dtype}")
            print(f"     Device: {inactive[:, :, :, :3].device}")
            print(f"     Mean: {inactive[:, :, :, :3].mean().item():.6f}")
            print(f"     Min: {inactive[:, :, :, :3].min().item():.6f}")
            print(f"     Max: {inactive[:, :, :, :3].max().item():.6f}")
            print(f"     Range: [{inactive[:, :, :, :3].min().item():.6f}, {inactive[:, :, :, :3].max().item():.6f}]")
            print(f"     Std: {inactive[:, :, :, :3].std().item():.6f}")
            print()
            
            print(f"   Reactive tensor:")
            print(f"     Shape: {reactive[:, :, :, :3].shape}")
            print(f"     Dtype: {reactive[:, :, :, :3].dtype}")
            print(f"     Device: {reactive[:, :, :, :3].device}")
            print(f"     Mean: {reactive[:, :, :, :3].mean().item():.6f}")
            print(f"     Min: {reactive[:, :, :, :3].min().item():.6f}")
            print(f"     Max: {reactive[:, :, :, :3].max().item():.6f}")
            print(f"     Range: [{reactive[:, :, :, :3].min().item():.6f}, {reactive[:, :, :, :3].max().item():.6f}]")
            print(f"     Std: {reactive[:, :, :, :3].std().item():.6f}")
            print()
            
            # Device verification: Ensure VAE model is on GPU
            self._verify_vae_device()
            
            # CRITICAL FIX: Convert to VAE format [1, 3, T, H, W] (motion pipeline VAE expects this)
            # Convert from [T, H, W, 3] to [1, 3, T, H, W] for VAE encoding
            inactive_5d = inactive[:, :, :, :3].permute(3, 0, 1, 2).unsqueeze(0)  # [T,H,W,3] -> [1,3,T,H,W]
            reactive_5d = reactive[:, :, :, :3].permute(3, 0, 1, 2).unsqueeze(0)  # [T,H,W,3] -> [1,3,T,H,W]
            
            inactive_latent = self.vae.encode(inactive_5d)
            
            # GPU Monitoring: Check GPU state after first encode
            self._log_gpu_state("AFTER INACTIVE VAE ENCODE")
            
            reactive_latent = self.vae.encode(reactive_5d)
            
            # GPU Monitoring: Check GPU state after second encode
            self._log_gpu_state("AFTER REACTIVE VAE ENCODE")
            
            control_video_latent = torch.cat((inactive_latent, reactive_latent), dim=1)
            
            # DETAILED RESULTS: Three VAE Encodes Analysis
            print(f"\n📊 THREE VAE ENCODES RESULTS:")
            print("=" * 60)
            
            # 1. Inactive Latent Results
            print(f"1️⃣ INACTIVE LATENT:")
            print(f"   Shape: {inactive_latent.shape}")
            print(f"   Mean: {inactive_latent.mean().item():.6f}")
            print(f"   Range: [{inactive_latent.min().item():.6f}, {inactive_latent.max().item():.6f}]")
            print(f"   Std: {inactive_latent.std().item():.6f}")
            flat_inactive = inactive_latent.flatten()
            first_5_inactive = [f"{flat_inactive[i].item():.6f}" for i in range(min(5, len(flat_inactive)))]
            print(f"   First 5 elements: {first_5_inactive}")
            
            # 2. Reactive Latent Results
            print(f"\n2️⃣ REACTIVE LATENT:")
            print(f"   Shape: {reactive_latent.shape}")
            print(f"   Mean: {reactive_latent.mean().item():.6f}")
            print(f"   Range: [{reactive_latent.min().item():.6f}, {reactive_latent.max().item():.6f}]")
            print(f"   Std: {reactive_latent.std().item():.6f}")
            flat_reactive = reactive_latent.flatten()
            first_5_reactive = [f"{flat_reactive[i].item():.6f}" for i in range(min(5, len(flat_reactive)))]
            print(f"   First 5 elements: {first_5_reactive}")
            
            # 3. Reference Image Latent Results (if available)
            if reference_image_latent is not None:
                print(f"\n3️⃣ REFERENCE IMAGE LATENT:")
                print(f"   Shape: {reference_image_latent.shape}")
                print(f"   Mean: {reference_image_latent.mean().item():.6f}")
                print(f"   Range: [{reference_image_latent.min().item():.6f}, {reference_image_latent.max().item():.6f}]")
                print(f"   Std: {reference_image_latent.std().item():.6f}")
                flat_reference = reference_image_latent.flatten()
                first_5_reference = [f"{flat_reference[i].item():.6f}" for i in range(min(5, len(flat_reference)))]
                print(f"   First 5 elements: {first_5_reference}")
            else:
                print(f"\n3️⃣ REFERENCE IMAGE LATENT: Not available")
            
            print("=" * 60)
            
        
        # Process reference image using ComfyUI-compatible method
        reference_image_latent = None
        if reference_image is not None:
            # Use ComfyUI's common_upscale with movedim (exact match to WanVaceToVideo)
            reference_image = common_upscale(
                reference_image[:1].movedim(-1, 1), 
                width, height, "bilinear", "center"
            ).movedim(1, -1)
            
            # Encode reference image
            with torch.no_grad():
                # GPU Monitoring: Check GPU state before reference image encoding
                self._log_gpu_state("BEFORE REFERENCE IMAGE VAE ENCODING")
                
                # DEBUG: Print tensor info before VAE encoding
                print(f"🔍 REFERENCE IMAGE TENSOR BEFORE VAE ENCODING:")
                print(f"   Shape: {reference_image[:, :, :, :3].shape}")
                print(f"   Dtype: {reference_image[:, :, :, :3].dtype}")
                print(f"   Device: {reference_image[:, :, :, :3].device}")
                print(f"   Mean: {reference_image[:, :, :, :3].mean().item():.6f}")
                print(f"   Min: {reference_image[:, :, :, :3].min().item():.6f}")
                print(f"   Max: {reference_image[:, :, :, :3].max().item():.6f}")
                print(f"   Range: [{reference_image[:, :, :, :3].min().item():.6f}, {reference_image[:, :, :, :3].max().item():.6f}]")
                print(f"   Std: {reference_image[:, :, :, :3].std().item():.6f}")
                print()
                
                # CRITICAL FIX: Convert to VAE format [1, 3, 1, H, W] (motion pipeline VAE expects this)
                # Convert from [1, H, W, 3] to [1, 3, 1, H, W] for VAE encoding
                reference_5d = reference_image[:, :, :, :3].permute(3, 0, 1, 2).unsqueeze(0)  # [1,H,W,3] -> [1,3,1,H,W]
                
                # CRITICAL FIX: Ensure reference image is in float32 for VAE encoding
                if reference_5d.dtype != torch.float32:
                    print(f"🔧 Converting reference image from {reference_5d.dtype} to float32")
                    reference_5d = reference_5d.float()
                
                # CRITICAL FIX: Use float32 encoding for reference image
                reference_image_latent = self.vae.encode(reference_5d)
                
                # GPU Monitoring: Check GPU state after reference image encoding
                self._log_gpu_state("AFTER REFERENCE IMAGE VAE ENCODING")
                
            
            # Add motion latent channels (WAN format) - exact match to ComfyUI WanVaceToVideo
            try:
                from wan_latent_format import Wan21_LatentFormat
                wan21_format = Wan21_LatentFormat()
                motion_channels = wan21_format.process_out(torch.zeros_like(reference_image_latent))
                reference_image_latent = torch.cat([reference_image_latent, motion_channels], dim=1)
                print(f"   ✅ Added WAN21 motion channels to reference image")
            except ImportError:
                print(f"   ⚠️  Wan21_LatentFormat not available, using standard format")
                pass  # Use standard format
        
        # Create final initial latent using ComfyUI-compatible method (exact match to WanVaceToVideo)
        vae_stride = 8
        latent_height = height // vae_stride
        latent_width = width // vae_stride
        latent_length = ((length - 1) // 4) + 1
        
        # Start with control video latent
        initial_latent = control_video_latent
        
        # Add reference image if provided (exact match to ComfyUI)
        if reference_image_latent is not None:
            initial_latent = torch.cat((reference_image_latent, control_video_latent), dim=2)
        
        # Create control mask in latent space using ComfyUI-compatible method (exact match to WanVaceToVideo)
        height_mask = height // vae_stride
        width_mask = width // vae_stride
        
        mask_latent = mask.view(length, height_mask, vae_stride, width_mask, vae_stride)
        mask_latent = mask_latent.permute(2, 4, 0, 1, 3)
        mask_latent = mask_latent.reshape(vae_stride * vae_stride, length, height_mask, width_mask)
        
        # Interpolate mask to latent temporal resolution using ComfyUI method
        mask_latent = torch.nn.functional.interpolate(
            mask_latent.unsqueeze(0), 
            size=(latent_length, height_mask, width_mask), 
            mode='nearest-exact'
        ).squeeze(0)
        
        # Handle reference image mask padding using ComfyUI method
        if reference_image_latent is not None:
            mask_pad = torch.zeros_like(mask_latent[:, :reference_image_latent.shape[2], :, :])
            mask_latent = torch.cat((mask_pad, mask_latent), dim=1)
            latent_length += reference_image_latent.shape[2]
        
        mask_latent = mask_latent.unsqueeze(0)  # Add batch dimension
        
        # Setup VACE Conditioning using ComfyUI-compatible method (exact match to WanVaceToVideo)
        # Create empty conditioning structures that will be populated with VACE data
        empty_text_tensor = torch.zeros([1, 77, 4096], device=self.device, dtype=torch.float32)
        
        # Use ComfyUI's conditioning_set_values equivalent structure
        positive = [
            empty_text_tensor,  # Placeholder - will be replaced with actual text encoding in Step 3
            {
                "pooled_output": None,
                "vace_frames": [initial_latent],
                "vace_mask": [mask_latent], 
                "vace_strength": [strength]
            }
        ]
        
        negative = [
            empty_text_tensor,  # Placeholder - will be replaced with actual text encoding in Step 3
            {
                "pooled_output": None,
                "vace_frames": [initial_latent],
                "vace_mask": [mask_latent], 
                "vace_strength": [strength]
            }
        ]
        
        # Mark step complete and return results
        self.step_completed[1] = True
        
        # Create WAN-format output latent using ComfyUI-compatible method (exact match to WanVaceToVideo)
        output_latent = torch.zeros([batch_size, 16, latent_length, latent_height, latent_width], 
                                   device=self.device, dtype=self.vae.vae_dtype)
        out_latent = {"samples": output_latent}
        
        # Calculate trim_latent using ComfyUI method (exact match to WanVaceToVideo)
        trim_latent = reference_image_latent.shape[2] if reference_image_latent is not None else 0
        
        # Return results
        step_1_results = {
            'positive': positive,
            'negative': negative,  
            'out_latent': out_latent,
            'trim_latent': trim_latent,
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
                'channels': 16,  # Fixed to match ComfyUI WanVaceToVideo
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
                'total_step_time': time.time() - start_time,
                'comfyui_style': True,
                'memory_management': 'comfyui_style'
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
                                               vace_positive_conditioning: Any = None,
                                               vace_negative_conditioning: Any = None,
                                               shift: float = 8.0,
                                               multiplier: int = 1000) -> Dict[str, Any]:
        """
        Step 3: Model Sampling + Text Encoding + VACE Conditioning Integration
        
        This step applies SD3 model sampling and encodes text prompts while preserving VACE conditioning:
        1. Applies ModelSamplingSD3 to the UNet model with shift parameter
        2. Initializes CLIP text encoder
        3. Encodes positive and negative text prompts  
        4. Combines text encoding with VACE conditioning from Step 1
        5. Returns combined conditioning tensors ready for sampling
        
        Args:
            positive_prompt: Positive text prompt for conditioning
            negative_prompt: Negative text prompt for conditioning
            vace_positive_conditioning: VACE conditioning from Step 1 (optional)
            vace_negative_conditioning: VACE conditioning from Step 1 (optional)
            shift: SD3 shift parameter (default 8.0)
            multiplier: SD3 multiplier parameter (default 1000)
            
        Returns:
            Dictionary containing combined conditioning and model information
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
                text_positive_cond = text_encoder.encode(self.clip, positive_prompt)
                positive_encoding_time = time.time() - positive_encoding_start
                print(f"   ✅ Positive prompt encoded in {positive_encoding_time:.3f}s")
            except Exception as e:
                print(f"   ❌ Positive prompt encoding failed: {e}")
                raise
            
            # Encode negative prompt
            negative_encoding_start = time.time()
            try:
                text_negative_cond = text_encoder.encode(self.clip, negative_prompt)
                negative_encoding_time = time.time() - negative_encoding_start
                print(f"   ✅ Negative prompt encoded in {negative_encoding_time:.3f}s")
            except Exception as e:
                print(f"   ❌ Negative prompt encoding failed: {e}")
                raise
            
            # Combine text encoding with VACE conditioning
            if vace_positive_conditioning is not None and vace_negative_conditioning is not None:
                print("   🔗 Combining text encoding with VACE conditioning...")
                
                # Extract text tensor from text conditioning
                if isinstance(text_positive_cond, (tuple, list)) and len(text_positive_cond) > 0:
                    text_positive_tensor = text_positive_cond[0]
                else:
                    text_positive_tensor = text_positive_cond
                
                if isinstance(text_negative_cond, (tuple, list)) and len(text_negative_cond) > 0:
                    text_negative_tensor = text_negative_cond[0]
                else:
                    text_negative_tensor = text_negative_cond
                
                # VACE conditioning should already be in ComfyUI-style list format from Step 1
                # Just replace the text tensor (first element) with the actual encoded text
                positive_cond = vace_positive_conditioning.copy()
                positive_cond[0] = text_positive_tensor
                
                negative_cond = vace_negative_conditioning.copy()
                negative_cond[0] = text_negative_tensor
                
                print("   ✅ VACE conditioning preserved and combined with text encoding")
                print(f"   📊 Positive conditioning structure: {len(positive_cond)} items")
                print(f"   📊 Negative conditioning structure: {len(negative_cond)} items")
            else:
                # No VACE conditioning provided, use text-only conditioning
                print("   ⚠️  No VACE conditioning provided, using text-only conditioning")
                positive_cond = text_positive_cond
                negative_cond = text_negative_cond
            
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
        Step 4: KSampler Denoising (ComfyUI Compatible)
        
        This step performs the core denoising process using ComfyUI-compatible KSampler:
        1. Fixes empty latent channels (ComfyUI pattern)
        2. Prepares noise for the initial latent
        3. Sets up the KSampler with specified parameters
        4. Performs denoising steps with zero tensor as latent_image (ComfyUI pattern)
        5. Returns the denoised latent ready for VAE decoding
        
        Args:
            initial_latent: Zero tensor from Step 1 (ComfyUI pattern)
            positive_conditioning: Positive conditioning with VACE from Step 3
            negative_conditioning: Negative conditioning with VACE from Step 3
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
        
        try:
            step_4_start = time.time()
            
            # Verify prerequisites from previous steps
            if not self.step_completed[3]:
                raise RuntimeError("Step 3 (Model Sampling + Text Encoding) must be completed before Step 4")
            
            if self.unet is None:
                raise RuntimeError("UNet model not loaded - Step 2 must be completed first")
            
            if self.clip is None:
                raise RuntimeError("CLIP model not loaded - Step 2 must be completed first")
            
            # DETAILED K-SAMPLER INPUT ANALYSIS
            analyze_ksampler_inputs(positive_conditioning, negative_conditioning, initial_latent)
            
            # Prepare noise for initial latent
            try:
                from comfy.sample import fix_empty_latent_channels
                initial_latent = fix_empty_latent_channels(self.unet, initial_latent)
            except ImportError:
                pass  # Use original latent
            
            noise = prepare_noise(initial_latent, seed, noise_inds)
            
            # Create KSampler instance
            ksampler = StandaloneKSampler(
                model=self.unet,
                steps=steps,
                device=self.device,
                sampler=sampler_name,
                scheduler=scheduler,
                denoise=denoise
            )
            
            # Perform denoising
            denoising_start = time.time()
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Perform sampling
                denoised_latent = ksampler.sample(
                    noise=noise,
                    positive=positive_conditioning,
                    negative=negative_conditioning,
                    cfg=cfg,
                latent_image=initial_latent,
                    start_step=None,
                    last_step=None,
                    force_full_denoise=False,
                    denoise_mask=None,
                    sigmas=None,
                callback=None,
                    disable_pbar=False,
                    seed=seed
                )
            
            denoising_time = time.time() - denoising_start
            
            # Ensure device consistency
            if initial_latent.device != denoised_latent.device:
                denoised_latent = denoised_latent.to(initial_latent.device)
            
            # Mark step complete
            self.step_completed[4] = True
            
            # UNet cleanup for memory management
            if hasattr(self.unet, 'cleanup'):
                self.unet.cleanup()
            elif hasattr(self.unet, 'unload'):
                self.unet.unload()
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
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
                'timing': {
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
            
            # Perform VAE decoding using ComfyUI-style memory management and OOM handling
            decoded_images = None
            try:
                # Calculate memory usage for decode operation
                memory_used = vae_model.memory_used_decode(trimmed_latent.shape, vae_model.vae_dtype)
                print(f"   📊 Memory required for decode: {memory_used / (1024**3):.2f} GB")
                
                # Get available free memory
                if torch.cuda.is_available():
                    free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                    free_memory = free_memory / (1024**3)  # Convert to GB
                else:
                    free_memory = 8.0  # Assume 8GB for CPU
                
                # Calculate optimal batch size based on available memory
                batch_number = int(free_memory * 1024**3 / max(1, memory_used))
                batch_number = max(1, min(4, batch_number))  # Limit to reasonable batch size
                print(f"   📊 Available memory: {free_memory:.2f} GB")
                print(f"   📊 Batch size: {batch_number}")
                
                # Process in batches to avoid OOM
                with torch.no_grad():
                    for x in range(0, trimmed_latent.shape[0], batch_number):
                        batch_latent = trimmed_latent[x:x+batch_number].to(vae_model.vae_dtype).to(vae_model.device)
                        
                        # Use first_stage_model.decode() to access actual WanVAE (following ComfyUI pattern)
                        batch_output = vae_model.first_stage_model.decode(batch_latent).to(vae_model.output_device).float()
                        
                        if decoded_images is None:
                            decoded_images = torch.empty((trimmed_latent.shape[0],) + tuple(batch_output.shape[1:]), 
                                                       device=vae_model.output_device)
                        decoded_images[x:x+batch_number] = batch_output
                        
            except torch.cuda.OutOfMemoryError:
                print(f"   ⚠️ GPU OOM during regular decode, retrying with tiled decode...")
                
                # Fallback to tiled decoding (following ComfyUI OOM handling)
                dims = trimmed_latent.ndim - 2
                if dims == 3:  # 3D tensor (B, C, T, H, W)
                    tile = 256 // vae_model.spacial_compression_decode() if hasattr(vae_model, 'spacial_compression_decode') else 32
                    overlap = tile // 4
                    print(f"   🔧 Using 3D tiled decode with tile={tile}, overlap={overlap}")
                    decoded_images = vae_model.decode_tiled_3d(trimmed_latent, tile_x=tile, tile_y=tile, overlap=(1, overlap, overlap))
                elif dims == 2:  # 2D tensor
                    print(f"   🔧 Using 2D tiled decode")
                    decoded_images = vae_model.decode_tiled_(trimmed_latent)
                elif dims == 1:  # 1D tensor
                    print(f"   🔧 Using 1D tiled decode")
                    decoded_images = vae_model.decode_tiled_1d(trimmed_latent)
                else:
                    raise RuntimeError(f"Unsupported tensor dimensions: {trimmed_latent.ndim}")
            
            # Apply output processing (following ComfyUI pattern)
            decoded_images = vae_model.process_output(decoded_images)
            
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
            from video_export import MotionVideoExporter as VideoExporter
            
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
        """Load control video from path as float tensor (T, H, W, 3) in [0,1] - ComfyUI compatible"""
        if not video_path or not os.path.exists(video_path):
            print(f"Warning: Video file not found: {video_path}")
            return None
        
        try:
            from torchvision.io import read_video
            print(f"   📹 Loading video from: {video_path}")
            
            # Use ComfyUI-compatible PTS unit (exact match to ComfyUI workflow)
            video, audio, info = read_video(video_path, pts_unit='pts')
            if video is None or video.numel() == 0:
                print(f"Warning: Empty video: {video_path}")
                return None
            
            # Apply ComfyUI-compatible frame limiting (exact match to ComfyUI workflow)
            max_frames = min(37, video.shape[0])  # Limit to first 37 frames like ComfyUI
            video = video[:max_frames]
            
            # Ensure 3 channels (ComfyUI expects uint8 format from torchvision)
            if video.shape[-1] > 3:
                video = video[..., :3]
            elif video.shape[-1] == 1:
                video = video.repeat(1, 1, 1, 3)
            
            # CRITICAL FIX: Normalize uint8 to [0,1] range (exact match to ComfyUI VHS_LoadVideo)
            if video.dtype == torch.uint8:
                print(f"   🔧 Normalizing video from uint8 to float32 [0,1] range")
                video = video.float() / 255.0
                print(f"   ✅ Video normalized: range [{video.min().item():.6f}, {video.max().item():.6f}], dtype: {video.dtype}")
            elif video.dtype != torch.float32:
                print(f"   🔧 Converting video to float32")
                video = video.float()
            
            print(f"   📊 Loaded video tensor: {tuple(video.shape)} (T,H,W,C) - limited to {max_frames} frames")
            return video
            
        except Exception as e:
            print(f"Error loading video '{video_path}': {e}")
            return None

    def load_image(self, image_path: str) -> Optional[torch.Tensor]:
        """Load reference image from path as float tensor (1, H, W, 3) in [0,1] - ComfyUI compatible"""
        if not image_path or not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}")
            return None
        
        try:
            from PIL import Image, ImageOps
            import numpy as np
            print(f"   🖼️  Loading image from: {image_path}")
            
            # Load image using ComfyUI-compatible method (exact match to ComfyUI LoadImage)
            img = Image.open(image_path)
            
            # Apply EXIF handling like ComfyUI LoadImage (exact match)
            img = ImageOps.exif_transpose(img)
            
            # Handle special image modes like ComfyUI LoadImage (exact match)
            if img.mode == 'I':
                img = img.point(lambda i: i * (1 / 255))
            
            # Convert to RGB
            img = img.convert('RGB')
            
            # Convert to numpy array and normalize
            arr = np.array(img).astype(np.float32) / 255.0
            
            # Add batch dimension using ComfyUI-compatible method (exact match)
            tensor = torch.from_numpy(arr)[None,]  # Use [None,] like ComfyUI LoadImage
            
            print(f"   📊 Loaded image tensor: {tuple(tensor.shape)} (1,H,W,3) - ComfyUI compatible")
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
            # Pass VACE conditioning from Step 1 to Step 3
            step_3_params['vace_positive_conditioning'] = step_1_results['positive']
            step_3_params['vace_negative_conditioning'] = step_1_results['negative']
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
        # Pass VACE conditioning from Step 1 to Step 3
        step_3_params['vace_positive_conditioning'] = step_1_results['positive']
        step_3_params['vace_negative_conditioning'] = step_1_results['negative']
        step_3_results = self.step_3_model_sampling_and_text_encoding(**step_3_params)
        return step_1_results, step_2_results, step_3_results
    
    def run_steps_1_2_3_and_4(self, step_1_params: Dict[str, Any], step_2_params: Dict[str, Any], step_3_params: Dict[str, Any], step_4_params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Run Steps 1, 2, 3, and 4 in sequence"""
        print("🚀 Running Steps 1, 2, 3, and 4 in sequence...")
        step_1_results = self.step_1_vae_and_latent_creation(**step_1_params)
        step_2_results = self.step_2_unet_clip_lora_loading(**step_2_params)
        # Pass VACE conditioning from Step 1 to Step 3
        step_3_params['vace_positive_conditioning'] = step_1_results['positive']
        step_3_params['vace_negative_conditioning'] = step_1_results['negative']
        step_3_results = self.step_3_model_sampling_and_text_encoding(**step_3_params)
        # Prepare Step 4 parameters
        step_4_params['initial_latent'] = step_1_results['out_latent']['samples']
        step_4_params['positive_conditioning'] = step_3_results['positive_conditioning']
        step_4_params['negative_conditioning'] = step_3_results['negative_conditioning']
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
    
    def _log_gpu_state(self, stage_name):
        """Log comprehensive GPU state information"""
        if not torch.cuda.is_available():
            print(f"🖥️  GPU STATE [{stage_name}]: CUDA not available - using CPU")
            return
        
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        max_allocated = torch.cuda.max_memory_allocated(device)
        max_reserved = torch.cuda.max_memory_reserved(device)
        
        # Convert bytes to MB
        allocated_mb = allocated / (1024 * 1024)
        reserved_mb = reserved / (1024 * 1024)
        max_allocated_mb = max_allocated / (1024 * 1024)
        max_reserved_mb = max_reserved / (1024 * 1024)
        
        # Get GPU properties
        gpu_props = torch.cuda.get_device_properties(device)
        total_memory_mb = gpu_props.total_memory / (1024 * 1024)
        memory_usage_percent = (allocated_mb / total_memory_mb) * 100
        
        print(f"🖥️  GPU STATE [{stage_name}]:")
        print(f"   Device: {device} ({gpu_props.name})")
        print(f"   Memory Allocated: {allocated_mb:.2f} MB ({memory_usage_percent:.1f}%)")
        print(f"   Memory Reserved: {reserved_mb:.2f} MB")
        print(f"   Max Allocated: {max_allocated_mb:.2f} MB")
        print(f"   Max Reserved: {max_reserved_mb:.2f} MB")
        print(f"   Total Memory: {total_memory_mb:.2f} MB")
        print(f"   Free Memory: {total_memory_mb - allocated_mb:.2f} MB")
        print()
    
    def _verify_vae_device(self):
        """Verify VAE model and all its parameters are on the correct device"""
        if not torch.cuda.is_available():
            print(f"🔍 VAE DEVICE VERIFICATION: CUDA not available - VAE should be on CPU")
            return
        
        device = torch.cuda.current_device()
        vae_device = next(self.vae.first_stage_model.parameters()).device
        
        print(f"🔍 VAE DEVICE VERIFICATION:")
        print(f"   Expected device: cuda:{device}")
        print(f"   VAE device: {vae_device}")
        
        if vae_device.type == 'cuda' and vae_device.index == device:
            print(f"   ✅ VAE is correctly on GPU")
        elif vae_device.type == 'cpu':
            print(f"   ⚠️  VAE is on CPU - this may cause performance issues")
        else:
            print(f"   ❌ VAE device mismatch - expected cuda:{device}, got {vae_device}")
        
        # Check all VAE parameters
        all_on_gpu = True
        cpu_params = []
        for name, param in self.vae.first_stage_model.named_parameters():
            if param.device.type != 'cuda':
                all_on_gpu = False
                cpu_params.append(name)
        
        if all_on_gpu:
            print(f"   ✅ All VAE parameters are on GPU")
        else:
            print(f"   ❌ Some VAE parameters are on CPU: {cpu_params[:5]}...")
        
        print()
    

# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

def main(debug_mode=False):
    """Simple complete WAN Video Pipeline - All 7 Steps or Debug Mode (Steps 1-4)"""
    if debug_mode:
        print("🚀 WAN Video Pipeline - DEBUG MODE (Steps 1-4)")
    else:
        print("🚀 WAN Video Pipeline - Complete 7-Step Pipeline")
    print("="*60)
    
    # Initialize pipeline
    pipeline = WanVideoPipeline(models_dir="models")
    
    # Model file paths
    vae_model_path = "models/vaes/wan_vae.safetensors"
    unet_model_path = "models/diffusion_models/wan_2.1_diffusion_model.safetensors"
    clip_model_path = "models/text_encoders/wan_clip_model.safetensors"
    
    # Check if models exist
    if not os.path.exists(vae_model_path):
        print(f"❌ VAE model not found: {vae_model_path}")
        return
    if not os.path.exists(unet_model_path):
        print(f"❌ UNet model not found: {unet_model_path}")
        return
    if not os.path.exists(clip_model_path):
        print(f"❌ CLIP model not found: {clip_model_path}")
        return
    
    if debug_mode:
        print("✅ All models found - running debug mode (Steps 1-4)")
    else:
        print("✅ All models found - running complete pipeline")
    print("="*60)
            
    try:
        # Step 1: VAE Loading and Latent Creation
        print("🎬 STEP 1: VAE LOADING AND LATENT CREATION")
        step_1_results = pipeline.step_1_vae_and_latent_creation(
            vae_model_path=vae_model_path,
            positive_prompt="very cinematic video",
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量",
            control_video_path="safu.mp4" if os.path.exists("safu.mp4") else None,
            reference_image_path="safu.jpg" if os.path.exists("safu.jpg") else None,
            width=480, height=832, length=37, batch_size=1, strength=1.0
        )
            
        # Step 2: UNet + CLIP Loading
        print("\n🧠 STEP 2: UNET + CLIP LOADING")
        step_2_results = pipeline.step_2_unet_clip_lora_loading(
            unet_model_path=unet_model_path,
            clip_model_path=clip_model_path,
            lora_model_path=None,
            strength_model=1.0, strength_clip=0.0
        )
            
        # Step 3: Model Sampling + Text Encoding
        print("\n📝 STEP 3: MODEL SAMPLING + TEXT ENCODING")
        step_3_results = pipeline.step_3_model_sampling_and_text_encoding(
            positive_prompt="very cinematic video",
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量",
            vace_positive_conditioning=step_1_results['positive'],
            vace_negative_conditioning=step_1_results['negative'],
            shift=8.0, multiplier=1000
        )
            
        # Step 4: KSampler Denoising
        print("\n🎯 STEP 4: KSAMPLER DENOISING")
        step_4_results = pipeline.step_4_ksampler_denoising(
            initial_latent=step_1_results['out_latent']['samples'],
            positive_conditioning=step_3_results['positive_conditioning'],
            negative_conditioning=step_3_results['negative_conditioning'],
            seed=42, steps=4, cfg=7.0, sampler_name='euler',
            scheduler='normal', denoise=1.0, noise_inds=None
        )
        
        # Debug mode: Save Step 4 output and exit
        if debug_mode:
            print("\n🔍 DEBUG MODE: Saving Step 4 output...")
            denoised_latent = step_4_results['denoised_latent']
            
            # Print tensor shape
            print(f"📊 Step 4 Output Tensor Shape: {denoised_latent.shape}")
            print(f"📊 Step 4 Output Tensor Dtype: {denoised_latent.dtype}")
            print(f"📊 Step 4 Output Tensor Device: {denoised_latent.device}")
            
            # Save as .npy file
            output_path = "debug_step4_output.npy"
            import numpy as np
            
            # Convert to CPU and numpy if needed
            if denoised_latent.is_cuda:
                denoised_latent_cpu = denoised_latent.cpu()
            else:
                denoised_latent_cpu = denoised_latent
            
            # Convert to numpy
            denoised_latent_np = denoised_latent_cpu.numpy()
            
            # Save as .npy file
            np.save(output_path, denoised_latent_np)
            print(f"✅ Step 4 output saved to: {output_path}")
            print(f"📊 File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
            
            print(f"\n🎉 DEBUG MODE COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"✅ Steps 1-4 completed")
            print(f"✅ Step 4 output saved to: {output_path}")
            print(f"📊 Tensor shape: {denoised_latent.shape}")
            return
        
        # Step 5: Trim Video Latent
        print("\n🎬 STEP 5: TRIM VIDEO LATENT")
        step_5_results = pipeline.step_5_trim_latent(
            denoised_latent=step_4_results['denoised_latent'],
            trim_amount=0
        )
        
        # Step 6: VAE Decode
        print("\n🎨 STEP 6: VAE DECODE")
        step_6_results = pipeline.step_6_vae_decode(
            trimmed_latent=step_5_results['trimmed_latent'],
            vae_model=None
        )
        
        # Step 7: Video Export
        print("\n🎬 STEP 7: VIDEO EXPORT")
        step_7_results = pipeline.step_7_video_export(
            decoded_images=step_6_results['decoded_images'],
            output_path='output_video.mp4', fps=24
        )
        
        print(f"\n🎉 COMPLETE PIPELINE SUCCESS!")
        print("="*60)
        print(f"✅ Video exported to: {step_7_results['exported_path']}")
        print(f"📊 Video duration: {step_7_results.get('duration_seconds', 0):.2f} seconds")
        print(f"📊 File size: {step_7_results.get('file_size_mb', 0):.2f} MB")
        
    except Exception as e:
        print(f"\n❌ PIPELINE FAILED: {str(e)}")
        print(f"   Error Type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
    

if __name__ == "__main__":
    import sys
    
    # Check for debug mode argument
    if len(sys.argv) > 1 and sys.argv[1] == "--debug":
        print("🔍 Running in DEBUG MODE (Steps 1-4 only)")
        main(debug_mode=True)
    else:
        print("🚀 Running COMPLETE PIPELINE (All 7 steps)")
        main(debug_mode=False)
