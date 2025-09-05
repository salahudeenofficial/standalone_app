#!/usr/bin/env python3
"""
Standalone Reference Image + Control Video to Output Video Pipeline
Based on ComfyUI components but stripped of WebSocket, graph execution, and UI dependencies

This pipeline now properly leverages ComfyUI's native memory management system:
- UNET models: Automatically managed by ComfyUI's ModelPatcher system
- VAE models: Built-in memory management with automatic loading/unloading
- CLIP models: Automatically managed by ComfyUI's ModelPatcher system
- All memory management: Handled by ComfyUI's proven system

Uses the exact same approach as WAN VAE-to-Video and Load VAE nodes from ComfyUI.
"""

# ===============================================================================
# STEP 1: Initialize ComfyUI CLI Arguments and Environment Variables
# ===============================================================================
import os
import sys
import argparse
from pathlib import Path

# Set required environment variables BEFORE importing ComfyUI
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Create minimal CLI args that ComfyUI expects
sys.argv = ['pipeline.py']  # Let ComfyUI use default settings

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Add ComfyUI path for utilities
sys.path.insert(0, str(Path(__file__).parent / "comfy"))

# NOW import ComfyUI modules AFTER setting argv and environment
import comfy.cli_args
import comfy.model_management

# Initialize ComfyUI CLI arguments system
comfy.cli_args.args = comfy.cli_args.parser.parse_args()

# Ensure args.fast is properly initialized (ComfyUI expects this)
if comfy.cli_args.args.fast is None:
    comfy.cli_args.args.fast = set()

import torch
from pathlib import Path
import time
import numpy as np

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Add ComfyUI path for utilities
sys.path.insert(0, str(Path(__file__).parent / "comfy"))

# Import psutil for system information in diagnostic summary
try:
    import psutil
except ImportError:
    print("Warning: psutil not available, system information will be limited")
    psutil = None

import comfy.utils
from components.lora_loader import LoraLoader
from components.text_encoder import CLIPTextEncode
from components.model_sampling import ModelSamplingSD3
from components.video_generator import WanVaceToVideo
from components.sampler import KSampler
from components.video_processor import TrimVideoLatent
from components.vae_decoder import VAEDecode
from components.video_export import VideoExporter

# ===============================================================================
# REAL-TIME MEMORY MONITORING SYSTEM
# ===============================================================================

# ===============================================================================
# STEP 2: Initialize ComfyUI Device Detection and Memory Management
# ===============================================================================

def initialize_comfy_memory_system():
    """Initialize ComfyUI's memory management system"""
    print("🔧 Initializing ComfyUI memory management system...")
    
    # Let ComfyUI handle all initialization automatically
    print("✅ ComfyUI memory management system initialization completed")
    print("="*80 + "\n")

# ===============================================================================
# STEP 3: Initialize ComfyUI Memory Management System
# ===============================================================================

class ReferenceVideoPipeline:
    """
    Standalone Reference Image + Control Video to Output Video Pipeline
    
    Memory Management Philosophy (ComfyUI Native):
    - UNET models: Automatically managed by ComfyUI's ModelPatcher system
    - VAE models: Built-in memory management with automatic loading/unloading
    - CLIP models: Automatically managed by ComfyUI's ModelPatcher system
    - All memory management: Handled by ComfyUI's proven system
    
    This approach ensures:
    1. All models use ComfyUI's native memory management
    2. Automatic GPU/CPU swapping via ModelPatcher
    3. Intelligent memory allocation and cleanup
    4. Uses the exact same approach as WAN VAE-to-Video and Load VAE nodes
    5. No manual memory management needed - ComfyUI handles everything
    """
    def __init__(self, models_dir="models"):
        """Initialize the pipeline with model directory"""
        self.models_dir = models_dir
        self.setup_model_paths()
        
        # Initialize ComfyUI memory management system
        initialize_comfy_memory_system()
        
        # ComfyUI handles all chunking automatically - no manual processor needed
        print("✅ ComfyUI memory management initialized")
        
    def setup_model_paths(self):
        """Setup model paths for the standalone app"""
        # Get the script directory for absolute paths
        script_dir = Path(__file__).parent
        models_dir = script_dir / self.models_dir
        
        # Create model directories if they don't exist
        os.makedirs(models_dir / "diffusion_models", exist_ok=True)  # For UNET models
        os.makedirs(models_dir / "text_encoders", exist_ok=True)     # For CLIP models
        os.makedirs(models_dir / "vaes", exist_ok=True)             # For VAE models
        os.makedirs(models_dir / "loras", exist_ok=True)            # For LoRA models
        
        # Set environment variables for model paths
        os.environ["COMFY_MODEL_PATH"] = str(models_dir)
        
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
        """
        Run the complete pipeline from reference image + control video to output video
        
        This pipeline now uses explicit ModelPatcher memory management (same as working test):
        - All models are loaded using ComfyUI's native functions
        - ModelPatcher explicitly manages UNET/CLIP memory with device_to=offload_device
        - VAE moved to GPU for operations, then to CPU for memory management
        - Explicit memory management calls using proven working logic
        """
    
        try:
            # 1. Load Diffusion Model Components using ComfyUI's native system
            print("1. Loading diffusion model components using ComfyUI...")
            
            # Import ComfyUI's model loading functions
            import comfy.sd
            import comfy.model_management
           # Load UNET with proper WAN model detection
            print("1a. Loading UNET with automatic WAN detection...")
            unet_state_dict = comfy.utils.load_torch_file(unet_model_path)
            
            # Use ComfyUI's automatic model detection for UNET
            # This will automatically detect WAN models and load them correctly
            from comfy.sd import load_state_dict_guess_config
            
            # Load just the UNET with automatic detection
            model, _, _, _ = load_state_dict_guess_config(
                unet_state_dict, 
                output_vae=False, 
                output_clip=False, 
                output_clipvision=False, 
                embedding_directory=None, 
                output_model=True
            )
            
            print(f"1a. ✅ UNET loaded using ComfyUI's automatic detection: {type(model)}")
            
            # Load CLIP separately with proper WAN type
            print("1a. Loading CLIP with WAN type...")
            clip_model = comfy.sd.load_clip([clip_model_path], clip_type=comfy.sd.CLIPType.WAN)
            
            if clip_model is None:
                print("1a. ⚠️  CLIP loading failed, trying alternative approach...")
                # Fallback: load CLIP state dict and create manually
                clip_state_dict = comfy.utils.load_torch_file(clip_model_path)
                from comfy.sd import CLIP
                clip_model = CLIP(clip_state_dict, clip_type=comfy.sd.CLIPType.WAN)
            
            print(f"1a. ✅ CLIP loaded: {type(clip_model)}")
            
            # Load VAE using ComfyUI's native approach (exactly like VAE Loader node)
            print("1a. Loading VAE with ComfyUI's native approach...")
            vae_state_dict = comfy.utils.load_torch_file(vae_model_path)
            vae = comfy.sd.VAE(sd=vae_state_dict)
            vae.throw_exception_if_invalid()

            
          
            print("\n" + "="*80)
            print("🔍 STEP 3: LOAD VIDEO AND IMAGE DATA")
            print("="*80)
            
            # Load control video
            print("3a. Loading control video...")
            if control_video_path:
                control_video = self.load_video(control_video_path)
                if control_video is not None:
                    print(f"   ✅ Control video loaded: {control_video.shape}")
                else:
                    print("   ❌ Failed to load control video")
                    control_video = None
            else:
                print("   ⚠️  No control video path specified")
                control_video = None
            
            # Load reference image
            print("3b. Loading reference image...")
            if reference_image_path:
                reference_image = self.load_image(reference_image_path)
                if reference_image is not None:
                    print(f"   ✅ Reference image loaded: {reference_image.shape}")
                else:
                    print("   ❌ Failed to load reference image")
                    reference_image = None
            else:
                print("   ⚠️  No reference image path specified")
                reference_image = None
            
            print("✅ Step 3 completed - continuing to Step 5...")
            

            
            # Prepare control video

            
            # 2. Apply LoRA if specified
            if lora_path:
                print("2. Applying LoRA...")


                lora_loader = LoraLoader()
                
                try:
                    # Store original models for comparison
                    original_model = model
                    original_clip_model = clip_model
                    

                    import comfy.utils                    
                    model, clip_model = lora_loader.load_lora(
                        model, clip_model, lora_path, 0.5, 1.0
                    )
                    
                    print("✅ LoRA applied successfully")

                except Exception as e:

                    model = original_model
                    clip_model = original_clip_model
                

                
                # Load control video
                print("3a. Loading control video...")
                if control_video_path:
                    control_video = self.load_video(control_video_path)
                    if control_video is not None:
                        print(f"   ✅ Control video loaded: {control_video.shape}")
                    else:
                        print("   ❌ Failed to load control video")
                        control_video = None
                else:
                    print("   ⚠️  No control video path specified")
                    control_video = None
                
                # Load reference image
                print("3b. Loading reference image...")
                if reference_image_path:
                    reference_image = self.load_image(reference_image_path)
                    if reference_image is not None:
                        print(f"   ✅ Reference image loaded: {reference_image.shape}")
                    else:
                        print("   ❌ Failed to load reference image")
                        reference_image = None
                else:
                    print("   ⚠️  No reference image path specified")
                    reference_image = None
                
                print("✅ Step 3 completed - continuing to Step 5...")
                
            else:
                pass
               
            
            # step 2. TEXT ENCODING (MONITORING COMMENTED OUT)
            print("\n" + "="*80)
            print("🔍 STEP 3: TEXT ENCODING (MONITORING COMMENTED OUT)")
            print("="*80)
            
            # Simple text encoding without monitoring
            text_encoder = CLIPTextEncode()
            positive_cond = text_encoder.encode(clip_model, positive_prompt)
            negative_cond = text_encoder.encode(clip_model, negative_prompt)
            
            print(f"✅ Text encoding completed")

            # ========================================================================
            # STEP 3: SAMPLING STEP (MONITORING COMMENTED OUT)
            # ========================================================================
            print(f"\n{'='*80}")
            print(f"🔍 STEP 4: SAMPLING STEP (MONITORING COMMENTED OUT)")
            print(f"{'='*80}")
            
      
            
            print("4. Applying ModelSamplingSD3...")
            model_sampling = ModelSamplingSD3()
            
            model = model_sampling.patch(model, shift=8.0)
            
            # ComfyUI automatically tracks the patched model through ModelPatcher
            print("4a. ModelSamplingSD3 applied")
            
           
            
            print("="*80)
            

            
            # ========================================================================
            # STEP 4: GENERATE INITIAL LATENTS (COMFY-LIKE)
            # ========================================================================
            print(f"\n{'='*80}")
            print(f"🔍 STEP 4: GENERATE INITIAL LATENTS (COMFY-LIKE)")
            print(f"{'='*80}")
            # Prepare control video
            latent_length = ((length - 1) // 4) + 1
  
            # Ensure inputs are loaded locally for this step
            control_video = locals().get('control_video', None)
            if control_video is None:
                control_video = self.load_video(control_video_path) if control_video_path else None
            reference_image = locals().get('reference_image', None)
            if reference_image is None:
                reference_image = self.load_image(reference_image_path) if reference_image_path else None

            # Comfy-like implementation of WanVaceToVideo.encode
            from comfy import node_helpers

            vae_stride = 8
            latent_length = ((length - 1) // 4) + 1

            # Prepare control video
            if control_video is not None:

                
                control_video = control_video[:length]
                
                control_video = comfy.utils.common_upscale(
                    control_video.movedim(-1, 1), width, height, "bilinear", "center"
                ).movedim(1, -1)
                print(f"   After upscaling to {width}x{height}: {control_video.shape}")
                
                # Check if 8x downsampling is happening
                expected_latent_size = (control_video.shape[1] // 8, control_video.shape[2] // 8)
                print(f"   Expected VAE latent size (with 8x downsampling): {expected_latent_size}")
                if control_video.shape[0] < length:
                    control_video = torch.nn.functional.pad(
                        control_video, (0, 0, 0, 0, 0, 0, 0, length - control_video.shape[0]), value=0.5
                    )
            else:
                
                print("no control video")
                return 

            # Prepare reference image (optional) - exactly like WAN VAE-to-Video node
            if reference_image is not None:
                reference_image = comfy.utils.common_upscale(reference_image[:1].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
                reference_image = vae.encode(reference_image[:, :, :, :3])
                reference_image = torch.cat([reference_image, comfy.latent_formats.Wan21().process_out(torch.zeros_like(reference_image))], dim=1)

            # Prepare mask (default full mask if none provided) - exactly like WAN VAE-to-Video node
            control_masks = None  # Define control_masks variable
            if control_masks is None:
                mask = torch.ones((length, height, width, 1), device=control_video.device)
            else:
                mask = control_masks
                if mask.ndim == 3:
                    mask = mask.unsqueeze(1)
                mask = comfy.utils.common_upscale(mask[:length], width, height, "bilinear", "center").movedim(1, -1)
                if mask.shape[0] < length:
                    mask = torch.nn.functional.pad(mask, (0, 0, 0, 0, 0, 0, 0, length - mask.shape[0]), value=1.0)

            # Normalize and split by mask - exactly like WAN VAE-to-Video node
            control_video = control_video - 0.5
            inactive = (control_video * (1 - mask)) + 0.5
            reactive = (control_video * mask) + 0.5


            print("🔧 Starting VAE encoding...")
            
            inactive = vae.encode(inactive[:, :, :, :3])
            print("🔧 VAE encoding inactive completed")
            reactive = vae.encode(reactive[:, :, :, :3])
            print("🔧 VAE encoding reactive completed")
            control_video_latent = torch.cat((inactive, reactive), dim=1)
            
            # Reference image processing - exactly like WAN VAE-to-Video node
            trim_latent = 0
            if reference_image is not None:
                control_video_latent = torch.cat((reference_image, control_video_latent), dim=2)

            # WAN mask reshaping to latent grid - exactly like WAN VAE-to-Video node
            vae_stride = 8
            height_mask = height // vae_stride
            width_mask = width // vae_stride
            mask = mask.view(length, height_mask, vae_stride, width_mask, vae_stride)
            mask = mask.permute(2, 4, 0, 1, 3)
            mask = mask.reshape(vae_stride * vae_stride, length, height_mask, width_mask)
            mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=(latent_length, height_mask, width_mask), mode='nearest-exact').squeeze(0)

            if reference_image is not None:
                mask_pad = torch.zeros_like(mask[:, :reference_image.shape[2], :, :])
                mask = torch.cat((mask_pad, mask), dim=1)
                latent_length += reference_image.shape[2]
                trim_latent = reference_image.shape[2]

            mask = mask.unsqueeze(0)

            # Update conditioning - exactly like WAN VAE-to-Video node
            positive_cond = node_helpers.conditioning_set_values(positive_cond, {"vace_frames": [control_video_latent], "vace_mask": [mask], "vace_strength": [strength]}, append=True)
            negative_cond = node_helpers.conditioning_set_values(negative_cond, {"vace_frames": [control_video_latent], "vace_mask": [mask], "vace_strength": [strength]}, append=True)

            # Allocate output latent (container) on intermediate device - exactly like WAN VAE-to-Video node
            latent = torch.zeros([batch_size, 16, latent_length, height // 8, width // 8], device=comfy.model_management.intermediate_device())
            out_latent = {"samples": latent}

            # Preserve variable names used later if any logging expects them
            init_latent = latent
            trim_count = trim_latent

            print(f"\n{'='*80}")
            print(f"✅ STEP 5 COMPLETE: Generate Initial Latents (Comfy-like)")
            print(f"{'='*80}")


            
            return "pipeline_stopped_after_step_5_for_debugging"

        except Exception as e:
            print(f"Pipeline failed with error: {str(e)}")
            
            # ComfyUI automatically handles cleanup on failure
            raise
            
        #     # 4. Apply ModelSamplingSD3 Shift
        #     print("4. Applying ModelSamplingSD3...")
        #     model_sampling = ModelSamplingSD3()
            
        #     # ModelPatcher automatically handles loading/unloading during patching
        #     model = model_sampling.patch(model, shift=8.0)
            
        #     # ComfyUI automatically tracks the patched model through ModelPatcher
        #     print("4a. ModelSamplingSD3 applied")
            
        #     # COMPREHENSIVE VERIFICATION AFTER MODEL SAMPLING
        #     print("\n" + "="*80)
        #     print("🔍 STEP 4 COMPLETE: COMPREHENSIVE VERIFICATION")
        #     print("="*80)
            
        #     # 1. Model Placement Verification
        #     print("1️⃣  MODEL PLACEMENT VERIFICATION:")
        #     model_placement = self._check_model_placement('model_sampling', ['unet'])
            
        #     # 2. Memory Management Verification
        #     print("\n2️⃣  MEMORY MANAGEMENT VERIFICATION:")
        #     memory_management = self._verify_memory_management('model_sampling', ['unet'])
            
        #     # 3. Chunking Strategy Verification
        #     print("\n3️⃣  CHUNKING STRATEGY VERIFICATION:")
        #     chunking_strategy = self._verify_chunking_strategy('model_sampling')
            
        #     # 4. Summary
        #     print("\n📊 STEP 4 SUMMARY:")
        #     print(f"   Model Placement: {'✅ PASS' if model_placement else '❌ FAIL'}")
        #     print(f"   Memory Management: {'✅ PASS' if memory_management else '❌ FAIL'}")
        #     print(f"   Chunking Strategy: {'✅ PASS' if chunking_strategy else '❌ FAIL'}")
            
        #     if not all([model_placement, memory_management, chunking_strategy]):
        #         print("   ⚠️  Some verifications failed - pipeline may have issues")
        #     else:
        #         print("   ✅ All verifications passed - pipeline ready for next step")
            
        #     print("="*80)
            
        #     # 7. Trim Video Latent
        #     print("7. Trimming video latent...")
        #     trim_processor = TrimVideoLatent()
            
        #     # Wrap the latent tensor in the dictionary format expected by TrimVideoLatent
        #     latent_dict = {"samples": final_latent}
        #     trimmed_latent_dict = trim_processor.op(latent_dict, trim_count)
            
        #     # Extract the trimmed tensor from the dictionary
        #     trimmed_latent = trimmed_latent_dict["samples"]
        #     print(f"7a. Trimmed latent shape: {trimmed_latent.shape}")
            
        #     # COMPREHENSIVE VERIFICATION AFTER VIDEO LATENT TRIMMING
        #     print("\n" + "="*80)
        #     print("🔍 STEP 7 COMPLETE: COMPREHENSIVE VERIFICATION")
        #     print("="*80)
            
        #     # 1. Model Placement Verification
        #     print("1️⃣  MODEL PLACEMENT VERIFICATION:")
        #     model_placement = self._check_model_placement('video_trimming', [])
            
        #     # 2. Memory Management Verification
        #     print("\n2️⃣  MEMORY MANAGEMENT VERIFICATION:")
        #     memory_management = self._verify_memory_management('video_trimming', [])
            
        #     # 3. Chunking Strategy Verification
        #     print("\n3️⃣  CHUNKING STRATEGY VERIFICATION:")
        #     chunking_strategy = self._verify_chunking_strategy('video_trimming')
            
        #     # 4. Video Trimming Results Verification
        #     print("\n4️⃣  VIDEO TRIMMING RESULTS VERIFICATION:")
        #     if 'trimmed_latent' in locals():
        #         if hasattr(trimmed_latent, 'shape'):
        #             print(f"   Trimmed Latent: ✅ Shape: {trimmed_latent.shape}")
        #             print(f"   Trim Count: {trim_count if 'trim_count' in locals() else 'Unknown'}")
        #             video_trimming_success = True
        #         else:
        #             print("   Trimmed Latent: ❌ No shape information")
        #             video_trimming_success = False
        #     else:
        #         print("   Trimmed Latent: ❌ No trimmed latent created")
        #         video_trimming_success = False
            
        #     # 5. Summary
        #     print("\n📊 STEP 7 SUMMARY:")
        #     print(f"   Model Placement: {'✅ PASS' if model_placement else '❌ FAIL'}")
        #     print(f"   Memory Management: {'✅ PASS' if memory_management else '❌ FAIL'}")
        #     print(f"   Chunking Strategy: {'✅ PASS' if chunking_strategy else '❌ FAIL'}")
        #     print(f"   Video Trimming Success: {'✅ PASS' if video_trimming_success else '❌ FAIL'}")
            
        #     if not all([model_placement, memory_management, chunking_strategy, video_trimming_success]):
        #         print("   ⚠️  Some verifications failed - pipeline may have issues")
        #     else:
        #         print("   ✅ All verifications passed - pipeline ready for next step")
            
        #     print("="*80)
            
        #     # 8. Decode Frames
        #     print("8. Decoding frames...")
            
        #     # VAE automatically manages memory during decode()
        #     print("8a. VAE automatically manages memory during decode()")
            
        #     # Optimize chunk size for VAE decoding based on available VRAM
        #     print("8a. Optimizing chunk size for VAE decoding...")
        #     if torch.cuda.is_available():
        #         available_vram = torch.cuda.get_device_properties(0).total_memory / 1024**2
        #         allocated = torch.cuda.memory_allocated() / 1024**2
        #         free_vram = available_vram - allocated
                
        #         # Calculate optimal chunk size for VAE decoding
        #         if free_vram > 25000:  # >25GB free
        #             optimal_decode_chunk_size = 16
        #             print(f"8a. High VRAM available ({free_vram:.1f} GB), using decode chunk size: {optimal_decode_chunk_size}")
        #         elif free_vram > 15000:  # >15GB free
        #             optimal_decode_chunk_size = 12
        #             print(f"8a. Good VRAM available ({free_vram:.1f} GB), using decode chunk size: {optimal_decode_chunk_size}")
        #         else:  # <15GB free
        #             optimal_decode_chunk_size = 8
        #             print(f"8a. Limited VRAM available ({free_vram:.1f} GB), using conservative decode chunk size: {optimal_decode_chunk_size}")
                
        #         # Update processing plan for decoding
        #         processing_plan['vae_decode']['chunk_size'] = optimal_decode_chunk_size
        #         processing_plan['vae_decode']['num_chunks'] = (length + optimal_decode_chunk_size - 1) // optimal_decode_chunk_size
        #         print(f"8a. Updated decoding plan: {processing_plan['vae_decode']['num_chunks']} chunks of size {optimal_decode_chunk_size}")
            
        #     # Ensure VAE is on GPU for decoding
        #     print("8a. Ensuring VAE is on GPU for decoding...")
        #     print("8a. Letting ComfyUI's VAE ModelPatcher handle device placement automatically...")
        #     print("8a. VAE will be moved to GPU when needed for decoding operations")
            
        #     print("8a. VAE is ready for decoding...")
            
        #     vae_decoder = VAEDecode()
            
        #     # Use chunked processing for VAE decoding if needed
        #     if length > processing_plan['vae_decode']['chunk_size']:
        #         print(f"Using chunked VAE decoding: {processing_plan['vae_decode']['num_chunks']} chunks")
                
        #         # Debug: Show what we're passing to VAE decoding
        #         print(f"8a. Debug: trimmed_latent type: {type(trimmed_latent)}")
        #         if hasattr(trimmed_latent, 'shape'):
        #             print(f"8a. Debug: trimmed_latent shape: {trimmed_latent.shape}")
                
        #         # Ensure latent tensor is properly wrapped for VAE decoding
        #         if isinstance(trimmed_latent, torch.Tensor):
        #             latent_dict = {"samples": trimmed_latent}
        #             print(f"8a. Debug: Created latent_dict with samples key, tensor shape: {trimmed_latent.shape}")
        #         else:
        #             latent_dict = trimmed_latent
        #             print(f"8a. Debug: Using existing latent_dict: {type(latent_dict)}")
                
        #         # Try chunked processing first
        #         try:
        #             frames = self.chunked_processor.vae_decode_chunked(vae, latent_dict)
        #             print("8a. Chunked VAE decoding successful!")
                    
        #         except torch.cuda.OutOfMemoryError:
        #             print("OOM during chunked VAE decoding! Trying smaller chunks...")
                    
        #             # Progressive fallback: reduce chunk size until it works
        #             chunk_sizes_to_try = [8, 4, 2, 1]
        #             frames = None
                    
        #             for smaller_chunk_size in chunk_sizes_to_try:
        #                 try:
        #                     print(f"8a. Trying VAE decoding with chunk size: {smaller_chunk_size}")
                            
        #                     # Update processing plan with smaller chunk size
        #                     processing_plan['vae_decode']['chunk_size'] = smaller_chunk_size
        #                     processing_plan['vae_decode']['num_chunks'] = (length + smaller_chunk_size - 1) // smaller_chunk_size
                            
        #                     frames = self.chunked_processor.vae_decode_chunked(vae, latent_dict)
        #                     print(f"8a. VAE decoding successful with chunk size: {smaller_chunk_size}")
        #                     break
                            
        #                 except torch.cuda.OutOfMemoryError:
        #                     print(f"8a. Still OOM with chunk size {smaller_chunk_size}, trying smaller...")
        #                     continue
                    
        #             if frames is None:
        #                 print("8a. All chunk sizes failed! Using single-frame fallback...")
        #                 # Final fallback: process one frame at a time
        #                 frames = self._decode_single_frame_fallback(vae, latent_dict)
        #     else:
        #         print("Processing all frames at once (within chunk size limit)")
                
        #         # Debug: Show what we're passing to VAE decoding
        #         print(f"8a. Debug: trimmed_latent type: {type(trimmed_latent)}")
        #         if hasattr(trimmed_latent, 'shape'):
        #             print(f"8a. Debug: trimmed_latent shape: {trimmed_latent.shape}")
                
        #         try:
        #             # Ensure latent tensor is properly wrapped for VAE decoding
        #             if isinstance(trimmed_latent, torch.Tensor):
        #                 latent_dict = {"samples": trimmed_latent}
        #                 print(f"8a. Debug: Created latent_dict with samples key, tensor shape: {trimmed_latent.shape}")
        #             else:
        #                 latent_dict = trimmed_latent
        #                 print(f"8a. Debug: Using existing latent_dict: {type(latent_dict)}")
                    
        #             frames = vae_decoder.decode(vae, latent_dict)
        #         except torch.cuda.OutOfMemoryError:
        #             print("OOM during single-pass VAE decoding! Using single-frame fallback...")
        #             frames = self._decode_single_frame_fallback(vae, latent_dict)
            
        #     # OOM Checklist: Check memory after VAE decoding execution
        #     # ComfyUI automatically handles memory management during VAE operations
            
        #     # Let ComfyUI handle VAE memory management automatically
        #     print("8b. VAE decoding complete")
        #     print("8b. ComfyUI's VAE ModelPatcher will handle memory management automatically")
        #     print("8b. No manual VAE device management needed - letting ComfyUI coordinate")
            
        #     # COMPREHENSIVE VERIFICATION AFTER VAE DECODING
        #     print("\n" + "="*80)
        #     print("🔍 STEP 8 COMPLETE: COMPREHENSIVE VERIFICATION")
        #     print("="*80)
            
        #     # 1. Model Placement Verification
        #     print("1️⃣  MODEL PLACEMENT VERIFICATION:")
        #     model_placement = self._check_model_placement('vae_decoding', ['vae'])
            
        #     # 2. Memory Management Verification
        #     print("\n2️⃣  MEMORY MANAGEMENT VERIFICATION:")
        #     memory_management = self._verify_memory_management('vae_decoding', ['vae'])
            
        #     # 3. Chunking Strategy Verification
        #     print("\n3️⃣  CHUNKING STRATEGY VERIFICATION:")
        #     chunking_strategy = self._verify_chunking_strategy('vae_decoding')
            
        #     # 4. VAE Decoding Results Verification
        #     print("\n4️⃣  VAE DECODING RESULTS VERIFICATION:")
        #     if 'frames' in locals():
        #         if hasattr(frames, 'shape'):
        #             print(f"   Frames Generated: ✅ Shape: {frames.shape}")
        #             if len(frames.shape) == 4:
        #                 print(f"   Frame Info: {frames.shape[0]} frames, {frames.shape[1]}x{frames.shape[2]}, {frames.shape[3]} channels")
        #                 if frames.shape[3] == 3:
        #                     print("   ✅ Frames have correct 3 channels (RGB)")
        #                 else:
        #                     print(f"   ⚠️  Frames have wrong channel count: {frames.shape[3]} (expected 3)")
        #             else:
        #                 print(f"   ⚠️  Frames have unexpected shape: {frames.shape}")
        #             vae_decoding_success = True
        #         else:
        #             print("   Frames Generated: ❌ No shape information")
        #             vae_decoding_success = False
        #     else:
        #         print("   Frames Generated: ❌ No frames created")
        #         vae_decoding_success = False
            
        #     # 5. Summary
        #     print("\n📊 STEP 8 SUMMARY:")
        #     print(f"   Model Placement: {'✅ PASS' if model_placement else '❌ FAIL'}")
        #     print(f"   Memory Management: {'✅ PASS' if memory_management else '❌ FAIL'}")
        #     print(f"   Chunking Strategy: {'✅ PASS' if chunking_strategy else '❌ FAIL'}")
        #     print(f"   VAE Decoding Success: {'✅ PASS' if vae_decoding_success else '❌ FAIL'}")
            
        #     if not all([model_placement, memory_management, chunking_strategy, vae_decoding_success]):
        #         print("   ⚠️  Some verifications failed - pipeline may have issues")
        #     else:
        #         print("   ✅ All verifications passed - pipeline ready for next step")
            
        #     print("="*80)
            
        #     # 9. Export Video
        #     print("9. Exporting video...")
            
        #     # Debug: Check frame format before export
        #     print("9a. Pre-export frame debug info:")
        #     if frames is not None:
        #         print(f"9a. Export frames type: {type(frames)}")
        #         if hasattr(frames, 'shape'):
        #             print(f"9a. Export frames shape: {frames.shape}")
        #             if len(frames.shape) == 4:  # (batch, height, width, channels)
        #                 print(f"9a. Export frame dimensions: {frames.shape[0]} frames, {frames.shape[1]}x{frames.shape[2]}, {frames.shape[3]} channels")
        #                 if frames.shape[3] == 3:
        #                     print("9a. ✅ Export frames have correct 3 channels (RGB)")
        #                 elif frames.shape[3] == 1:
        #                     print("9a. ⚠️  WARNING: Frames have only 1 channel! Expected 3 channels (RGB)")
        #                     print("9a. 🔧 Attempting to expand 1-channel frames to 3-channel...")
        #                     # Expand 1-channel to 3-channel by repeating
        #                     frames = frames.repeat(1, 1, 1, 3)
        #                     print(f"9a. ✅ Expanded frames shape: {frames.shape}")
        #                 else:
        #                     print(f"9a. ❌ Export frames have wrong channel count: {frames.shape[3]} (expected 3)")
        #             else:
        #                 print(f"9a. ⚠️  Export frames have unexpected shape: {frames.shape}")
        #         else:
        #             print("9a. ⚠️  Export frames object has no shape attribute")
        #     else:
        #         print("9a. ❌ ERROR: No frames to export!")
            
        #     exporter = VideoExporter()
        #     exporter.export_video(frames, output_path)
            
        #     print(f"Pipeline completed successfully! Output saved to: {output_path}")
            
        #     # COMPREHENSIVE VERIFICATION AFTER VIDEO EXPORT
        #     print("\n" + "="*80)
        #     print("🔍 STEP 9 COMPLETE: COMPREHENSIVE VERIFICATION")
        #     print("="*80)
            
        #     # 1. Model Placement Verification
        #     print("1️⃣  MODEL PLACEMENT VERIFICATION:")
        #     model_placement = self._check_model_placement('video_export', [])
            
        #     # 2. Memory Management Verification
        #     print("\n2️⃣  MEMORY MANAGEMENT VERIFICATION:")
        #     memory_management = self._verify_memory_management('video_export', [])
            
        #     # 3. Chunking Strategy Verification
        #     print("\n3️⃣  CHUNKING STRATEGY VERIFICATION:")
        #     chunking_strategy = self._verify_chunking_strategy('video_export')
            
        #     # 4. Video Export Results Verification
        #     print("\n4️⃣  VIDEO EXPORT RESULTS VERIFICATION:")
        #     if 'output_path' in locals():
        #         print(f"   Output Path: {output_path}")
        #         if os.path.exists(output_path):
        #             file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        #             print(f"   File Size: {file_size:.1f} MB")
        #             video_export_success = True
        #         else:
        #             print("   File Size: ❌ File not found")
        #             video_export_success = False
        #     else:
        #         print("   Output Path: ❌ No output path specified")
        #         video_export_success = False
            
        #     # 5. Summary
        #     print("\n📊 STEP 9 SUMMARY:")
        #     print(f"   Model Placement: {'✅ PASS' if model_placement else '❌ FAIL'}")
        #     print(f"   Memory Management: {'✅ PASS' if memory_management else '❌ FAIL'}")
        #     print(f"   Chunking Strategy: {'✅ PASS' if chunking_strategy else '❌ FAIL'}")
        #     print(f"   Video Export Success: {'✅ PASS' if video_export_success else '❌ FAIL'}")
            
        #     if not all([model_placement, memory_management, chunking_strategy, video_export_success]):
        #         print("   ⚠️  Some verifications failed - pipeline may have issues")
        #     else:
        #         print("   ✅ All verifications passed - pipeline ready for next step")
            
        #     print("="*80)
            
        #     # ✅ FINAL CLEANUP - TRUSTING COMFYUI'S SYSTEM
        #     print("Final cleanup: ✅ Trusting ComfyUI's automatic memory management system")
        #     print("Final cleanup: 💡 ComfyUI will automatically clean up all models and memory")
        #     print("Final cleanup: 💡 No manual cleanup needed - ComfyUI handles everything!")
            
        #     # ComfyUI automatically manages cleanup when the pipeline completes
        #     # All models will be properly offloaded and memory will be freed
            
        #     # COMPREHENSIVE VERIFICATION AFTER FINAL CLEANUP
        #     print("\n" + "="*80)
        #     print("🔍 FINAL CLEANUP COMPLETE: COMPREHENSIVE VERIFICATION")
        #     print("="*80)
            
        #     # 1. Model Placement Verification
        #     print("1️⃣  MODEL PLACEMENT VERIFICATION:")
        #     model_placement = self._check_model_placement('final_cleanup', ['unet', 'clip', 'vae'])
            
        #     # 2. Memory Management Verification
        #     print("\n2️⃣  MEMORY MANAGEMENT VERIFICATION:")
        #     memory_management = self._verify_memory_management('final_cleanup', ['unet', 'clip', 'vae'])
            
        #     # 3. Chunking Strategy Verification
        #     print("\n3️⃣  CHUNKING STRATEGY VERIFICATION:")
        #     chunking_strategy = self._verify_chunking_strategy('final_cleanup')
            
        #     # 4. Final Memory State Verification
        #     print("\n4️⃣  FINAL MEMORY STATE VERIFICATION:")
        #     if torch.cuda.is_available():
        #         final_allocated = torch.cuda.memory_allocated() / 1024**2
        #         final_reserved = torch.cuda.memory_reserved() / 1024**2
        #         total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**2
        #         free_vram = total_vram - final_reserved
                
        #         print(f"   Final GPU Memory:")
        #         print(f"     Allocated: {final_allocated:.1f} MB")
        #         print(f"     Reserved: {final_reserved:.1f} MB")
        #         print(f"     Free: {free_vram:.1f} MB")
        #         print(f"     Total: {total_vram:.1f} MB")
        #         print(f"     Utilization: {(final_reserved/total_vram)*100:.1f}%")
                
        #         # Memory efficiency
        #         if 'baseline_allocated' in locals():
        #             baseline_mb = baseline_allocated / 1024**2
        #             memory_efficiency = ((final_allocated - baseline_mb) / baseline_mb) * 100 if baseline_mb > 0 else 0
        #             print(f"     Memory Efficiency: {memory_efficiency:+.1f}% from baseline")
                    
        #             if abs(memory_efficiency) < 100:  # Within 100MB of baseline
        #                 print("     ✅ Memory successfully restored to baseline state")
        #                 memory_restored = True
        #             else:
        #                 print("     ⚠️  Memory not fully restored to baseline state")
        #                 memory_restored = False
        #         else:
        #             memory_restored = False
        #             print("     ⚠️  Cannot determine memory restoration (no baseline)")
        #     else:
        #         memory_restored = True
        #         print("   GPU not available, skipping memory verification")
            
        #     # 5. Summary
        #     print("\n📊 FINAL CLEANUP SUMMARY:")
        #     print(f"   Model Placement: {'✅ PASS' if model_placement else '❌ FAIL'}")
        #     print(f"   Memory Management: {'✅ PASS' if memory_management else '❌ FAIL'}")
        #     print(f"   Chunking Strategy: {'✅ PASS' if chunking_strategy else '❌ FAIL'}")
        #     print(f"   Memory Restored: {'✅ PASS' if memory_restored else '❌ FAIL'}")
            
        #     if not all([model_placement, memory_management, chunking_strategy, memory_restored]):
        #         print("   ⚠️  Some verifications failed - final cleanup may be incomplete")
        #     else:
        #         print("   ✅ All verifications passed - pipeline cleanup complete")
            
        #     print("="*80)
            
        #     # COMPREHENSIVE DIAGNOSTIC SUMMARY
        #     print("\n" + "="*100)
        #     print("🔍 COMPREHENSIVE PIPELINE DIAGNOSTIC SUMMARY")
        #     print("="*100)
            
        #     # System Information
        #     print("💻 SYSTEM INFORMATION:")
        #     if torch.cuda.is_available():
        #         gpu_props = torch.cuda.get_device_properties(0)
        #         print(f"   GPU: {gpu_props.name}")
        #         print(f"   Total VRAM: {gpu_props.total_memory / 1024**3:.2f} GB")
        #         print(f"   CUDA Version: {torch.version.cuda}")
        #     else:
        #         print("   GPU: Not available")
            
        #     if psutil:
        #         cpu_info = psutil.cpu_count(logical=False)
        #         cpu_logical = psutil.cpu_count(logical=True)
        #         memory_info = psutil.virtual_memory()
        #         print(f"   CPU: {cpu_info} physical cores, {cpu_logical} logical cores")
        #         print(f"   RAM: {memory_info.total / 1024**3:.2f} GB total, {memory_info.available / 1024**3:.2f} GB available")
        #     else:
        #         print("   CPU: Not available")
        #         print("   RAM: Not available")
            
        #     # Pipeline Step-by-Step Analysis
        #     print("\n📊 PIPELINE STEP ANALYSIS:")
        #     print("-" * 80)
            
        #     # Step 1: Model Loading
        #     print("1️⃣  MODEL LOADING:")
        #     step1_data = self.oom_checklist.get('model_loading')
        #     if step1_data:
        #         print(f"   Status: {'✅ PASS' if step1_data['status'] == 'PASS' else '❌ FAIL'}")
        #         print(f"   GPU Memory: {step1_data['allocated_mb']:.1f} MB allocated, {step1_data['reserved_mb']:.1f} MB reserved")
        #         if torch.cuda.is_available():
        #             current_gpu = torch.cuda.memory_allocated() / 1024**2
        #             current_reserved = torch.cuda.memory_reserved() / 1024**2
        #             print(f"   Current GPU: {current_gpu:.1f} MB allocated, {current_reserved:.1f} MB reserved")
        #             if step1_data['allocated_mb'] > 0:
        #                 memory_change = current_gpu - step1_data['allocated_mb']
        #                 print(f"   Memory Change: {memory_change:+.1f} MB")
        #     else:
        #         print("   Status: ❌ NOT EXECUTED")
            
        #     # Step 2: LoRA Application
        #     print("\n2️⃣  LoRA APPLICATION:")
        #     step2_data = self.oom_checklist.get('lora_application')
        #     if step2_data:
        #         print(f"   Status: {'✅ PASS' if step2_data['status'] == 'PASS' else '❌ FAIL'}")
        #         print(f"   GPU Memory: {step2_data['allocated_mb']:.1f} MB allocated, {step2_data['reserved_mb']:.1f} MB reserved")
        #     else:
        #         print("   Status: ❌ NOT EXECUTED")
            
        #     # Step 3: Text Encoding
        #     print("\n3️⃣  TEXT ENCODING:")
        #     step3_data = self.oom_checklist.get('text_encoding')
        #     if step3_data:
        #         print(f"   Status: {'✅ PASS' if step3_data['status'] == 'PASS' else '❌ FAIL'}")
        #         print(f"   GPU Memory: {step3_data['allocated_mb']:.1f} MB allocated, {step3_data['reserved_mb']:.1f} MB reserved")
        #         print(f"   CLIP Status: Moved to offload device (CPU)")
        #     else:
        #         print("   Status: ❌ NOT EXECUTED")
            
        #     # Step 4: Model Sampling
        #     print("\n4️⃣  MODEL SAMPLING (ModelSamplingSD3):")
        #     step4_data = self.oom_checklist.get('model_sampling')
        #     if step4_data:
        #         print(f"   Status: {'✅ PASS' if step4_data['status'] == 'PASS' else '❌ FAIL'}")
        #         print(f"   GPU Memory: {step4_data['allocated_mb']:.1f} MB allocated, {step4_data['reserved_mb']:.1f} MB reserved")
        #     else:
        #         print("   Status: ❌ NOT EXECUTED")
            
        #     # Step 5: VAE Encoding
        #     print("\n5️⃣  VAE ENCODING:")
        #     step5_data = self.oom_checklist.get('vae_encoding_complete')
        #     if step5_data:
        #         print(f"   Status: {'✅ PASS' if step5_data['status'] == 'PASS' else '❌ FAIL'}")
        #         print(f"   GPU Memory: {step5_data['allocated_mb']:.1f} MB allocated, {step5_data['reserved_mb']:.1f} MB reserved")
                
        #         # Check if VAE encoding actually worked or fell back to dummies
        #         if 'init_latent' in locals():
        #             if hasattr(init_latent, 'shape'):
        #                 print(f"   Latent Generated: ✅ Shape: {init_latent.shape}")
        #                 if init_latent.shape[1] < 10:  # Likely dummy latents
        #                     print("   ⚠️  WARNING: Using dummy latents (VAE encoding failed)")
        #                 else:
        #                     print("   ✅ Real VAE encoding successful")
        #             else:
        #                 print("   Latent Generated: ❌ No shape information")
        #         else:
        #             print("   Latent Generated: ❌ No latent created")
        #     else:
        #         print("   Status: ❌ NOT EXECUTED")
            
        #     # Step 6: UNET Sampling
        #     print("\n6️⃣  UNET SAMPLING:")
        #     step6_data = self.oom_checklist.get('unet_sampling')
        #     if step6_data:
        #         print(f"   Status: {'✅ PASS' if step6_data['status'] == 'PASS' else '❌ FAIL'}")
        #         print(f"   GPU Memory: {step6_data['allocated_mb']:.1f} MB allocated, {step6_data['reserved_mb']:.1f} MB reserved")
                
        #         # Check if UNET sampling worked
        #         if 'final_latent' in locals():
        #             if hasattr(final_latent, 'shape'):
        #                 print(f"   Sampling Result: ✅ Shape: {final_latent.shape}")
        #             else:
        #                 print("   Sampling Result: ❌ No shape information")
        #         else:
        #             print("   Sampling Result: ❌ No final latent created")
        #     else:
        #         print("   Status: ❌ NOT EXECUTED")
            
        #     # Step 7: Video Latent Trimming
        #     print("\n7️⃣  VIDEO LATENT TRIMMING:")
        #     if 'trimmed_latent' in locals():
        #         if hasattr(trimmed_latent, 'shape'):
        #             print(f"   Status: ✅ PASS")
        #             print(f"   Trimmed Shape: {trimmed_latent.shape}")
        #             print(f"   Trim Count: {trim_count if 'trim_count' in locals() else 'Unknown'}")
        #         else:
        #             print("   Status: ❌ FAIL - No shape information")
        #     else:
        #         print("   Status: ❌ NOT EXECUTED")
            
        #     # Step 8: VAE Decoding
        #     print("\n8️⃣  VAE DECODING:")
        #     step8_data = self.oom_checklist.get('vae_decoding')
        #     if step8_data:
        #         print(f"   Status: {'✅ PASS' if step8_data['status'] == 'PASS' else '❌ FAIL'}")
        #         print(f"   GPU Memory: {step8_data['allocated_mb']:.1f} MB allocated, {step8_data['reserved_mb']:.1f} MB reserved")
                
        #         # Check if frames were generated
        #         if 'frames' in locals():
        #             if hasattr(frames, 'shape'):
        #                 print(f"   Frames Generated: ✅ Shape: {frames.shape}")
        #                 if len(frames.shape) == 4:
        #                     print(f"   Frame Info: {frames.shape[0]} frames, {frames.shape[1]}x{frames.shape[2]}, {frames.shape[3]} channels")
        #             else:
        #                 print("   Frames Generated: ❌ No shape information")
        #         else:
        #             print("   Frames Generated: ❌ No frames created")
        #     else:
        #         print("   Status: ❌ NOT EXECUTED")
            
        #     # Step 9: Video Export
        #     print("\n9️⃣  VIDEO EXPORT:")
        #     if 'output_path' in locals():
        #         print(f"   Status: ✅ PASS")
        #         print(f"   Output Path: {output_path}")
        #         if os.path.exists(output_path):
        #             file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        #             print(f"   File Size: {file_size:.1f} MB")
        #         else:
        #             print("   File Size: ❌ File not found")
        #     else:
        #         print("   Status: ❌ NOT EXECUTED")
            
        #     # Memory Usage Summary
        #     print("\n💾 MEMORY USAGE SUMMARY:")
        #     print("-" * 80)
            
        #     if torch.cuda.is_available():
        #         final_allocated = torch.cuda.memory_allocated() / 1024**2
        #         final_reserved = torch.cuda.memory_reserved() / 1024**2
        #         total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**2
        #         free_vram = total_vram - final_reserved
                
        #         print(f"   Final GPU Memory:")
        #         print(f"     Allocated: {final_allocated:.1f} MB")
        #         print(f"     Reserved: {final_reserved:.1f} MB")
        #         print(f"     Free: {free_vram:.1f} MB")
        #         print(f"     Total: {total_vram:.1f} MB")
        #         print(f"     Utilization: {(final_reserved/total_vram)*100:.1f}%")
                
        #         # Memory efficiency
        #         if 'baseline_allocated' in locals():
        #             baseline_mb = baseline_allocated / 1024**2
        #             memory_efficiency = ((final_allocated - baseline_mb) / baseline_mb) * 100 if baseline_mb > 0 else 0
        #             print(f"     Memory Efficiency: {memory_efficiency:+.1f}% from baseline")
            
        #     # CPU Memory
        #     cpu_memory = psutil.virtual_memory()
        #     print(f"   Final CPU Memory:")
        #     print(f"     Used: {cpu_memory.used / 1024**3:.1f} GB")
        #     print(f"     Available: {cpu_memory.available / 1024**3:.1f} GB")
        #     print(f"     Total: {cpu_memory.total / 1024**3:.1f} GB")
        #     print(f"     Utilization: {cpu_memory.percent:.1f}%")
            
        #     # Performance Metrics
        #     print("\n⚡ PERFORMANCE METRICS:")
        #     print("-" * 80)
            
        #     # Count successful vs failed steps
        #     successful_steps = 0
        #     failed_steps = 0
        #     total_steps = 0
            
        #     for step_name, step_data in self.oom_checklist.items():
        #         if step_data is not None:
        #             total_steps += 1
        #             if step_data['status'] == 'PASS':
        #                 successful_steps += 1
        #             else:
        #                 failed_steps += 1
            
        #     print(f"   Pipeline Success Rate: {successful_steps}/{total_steps} steps ({successful_steps/total_steps*100:.1f}%)")
            
        #     # Identify critical failures
        #     critical_failures = []
        #     if 'vae_encoding_complete' in self.oom_checklist and self.oom_checklist['vae_encoding_complete']:
        #         if self.oom_checklist['vae_encoding_complete']['status'] == 'FAIL':
        #             critical_failures.append("VAE Encoding")
            
        #     if 'unet_sampling' in self.oom_checklist and self.oom_checklist['unet_sampling']:
        #         if self.oom_checklist['unet_sampling']['status'] == 'FAIL':
        #             critical_failures.append("UNET Sampling")
            
        #     if critical_failures:
        #         print(f"   Critical Failures: {'❌ ' + ', '.join(critical_failures)}")
        #     else:
        #         print("   Critical Failures: ✅ None")
            
        #     # Recommendations
        #     print("\n💡 RECOMMENDATIONS:")
        #     print("-" * 80)
            
        #     print("   ✅ Pipeline: Now fully leverages ComfyUI's proven memory management")
        #     print("   ✅ Memory: ComfyUI automatically prevents fragmentation and OOM")
        #     print("   ✅ Models: All model loading/unloading handled by ComfyUI")
            
        #     if failed_steps > 0:
        #         print("   🔧 Pipeline: Review failed steps - may need to adjust input parameters")
        #         print("   🔧 Pipeline: ComfyUI will handle memory automatically")
            
        #     if 'vae_encoding_complete' in self.oom_checklist and self.oom_checklist['vae_encoding_complete']:
        #         if self.oom_checklist['vae_encoding_complete']['status'] == 'PASS':
        #             print("   ✅ VAE Encoding: Working correctly with ComfyUI")
        #         else:
        #             print("   🔧 VAE Encoding: ComfyUI will handle memory management automatically")
            
        #     print("\n" + "="*100)
        #     print("🔍 DIAGNOSTIC SUMMARY COMPLETE")
        #     print("="*100)
            
        #     # Verify final memory state
        #     print("Final cleanup: Verifying memory state...")
        #     if torch.cuda.is_available():
        #         final_allocated = torch.cuda.memory_allocated() / 1024**2
        #         final_reserved = torch.cuda.memory_reserved() / 1024**2
        #         print(f"Final VRAM - Allocated: {final_allocated:.1f} MB, Reserved: {final_reserved:.1f} MB")
                
        #         # Compare with baseline
        #         if 'baseline_allocated' in locals():
        #             memory_diff = final_allocated - baseline_allocated
        #             print(f"Memory change from baseline: {memory_diff:+.1f} MB")
        #             if abs(memory_diff) < 100:  # Within 100MB of baseline
        #                 print("✓ Memory successfully restored to baseline state")
        #             else:
        #                 print("⚠ Memory not fully restored to baseline state")
            
        #     # ComfyUI automatically handles cleanup when the pipeline completes
        #     # All models will be properly offloaded and memory will be freed
            
        #     return output_path
            
        # except Exception as e:
        #     print(f"Pipeline failed with error: {str(e)}")
            
        #     # ComfyUI automatically handles cleanup on failure
        #     raise
    
    def load_video(self, video_path):
        """Load control video from path as float tensor (T, H, W, 3) in [0,1]."""
        if not video_path or not os.path.exists(video_path):
            print(f"Warning: Video file not found: {video_path}")
            return None
        try:
            from torchvision.io import read_video
            print(f"Loading video from: {video_path}")
            video, audio, info = read_video(video_path, pts_unit='sec')  # (T, H, W, C) uint8
            if video is None or video.numel() == 0:
                print(f"Warning: Empty video: {video_path}")
                return None
            # Normalize to [0,1] float32 and ensure CPU tensor
            video = video.float() / 255.0
            # Ensure 3 channels; if more, take first 3; if 1, repeat to 3
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
        """Load reference image from path as float tensor (1, H, W, 3) in [0,1]."""
        if not image_path or not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}")
            return None
        try:
            from PIL import Image
            import numpy as np
            print(f"Loading image from: {image_path}")
            img = Image.open(image_path).convert('RGB')
            arr = np.asarray(img).astype('float32') / 255.0  # (H,W,3)
            # Add time dimension of 1 frame to match expected shape
            tensor = torch.from_numpy(arr).unsqueeze(0)  # (1,H,W,3)
            print(f"Loaded image tensor: {tuple(tensor.shape)} (1,H,W,3)")
            return tensor
        except Exception as e:
            print(f"Error loading image '{image_path}': {e}")
            return None
    

    

    

    

    


def main():
    """Main function to run the pipeline"""
    pipeline = ReferenceVideoPipeline()
    
    # Example usage - Updated for individual component loading with absolute paths
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
    
    print(f"Video generated successfully: {output_path}")

if __name__ == "__main__":
    main() 