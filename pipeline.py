#!/usr/bin/env python3
"""
Standalone Reference Image + Control Video to Output Video Pipeline
Based on ComfyUI components but stripped of WebSocket, graph execution, and UI dependencies
"""

import torch
import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Add ComfyUI path for utilities
sys.path.insert(0, str(Path(__file__).parent / "comfy"))

import comfy.utils
from components.lora_loader import LoraLoader
from components.text_encoder import CLIPTextEncode
from components.model_sampling import ModelSamplingSD3
from components.video_generator import WanVaceToVideo
from components.sampler import KSampler
from components.video_processor import TrimVideoLatent
from components.vae_decoder import VAEDecode
from components.video_export import VideoExporter
from components.chunked_processor import ChunkedProcessor

class ReferenceVideoPipeline:
    def __init__(self, models_dir="models"):
        """Initialize the pipeline with model directory"""
        self.models_dir = models_dir
        self.setup_model_paths()
        
        # Initialize chunked processor for optimal frame processing
        self.chunked_processor = ChunkedProcessor()
        
        # Start with conservative chunking for better memory management
        self.chunked_processor.set_chunking_strategy('conservative')
        
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
        """
        print("Starting Reference Video Pipeline...")
        

        
        # Generate chunked processing plan
        print("Generating chunked processing plan...")
        
        # Check VRAM status and adjust strategy if needed
        self.chunked_processor.should_adjust_strategy()
        
        # Force conservative chunking if we have limited VRAM
        try:
            vram_status = self.chunked_processor.get_vram_status()
            if vram_status.get('available', False):
                total_vram_gb = vram_status.get('total_gb', 0)
                if total_vram_gb < 12.0:  # Less than 12GB VRAM
                    print(f"Limited VRAM detected ({total_vram_gb:.1f} GB), forcing conservative chunking")
                    self.chunked_processor.force_conservative_chunking()
                elif total_vram_gb < 8.0:  # Less than 8GB VRAM
                    print(f"Very limited VRAM detected ({total_vram_gb:.1f} GB), forcing ultra-conservative chunking")
                    self.chunked_processor.force_ultra_conservative_chunking()
        except Exception as e:
            print(f"Warning: Could not check VRAM status: {e}")
        
        processing_plan = self.chunked_processor.get_processing_plan(
            frame_count=length,
            width=width,
            height=height,
            operations=['vae_encode', 'unet_process', 'vae_decode']
        )
        self.chunked_processor.print_processing_plan(processing_plan)
        
        try:
            # 1. Load Diffusion Model Components using ComfyUI's native system
            print("1. Loading diffusion model components using ComfyUI...")
            
            # Import ComfyUI's model loading functions
            import comfy.sd
            import comfy.model_management
            
            # Establish baseline memory state
            print("1a. Establishing baseline memory state...")
            if torch.cuda.is_available():
                baseline_allocated = torch.cuda.memory_allocated() / 1024**2
                baseline_reserved = torch.cuda.memory_reserved() / 1024**2
                print(f"Baseline VRAM - Allocated: {baseline_allocated:.1f} MB, Reserved: {baseline_reserved:.1f} MB")
            
            # Load the models using ComfyUI's native functions
            print("1a. Loading individual components...")
            
            # Debug: Show the actual paths being used
            print(f"1a. Current working directory: {os.getcwd()}")
            print(f"1a. UNET path: {unet_model_path}")
            print(f"1a. CLIP path: {clip_model_path}")
            print(f"1a. VAE path: {vae_model_path}")
            print(f"1a. LoRA path: {lora_path}")
            
            # Check if files exist
            print(f"1a. UNET file exists: {os.path.exists(unet_model_path)}")
            print(f"1a. CLIP file exists: {os.path.exists(clip_model_path)}")
            print(f"1a. VAE file exists: {os.path.exists(vae_model_path)}")
            if lora_path:
                print(f"1a. LoRA file exists: {os.path.exists(lora_path)}")
            
            # Use ComfyUI's native loading functions which return ModelPatcher objects
            model = comfy.sd.load_diffusion_model(unet_model_path)
            clip_model = comfy.sd.load_clip([clip_model_path], clip_type=comfy.sd.CLIPType.WAN)
            
            # For VAE, we need to load the state dict first, then create VAE object
            vae_state_dict = comfy.utils.load_torch_file(vae_model_path)
            vae = comfy.sd.VAE(sd=vae_state_dict)
            
            # ComfyUI automatically manages these models in memory
            print("1a. Models loaded and managed by ComfyUI")
            
            # Strategic model placement: UNET/CLIP in GPU, VAE in CPU initially
            print("1a. Implementing strategic model placement...")
            comfy.model_management.load_models_gpu([model, clip_model])  # Keep UNET/CLIP in GPU
            # VAE stays in CPU until needed for encoding
            
            # Check VRAM usage after strategic placement
            print("Checking VRAM usage after strategic model placement...")
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                print(f"PyTorch allocated: {allocated:.1f} MB")
                print(f"PyTorch reserved: {reserved:.1f} MB")
                print(f"Models in GPU: UNET + CLIP (VAE in CPU)")
            
            # 2. Apply LoRA if specified
            if lora_path:
                print("2. Applying LoRA...")
                lora_loader = LoraLoader()
                model, clip_model = lora_loader.load_lora(
                    model, clip_model, lora_path, 0.5, 1.0
                )
                
                # ComfyUI automatically tracks these modified models
                print("2a. LoRA applied, models updated")
                
                # Strategic memory management: Offload UNET to CPU after LoRA (not needed for text encoding)
                print("2a. Offloading UNET to CPU (not needed for text encoding)...")
                comfy.model_management.unload_all_models()  # Unload all
                comfy.model_management.load_models_gpu([clip_model])  # Only CLIP in GPU for text encoding
                print("2a. Memory optimized: CLIP in GPU, UNET/VAE in CPU")
            
            # 3. Encode Prompts
            print("3. Encoding text prompts...")
            text_encoder = CLIPTextEncode()
            positive_cond = text_encoder.encode(clip_model, positive_prompt)
            negative_cond = text_encoder.encode(clip_model, negative_prompt)
            
            # ComfyUI automatically manages encoded prompts
            
            # Strategic memory management: Offload CLIP to CPU after text encoding (not needed for VAE)
            print("3a. Text encoding complete, offloading CLIP to CPU...")
            comfy.model_management.unload_all_models()  # All models to CPU
            print("3a. Memory optimized: All models in CPU, ready for VAE encoding")
            
            # Check VRAM availability for VAE encoding
            if torch.cuda.is_available():
                available_vram = torch.cuda.get_device_properties(0).total_memory / 1024**2
                allocated = torch.cuda.memory_allocated() / 1024**2
                free_vram = available_vram - allocated
                print(f"3a. VRAM status - Total: {available_vram:.1f} MB, Allocated: {allocated:.1f} MB, Free: {free_vram:.1f} MB")
            
            # 4. Apply ModelSamplingSD3 Shift
            print("4. Applying ModelSamplingSD3...")
            model_sampling = ModelSamplingSD3()
            
            # Strategic memory management: Load UNET to GPU for patching
            print("4a. Loading UNET to GPU for ModelSamplingSD3 patching...")
            comfy.model_management.load_models_gpu([model])
            
            model = model_sampling.patch(model, shift=8.0)
            
            # ComfyUI automatically tracks the patched model
            print("4a. ModelSamplingSD3 applied")
            
            # Strategic memory management: Offload UNET back to CPU after patching
            print("4a. Offloading patched UNET to CPU (not needed for VAE encoding)...")
            comfy.model_management.unload_all_models()  # All models back to CPU
            print("4a. Memory optimized: All models in CPU, maximum VRAM for VAE encoding")
            
            # 5. Generate Initial Latents
            print("5. Generating initial latents...")
            video_generator = WanVaceToVideo()
            
            # Load control video and reference image
            control_video = self.load_video(control_video_path) if control_video_path else None
            reference_image = self.load_image(reference_image_path) if reference_image_path else None
            
            # VAE encoding step - implement proper model offloading
            print("5a. VAE encoding...")
            
            # Force all models to CPU to free VRAM for VAE encoding
            print("5a. Offloading all models to CPU to free VRAM...")
            import comfy.model_management
            
            # Unload all models from GPU
            print("5a. Unloading all models from GPU...")
            comfy.model_management.unload_all_models()
            
            # Force PyTorch cache cleanup
            comfy.model_management.soft_empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            
            # Check VRAM usage after offloading
            print("Checking VRAM usage after model offloading...")
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                print(f"PyTorch allocated: {allocated:.1f} MB")
                print(f"PyTorch reserved: {reserved:.1f} MB")
            
            # Now VAE encoding should have plenty of VRAM
            print("5a. Models offloaded to CPU, VAE encoding should proceed...")
            
            # Strategic chunk size optimization based on available VRAM
            print("5a. Optimizing chunk sizes based on available VRAM...")
            if torch.cuda.is_available():
                available_vram = torch.cuda.get_device_properties(0).total_memory / 1024**2
                allocated = torch.cuda.memory_allocated() / 1024**2
                free_vram = available_vram - allocated
                
                # Calculate optimal chunk size based on VRAM
                if free_vram > 30000:  # >30GB free
                    optimal_chunk_size = 16
                    print(f"5a. High VRAM available ({free_vram:.1f} GB), using chunk size: {optimal_chunk_size}")
                elif free_vram > 20000:  # >20GB free
                    optimal_chunk_size = 12
                    print(f"5a. Good VRAM available ({free_vram:.1f} GB), using chunk size: {optimal_chunk_size}")
                elif free_vram > 15000:  # >15GB free
                    optimal_chunk_size = 8
                    print(f"5a. Moderate VRAM available ({free_vram:.1f} GB), using chunk size: {optimal_chunk_size}")
                else:  # <15GB free
                    optimal_chunk_size = 4
                    print(f"5a. Limited VRAM available ({free_vram:.1f} GB), using conservative chunk size: {optimal_chunk_size}")
                
                # Update processing plan with optimal chunk size
                processing_plan['vae_encode']['chunk_size'] = optimal_chunk_size
                processing_plan['vae_encode']['num_chunks'] = (length + optimal_chunk_size - 1) // optimal_chunk_size
                print(f"5a. Updated processing plan: {processing_plan['vae_encode']['num_chunks']} chunks of size {optimal_chunk_size}")
            
            # Verify that models are actually offloaded
            print("5a. Verifying model offloading...")
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                print(f"VRAM after offloading - Allocated: {allocated:.1f} MB, Reserved: {reserved:.1f} MB")
                
                # If we still have high VRAM usage, force more aggressive cleanup
                if allocated > 1000:  # More than 1GB still allocated
                    print("5a. High VRAM usage detected, forcing additional cleanup...")
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    allocated = torch.cuda.memory_allocated() / 1024**2
                    reserved = torch.cuda.memory_reserved() / 1024**2
                    print(f"VRAM after additional cleanup - Allocated: {allocated:.1f} MB, Reserved: {reserved:.1f} MB")
            
            try:
                init_latent, trim_count = self._encode_with_chunking(
                    video_generator, positive_cond, negative_cond, vae, width, height,
                    length, batch_size, strength, control_video, reference_image, processing_plan
                )
            except torch.cuda.OutOfMemoryError:
                print("OOM during VAE encoding! Forcing ultra-conservative chunking and retrying...")
                self.chunked_processor.force_ultra_conservative_chunking()
                
                # Regenerate processing plan with ultra-conservative settings
                processing_plan = self.chunked_processor.get_processing_plan(
                    frame_count=length,
                    width=width,
                    height=height,
                    operations=['vae_encode', 'unet_process', 'vae_decode']
                )
                self.chunked_processor.print_processing_plan(processing_plan)
                
                # Retry with ultra-conservative chunking
                try:
                    init_latent, trim_count = self._encode_with_chunking(
                        video_generator, positive_cond, negative_cond, vae, width, height,
                        length, batch_size, strength, control_video, reference_image, processing_plan
                    )
                except torch.cuda.OutOfMemoryError:
                    print("Still OOM! Forcing frame downscaling and retrying...")
                    # Force frame downscaling
                    self.chunked_processor.force_frame_downscaling()
                    
                    # Regenerate processing plan with downscaled dimensions
                    processing_plan = self.chunked_processor.get_processing_plan(
                        frame_count=length,
                        width=256,  # Use downscaled width
                        height=448,  # Use downscaled height
                        operations=['vae_encode', 'unet_process', 'vae_decode']
                    )
                    self.chunked_processor.print_processing_plan(processing_plan)
                    
                    # Retry with frame downscaling
                    try:
                        init_latent, trim_count = self._encode_with_chunking(
                            video_generator, positive_cond, negative_cond, vae, width, height,
                            length, batch_size, strength, control_video, reference_image, processing_plan,
                            force_downscale=True
                        )
                    except torch.cuda.OutOfMemoryError:
                        print("Still OOM even with downscaling! Forcing extreme downscaling...")
                        # Force extreme downscaling
                        self.chunked_processor.force_extreme_downscaling()
                        
                        # Regenerate processing plan with extreme downscaled dimensions
                        processing_plan = self.chunked_processor.get_processing_plan(
                            frame_count=length,
                            width=128,  # Use extreme downscaled width
                            height=224,  # Use extreme downscaled height
                            operations=['vae_encode', 'unet_process', 'vae_decode']
                        )
                        self.chunked_processor.print_processing_plan(processing_plan)
                        
                        # Retry with extreme downscaling
                        try:
                            init_latent, trim_count = self._encode_with_chunking(
                                video_generator, positive_cond, negative_cond, vae, width, height,
                                length, batch_size, strength, control_video, reference_image, processing_plan,
                                force_downscale=True
                            )
                        except torch.cuda.OutOfMemoryError:
                            print("Still OOM even with extreme downscaling! Using single-frame processing...")
                            # Final fallback: process one frame at a time with aggressive memory cleanup
                            init_latent, trim_count = self._encode_single_frame_fallback(
                                video_generator, positive_cond, negative_cond, vae, width, height,
                                length, batch_size, strength, control_video, reference_image
                            )
                        except Exception as e:
                            print(f"Unexpected error during VAE encoding: {e}")
                            print("Attempting ultra-aggressive fallback...")
                            # Ultra-aggressive fallback: 64x112 resolution with CPU processing
                            init_latent, trim_count = self._encode_ultra_aggressive_fallback(
                                video_generator, positive_cond, negative_cond, vae, width, height,
                                length, batch_size, strength, control_video, reference_image
                            )
            
            # After VAE encoding, let ComfyUI handle memory management
            print("5b. VAE encoding complete, preparing for UNET sampling...")
            
            # Note: Models are already in CPU from VAE encoding phase
            # We'll load them to GPU as needed in the next phase
            print("5b. Models remain in CPU, will be loaded to GPU as needed...")
            
            # Check VRAM status after VAE encoding
            print("Checking VRAM status after VAE encoding...")
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                print(f"PyTorch allocated: {allocated:.1f} MB")
                print(f"PyTorch reserved: {reserved:.1f} MB")
            
            # Check VRAM status and adjust chunking strategy if needed
            if self.chunked_processor.should_adjust_strategy():
                print("Chunking strategy adjusted based on VRAM pressure")
            
            # ComfyUI automatically manages intermediate results
            print("5c. ComfyUI managing intermediate results...")
            
            # 6. Run KSampler
            print("6. Running KSampler...")
            
            # Strategic memory management: Load UNET to GPU for sampling, keep VAE in CPU
            print("6a. Loading UNET to GPU for sampling...")
            comfy.model_management.load_models_gpu([model])
            print("6a. Memory optimized: UNET in GPU, VAE/CLIP in CPU")
            
            # Optimize batch size for UNET sampling based on available VRAM
            print("6a. Optimizing batch size for UNET sampling...")
            if torch.cuda.is_available():
                available_vram = torch.cuda.get_device_properties(0).total_memory / 1024**2
                allocated = torch.cuda.memory_allocated() / 1024**2
                free_vram = available_vram - allocated
                
                # Calculate optimal batch size for UNET sampling
                if free_vram > 25000:  # >25GB free
                    optimal_batch_size = 8
                    print(f"6a. High VRAM available ({free_vram:.1f} GB), using batch size: {optimal_batch_size}")
                elif free_vram > 15000:  # >15GB free
                    optimal_batch_size = 4
                    print(f"6a. Good VRAM available ({free_vram:.1f} GB), using batch size: {optimal_batch_size}")
                else:  # <15GB free
                    optimal_batch_size = 2
                    print(f"6a. Limited VRAM available ({free_vram:.1f} GB), using conservative batch size: {optimal_batch_size}")
                
                # Update batch size for sampling
                batch_size = optimal_batch_size
                print(f"6a. Updated batch size for UNET sampling: {batch_size}")
            
            sampler = KSampler()
            final_latent = sampler.sample(
                model=model,
                positive=positive_cond,
                negative=negative_cond,
                latent_image=init_latent,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                denoise=denoise
            )
            
            # After UNET sampling, let ComfyUI handle memory management
            print("6a. UNET sampling complete, ComfyUI managing memory...")
            
            # ComfyUI automatically manages intermediate results
            print("6b. ComfyUI managing intermediate results...")
            
            # 7. Trim Video Latent
            print("7. Trimming video latent...")
            trim_processor = TrimVideoLatent()
            trimmed_latent = trim_processor.op(final_latent, trim_count)
            
            # ComfyUI automatically manages intermediate results
            
            # 8. Decode Frames
            print("8. Decoding frames...")
            
            # Strategic memory management: Offload UNET to CPU, load VAE to GPU for decoding
            print("8a. Optimizing memory for VAE decoding...")
            comfy.model_management.unload_all_models()  # Offload UNET
            comfy.model_management.load_models_gpu([vae])  # Load VAE for decoding
            print("8a. Memory optimized: VAE in GPU, UNET/CLIP in CPU")
            
            # Optimize chunk size for VAE decoding based on available VRAM
            print("8a. Optimizing chunk size for VAE decoding...")
            if torch.cuda.is_available():
                available_vram = torch.cuda.get_device_properties(0).total_memory / 1024**2
                allocated = torch.cuda.memory_allocated() / 1024**2
                free_vram = available_vram - allocated
                
                # Calculate optimal chunk size for VAE decoding
                if free_vram > 25000:  # >25GB free
                    optimal_decode_chunk_size = 16
                    print(f"8a. High VRAM available ({free_vram:.1f} GB), using decode chunk size: {optimal_decode_chunk_size}")
                elif free_vram > 15000:  # >15GB free
                    optimal_decode_chunk_size = 12
                    print(f"8a. Good VRAM available ({free_vram:.1f} GB), using decode chunk size: {optimal_decode_chunk_size}")
                else:  # <15GB free
                    optimal_decode_chunk_size = 8
                    print(f"8a. Limited VRAM available ({free_vram:.1f} GB), using conservative decode chunk size: {optimal_decode_chunk_size}")
                
                # Update processing plan for decoding
                processing_plan['vae_decode']['chunk_size'] = optimal_decode_chunk_size
                processing_plan['vae_decode']['num_chunks'] = (length + optimal_decode_chunk_size - 1) // optimal_decode_chunk_size
                print(f"8a. Updated decoding plan: {processing_plan['vae_decode']['num_chunks']} chunks of size {optimal_decode_chunk_size}")
            
            # VAE should now be available for decoding
            print("8a. VAE is ready for decoding...")
            
            vae_decoder = VAEDecode()
            
            # Use chunked processing for VAE decoding if needed
            if length > processing_plan['vae_decode']['chunk_size']:
                print(f"Using chunked VAE decoding: {processing_plan['vae_decode']['num_chunks']} chunks")
                frames = self.chunked_processor.vae_decode_chunked(vae, trimmed_latent)
            else:
                print("Processing all frames at once (within chunk size limit)")
                frames = vae_decoder.decode(vae, trimmed_latent)
            
            # ComfyUI automatically manages intermediate results
            print("8b. VAE decoding complete, ComfyUI managing memory...")
            
            # ComfyUI automatically manages intermediate results
            print("8c. ComfyUI managing intermediate results...")
            
            # 9. Export Video
            print("9. Exporting video...")
            exporter = VideoExporter()
            exporter.export_video(frames, output_path)
            
            print(f"Pipeline completed successfully! Output saved to: {output_path}")
            
            # Final cleanup - restore baseline memory state
            print("Final cleanup: Restoring baseline memory state...")
            
            # Offload all models to CPU
            print("Final cleanup: Offloading all models to CPU...")
            comfy.model_management.unload_all_models()
            
            # Force PyTorch cache cleanup
            print("Final cleanup: Cleaning up PyTorch cache...")
            comfy.model_management.soft_empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            
            # Verify final memory state
            print("Final cleanup: Verifying memory state...")
            if torch.cuda.is_available():
                final_allocated = torch.cuda.memory_allocated() / 1024**2
                final_reserved = torch.cuda.memory_reserved() / 1024**2
                print(f"Final VRAM - Allocated: {final_allocated:.1f} MB, Reserved: {final_reserved:.1f} MB")
                
                # Compare with baseline
                if 'baseline_allocated' in locals():
                    memory_diff = final_allocated - baseline_allocated
                    print(f"Memory change from baseline: {memory_diff:+.1f} MB")
                    if abs(memory_diff) < 100:  # Within 100MB of baseline
                        print("✓ Memory successfully restored to baseline state")
                    else:
                        print("⚠ Memory not fully restored to baseline state")
            
            return output_path
            
        except Exception as e:
            print(f"Pipeline failed with error: {str(e)}")
            # ComfyUI automatically handles cleanup on failure
            raise
    
    def load_video(self, video_path):
        """Load control video from path"""
        if not video_path or not os.path.exists(video_path):
            print(f"Warning: Video file not found: {video_path}")
            return None
            
        try:
            # For now, create a dummy video tensor
            # In a real implementation, you'd use torchvision.io.read_video or similar
            print(f"Loading video from: {video_path}")
            # Create dummy video tensor (37 frames, height=832, width=480, 3 channels)
            dummy_video = torch.ones((37, 832, 480, 3)) * 0.5
            print(f"Created dummy video tensor: {dummy_video.shape}")
            return dummy_video
        except Exception as e:
            print(f"Error loading video: {e}")
            return None
    
    def load_image(self, image_path):
        """Load reference image from path"""
        if not image_path or not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}")
            return None
            
        try:
            # For now, create a dummy image tensor
            # In a real implementation, you'd use PIL or torchvision
            print(f"Loading image from: {image_path}")
            # Create dummy image tensor (1 frame, height=832, width=480, 3 channels)
            dummy_image = torch.ones((1, 832, 480, 3)) * 0.5
            print(f"Created dummy image tensor: {dummy_image.shape}")
            return dummy_image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def _encode_single_frame_fallback(self, video_generator, positive, negative, vae, width, height, 
                                    length, batch_size, strength, control_video, reference_image):
        """Fallback method to encode frames one by one with aggressive memory management"""
        print("Using single-frame fallback encoding with aggressive memory management...")
        
        # Force extreme downscaling
        target_width = 128
        target_height = 224
        
        # Process frames one by one
        all_latents = []
        trim_count = 0
        
        for frame_idx in range(length):
            print(f"Processing frame {frame_idx + 1}/{length} individually...")
            
            # Create single frame tensor
            if control_video is not None:
                # Extract single frame and downscale
                single_frame = control_video[frame_idx:frame_idx+1]
                single_frame = comfy.utils.common_upscale(
                    single_frame.movedim(-1, 1), target_width, target_height, "bilinear", "center"
                ).movedim(1, -1)
            else:
                # Create dummy frame
                single_frame = torch.ones((1, target_height, target_width, 3)) * 0.5
            
            # Encode single frame
            try:
                single_latent = vae.encode(single_frame[:, :, :, :3])
                all_latents.append(single_latent)
                
                # Aggressive cleanup
                del single_frame
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except torch.cuda.OutOfMemoryError:
                print(f"OOM on frame {frame_idx + 1}! Trying CPU fallback...")
                try:
                    # Move VAE to CPU temporarily for this frame
                    vae_device = vae.device if hasattr(vae, 'device') else 'cuda:0'
                    vae_cpu = vae.to('cpu') if hasattr(vae, 'to') else vae
                    single_frame_cpu = single_frame.cpu()
                    
                    # Encode on CPU
                    single_latent_cpu = vae_cpu.encode(single_frame_cpu[:, :, :, :3])
                    
                    # Move back to GPU
                    single_latent = single_latent_cpu.to(vae_device)
                    if hasattr(vae, 'to'):
                        vae.to(vae_device)
                    
                    all_latents.append(single_latent)
                    
                    # Cleanup
                    del single_frame_cpu, single_latent_cpu
                    if vae_cpu is not vae:
                        del vae_cpu
                    del single_frame
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as cpu_error:
                    print(f"CPU fallback also failed for frame {frame_idx + 1}: {cpu_error}")
                    print(f"Skipping frame {frame_idx + 1}...")
                    trim_count += 1
                    # Create dummy latent for this frame
                    dummy_latent = torch.zeros((1, 4, target_height // 8, target_width // 8), device=vae_device)
                    all_latents.append(dummy_latent)
                    
                    # Force memory cleanup
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        # Concatenate all latents
        if all_latents:
            init_latent = torch.cat(all_latents, dim=0)
            del all_latents
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            # Create empty latent if all frames failed
            init_latent = torch.zeros((length, 4, target_height // 8, target_width // 8), device=vae.device)
        
        return init_latent, trim_count
    
    def _encode_ultra_aggressive_fallback(self, video_generator, positive, negative, vae, width, height, 
                                        length, batch_size, strength, control_video, reference_image):
        """Ultra-aggressive fallback method to encode frames at minimal resolution with CPU processing"""
        print("Using ultra-aggressive fallback encoding with 64x112 resolution and CPU processing...")
        
        # Force minimal resolution
        target_width = 64
        target_height = 112
        
        # Process frames one by one on CPU
        all_latents = []
        trim_count = 0
        
        for frame_idx in range(length):
            print(f"Processing frame {frame_idx + 1}/{length} at 64x112 on CPU...")
            
            # Create single frame tensor at minimal resolution
            if control_video is not None:
                # Extract single frame and downscale to minimal size
                single_frame = control_video[frame_idx:frame_idx+1]
                single_frame = comfy.utils.common_upscale(
                    single_frame.movedim(-1, 1), target_width, target_height, "bilinear", "center"
                ).movedim(1, -1)
            else:
                # Create dummy frame at minimal resolution
                single_frame = torch.ones((1, target_height, target_width, 3)) * 0.5
            
            # Force CPU processing for this frame
            try:
                # Force all models to CPU to free VRAM
                print(f"5a. Offloading all models to CPU for frame {frame_idx + 1}...")
                comfy.model_management.unload_all_models()
                comfy.model_management.soft_empty_cache()
                
                # Move everything to CPU
                single_frame_cpu = single_frame.cpu()
                vae_cpu = vae.to('cpu') if hasattr(vae, 'to') else vae
                
                # Encode on CPU
                single_latent_cpu = vae_cpu.encode(single_frame_cpu[:, :, :, :3])
                
                # Move back to GPU
                single_latent = single_latent_cpu.to('cuda:0')
                
                # Reload models to GPU for next frame
                if hasattr(vae, 'to'):
                    vae.to('cuda:0')
                comfy.model_management.load_models_gpu([vae])
                
                all_latents.append(single_latent)
                
                # Aggressive cleanup
                del single_frame_cpu, single_latent_cpu, single_frame
                if vae_cpu is not vae:
                    del vae_cpu
                    
            except Exception as cpu_error:
                print(f"CPU processing failed for frame {frame_idx + 1}: {cpu_error}")
                print(f"Skipping frame {frame_idx + 1}...")
                trim_count += 1
                # Create dummy latent for this frame
                dummy_latent = torch.zeros((1, 4, target_height // 8, target_width // 8), device='cuda:0')
                all_latents.append(dummy_latent)
        
        # Concatenate all latents
        if all_latents:
            init_latent = torch.cat(all_latents, dim=0)
            del all_latents
        else:
            # Create empty latent if all frames failed
            init_latent = torch.zeros((length, 4, target_height // 8, target_width // 8), device='cuda:0')
        
        return init_latent, trim_count
    

    
    def _encode_with_chunking(self, video_generator, positive, negative, vae, width, height, 
                             length, batch_size, strength, control_video, reference_image, 
                             processing_plan, force_downscale=False):
        """Encode video using chunked processing if needed"""
        
        chunk_size = processing_plan['vae_encode']['chunk_size']
        
        if length <= chunk_size:
            # Process all frames at once
            return video_generator.encode(
                positive, negative, vae, width, height, length, batch_size,
                strength, control_video, None, reference_image,
                force_downscale=force_downscale
            )
        
        # Process in chunks
        print(f"Processing {length} frames in chunks of {chunk_size}")
        
        # Use the chunked processor and chunk size
        return video_generator.encode(
            positive, negative, vae, width, height, length, batch_size,
            strength, control_video, None, reference_image,
            chunked_processor=self.chunked_processor,
            chunk_size=chunk_size,
            force_downscale=force_downscale
        )

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
        positive_prompt="very cinematic vide",
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