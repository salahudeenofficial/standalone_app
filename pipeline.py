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

from components.model_loader import UNETLoader, CLIPLoader, VAELoader
from components.lora_loader import LoraLoader
from components.text_encoder import CLIPTextEncode
from components.model_sampling import ModelSamplingSD3
from components.video_generator import WanVaceToVideo
from components.sampler import KSampler
from components.video_processor import TrimVideoLatent
from components.vae_decoder import VAEDecode
from components.video_export import VideoExporter
from components.model_manager import ModelManager
from components.memory_manager import MemoryManager
from components.chunked_processor import ChunkedProcessor

class ReferenceVideoPipeline:
    def __init__(self, models_dir="models"):
        """Initialize the pipeline with model directory"""
        self.models_dir = models_dir
        self.setup_model_paths()
        
        # Initialize model manager for CPU/GPU swapping
        self.model_manager = ModelManager()
        
        # Initialize memory manager for intermediate tensor cleanup
        # Set auto-cleanup threshold to 2GB to prevent memory buildup
        self.memory_manager = MemoryManager(auto_cleanup_threshold_mb=2000.0)
        
        # Add cleanup callback to print memory stats
        self.memory_manager.add_cleanup_callback(self._print_memory_callback)
        
        # Initialize chunked processor for optimal frame processing
        self.chunked_processor = ChunkedProcessor(
            memory_manager=self.memory_manager,
            model_manager=self.model_manager
        )
        
        # Start with conservative chunking for better memory management
        self.chunked_processor.set_chunking_strategy('conservative')
        
    def setup_model_paths(self):
        """Setup model paths for the standalone app"""
        # Create model directories if they don't exist
        os.makedirs(f"{self.models_dir}/diffusion_models", exist_ok=True)
        os.makedirs(f"{self.models_dir}/text_encoders", exist_ok=True)
        os.makedirs(f"{self.models_dir}/vaes", exist_ok=True)
        os.makedirs(f"{self.models_dir}/loras", exist_ok=True)
        
        # Set environment variables for model paths
        os.environ["COMFY_MODEL_PATH"] = self.models_dir
        
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
        
        # Add cleanup point for pipeline start
        self.memory_manager.add_cleanup_point("pipeline_start", "Pipeline initialization")
        
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
            # 1. Load Diffusion Model Components
            print("1. Loading diffusion model components...")
            self.memory_manager.add_cleanup_point("model_loading", "Model loading phase")
            unet_loader = UNETLoader()
            clip_loader = CLIPLoader()
            vae_loader = VAELoader()
            
            model = unet_loader.load_unet(unet_model_path, "default")
            clip_model = clip_loader.load_clip(clip_model_path, "wan")
            vae = vae_loader.load_vae(vae_model_path)
            
            # Load models to GPU with smart memory management
            print("1a. Loading models to GPU...")
            self.model_manager.load_model_gpu(model, "UNET")
            self.model_manager.load_model_gpu(clip_model, "CLIP")
            self.model_manager.load_model_gpu(vae, "VAE")
            
            # Print current memory status
            self.model_manager.print_status()
            
            # Check detailed VRAM usage after model loading
            print("Checking VRAM usage after model loading...")
            self.memory_manager.print_detailed_vram_usage()
            
            # Mark model loading complete
            self.memory_manager.mark_cleanup_point_complete("model_loading")
            
            # 2. Apply LoRA if specified
            if lora_path:
                print("2. Applying LoRA...")
                lora_loader = LoraLoader()
                model, clip_model = lora_loader.load_lora(
                    model, clip_model, lora_path, 0.5, 1.0
                )
                
                # Update model manager tracking for modified models
                self.model_manager.loaded_models["UNET"] = {
                    'model': model, 
                    'device': self.model_manager.device, 
                    'size_gb': self.model_manager._estimate_model_size(model)
                }
                self.model_manager.loaded_models["CLIP"] = {
                    'model': clip_model, 
                    'device': self.model_manager.device, 
                    'size_gb': self.model_manager._estimate_model_size(clip_model)
                }
            
            # 3. Encode Prompts
            print("3. Encoding text prompts...")
            text_encoder = CLIPTextEncode()
            positive_cond = text_encoder.encode(clip_model, positive_prompt)
            negative_cond = text_encoder.encode(clip_model, negative_prompt)
            
            # Track encoded prompts for cleanup
            self.memory_manager.track_tensor(positive_cond, "positive_conditioning", cleanup_priority=2)
            self.memory_manager.track_tensor(negative_cond, "negative_conditioning", cleanup_priority=2)
            
            # 4. Apply ModelSamplingSD3 Shift
            print("4. Applying ModelSamplingSD3...")
            model_sampling = ModelSamplingSD3()
            model = model_sampling.patch(model, shift=8.0)
            
            # Update model manager tracking for patched model
            self.model_manager.loaded_models["UNET"] = {
                'model': model, 
                'device': self.model_manager.device, 
                'size_gb': self.model_manager._estimate_model_size(model)
            }
            
            # 5. Generate Initial Latents
            print("5. Generating initial latents...")
            video_generator = WanVaceToVideo()
            
            # Load control video and reference image
            control_video = self.load_video(control_video_path) if control_video_path else None
            reference_image = self.load_image(reference_image_path) if reference_image_path else None
            
            # VAE encoding step - use ComfyUI's native memory management
            print("5a. VAE encoding...")
            
            # CRITICAL: Use ComfyUI's native memory management to free VRAM
            print("5a. Using ComfyUI's native memory management to free VRAM...")
            
            # Import ComfyUI's memory management
            import comfy.model_management
            
            # CRITICAL: First, we need to unload UNET and CLIP to free VRAM for VAE encoding
            print("5a. Unloading UNET and CLIP to free VRAM for VAE encoding...")
            
            # Get the current loaded models from ComfyUI
            current_models = comfy.model_management.loaded_models()
            print(f"5a. Currently loaded models: {[m.__class__.__name__ for m in current_models]}")
            
            # Force unload all models except VAE
            print("5a. Force unloading all models to free maximum VRAM...")
            comfy.model_management.unload_all_models()
            
            # Force aggressive CUDA cleanup
            print("5a. Forcing aggressive CUDA cleanup...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            
            # Now reload only VAE for encoding
            print("5a. Reloading VAE for encoding...")
            comfy.model_management.load_models_gpu([vae])
            
            # Check detailed VRAM usage before VAE encoding
            print("Checking VRAM usage before VAE encoding...")
            self.memory_manager.print_detailed_vram_usage()
            
            # Use chunked processing for VAE encoding
            # VAE should already be on GPU from model loading
            print("5a. VAE is ready for encoding...")
            
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
            
            # Track initial latent for cleanup
            self.memory_manager.track_tensor(init_latent, "initial_latent", cleanup_priority=1)
            
            # After VAE encoding, unload VAE and reload UNET for sampling
            print("5b. Unloading VAE and reloading UNET for sampling...")
            
            # Unload all models
            comfy.model_management.unload_all_models()
            
            # Force aggressive CUDA cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            
            # Reload UNET for sampling
            print("5b. Reloading UNET for sampling...")
            comfy.model_management.load_models_gpu([model])
            
            # Check VRAM status and adjust chunking strategy if needed
            if self.chunked_processor.should_adjust_strategy():
                print("Chunking strategy adjusted based on VRAM pressure")
            
            # Clean up intermediate results after VAE encoding
            print("5c. Cleaning up intermediate results after VAE encoding...")
            self.memory_manager.cleanup_intermediate_results("VAE_encoding", keep_tensors=["initial_latent", "positive_conditioning", "negative_conditioning"])
            self.memory_manager.print_memory_stats()
            
            # 6. Run KSampler
            print("6. Running KSampler...")
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
            
            # Track final latent and cleanup initial latent
            self.memory_manager.track_tensor(final_latent, "final_latent", cleanup_priority=1)
            self.memory_manager.cleanup_tensor(init_latent, "initial_latent")
            
            # After UNET sampling, unload UNET and reload VAE for decoding
            print("6a. Unloading UNET and reloading VAE for decoding...")
            
            # Unload all models
            comfy.model_management.unload_all_models()
            
            # Force aggressive CUDA cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            
            # Reload VAE for decoding
            print("6a. Reloading VAE for decoding...")
            comfy.model_management.load_models_gpu([vae])
            
            # Clean up intermediate results after UNET sampling
            print("6b. Cleaning up intermediate results after UNET sampling...")
            self.memory_manager.cleanup_intermediate_results("UNET_sampling", keep_tensors=["final_latent"])
            self.memory_manager.print_memory_stats()
            
            # 7. Trim Video Latent
            print("7. Trimming video latent...")
            trim_processor = TrimVideoLatent()
            trimmed_latent = trim_processor.op(final_latent, trim_count)
            
            # Track trimmed latent and cleanup final latent
            self.memory_manager.track_tensor(trimmed_latent, "trimmed_latent", cleanup_priority=1)
            self.memory_manager.cleanup_tensor(final_latent, "final_latent")
            
            # 8. Decode Frames
            print("8. Decoding frames...")
            # VAE should still be available for decoding
            print("8a. VAE is ready for decoding...")
            
            vae_decoder = VAEDecode()
            
            # Use chunked processing for VAE decoding if needed
            if length > processing_plan['vae_decode']['chunk_size']:
                print(f"Using chunked VAE decoding: {processing_plan['vae_decode']['num_chunks']} chunks")
                frames = self.chunked_processor.vae_decode_chunked(vae, trimmed_latent)
            else:
                print("Processing all frames at once (within chunk size limit)")
                frames = vae_decoder.decode(vae, trimmed_latent)
            
            # Track decoded frames and cleanup trimmed latent
            self.memory_manager.track_tensor(frames, "decoded_frames", cleanup_priority=1)
            self.memory_manager.cleanup_tensor(trimmed_latent, "trimmed_latent")
            
            # After VAE decoding, unload VAE for final cleanup
            print("8b. Unloading VAE for final cleanup...")
            
            # Unload all models
            comfy.model_management.unload_all_models()
            
            # Force aggressive CUDA cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            
            # Clean up intermediate results after VAE decoding
            print("8c. Cleaning up intermediate results after VAE decoding...")
            self.memory_manager.cleanup_intermediate_results("VAE_decoding", keep_tensors=["decoded_frames"])
            self.memory_manager.print_memory_stats()
            
            # 9. Export Video
            print("9. Exporting video...")
            exporter = VideoExporter()
            exporter.export_video(frames, output_path)
            
            # Clean up decoded frames after export
            print("9a. Cleaning up decoded frames after export...")
            self.memory_manager.cleanup_tensor(frames, "decoded_frames")
            
            print(f"Pipeline completed successfully! Output saved to: {output_path}")
            
            # Final cleanup - use ComfyUI's memory management and cleanup all tracked tensors
            print("Final cleanup: Using ComfyUI's memory management and cleaning up tensors...")
            
            # Force unload all models using ComfyUI
            comfy.model_management.unload_all_models()
            
            # Force aggressive CUDA cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            
            self.memory_manager.cleanup_all_tracked_tensors()
            
            return output_path
            
        except Exception as e:
            print(f"Pipeline failed with error: {str(e)}")
            # Ensure models and memory are cleaned up even on failure
            try:
                # Use ComfyUI's memory management for cleanup
                import comfy.model_management
                comfy.model_management.unload_all_models()
                
                # Force aggressive CUDA cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                
                self.memory_manager.cleanup_all_tracked_tensors()
            except:
                pass
            raise
    
    def load_video(self, video_path):
        """Load control video from path"""
        # Implementation for video loading
        # This would use torchvision or similar to load video frames
        pass
    
    def load_image(self, image_path):
        """Load reference image from path"""
        # Implementation for image loading
        # This would use PIL or torchvision to load image
        pass
    
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
    
    def _print_memory_callback(self):
        """Callback function to print memory stats during cleanup"""
        print("Memory cleanup callback triggered - checking current status...")
        self.memory_manager.print_memory_stats()
    
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
    
    # Example usage
    output_path = pipeline.run_pipeline(
        unet_model_path="wan_2.1_diffusion_model.safetensors",
        clip_model_path="wan_clip_model.safetensors",
        vae_model_path="wan_vae.safetensors",
        lora_path="Wan21_CausVid_14B_T2V_lora_rank32.safetensors",
        positive_prompt="very cinematic vide",
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走 , extra hands, extra arms, extra legs",
        control_video_path="safu.mp4",
        reference_image_path="safu.jpg",
        width=480,
        height=832,
        length=37,
        output_path="generated_video.mp4"
    )
    
    print(f"Video generated successfully: {output_path}")

if __name__ == "__main__":
    main() 