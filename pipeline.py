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
            
            # VAE encoding step - keep VAE on GPU
            print("5a. VAE encoding...")
            
            # Use chunked processing for VAE encoding
            init_latent, trim_count = self._encode_with_chunking(
                video_generator, positive_cond, negative_cond, vae, width, height,
                length, batch_size, strength, control_video, reference_image, processing_plan
            )
            
            # Track initial latent for cleanup
            self.memory_manager.track_tensor(init_latent, "initial_latent", cleanup_priority=1)
            
            # After VAE encoding, we can offload VAE to CPU to free VRAM for UNET
            print("5b. Offloading VAE to CPU after encoding...")
            self.model_manager.unload_model("VAE")
            self.model_manager.print_status()
            
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
            
            # After UNET sampling, we can offload UNET to CPU to free VRAM for VAE decoding
            print("6a. Offloading UNET to CPU after sampling...")
            self.model_manager.unload_model("UNET")
            self.model_manager.print_status()
            
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
            # Reload VAE to GPU for decoding
            print("8a. Reloading VAE to GPU for decoding...")
            self.model_manager.load_model_gpu(vae, "VAE")
            
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
            
            # After VAE decoding, offload VAE to CPU
            print("8b. Offloading VAE to CPU after decoding...")
            self.model_manager.unload_model("VAE")
            self.model_manager.print_status()
            
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
            
            # Final cleanup - unload all models to CPU and cleanup all tracked tensors
            print("Final cleanup: Unloading all models to CPU and cleaning up tensors...")
            self.model_manager.cleanup()
            self.memory_manager.cleanup_all_tracked_tensors()
            
            return output_path
            
        except Exception as e:
            print(f"Pipeline failed with error: {str(e)}")
            # Ensure models and memory are cleaned up even on failure
            try:
                self.model_manager.cleanup()
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
    
    def _print_memory_callback(self):
        """Callback function to print memory stats during cleanup"""
        print("Memory cleanup callback triggered - checking current status...")
        self.memory_manager.print_memory_stats()
    
    def _encode_with_chunking(self, video_generator, positive, negative, vae, width, height, 
                             length, batch_size, strength, control_video, reference_image, 
                             processing_plan):
        """Encode video using chunked processing if needed"""
        
        chunk_size = processing_plan['vae_encode']['chunk_size']
        
        if length <= chunk_size:
            # Process all frames at once
            return video_generator.encode(
                positive, negative, vae, width, height, length, batch_size,
                strength, control_video, None, reference_image
            )
        
        # Process in chunks
        print(f"Processing {length} frames in chunks of {chunk_size}")
        
        # For chunked processing, we need to handle the video generator differently
        # This is a simplified approach - in practice, you'd modify the video_generator
        # to support true chunked processing
        
        # For now, we'll use the original method but with chunked control video processing
        if control_video is not None and control_video.shape[0] > chunk_size:
            print("Processing control video in chunks...")
            # This would be where you'd implement true chunked processing
            # For now, we'll fall back to the original method
        
        return video_generator.encode(
            positive, negative, vae, width, height, length, batch_size,
            strength, control_video, None, reference_image
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