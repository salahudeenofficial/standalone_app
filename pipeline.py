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

class ReferenceVideoPipeline:
    def __init__(self, models_dir="models"):
        """Initialize the pipeline with model directory"""
        self.models_dir = models_dir
        self.setup_model_paths()
        
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
        
        try:
            # 1. Load Diffusion Model Components
            print("1. Loading diffusion model components...")
            unet_loader = UNETLoader()
            clip_loader = CLIPLoader()
            vae_loader = VAELoader()
            
            model = unet_loader.load_unet(unet_model_path, "default")
            clip_model = clip_loader.load_clip(clip_model_path, "wan")
            vae = vae_loader.load_vae(vae_model_path)
            
            # 2. Apply LoRA if specified
            if lora_path:
                print("2. Applying LoRA...")
                lora_loader = LoraLoader()
                model, clip_model = lora_loader.load_lora(
                    model, clip_model, lora_path, 0.5, 1.0
                )
            
            # 3. Encode Prompts
            print("3. Encoding text prompts...")
            text_encoder = CLIPTextEncode()
            positive_cond = text_encoder.encode(clip_model, positive_prompt)
            negative_cond = text_encoder.encode(clip_model, negative_prompt)
            
            # 4. Apply ModelSamplingSD3 Shift
            print("4. Applying ModelSamplingSD3...")
            model_sampling = ModelSamplingSD3()
            model = model_sampling.patch(model, shift=8.0)
            
            # 5. Generate Initial Latents
            print("5. Generating initial latents...")
            video_generator = WanVaceToVideo()
            
            # Load control video and reference image
            control_video = self.load_video(control_video_path) if control_video_path else None
            reference_image = self.load_image(reference_image_path) if reference_image_path else None
            
            init_latent, trim_count = video_generator.encode(
                positive=positive_cond,
                negative=negative_cond,
                vae=vae,
                width=width,
                height=height,
                length=length,
                batch_size=batch_size,
                strength=strength,
                control_video=control_video,
                reference_image=reference_image
            )
            
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
            
            # 7. Trim Video Latent
            print("7. Trimming video latent...")
            trim_processor = TrimVideoLatent()
            trimmed_latent = trim_processor.op(final_latent, trim_count)
            
            # 8. Decode Frames
            print("8. Decoding frames...")
            vae_decoder = VAEDecode()
            frames = vae_decoder.decode(vae, trimmed_latent)
            
            # 9. Export Video
            print("9. Exporting video...")
            exporter = VideoExporter()
            exporter.export_video(frames, output_path)
            
            print(f"Pipeline completed successfully! Output saved to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Pipeline failed with error: {str(e)}")
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

def main():
    """Main function to run the pipeline"""
    pipeline = ReferenceVideoPipeline()
    
    # Example usage
    output_path = pipeline.run_pipeline(
        unet_model_path="wan_2.1_diffusion_model.safetensors",
        clip_model_path="wan_clip_model.safetensors",
        vae_model_path="wan_vae.safetensors",
        lora_path="Wan21_CausVid_14B_T2V_lora_rank32.safetensors",
        positive_prompt="A beautiful landscape",
        negative_prompt="blurry, low quality",
        control_video_path="control_video.mp4",
        reference_image_path="reference.jpg",
        width=480,
        height=832,
        length=37,
        output_path="generated_video.mp4"
    )
    
    print(f"Video generated successfully: {output_path}")

if __name__ == "__main__":
    main() 