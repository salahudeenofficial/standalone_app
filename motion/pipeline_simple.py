def main():
    """Simple complete WAN Video Pipeline - All 7 Steps"""
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
    main()
