#!/usr/bin/env python3
"""
Modified main function for Steps 1 and 2 with multithreaded memory tracking
"""

import os
import sys
from pathlib import Path

# Add motion directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import WanVideoPipeline
from memory_tracker import create_memory_tracker, track_memory_during_operation

def main():
    """Example usage of Steps 1 and 2 with multithreaded memory tracking"""
    print("🚀 WAN Video Pipeline - Steps 1 and 2 Test with Memory Tracking")
    print("="*80)
    
    # Create memory tracker
    tracker = create_memory_tracker(interval=0.05, log_file="step1_step2_memory.log")
    
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
                print("\n🔍 Starting memory tracking for Step 1 test...")
                tracker.start_tracking()
                
                # Track Step 1 execution
                @track_memory_during_operation(tracker, "step1_vae_latent", "step1", {"test_mode": True})
                def test_step1():
                    return pipeline.run_step_1_only(**step_1_params)
                
                results = test_step1()
                
                tracker.stop_tracking()
                tracker.print_memory_summary()
                
                print("\n✅ Step 1 test completed - ready for Step 2 when models are available")
            except Exception as e:
                print(f"\n❌ STEP 1 TEST FAILED: {str(e)}")
                tracker.stop_tracking()
        return
    
    try:
        # Start memory tracking
        print("\n🔍 Starting multithreaded memory tracking...")
        tracker.start_tracking()
        
        # Track Step 1 execution
        print("\n🧪 STEP 1: VAE + Latent Creation (with memory tracking)")
        @track_memory_during_operation(tracker, "step1_vae_latent", "step1", {
            "width": step_1_params['width'], 
            "height": step_1_params['height'],
            "length": step_1_params['length']
        })
        def execute_step1():
            return pipeline.step_1_vae_and_latent_creation(**step_1_params)
        
        step_1_results = execute_step1()
        
        # Track Step 2 execution
        print("\n🧪 STEP 2: UNet + CLIP + LoRA Loading (with memory tracking)")
        @track_memory_during_operation(tracker, "step2_model_loading", "step2", {
            "unet_path": step_2_params['unet_model_path'],
            "clip_path": step_2_params['clip_model_path'],
            "lora_path": step_2_params['lora_model_path']
        })
        def execute_step2():
            return pipeline.step_2_unet_clip_lora_loading(**step_2_params)
        
        step_2_results = execute_step2()
        
        # Stop memory tracking
        tracker.stop_tracking()
        
        # Print memory summary
        tracker.print_memory_summary()
        
        print("\n🎉 STEPS 1 AND 2 COMPLETED WITH MEMORY TRACKING!")
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
            
            # Check for dynamic loading setup
            unet_model = step_2_results['unet'].model if hasattr(step_2_results['unet'], 'model') else step_2_results['unet']
            if hasattr(unet_model, '_dynamic_loading_info'):
                modules_count = len(unet_model._dynamic_loading_info.get('modules_info', []))
                print(f"   Dynamic Loading: ✅ Enabled ({modules_count} modules)")
            else:
                print(f"   Dynamic Loading: ❌ Not available")
        
        print("\n✅ Steps 1 and 2 completed - Memory tracking data saved to step1_step2_memory.log")
        print("✅ Ready for Step 3 when needed!")
        
    except Exception as e:
        print(f"\n❌ PIPELINE TEST FAILED: {str(e)}")
        tracker.stop_tracking()
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
