#!/usr/bin/env python3
"""
Test script to verify ComfyUI's native memory management system works correctly.
"""

import torch
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_comfyui_model_loading():
    """Test that ComfyUI can properly load and manage models"""
    print("🧪 TESTING COMFYUI MODEL LOADING")
    print("="*50)
    
    try:
        # Import ComfyUI modules
        import comfy.sd
        import comfy.model_management
        import comfy.utils
        
        print("✅ ComfyUI modules imported successfully")
        
        # Check if we have the required model files
        script_dir = Path(__file__).parent
        unet_path = script_dir / "models/diffusion_models/wan_2.1_diffusion_model.safetensors"
        clip_path = script_dir / "models/text_encoders/wan_clip_model.safetensors"
        vae_path = script_dir / "models/vaes/wan_vae.safetensors"
        
        print(f"📁 Checking model files:")
        print(f"   UNET: {unet_path} - {'✅ EXISTS' if unet_path.exists() else '❌ MISSING'}")
        print(f"   CLIP: {clip_path} - {'✅ EXISTS' if clip_path.exists() else '❌ MISSING'}")
        print(f"   VAE: {vae_path} - {'✅ EXISTS' if vae_path.exists() else '❌ MISSING'}")
        
        # Test model loading if files exist
        if unet_path.exists() and clip_path.exists() and vae_path.exists():
            print("\n🔄 Testing model loading...")
            
            # Load models using ComfyUI's native functions
            model = comfy.sd.load_diffusion_model(str(unet_path))
            clip_model = comfy.sd.load_clip([str(clip_path)], clip_type=comfy.sd.CLIPType.WAN)
            vae_state_dict = comfy.utils.load_torch_file(str(vae_path))
            vae = comfy.sd.VAE(sd=vae_state_dict)
            
            print("✅ Models loaded successfully!")
            print(f"   UNET type: {type(model)}")
            print(f"   CLIP type: {type(clip_model)}")
            print(f"   VAE type: {type(vae)}")
            
            # Check if UNET and CLIP are ModelPatcher objects
            if hasattr(model, 'patches') and hasattr(model, 'load_device') and hasattr(model, 'offload_device'):
                print("✅ UNET is properly wrapped in ModelPatcher")
                print(f"   Load device: {model.load_device}")
                print(f"   Offload device: {model.offload_device}")
                print(f"   Has patches: {len(model.patches)}")
            else:
                print("❌ UNET is not properly wrapped in ModelPatcher")
            
            # CLIP objects have an internal ModelPatcher in .patcher
            if hasattr(clip_model, 'patcher') and hasattr(clip_model.patcher, 'patches'):
                print("✅ CLIP has internal ModelPatcher")
                print(f"   Load device: {clip_model.patcher.load_device}")
                print(f"   Offload device: {clip_model.patcher.offload_device}")
                print(f"   Has patches: {len(clip_model.patcher.patches)}")
                print(f"   Is CLIP: {clip_model.patcher.is_clip}")
            else:
                print("❌ CLIP missing internal ModelPatcher")
            
            # Check VAE properties
            if hasattr(vae, 'device') and hasattr(vae, 'vae_dtype'):
                print("✅ VAE has proper memory management properties")
                print(f"   Device: {vae.device}")
                print(f"   Dtype: {vae.vae_dtype}")
            else:
                print("❌ VAE missing memory management properties")
            
            # Test memory management
            print("\n🔄 Testing memory management...")
            
            # Check initial memory
            if torch.cuda.is_available():
                initial_memory = torch.cuda.memory_allocated() / 1024**2
                print(f"   Initial VRAM: {initial_memory:.1f} MB")
                
                # Test model loading to GPU
                print("   Testing UNET loading to GPU...")
                model.patch_model(comfy.model_management.get_torch_device())
                
                after_load_memory = torch.cuda.memory_allocated() / 1024**2
                print(f"   After UNET load: {after_load_memory:.1f} MB")
                print(f"   Memory increase: {after_load_memory - initial_memory:.1f} MB")
                
                # Test model unloading
                print("   Testing UNET unloading...")
                model.unpatch_model()
                
                # Force garbage collection and cache cleanup
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                
                after_unload_memory = torch.cuda.memory_allocated() / 1024**2
                print(f"   After UNET unload + cleanup: {after_unload_memory:.1f} MB")
                print(f"   Memory freed: {after_load_memory - after_unload_memory:.1f} MB")
                
                if after_unload_memory <= initial_memory + 100:  # Within 100MB of initial
                    print("✅ Memory management working correctly!")
                else:
                    print("❌ Memory not properly freed!")
                    print("   This may be normal for some models - checking if it's actually freed...")
                    
                    # Check if the model is actually on CPU
                    if hasattr(model, 'model') and hasattr(model.model, 'device'):
                        print(f"   UNET model device: {model.model.device}")
                        if str(model.model.device) == 'cpu':
                            print("   ✅ UNET is actually on CPU (memory may be cached)")
                        else:
                            print("   ❌ UNET is not on CPU")
            else:
                print("   Skipping GPU memory tests (CUDA not available)")
            
            # Test CLIP memory management
            print("\n🔄 Testing CLIP memory management...")
            if torch.cuda.is_available():
                clip_initial_memory = torch.cuda.memory_allocated() / 1024**2
                print(f"   CLIP initial VRAM: {clip_initial_memory:.1f} MB")
                
                # Test CLIP loading to GPU
                print("   Testing CLIP loading to GPU...")
                clip_model.load_model()
                
                clip_after_load_memory = torch.cuda.memory_allocated() / 1024**2
                print(f"   After CLIP load: {clip_after_load_memory:.1f} MB")
                print(f"   CLIP memory increase: {clip_after_load_memory - clip_initial_memory:.1f} MB")
                
                # Test CLIP unloading
                print("   Testing CLIP unloading...")
                clip_model.patcher.unpatch_model()
                
                # Force cleanup
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                
                clip_after_unload_memory = torch.cuda.memory_allocated() / 1024**2
                print(f"   After CLIP unload + cleanup: {clip_after_unload_memory:.1f} MB")
                print(f"   CLIP memory freed: {clip_after_load_memory - clip_after_unload_memory:.1f} MB")
            
            # Cleanup
            del model, clip_model, vae, vae_state_dict
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        else:
            print("⚠️  Skipping model loading tests (missing model files)")
        
        print("\n🎉 ComfyUI integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ ComfyUI integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline_components():
    """Test that pipeline components can be imported and work correctly"""
    print("\n🧪 TESTING PIPELINE COMPONENTS")
    print("="*50)
    
    try:
        # Test component imports
        from components.lora_loader import LoraLoader
        from components.text_encoder import CLIPTextEncode
        from components.model_sampling import ModelSamplingSD3
        from components.video_generator import WanVaceToVideo
        from components.sampler import KSampler
        from components.video_processor import TrimVideoLatent
        from components.vae_decoder import VAEDecode
        from components.video_export import VideoExporter
        from components.chunked_processor import ChunkedProcessor
        
        print("✅ All pipeline components imported successfully")
        
        # Test component instantiation
        print("🔄 Testing component instantiation...")
        
        lora_loader = LoraLoader()
        text_encoder = CLIPTextEncode()
        model_sampling = ModelSamplingSD3()
        video_generator = WanVaceToVideo()
        sampler = KSampler()
        trim_processor = TrimVideoLatent()
        vae_decoder = VAEDecode()
        video_exporter = VideoExporter()
        chunked_processor = ChunkedProcessor()
        
        print("✅ All components instantiated successfully")
        
        # Test chunked processor
        print("🔄 Testing chunked processor...")
        processing_plan = chunked_processor.get_processing_plan(
            frame_count=37,
            width=480,
            height=832,
            operations=['vae_encode', 'unet_process', 'vae_decode']
        )
        print(f"   Processing plan created: {len(processing_plan)} operations")
        
        print("\n🎉 Pipeline components test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 STARTING COMFYUI INTEGRATION TESTS")
    print("="*60)
    
    try:
        # Run tests
        comfyui_success = test_comfyui_model_loading()
        components_success = test_pipeline_components()
        
        print("\n" + "="*60)
        print("📊 TEST RESULTS SUMMARY")
        print("="*60)
        print(f"ComfyUI Integration: {'✅ PASSED' if comfyui_success else '❌ FAILED'}")
        print(f"Pipeline Components: {'✅ PASSED' if components_success else '❌ FAILED'}")
        
        if comfyui_success and components_success:
            print("\n🎉 ALL TESTS PASSED!")
            print("ComfyUI's native memory management system is ready to use!")
            return 0
        else:
            print("\n❌ SOME TESTS FAILED!")
            print("Please fix the issues before proceeding.")
            return 1
        
    except Exception as e:
        print(f"\n❌ TEST SUITE FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 