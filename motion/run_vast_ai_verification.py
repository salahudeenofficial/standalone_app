#!/usr/bin/env python3
"""
Enhanced WAN 2.1 VACE 16B Model Verification Script for VAST AI
This script should be run from the directory containing your models/ folder
"""

import sys
import os
from pathlib import Path
from test_wan21_vace_16b_complete import WAN21VACEVerifier

def find_model_file():
    """Find the WAN 2.1 model file in various possible locations"""
    possible_paths = [
        "models/diffusion_models/wan_2.1_diffusion_model.safetensors",
        "./models/diffusion_models/wan_2.1_diffusion_model.safetensors", 
        "../models/diffusion_models/wan_2.1_diffusion_model.safetensors",
        "../../models/diffusion_models/wan_2.1_diffusion_model.safetensors",
        "/workspace/models/diffusion_models/wan_2.1_diffusion_model.safetensors",
        "/workspace/standalone_app/models/diffusion_models/wan_2.1_diffusion_model.safetensors"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            size_gb = os.path.getsize(path) / (1024**3)
            print(f"📁 Found model: {path}")
            print(f"📏 Size: {size_gb:.1f} GB")
            
            # Check if this is the actual large model (should be >25GB)
            if size_gb > 25:
                return path
            else:
                print(f"⚠️  File too small ({size_gb:.1f} GB), continuing search...")
    
    return None

def main():
    """Run enhanced verification on the WAN 2.1 VACE model"""
    
    print("🚀 WAN 2.1 VACE 16B Enhanced Model Verification")
    print("=" * 60)
    print("🔍 Searching for WAN 2.1 model file...")
    print()
    
    # Find the model file
    model_path = find_model_file()
    
    if not model_path:
        print("❌ WAN 2.1 model file not found!")
        print()
        print("🔍 Please ensure your model is located at one of these paths:")
        print("   • models/diffusion_models/wan_2.1_diffusion_model.safetensors")
        print("   • ./models/diffusion_models/wan_2.1_diffusion_model.safetensors")
        print("   • /workspace/models/diffusion_models/wan_2.1_diffusion_model.safetensors")
        print()
        print("📝 Or run this script from the directory containing the models/ folder")
        return 1
    
    print(f"✅ Using model: {model_path}")
    print()
    
    # Initialize verifier
    verifier = WAN21VACEVerifier(model_path)
    
    # Run complete verification
    print("🔍 Starting ENHANCED model verification with:")
    print("   🔄 ModelPatcher integration (normal + low VRAM fallback)")
    print("   🧪 Partial loading capabilities")
    print("   🎯 KSampler inference pipeline")
    print("   📊 14 comprehensive tests")
    print()
    print("⏳ This may take several minutes for a 16B parameter model...")
    print()
    
    try:
        report = verifier.run_complete_verification()
        
        # Print enhanced summary
        success_rate = report.get('success_rate', 0)
        passed_tests = report.get('passed_tests', 0)
        total_tests = report.get('total_tests', 0)
        
        print()
        print("=" * 60)
        if report.get('overall_success', False):
            print("🎉 SUCCESS! Enhanced verification completed successfully.")
            print(f"✅ Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests})")
            print()
            print("🚀 Your WAN 2.1 VACE model is ready for production:")
            print("   📹 Text-to-Video generation")
            print("   🖼️  Image-to-Video generation") 
            print("   🎬 VACE video editing")
            print("   🔄 ModelPatcher operations (memory efficient)")
            print("   🧠 KSampler inference pipeline")
            print("   ⚡ GPU/CPU dynamic loading")
        else:
            print("⚠️  Enhanced verification completed with some issues.")
            print(f"📊 Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests})")
            print()
            if success_rate >= 75:
                print("✨ Model is mostly functional - minor issues detected")
            elif success_rate >= 50:
                print("🔧 Model has moderate issues - check detailed report")
            else:
                print("❌ Model has significant issues - review required")
        
        # Show enhanced model info
        if 'model_info' in report:
            info = report['model_info']
            print()
            print("📊 Enhanced Model Analysis:")
            print(f"   Parameters: {info.get('total_parameters', 'Unknown'):,}")
            print(f"   Model Type: {info.get('model_type', 'Unknown')}")
            print(f"   VACE Support: {'✅' if info.get('is_vace_model', False) else '❌'}")
            print(f"   Loading Mode: {info.get('loading_mode', 'Unknown')}")
            print(f"   Uses Patcher: {'✅' if info.get('uses_patcher', False) else '❌'}")
        
        # Show test breakdown
        if 'verification_results' in report:
            results = report['verification_results']
            print()
            print("🔍 Detailed Test Results:")
            
            categories = {
                'Core Loading': ['file_verification', 'patcher_loading', 'model_loading'],
                'Architecture': ['architecture_verification', 'parameter_verification', 'component_verification'],
                'Performance': ['dtype_verification', 'memory_verification', 'partial_loading'],
                'Inference': ['forward_pass_t2v', 'forward_pass_i2v', 'forward_pass_vace', 'ksampler_inference'],
                'Structure': ['state_dict_verification']
            }
            
            for category, tests in categories.items():
                passed = sum(1 for test in tests if results.get(test, False))
                total = len(tests)
                status = "✅" if passed == total else "⚠️" if passed > 0 else "❌"
                print(f"   {status} {category}: {passed}/{total}")
        
        print()
        if 'total_time' in report:
            print(f"⏱️  Total verification time: {report['total_time']:.1f} seconds")
        
        return 0 if report.get('overall_success', False) else 1
        
    except Exception as e:
        print(f"❌ Enhanced verification failed with error: {e}")
        print()
        print("🔧 This could indicate:")
        print("   • Model file corruption")
        print("   • Insufficient GPU memory") 
        print("   • Missing dependencies")
        print("   • Hardware compatibility issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())
