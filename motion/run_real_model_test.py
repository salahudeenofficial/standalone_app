#!/usr/bin/env python3
"""
Quick script to run the enhanced verification on your real WAN 2.1 VACE 16B model
"""

import sys
import os
from test_wan21_vace_16b_complete import WAN21VACEVerifier

def main():
    """Run verification on the real model"""
    
    # Your model path - let's try the 14B version that seems to be complete
    model_path = "../wan2.1_vace_14B_fp16.safetensors"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("📁 Please ensure the model file exists at the specified path")
        return 1
    
    print("🚀 WAN 2.1 VACE 16B Model Verification on VAST AI")
    print("=" * 50)
    print(f"📁 Model file: {model_path}")
    print(f"📏 File size: {os.path.getsize(model_path) / (1024**3):.0f}G")
    print()
    
    # Initialize verifier
    verifier = WAN21VACEVerifier(model_path)
    
    # Run complete verification
    print("🔍 Starting enhanced model verification...")
    print("This may take several minutes for a 16B parameter model...")
    print()
    
    try:
        report = verifier.run_complete_verification()
        
        # Print summary
        success_rate = report.get('success_rate', 0)
        passed_tests = report.get('passed_tests', 0)
        total_tests = report.get('total_tests', 0)
        
        if report.get('overall_success', False):
            print("🎉 SUCCESS! Enhanced verification completed successfully.")
            print(f"✅ Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests})")
            print()
            print("🚀 Your WAN 2.1 VACE 16B model is ready for:")
            print("   📹 Text-to-Video generation")
            print("   🖼️  Image-to-Video generation") 
            print("   🎬 VACE video editing")
            print("   🔄 ModelPatcher operations")
            print("   🧠 KSampler inference")
        else:
            print("⚠️  Enhanced verification completed with some issues.")
            print(f"📊 Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests})")
            print("🔧 Check the detailed report for specific issues")
        
        # Show key info
        if 'model_info' in report:
            info = report['model_info']
            print()
            print("📊 Model Information:")
            print(f"   Parameters: {info.get('total_parameters', 'Unknown'):,}")
            print(f"   Type: {info.get('model_type', 'Unknown')}")
            print(f"   VACE: {info.get('is_vace_model', False)}")
            print(f"   Loading: {info.get('loading_mode', 'Unknown')}")
            print(f"   Patcher: {info.get('uses_patcher', False)}")
        
        return 0 if report.get('overall_success', False) else 1
        
    except Exception as e:
        print(f"❌ Verification failed with error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
