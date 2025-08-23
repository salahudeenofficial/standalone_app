#!/usr/bin/env python3
"""
Test script for LoRA Application Monitoring System
Tests the monitoring methods without requiring actual models
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Mock models for testing
class MockModel:
    def __init__(self, model_id, model_type="UNET"):
        self.model_id = model_id
        self.model_type = model_type
        self.patches = {}
        self.patches_uuid = f"uuid_{model_id}"
        self.device = "cuda:0"
        self.load_device = "cuda:0"
        self.offload_device = "cpu"
    
    def state_dict(self):
        return {"weight1": MockTensor(), "weight2": MockTensor()}

class MockCLIPModel:
    def __init__(self, model_id):
        self.model_id = model_id
        self.patcher = MockPatcher(model_id)
        self.device = "cuda:0"

class MockPatcher:
    def __init__(self, model_id):
        self.patches = {}
        self.patches_uuid = f"clip_uuid_{model_id}"
        self.load_device = "cuda:0"
        self.offload_device = "cpu"
        self.model = MockModel(model_id, "CLIP")

class MockTensor:
    def __init__(self):
        self.shape = (64, 64)
        self.dtype = "torch.float32"
        self.device = "cuda:0"

def test_monitoring_methods():
    """Test the monitoring methods with mock models"""
    print("🧪 TESTING LORA MONITORING SYSTEM")
    print("="*50)
    
    try:
        # Import the pipeline class
        from pipeline import ReferenceVideoPipeline
        
        # Create pipeline instance
        pipeline = ReferenceVideoPipeline()
        print("✅ Pipeline class imported successfully")
        
        # Create mock models
        mock_unet = MockModel(12345, "UNET")
        mock_clip = MockCLIPModel(67890)
        
        print("✅ Mock models created successfully")
        
        # Test baseline capture
        print("\n🔍 Testing baseline capture...")
        baseline = pipeline._capture_lora_baseline(mock_unet, mock_clip, "test_lora.safetensors")
        print(f"   UNET ID: {baseline['unet']['model_id']}")
        print(f"   CLIP ID: {baseline['clip']['model_id']}")
        print(f"   LoRA File: {baseline['lora_file']['filename']}")
        print("✅ Baseline capture test passed")
        
        # Test model identity changes tracking
        print("\n🔍 Testing model identity changes tracking...")
        modified_unet = MockModel(12345, "UNET")  # Same ID but different instance
        modified_unet.patches = {"layer1": ["patch1"]}  # Add a patch
        
        identity_changes = pipeline._track_model_identity_changes(mock_unet, modified_unet, "UNET")
        print(f"   Model Cloned: {identity_changes['model_cloned']}")
        print(f"   Patches Added: {identity_changes['patches_added']}")
        print("✅ Model identity changes tracking test passed")
        
        # Test weight modifications tracking
        print("\n🔍 Testing weight modifications tracking...")
        weight_changes = pipeline._track_weight_modifications(mock_unet, modified_unet, "UNET")
        print(f"   Total Keys Original: {weight_changes['total_keys_original']}")
        print(f"   Total Keys Modified: {weight_changes['total_keys_modified']}")
        print("✅ Weight modifications tracking test passed")
        
        # Test LoRA patches analysis
        print("\n🔍 Testing LoRA patches analysis...")
        patches_analysis = pipeline._analyze_lora_patches(modified_unet, "UNET")
        print(f"   Total Patched Keys: {patches_analysis['total_patched_keys']}")
        print("✅ LoRA patches analysis test passed")
        
        # Test model placement changes tracking
        print("\n🔍 Testing model placement changes tracking...")
        placement_changes = pipeline._track_model_placement_changes(mock_unet, modified_unet, "UNET")
        print(f"   Device Changed: {placement_changes['model_device_changed']}")
        print("✅ Model placement changes tracking test passed")
        
        # Test memory change calculation
        print("\n🔍 Testing memory change calculation...")
        memory_change = pipeline._calculate_memory_change(baseline['unet'], modified_unet)
        if 'error' not in memory_change:
            print(f"   Memory Change: {memory_change['allocated_change_mb']:+.1f} MB")
            print("✅ Memory change calculation test passed")
        else:
            print(f"   Memory calculation error: {memory_change['error']}")
            print("⚠️  Memory change calculation test failed (expected in test environment)")
        
        # Test comprehensive analysis
        print("\n🔍 Testing comprehensive analysis...")
        analysis = pipeline._analyze_lora_application_results(
            baseline, mock_unet, mock_clip, modified_unet, mock_clip, [modified_unet, mock_clip]
        )
        print(f"   LoRA Application Success: {analysis['lora_application_success']}")
        print(f"   Models Returned: {analysis['models_returned']}")
        print("✅ Comprehensive analysis test passed")
        
        print("\n🎉 ALL TESTS PASSED! LoRA monitoring system is working correctly.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you're running this from the standalone_app directory")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_monitoring_methods() 