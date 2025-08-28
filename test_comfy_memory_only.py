#!/usr/bin/env python3
"""
Focused Test Script for ComfyUI Memory Management Functions
Tests only the 3 critical functions without running the full pipeline
"""

import os
import sys
import argparse
from pathlib import Path

# Set required environment variables BEFORE importing ComfyUI
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Create minimal CLI args that ComfyUI expects
sys.argv = ['test_comfy_memory_only.py', '--cpu-vae']

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "comfy"))

print("🔧 Testing ComfyUI Memory Management Functions Only")
print("="*60)

# Test 1: Basic imports
print("\n📦 TEST 1: Basic ComfyUI Imports")
try:
    import comfy.cli_args
    print("   ✅ comfy.cli_args imported successfully")
except Exception as e:
    print(f"   ❌ comfy.cli_args import failed: {e}")
    sys.exit(1)

try:
    import comfy.model_management
    print("   ✅ comfy.model_management imported successfully")
except Exception as e:
    print(f"   ❌ comfy.model_management import failed: {e}")
    sys.exit(1)

try:
    import comfy.model_patcher
    print("   ✅ comfy.model_patcher imported successfully")
except Exception as e:
    print(f"   ❌ comfy.model_patcher import failed: {e}")
    sys.exit(1)

# Test 2: Device detection
print("\n🔍 TEST 2: Device Detection")
try:
    device = comfy.model_management.get_torch_device()
    print(f"   ✅ get_torch_device(): {device}")
except Exception as e:
    print(f"   ❌ get_torch_device() failed: {e}")

try:
    vae_device = comfy.model_management.vae_device()
    print(f"   ✅ vae_device(): {vae_device}")
except Exception as e:
    print(f"   ❌ vae_device() failed: {e}")

# Test 3: Memory functions
print("\n💾 TEST 3: Memory Management Functions")
try:
    device = comfy.model_management.get_torch_device()
    free_memory = comfy.model_management.get_free_memory(device)
    print(f"   ✅ get_free_memory(): {free_memory / (1024**2):.1f} MB")
except Exception as e:
    print(f"   ❌ get_free_memory() failed: {e}")

try:
    comfy.model_management.load_models_gpu([], memory_required=0)
    print("   ✅ load_models_gpu() with empty list: SUCCESS")
except Exception as e:
    print(f"   ❌ load_models_gpu() failed: {e}")

try:
    unloaded = comfy.model_management.free_memory(1024*1024, device)  # 1MB
    print(f"   ✅ free_memory(): {len(unloaded)} models unloaded")
except Exception as e:
    print(f"   ❌ free_memory() failed: {e}")

# Test 4: VRAM state
print("\n🎯 TEST 4: VRAM State Management")
try:
    if hasattr(comfy.model_management, 'VRAMState'):
        print(f"   ✅ VRAMState enum available: {comfy.model_management.VRAMState.NORMAL_VRAM}")
    else:
        print("   ❌ VRAMState enum not found")
except Exception as e:
    print(f"   ❌ VRAMState test failed: {e}")

try:
    if hasattr(comfy.model_management, 'CPUState'):
        print(f"   ✅ CPUState enum available: {comfy.model_management.CPUState.GPU}")
    else:
        print("   ❌ CPUState enum not found")
except Exception as e:
    print(f"   ❌ CPUState test failed: {e}")

# Test 5: Model tracking
print("\n📋 TEST 5: Model Tracking System")
try:
    if hasattr(comfy.model_management, 'current_loaded_models'):
        print(f"   ✅ current_loaded_models exists: {len(comfy.model_management.current_loaded_models)} models")
    else:
        print("   ❌ current_loaded_models not found")
except Exception as e:
    print(f"   ❌ Model tracking test failed: {e}")

# Test 6: LoadedModel class
print("\n🏗️  TEST 6: LoadedModel Class")
try:
    if hasattr(comfy.model_management, 'LoadedModel'):
        print("   ✅ LoadedModel class available")
    else:
        print("   ❌ LoadedModel class not found")
except Exception as e:
    print(f"   ❌ LoadedModel test failed: {e}")

print("\n" + "="*60)
print("🧪 ComfyUI Memory Management Test Complete!")
print("="*60)

# Summary
print("\n📊 SUMMARY:")
print("   If you see mostly ✅ marks, ComfyUI integration is working")
print("   If you see ❌ marks, those specific functions need fixing")
print("   Check the error messages above for specific issues") 