#!/usr/bin/env python3
"""
Test model-aware patching system
"""

import torch
import torch.nn as nn
import logging
from model_aware_patcher import ModelAwarePatcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_aware_patching():
    """Test model-aware patching with different model types"""
    print("🧪 Testing Model-Aware Patching System...")
    
    # Create a mock VaceWanModel-like structure
    class MockVaceBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.ModuleDict({
                'q': nn.Linear(512, 512),
                'k': nn.Linear(512, 512),
                'v': nn.Linear(512, 512),
                'o': nn.Linear(512, 512),
                'norm_q': nn.LayerNorm(512),
                'norm_k': nn.LayerNorm(512)
            })
            self.cross_attn = nn.ModuleDict({
                'q': nn.Linear(512, 512),
                'k': nn.Linear(512, 512),
                'v': nn.Linear(512, 512),
                'o': nn.Linear(512, 512)
            })
            self.mlp = nn.ModuleDict({
                'fc1': nn.Linear(512, 2048),
                'fc2': nn.Linear(2048, 512)
            })
            self.norm1 = nn.LayerNorm(512)
            self.norm2 = nn.LayerNorm(512)
            self.norm3 = nn.LayerNorm(512)
    
    class MockVaceWanModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.vace_blocks = nn.ModuleList([
                MockVaceBlock() for _ in range(4)
            ])
            self.head = nn.Linear(512, 10)
    
    # Test with mock VaceWanModel
    print("\n🔧 Testing with MockVaceWanModel...")
    model = MockVaceWanModel()
    
    # Count original layers
    original_linear_count = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    original_layernorm_count = sum(1 for m in model.modules() if isinstance(m, nn.LayerNorm))
    
    print(f"📊 Original model: {original_linear_count} Linear, {original_layernorm_count} LayerNorm layers")
    
    # Patch the model
    patcher = ModelAwarePatcher()
    patched_model = patcher.patch_model(model)
    
    # Count patched layers
    patched_linear_count = sum(1 for m in patched_model.modules() if hasattr(m, 'weight_function'))
    patched_layernorm_count = sum(1 for m in patched_model.modules() if hasattr(m, 'weight_function') and isinstance(m, nn.LayerNorm))
    
    print(f"📊 Patched model: {patched_linear_count} Linear, {patched_layernorm_count} LayerNorm layers with weight_function")
    
    # Test with generic model
    print("\n🔧 Testing with generic model...")
    generic_model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    original_count = sum(1 for m in generic_model.modules() if isinstance(m, nn.Linear))
    patched_generic = patcher.patch_model(generic_model)
    patched_count = sum(1 for m in patched_generic.modules() if hasattr(m, 'weight_function'))
    
    print(f"📊 Generic model: {original_count} → {patched_count} patched layers")
    
    print("\n✅ Model-aware patching test completed!")

if __name__ == "__main__":
    test_model_aware_patching()
