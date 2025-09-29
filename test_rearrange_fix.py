#!/usr/bin/env python3
"""Test rearrange fix"""

import sys
import os
sys.path.append('wan_vae_components')

from einops_replacement import rearrange
import torch

def test_rearrange_with_kwargs():
    """Test rearrange with keyword arguments"""
    print("🧪 Testing rearrange with keyword arguments...")
    
    # Test tensor: (b*t=8, c=3, h=32, w=32)
    tensor = torch.randn(8, 3, 32, 32)
    print(f"Input tensor shape: {tensor.shape}")
    
    # Test pattern 1: (b t) c h w -> b c t h w with t=t
    try:
        t_val = 4
        result1 = rearrange(tensor, '(b t) c h w -> b c t h w', t=t_val)
        print(f"✅ Pattern 1 success: {tensor.shape} -> {result1.shape}")
    except Exception as e:
        print(f"❌ Pattern 1 failed: {e}")
        return False
    
    # Test pattern 2: b c t h w -> (b t) c h w
    try:
        tensor2 = torch.randn(2, 3, 4, 32, 32)
        result2 = rearrange(tensor2, 'b c t h w -> (b t) c h w')
        print(f"✅ Pattern 2 success: {tensor2.shape} -> {result2.shape}")
    except Exception as e:
        print(f"❌ Pattern 2 failed: {e}")
        return False
    
    print("🎉 All rearrange tests passed!")
    return True

if __name__ == "__main__":
    success = test_rearrange_with_kwargs()
    sys.exit(0 if success else 1)
