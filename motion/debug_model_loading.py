#!/usr/bin/env python3
"""
Debug what happens during model loading that affects process_input
"""

import torch
from standalone_vae import VAE

def debug_step_by_step():
    print("🔍 DEBUGGING MODEL LOADING STEP BY STEP")
    print("=" * 60)
    
    # Create test tensor
    test_input = torch.tensor([[[[[1.0, 2.0], [3.0, 4.0]]]]])
    expected = torch.tensor([[[[[1.0, 3.0], [5.0, 7.0]]]]])  # x*2-1
    
    # Step 1: Test VAE before any loading
    print("\n1. BEFORE ANY MODEL LOADING:")
    vae = VAE(sd={})
    result1 = vae.process_input(test_input)
    match1 = torch.allclose(result1, expected, atol=1e-6)
    print(f"   result: {result1.flatten().tolist()}")
    print(f"   match: {match1}")
    
    # Step 2: Add WAN VAE detection but minimal state dict
    print("\n2. WITH WAN DETECTION BUT MINIMAL LOADING:")
    minimal_keys = {
        'decoder.conv1.weight': torch.randn(384, 16, 3, 3, 3),
        'decoder.conv1.bias': torch.randn(384)
    }
    
    try:
        vae2 = VAE(sd=minimal_keys)
        result2 = vae2.process_input(test_input)
        match2 = torch.allclose(result2, expected, atol=1e-6)
        print(f"   result: {result2.flatten().tolist()}")
        print(f"   match: {match2}")
        print(f"   first_stage_model: {type(vae2.first_stage_model).__name__}")
        
        # Now check what happens if we try to load more keys
        print("\n3. TESTING WITH MORE WAN KEYS:")
        more_keys = minimal_keys.copy()
        more_keys.update({
            'encoder.conv1.weight': torch.randn(3, 96, 3, 3, 3),
            'encoder.conv1.bias': torch.randn(96),
            'decoder.head.0.gamma': torch.randn(384),
            'decoder.head.2.weight': torch.randn(3, 96, 3, 3, 3),
            'decoder.head.2.bias': torch.randn(3)
        })
        
        # See what happens with more keys - but avoid the loading error
        vae3_obj = VAE.__new__(VAE)  # Create without calling __init__
        vae3_obj.__dict__.update(vae2.__dict__)  # Copy from working instance
        
        # Simulate what happens in _detect_and_init_vae with more keys
        print("   Simulating detection with more keys...")
        sd = more_keys
        
        if "decoder.head.0.gamma" in sd or "decoder.conv1.weight" in sd:
            print("   ✅ WAN VAE detection triggered")
            # The detection path should be the same
            print(f"   process_input before: {vae3_obj.process_input}")
            
            # Reset process_input to make sure it's right
            vae3_obj.process_input = lambda image: image * 2.0 - 1.0
            result3 = vae3_obj.process_input(test_input)
            match3 = torch.allclose(result3, expected, atol=1e-6)
            print(f"   result after reset: {result3.flatten().tolist()}")
            print(f"   match after reset: {match3}")
        
    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_step_by_step()
