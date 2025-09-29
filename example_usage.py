"""
Example usage of the standalone ModelPatcher with UNet loading
"""

import torch
import torch.nn as nn
from standalone_model_patcher import ModelPatcher, create_model_patcher, load_model_to_device, unload_model_from_device


def load_unet_only_standalone(unet_state_dict, model_options={}, metadata=None):
    """
    Minimal function to load only the UNet from a state dictionary using standalone ModelPatcher.
    Returns the ModelPatcher object for the loaded UNet.
    """
    # For this example, we'll create a simple UNet-like model
    # In practice, you would use your actual model detection and loading logic
    
    class SimpleUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(4, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
            self.conv4 = nn.Conv2d(64, 4, 3, padding=1)
            self.device = torch.device("cpu")
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = self.conv4(x)
            return x
        
        def memory_required(self, input_shape):
            # Simple memory estimation
            return self.conv1.weight.numel() * 4 + self.conv2.weight.numel() * 4
    
    # Create the model
    model = SimpleUNet()
    
    # Load weights from state dict (simplified)
    if unet_state_dict:
        model.load_state_dict(unet_state_dict, strict=False)
    
    # Create ModelPatcher
    load_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    offload_device = torch.device("cpu")
    
    model_patcher = ModelPatcher(
        model, 
        load_device=load_device, 
        offload_device=offload_device
    )
    
    # Load to GPU if not CPU
    if load_device != torch.device("cpu"):
        print("Loading diffusion model directly to GPU")
        load_model_to_device(model_patcher, force_full_load=True)
    
    return model_patcher


def main():
    """Example usage"""
    print("=== Standalone ModelPatcher Example ===")
    
    # Create some dummy state dict
    dummy_state_dict = {
        "conv1.weight": torch.randn(64, 4, 3, 3),
        "conv1.bias": torch.randn(64),
        "conv2.weight": torch.randn(128, 64, 3, 3),
        "conv2.bias": torch.randn(128),
        "conv3.weight": torch.randn(64, 128, 3, 3),
        "conv3.bias": torch.randn(64),
        "conv4.weight": torch.randn(4, 64, 3, 3),
        "conv4.bias": torch.randn(4),
    }
    
    # Load UNet using standalone ModelPatcher
    print("Loading UNet with standalone ModelPatcher...")
    model_patcher = load_unet_only_standalone(dummy_state_dict)
    
    print(f"Model size: {model_patcher.model_size() / (1024*1024):.2f} MB")
    print(f"Model loaded to: {model_patcher.current_loaded_device()}")
    
    # Test model inference
    print("\nTesting model inference...")
    with torch.no_grad():
        dummy_input = torch.randn(1, 4, 64, 64)
        output = model_patcher.model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
    
    # Add some patches (LoRA-like)
    print("\nAdding patches...")
    patches = {
        "conv1.weight": torch.randn(64, 4, 3, 3) * 0.1,
        "conv2.weight": torch.randn(128, 64, 3, 3) * 0.1,
    }
    patcher.add_patches(patches, strength_patch=0.5)
    print("Patches added successfully")
    
    # Test with patches
    print("\nTesting with patches...")
    with torch.no_grad():
        output_patched = model_patcher.model(dummy_input)
        print(f"Output with patches shape: {output_patched.shape}")
    
    # Memory management
    print("\nTesting memory management...")
    print(f"Loaded memory: {model_patcher.loaded_size() / (1024*1024):.2f} MB")
    
    # Partial unload
    memory_freed = model_patcher.partially_unload(torch.device("cpu"), memory_to_free=1024*1024)
    print(f"Memory freed: {memory_freed / (1024*1024):.2f} MB")
    
    # Partial load
    memory_used = model_patcher.partially_load(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Memory used: {memory_used / (1024*1024):.2f} MB")
    
    # Cleanup
    print("\nCleaning up...")
    unload_model_from_device(model_patcher)
    model_patcher.cleanup()
    print("Cleanup complete")
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()
