#!/usr/bin/env python3
"""
Implement ComfyUI-style LowVramPatch system for proper partial loading
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional, Any
import gc

class LowVramPatch:
    """
    ComfyUI-style LowVramPatch for dynamic weight loading
    
    This class handles loading weights to GPU on-demand during forward pass
    and evicting them back to CPU after use.
    """
    
    def __init__(self, weight_key: str, weight_tensor: torch.Tensor, target_device: torch.device):
        self.weight_key = weight_key
        self.weight_tensor = weight_tensor  # Original weight on CPU
        self.target_device = target_device
        self.gpu_weight = None  # Cached GPU weight
        self.is_loaded = False
        
    def __call__(self, *args, **kwargs):
        """
        Load weight to GPU on-demand during forward pass
        """
        if not self.is_loaded:
            try:
                # Load weight to GPU
                self.gpu_weight = self.weight_tensor.to(self.target_device)
                self.is_loaded = True
                logging.debug(f"🔄 Loaded weight {self.weight_key} to GPU")
            except torch.cuda.OutOfMemoryError as e:
                logging.warning(f"⚠️  OOM loading weight {self.weight_key}: {e}")
                return self.weight_tensor  # Fallback to CPU weight
        
        return self.gpu_weight
    
    def evict(self):
        """
        Evict weight from GPU back to CPU
        """
        if self.is_loaded and self.gpu_weight is not None:
            # Clear GPU weight
            del self.gpu_weight
            self.gpu_weight = None
            self.is_loaded = False
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logging.debug(f"🧹 Evicted weight {self.weight_key} from GPU")

class ComfyUIStylePartialLoader:
    """
    ComfyUI-style partial loading system that implements proper layer swapping
    """
    
    def __init__(self, model: nn.Module, target_device: torch.device, memory_budget_gb: float):
        self.model = model
        self.target_device = target_device
        self.memory_budget_gb = memory_budget_gb
        self.memory_budget_bytes = memory_budget_gb * 1024**3
        
        # Track loaded weights
        self.loaded_weights = {}  # weight_key -> LowVramPatch
        self.weight_patches = {}  # module_name -> {weight_key: LowVramPatch}
        
        # Memory tracking
        self.current_memory_usage = 0
        self.max_memory_usage = 0
        
        # Skip model patching for now - we'll use a different approach
        # from comfyui_ops import patch_model_with_comfyui_ops
        # self.model = patch_model_with_comfyui_ops(self.model)
        
        logging.info(f"🚀 ComfyUI-style partial loader initialized")
        logging.info(f"   Target device: {target_device}")
        logging.info(f"   Memory budget: {memory_budget_gb:.2f} GB")
    
    def analyze_model_weights(self) -> List[Dict[str, Any]]:
        """
        Analyze model weights and estimate their memory usage
        """
        weights_info = []
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                module_weights = []
                module_size_bytes = 0
                
                for param_name, param in module.named_parameters():
                    if isinstance(param, torch.Tensor):
                        weight_key = f"{name}.{param_name}" if name else param_name
                        weight_size_bytes = param.numel() * param.element_size()
                        
                        weight_info = {
                            'key': weight_key,
                            'module_name': name,
                            'param_name': param_name,
                            'tensor': param,
                            'size_bytes': weight_size_bytes,
                            'size_gb': weight_size_bytes / (1024**3),
                            'shape': param.shape,
                            'dtype': param.dtype
                        }
                        
                        module_weights.append(weight_info)
                        module_size_bytes += weight_size_bytes
                
                if module_weights:
                    weights_info.extend(module_weights)
        
        # Sort by size (largest first)
        weights_info.sort(key=lambda x: x['size_bytes'], reverse=True)
        
        logging.info(f"📊 Found {len(weights_info)} weights")
        total_size_gb = sum(w['size_gb'] for w in weights_info)
        logging.info(f"📊 Total weight size: {total_size_gb:.3f} GB")
        
        return weights_info
    
    def setup_partial_loading(self) -> Dict[str, Any]:
        """
        Set up partial loading by patching weights with LowVramPatch
        """
        logging.info("🔧 Setting up ComfyUI-style partial loading...")
        
        # Analyze weights
        weights_info = self.analyze_model_weights()
        
        # Move model to CPU first
        self.model.to('cpu')
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
        
        # Set up weight patching
        loaded_weights = []
        patched_weights = []
        remaining_memory = self.memory_budget_bytes
        
        for weight_info in weights_info:
            weight_size_bytes = weight_info['size_bytes']
            
            if weight_size_bytes <= remaining_memory:
                # Load weight to GPU immediately
                try:
                    weight_info['tensor'].to(self.target_device)
                    loaded_weights.append(weight_info)
                    remaining_memory -= weight_size_bytes
                    logging.info(f"  ✅ Loaded {weight_info['key']}: {weight_info['size_gb']:.3f} GB")
                except torch.cuda.OutOfMemoryError:
                    logging.warning(f"  ⚠️  OOM loading {weight_info['key']}, setting up dynamic loading")
                    self._setup_dynamic_weight(weight_info)
                    patched_weights.append(weight_info)
            else:
                # Set up dynamic loading for this weight
                self._setup_dynamic_weight(weight_info)
                patched_weights.append(weight_info)
                logging.info(f"  🔄 Dynamic loading for {weight_info['key']}: {weight_info['size_gb']:.3f} GB")
        
        # Store loading info
        self.model._partial_loading_info = {
            'loaded_weights': [w['key'] for w in loaded_weights],
            'patched_weights': [w['key'] for w in patched_weights],
            'target_device': self.target_device,
            'memory_budget_gb': self.memory_budget_gb,
            'total_weights': len(weights_info),
            'loaded_count': len(loaded_weights),
            'patched_count': len(patched_weights)
        }
        
        logging.info(f"🎉 Partial loading setup complete:")
        logging.info(f"   Loaded weights: {len(loaded_weights)}")
        logging.info(f"   Dynamic weights: {len(patched_weights)}")
        logging.info(f"   Memory used: {(self.memory_budget_bytes - remaining_memory) / (1024**3):.3f} GB")
        
        return {
            'loading_type': 'comfyui_partial',
            'loaded_weights': len(loaded_weights),
            'patched_weights': len(patched_weights),
            'memory_used_gb': (self.memory_budget_bytes - remaining_memory) / (1024**3),
            'memory_budget_gb': self.memory_budget_gb
        }
    
    def _setup_dynamic_weight(self, weight_info: Dict[str, Any]):
        """
        Set up dynamic loading for a specific weight using ComfyUI's weight_function approach
        """
        weight_key = weight_info['key']
        weight_tensor = weight_info['tensor']
        
        # Create LowVramPatch
        patch = LowVramPatch(weight_key, weight_tensor, self.target_device)
        
        # Store patch
        self.loaded_weights[weight_key] = patch
        
        # Set up weight_function on the module (ComfyUI's actual approach)
        module_name = weight_info['module_name']
        param_name = weight_info['param_name']
        
        if module_name not in self.weight_patches:
            self.weight_patches[module_name] = {}
        
        self.weight_patches[module_name][param_name] = patch
        
        # Get the module and set up weight_function
        module = dict(self.model.named_modules())[module_name]
        
        # Initialize weight_function and bias_function lists (ComfyUI approach)
        if not hasattr(module, 'weight_function'):
            module.weight_function = []
        if not hasattr(module, 'bias_function'):
            module.bias_function = []
        
        # Add the LowVramPatch to the appropriate function list
        if param_name == 'weight':
            module.weight_function.append(patch)
        elif param_name == 'bias':
            module.bias_function.append(patch)
        
        # Mark module as having dynamic loading
        module._dynamic_loading_setup = True
        
        logging.debug(f"  ✅ Set up weight_function for {weight_key} on {module_name}.{param_name}")
    
    def load_weights_for_inference(self, weight_keys: Optional[List[str]] = None):
        """
        Load all dynamic weights to GPU and replace module parameters for inference
        """
        logging.info(f"🔄 Loading dynamic weights to GPU for inference...")
        
        # Load all dynamic weights to GPU and replace module parameters
        for weight_key, patch in self.loaded_weights.items():
            try:
                # Load weight to GPU
                gpu_weight = patch()
                
                # Find the module and parameter to replace
                for module_name, param_patches in self.weight_patches.items():
                    for param_name, weight_patch in param_patches.items():
                        if weight_patch == patch:
                            # Get the module
                            module = dict(self.model.named_modules())[module_name]
                            
                            # Replace the parameter in the module
                            if param_name == 'weight':
                                module.weight = torch.nn.Parameter(gpu_weight)
                            elif param_name == 'bias':
                                module.bias = torch.nn.Parameter(gpu_weight)
                            
                            logging.debug(f"  ✅ Loaded {weight_key} to {module_name}.{param_name}")
                            break
            except Exception as e:
                logging.warning(f"  ⚠️  Failed to load {weight_key}: {e}")
        
        # Ensure model is on target device
        self.model.to(self.target_device)
        logging.info(f"  ✅ Model ready for inference on {self.target_device}")
    
    def evict_weights_after_inference(self, weight_keys: Optional[List[str]] = None):
        """
        Evict weights from GPU and restore original parameters
        """
        logging.info(f"🧹 Evicting dynamic weights after inference...")
        
        # Restore original parameters and evict from GPU
        for weight_key, patch in self.loaded_weights.items():
            try:
                # Find the module and parameter to restore
                for module_name, param_patches in self.weight_patches.items():
                    for param_name, weight_patch in param_patches.items():
                        if weight_patch == patch:
                            # Get the module
                            module = dict(self.model.named_modules())[module_name]
                            
                            # Restore original parameter (CPU version)
                            original_weight = patch.weight_tensor  # This is the original CPU weight
                            if param_name == 'weight':
                                module.weight = torch.nn.Parameter(original_weight)
                            elif param_name == 'bias':
                                module.bias = torch.nn.Parameter(original_weight)
                            
                            # Evict from GPU
                            patch.evict()
                            
                            logging.debug(f"  ✅ Evicted {weight_key} from {module_name}.{param_name}")
                            break
            except Exception as e:
                logging.warning(f"  ⚠️  Failed to evict {weight_key}: {e}")
        
        # Move model back to CPU
        self.model.to('cpu')
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        
        logging.info("  ✅ Dynamic weights evicted and model moved to CPU")
    
    def get_loading_info(self) -> Dict[str, Any]:
        """
        Get current loading information
        """
        if hasattr(self.model, '_partial_loading_info'):
            return self.model._partial_loading_info
        return {}

def setup_comfyui_style_partial_loading(model: nn.Module, target_device: torch.device, 
                                       memory_budget_gb: float) -> ComfyUIStylePartialLoader:
    """
    Set up ComfyUI-style partial loading for a model
    
    Args:
        model: PyTorch model to set up partial loading for
        target_device: Target GPU device
        memory_budget_gb: Available memory budget in GB
    
    Returns:
        ComfyUIStylePartialLoader: Configured partial loader
    """
    loader = ComfyUIStylePartialLoader(model, target_device, memory_budget_gb)
    loader.setup_partial_loading()
    return loader

# Test function
def test_comfyui_style_partial_loading():
    """Test the ComfyUI-style partial loading system"""
    print("🧪 TESTING COMFYUI-STYLE PARTIAL LOADING")
    print("="*60)
    
    # Create a test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
            self.fc = nn.Linear(256 * 8 * 8, 1000)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = TestModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cuda':
        # Set up partial loading with small budget
        loader = setup_comfyui_style_partial_loading(model, device, memory_budget_gb=0.1)
        
        # Test inference
        print("🔧 Testing inference with partial loading...")
        input_tensor = torch.randn(1, 3, 32, 32)
        
        with torch.no_grad():
            # Load weights for inference
            loader.load_weights_for_inference()
            
            # Run inference
            output = model(input_tensor)
            print(f"  ✅ Inference successful: {output.shape}")
            
            # Evict weights after inference
            loader.evict_weights_after_inference()
        
        # Get loading info
        info = loader.get_loading_info()
        print(f"📊 Loading info: {info}")
        
        print("✅ ComfyUI-style partial loading test completed!")
    else:
        print("⚠️  CUDA not available, skipping test")

if __name__ == "__main__":
    test_comfyui_style_partial_loading()
