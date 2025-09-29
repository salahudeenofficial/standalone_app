#!/usr/bin/env python3
"""
ComfyUI-style Memory Management with CPU-first loading and patcher assignment
Based on ComfyUI's actual model_patcher.py and model_management.py
"""

import torch
import logging
from enum import Enum
from typing import Optional, Dict, Any, Tuple, Union
from memory_utils import get_memory_info, clear_cuda_memory

class VRAMState(Enum):
    DISABLED = 0    # No vram present: no need to move models to vram
    NO_VRAM = 1     # Very low vram: enable all the options to save vram
    LOW_VRAM = 2
    NORMAL_VRAM = 3
    HIGH_VRAM = 4
    SHARED = 5      # No dedicated vram: memory shared between CPU and GPU

class LoadingMode(Enum):
    FULL_LOAD = "full_load"           # Complete loading to GPU
    PARTIAL_LOAD = "partial_load"     # Partial loading with dynamic weights
    CPU_ONLY = "cpu_only"            # CPU-only with dynamic loading

class ComfyUIPatcher:
    """
    ComfyUI-style patcher for model loading and memory management
    """
    
    def __init__(self, model, load_device, offload_device, size=0):
        self.model = model
        self.load_device = load_device
        self.offload_device = offload_device
        self.size = size
        self.patches = {}
        self.backup = {}
        self.model_options = {"transformer_options": {}}
        
        # ComfyUI-style attributes
        self.model.model_loaded_weight_memory = 0
        self.model.lowvram_patch_counter = 0
        self.model.model_lowvram = False
        self.model.current_weight_patches_uuid = None
        
        # Set initial device to CPU (ComfyUI approach)
        self.model.device = offload_device
        self.model.to(offload_device)
        
    def model_size(self):
        """Calculate model size"""
        if self.size > 0:
            return self.size
        self.size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        return self.size
        
    def loaded_size(self):
        """Get loaded model size"""
        return self.model.model_loaded_weight_memory
        
    def load(self, device_to, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        """
        ComfyUI-style model loading with partial loading support
        """
        logging.info(f"🔄 ComfyUI-style model loading to {device_to}")
        logging.info(f"  LowVRAM memory: {lowvram_model_memory / (1024**3):.2f} GB")
        logging.info(f"  Full load: {full_load}")
        
        if lowvram_model_memory == 0:
            full_load = True
            
        if full_load:
            # Complete loading
            self.model.to(device_to)
            self.model.device = device_to
            self.model.model_loaded_weight_memory = self.model_size()
            self.model.model_lowvram = False
            logging.info(f"✅ Complete loading to {device_to}")
        else:
            # Partial loading with dynamic weights
            self._partial_load(device_to, lowvram_model_memory, force_patch_weights)
            
    def _partial_load(self, device_to, lowvram_model_memory, force_patch_weights):
        """
        ComfyUI-style partial loading with dynamic weight functions
        """
        logging.info(f"🔄 Partial loading with {lowvram_model_memory / (1024**3):.2f} GB limit")
        
        # Get model modules sorted by size
        modules = []
        for name, module in self.model.named_modules():
            if len(list(module.parameters())) > 0:
                module_size = sum(p.numel() * p.element_size() for p in module.parameters())
                modules.append((module_size, name, module))
        
        modules.sort(reverse=True)  # Load largest modules first
        
        mem_counter = 0
        lowvram_counter = 0
        patch_counter = 0
        
        for module_size, name, module in modules:
            if mem_counter + module_size >= lowvram_model_memory:
                # Use dynamic loading for this module
                lowvram_counter += 1
                self._setup_dynamic_module(module, name)
                patch_counter += 1
            else:
                # Load module completely
                mem_counter += module_size
                module.to(device_to)
                
        self.model.device = device_to
        self.model.model_loaded_weight_memory = mem_counter
        self.model.model_lowvram = True
        self.model.lowvram_patch_counter = patch_counter
        
        logging.info(f"✅ Partial loading complete: {mem_counter / (1024**3):.2f} GB loaded, {lowvram_counter} modules dynamic")
        
    def _setup_dynamic_module(self, module, name):
        """
        Set up dynamic loading for a module (ComfyUI-style)
        """
        # Store original weights
        for param_name, param in module.named_parameters():
            key = f"{name}.{param_name}"
            if key not in self.backup:
                self.backup[key] = param.clone()
                
        # Set up weight functions for dynamic loading
        for param_name, param in module.named_parameters():
            key = f"{name}.{param_name}"
            if not hasattr(module, 'weight_function'):
                module.weight_function = []
            module.weight_function.append(self._create_weight_function(key))
            
    def _create_weight_function(self, key):
        """
        Create a weight function for dynamic loading
        """
        def weight_function():
            if key in self.backup:
                return self.backup[key]
            return None
        return weight_function
        
    def unload(self):
        """Unload model from GPU"""
        self.model.to(self.offload_device)
        self.model.device = self.offload_device
        self.model.model_loaded_weight_memory = 0
        self.model.model_lowvram = False
        self.model.lowvram_patch_counter = 0

class ComfyUIMemoryManager:
    """
    ComfyUI-style memory management with CPU-first loading and patcher assignment
    """
    
    def __init__(self):
        self.vram_state = VRAMState.NORMAL_VRAM
        self.total_vram = 0
        self.lowvram_available = True
        
        # Initialize memory info
        self._detect_system_capabilities()
        self._detect_vram_state()
        
    def _detect_system_capabilities(self):
        """Detect system capabilities like ComfyUI does"""
        if torch.cuda.is_available():
            self.total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)  # MB
        else:
            self.total_vram = 0
            
    def _detect_vram_state(self):
        """Detect VRAM state based on available memory - ComfyUI logic"""
        if not torch.cuda.is_available():
            self.vram_state = VRAMState.DISABLED
            return
            
        vram_gb = self.total_vram / 1024
        
        if vram_gb < 4:
            self.vram_state = VRAMState.NO_VRAM
        elif vram_gb < 8:
            self.vram_state = VRAMState.LOW_VRAM
        elif vram_gb < 16:
            self.vram_state = VRAMState.NORMAL_VRAM
        else:
            self.vram_state = VRAMState.HIGH_VRAM
            
        logging.info(f"Auto-detected VRAM state: {self.vram_state.name}")
        
    def get_torch_device(self):
        """Get the appropriate torch device"""
        if torch.cuda.is_available():
            return torch.device(torch.cuda.current_device())
        else:
            return torch.device("cpu")
            
    def create_model_patcher(self, model, state_dict=None):
        """
        Create a ComfyUI-style model patcher with CPU-first loading
        
        Args:
            model: PyTorch model
            state_dict: Optional state dict for memory estimation
            
        Returns:
            ComfyUIPatcher: Configured patcher
        """
        # Always start with CPU loading (ComfyUI approach)
        cpu_device = torch.device("cpu")
        gpu_device = self.get_torch_device()
        
        # Calculate model size
        if state_dict is not None:
            model_size = sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values())
        else:
            model_size = sum(p.numel() * p.element_size() for p in model.parameters())
            
        logging.info(f"🔧 Creating ComfyUI-style patcher")
        logging.info(f"  Model size: {model_size / (1024**3):.2f} GB")
        logging.info(f"  VRAM state: {self.vram_state.name}")
        
        # Create patcher with CPU-first loading
        patcher = ComfyUIPatcher(model, gpu_device, cpu_device, model_size)
        
        return patcher
        
    def determine_loading_strategy(self, model_size_gb, parameters, dtype):
        """
        Determine loading strategy based on ComfyUI's logic
        """
        logging.info(f"🎯 Determining loading strategy")
        logging.info(f"  Model size: {model_size_gb:.2f} GB")
        logging.info(f"  VRAM state: {self.vram_state.name}")
        
        if self.vram_state == VRAMState.DISABLED:
            return {
                'loading_mode': LoadingMode.CPU_ONLY,
                'reason': 'cuda_not_available',
                'lowvram_memory': 0
            }
            
        # Get available memory
        info = get_memory_info()
        available_memory_gb = info['cuda_free']
        
        # ComfyUI decision logic
        if self.vram_state == VRAMState.HIGH_VRAM:
            # High VRAM - try complete loading first
            if model_size_gb < available_memory_gb - 1.0:  # Reserve 1GB
                return {
                    'loading_mode': LoadingMode.FULL_LOAD,
                    'reason': 'high_vram_mode',
                    'lowvram_memory': 0
                }
            else:
                return {
                    'loading_mode': LoadingMode.PARTIAL_LOAD,
                    'reason': 'model_too_large_for_full_load',
                    'lowvram_memory': int((available_memory_gb - 1.0) * 1024**3)  # Convert to bytes
                }
        else:
            # Low/Normal VRAM - use partial loading
            reserved_memory_gb = 1.0
            usable_memory_gb = available_memory_gb - reserved_memory_gb
            
            if model_size_gb < usable_memory_gb:
                return {
                    'loading_mode': LoadingMode.FULL_LOAD,
                    'reason': 'fits_in_available_memory',
                    'lowvram_memory': 0
                }
            else:
                return {
                    'loading_mode': LoadingMode.PARTIAL_LOAD,
                    'reason': 'requires_partial_loading',
                    'lowvram_memory': int(usable_memory_gb * 1024**3)  # Convert to bytes
                }
                
    def load_model_with_patcher(self, model, state_dict=None):
        """
        Load model with ComfyUI-style patcher and automatic strategy selection
        
        Args:
            model: PyTorch model
            state_dict: Optional state dict for memory estimation
            
        Returns:
            tuple: (model, patcher, loading_info)
        """
        # Create patcher with CPU-first loading
        patcher = self.create_model_patcher(model, state_dict)
        
        # Determine loading strategy
        if state_dict is not None:
            model_size_gb = sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values()) / (1024**3)
            parameters = sum(tensor.numel() for tensor in state_dict.values())
        else:
            model_size_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)
            parameters = sum(p.numel() for p in model.parameters())
            
        strategy = self.determine_loading_strategy(model_size_gb, parameters, torch.float16)
        
        # Apply loading strategy
        if strategy['loading_mode'] == LoadingMode.FULL_LOAD:
            patcher.load(patcher.load_device, full_load=True)
        elif strategy['loading_mode'] == LoadingMode.PARTIAL_LOAD:
            patcher.load(patcher.load_device, lowvram_model_memory=strategy['lowvram_memory'])
        else:  # CPU_ONLY
            patcher.load(patcher.offload_device, full_load=True)
            
        loading_info = {
            'loading_mode': strategy['loading_mode'].value,
            'reason': strategy['reason'],
            'vram_state': self.vram_state.name,
            'model_size_gb': model_size_gb,
            'lowvram_memory_gb': strategy['lowvram_memory'] / (1024**3) if strategy['lowvram_memory'] > 0 else 0
        }
        
        return model, patcher, loading_info

def test_comfyui_patcher_system():
    """Test ComfyUI-style patcher system"""
    print("🧪 Testing ComfyUI-style Patcher System")
    print("=" * 60)
    
    # Initialize memory manager
    mem_manager = ComfyUIMemoryManager()
    
    # Test with your 32GB WAN model
    print(f"\n🎯 Testing with WAN 2.1 VACE 16B Model:")
    print(f"  Parameters: 17,337,592,896")
    print(f"  Dtype: torch.float16")
    print(f"  Expected size: ~32.29 GB")
    
    # Create a simple dummy model for testing
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = torch.nn.Linear(1000, 1000)
            self.layer2 = torch.nn.Linear(1000, 1000)
            
        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            return x
    
    # Create dummy model
    dummy_model = DummyModel()
    
    # Test patcher creation and loading
    print(f"\n🚀 Testing Patcher Creation and Loading:")
    
    try:
        model, patcher, loading_info = mem_manager.load_model_with_patcher(dummy_model)
        
        print(f"  Loading mode: {loading_info['loading_mode']}")
        print(f"  Reason: {loading_info['reason']}")
        print(f"  VRAM state: {loading_info['vram_state']}")
        print(f"  Model size: {loading_info['model_size_gb']:.2f} GB")
        print(f"  LowVRAM memory: {loading_info['lowvram_memory_gb']:.2f} GB")
        
        print(f"  ✅ SUCCESS: Model loaded with ComfyUI-style patcher!")
        
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
    
    # Test with different VRAM states
    print(f"\n🔬 Testing different VRAM states:")
    original_vram_state = mem_manager.vram_state
    
    for vram_state in [VRAMState.NO_VRAM, VRAMState.LOW_VRAM, VRAMState.NORMAL_VRAM, VRAMState.HIGH_VRAM]:
        mem_manager.vram_state = vram_state
        strategy = mem_manager.determine_loading_strategy(32.29, 17_337_592_896, torch.float16)
        print(f"  {vram_state.name}: {strategy['loading_mode'].value} -> {strategy['reason']}")
    
    mem_manager.vram_state = original_vram_state

if __name__ == "__main__":
    test_comfyui_patcher_system()
