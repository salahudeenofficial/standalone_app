#!/usr/bin/env python3
"""
ComfyUI-style memory management and auto mode implementation
Based on ComfyUI's actual model_management.py logic
"""

import torch
import psutil
import logging
from enum import Enum
from typing import Optional, Dict, Any, Tuple

class VRAMState(Enum):
    DISABLED = 0    # No vram present: no need to move models to vram
    NO_VRAM = 1     # Very low vram: enable all the options to save vram
    LOW_VRAM = 2
    NORMAL_VRAM = 3
    HIGH_VRAM = 4
    SHARED = 5      # No dedicated vram: memory shared between CPU and GPU

class CPUState(Enum):
    GPU = 0
    CPU = 1
    MPS = 2

class ComfyUIMemoryManager:
    """
    ComfyUI-style memory management with automatic VRAM state detection
    """
    
    def __init__(self):
        self.vram_state = VRAMState.NORMAL_VRAM
        self.cpu_state = CPUState.GPU
        self.total_vram = 0
        self.total_ram = 0
        self.lowvram_available = True
        self.disable_smart_memory = False
        
        # Initialize memory info
        self._detect_system_capabilities()
        self._detect_vram_state()
        
    def _detect_system_capabilities(self):
        """Detect system capabilities like ComfyUI does"""
        if torch.cuda.is_available():
            self.cpu_state = CPUState.GPU
            self.total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)  # MB
        else:
            self.cpu_state = CPUState.CPU
            
        self.total_ram = psutil.virtual_memory().total / (1024 * 1024)  # MB
        
        logging.info(f"Total VRAM {self.total_vram:0.0f} MB, total RAM {self.total_ram:0.0f} MB")
        
    def _detect_vram_state(self):
        """Detect VRAM state based on available memory - ComfyUI logic"""
        if self.cpu_state != CPUState.GPU:
            self.vram_state = VRAMState.DISABLED
            return
            
        # ComfyUI's VRAM state detection logic
        vram_gb = self.total_vram / 1024
        
        if vram_gb < 4:  # Less than 4GB VRAM
            self.vram_state = VRAMState.NO_VRAM
        elif vram_gb < 8:  # Less than 8GB VRAM
            self.vram_state = VRAMState.LOW_VRAM
        elif vram_gb < 16:  # Less than 16GB VRAM
            self.vram_state = VRAMState.NORMAL_VRAM
        else:  # 16GB+ VRAM
            self.vram_state = VRAMState.HIGH_VRAM
            
        logging.info(f"Auto-detected VRAM state: {self.vram_state.name}")
        
    def get_torch_device(self):
        """Get the appropriate torch device"""
        if self.cpu_state == CPUState.CPU:
            return torch.device("cpu")
        elif self.cpu_state == CPUState.MPS:
            return torch.device("mps")
        else:
            return torch.device(torch.cuda.current_device())
            
    def get_free_memory(self, device):
        """Get free memory for a device"""
        if device.type == 'cpu':
            return psutil.virtual_memory().available / (1024**3)  # GB
        elif device.type == 'cuda':
            return torch.cuda.get_device_properties(device).total_memory / (1024**3) - torch.cuda.memory_allocated(device) / (1024**3)
        else:
            return 0
            
    def dtype_size(self, dtype):
        """Get size of dtype in bytes"""
        if dtype == torch.float16 or dtype == torch.bfloat16:
            return 2
        elif dtype == torch.float32:
            return 4
        else:
            try:
                return dtype.itemsize
            except:
                return 4
                
    def unet_initial_load_device(self, parameters, dtype):
        """
        ComfyUI's unet_initial_load_device logic
        Determines where to initially load the model based on VRAM state and memory
        """
        torch_dev = self.get_torch_device()
        
        # High VRAM or shared memory - always use GPU
        if self.vram_state == VRAMState.HIGH_VRAM or self.vram_state == VRAMState.SHARED:
            return torch_dev
            
        cpu_dev = torch.device("cpu")
        
        # No VRAM or smart memory disabled - use CPU
        if self.disable_smart_memory or self.vram_state == VRAMState.NO_VRAM:
            return cpu_dev
            
        # Calculate model size
        model_size = self.dtype_size(dtype) * parameters
        
        # Get available memory
        mem_dev = self.get_free_memory(torch_dev)
        mem_cpu = self.get_free_memory(cpu_dev)
        
        # ComfyUI decision: If GPU has more memory AND model fits, use GPU
        if mem_dev > mem_cpu and model_size < mem_dev:
            return torch_dev
        else:
            return cpu_dev
            
    def determine_loading_strategy(self, model_size_gb, parameters, dtype):
        """
        Determine loading strategy based on ComfyUI's logic
        """
        torch_dev = self.get_torch_device()
        
        # Calculate model size in bytes
        model_size_bytes = self.dtype_size(dtype) * parameters
        model_size_gb_calculated = model_size_bytes / (1024**3)
        
        logging.info(f"🔍 ComfyUI-style loading strategy determination:")
        logging.info(f"  VRAM State: {self.vram_state.name}")
        logging.info(f"  Model size: {model_size_gb_calculated:.2f} GB")
        logging.info(f"  Total VRAM: {self.total_vram/1024:.1f} GB")
        
        # Get initial load device using ComfyUI logic
        initial_device = self.unet_initial_load_device(parameters, dtype)
        
        if initial_device.type == 'cuda':
            logging.info(f"✅ ComfyUI decision: Load to GPU")
            return {
                'loading_type': 'full_gpu',
                'device': initial_device,
                'reason': 'fits_in_gpu_memory',
                'vram_state': self.vram_state.name
            }
        else:
            logging.info(f"📱 ComfyUI decision: Load to CPU")
            return {
                'loading_type': 'cpu_with_dynamic',
                'device': initial_device,
                'reason': 'insufficient_gpu_memory',
                'vram_state': self.vram_state.name
            }

def test_comfyui_memory_management():
    """Test ComfyUI-style memory management"""
    print("🧪 Testing ComfyUI-style Memory Management")
    print("=" * 60)
    
    # Initialize ComfyUI memory manager
    mem_manager = ComfyUIMemoryManager()
    
    # Test with your 32GB WAN model
    model_params = 17_000_000_000  # 17B parameters
    dtype = torch.float16
    
    print(f"\n📊 System Information:")
    print(f"  VRAM State: {mem_manager.vram_state.name}")
    print(f"  Total VRAM: {mem_manager.total_vram/1024:.1f} GB")
    print(f"  Total RAM: {mem_manager.total_ram/1024:.1f} GB")
    
    print(f"\n🎯 Model Information:")
    print(f"  Parameters: {model_params:,}")
    print(f"  Dtype: {dtype}")
    print(f"  Model size: {mem_manager.dtype_size(dtype) * model_params / (1024**3):.2f} GB")
    
    # Test loading strategy
    strategy = mem_manager.determine_loading_strategy(32.0, model_params, dtype)
    
    print(f"\n🚀 Loading Strategy:")
    print(f"  Type: {strategy['loading_type']}")
    print(f"  Device: {strategy['device']}")
    print(f"  Reason: {strategy['reason']}")
    print(f"  VRAM State: {strategy['vram_state']}")
    
    # Test with different VRAM states
    print(f"\n🔬 Testing different VRAM states:")
    original_vram_state = mem_manager.vram_state
    
    for vram_state in [VRAMState.NO_VRAM, VRAMState.LOW_VRAM, VRAMState.NORMAL_VRAM, VRAMState.HIGH_VRAM]:
        mem_manager.vram_state = vram_state
        strategy = mem_manager.determine_loading_strategy(32.0, model_params, dtype)
        print(f"  {vram_state.name}: {strategy['loading_type']} -> {strategy['device']}")
    
    mem_manager.vram_state = original_vram_state

if __name__ == "__main__":
    test_comfyui_memory_management()
