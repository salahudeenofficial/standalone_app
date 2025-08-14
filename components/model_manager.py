#!/usr/bin/env python3
"""
Model Manager for CPU/GPU Swapping
Manages model loading/unloading to reduce peak VRAM usage
"""

import torch
import gc
import logging
from typing import List, Optional, Union
from pathlib import Path

class ModelManager:
    """Manages model loading/unloading between CPU and GPU to optimize VRAM usage"""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cpu_device = torch.device("cpu")
        self.loaded_models = {}  # Track loaded models and their states
        self.vram_threshold = 0.8  # Use 80% of available VRAM before offloading
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Get VRAM info
        if torch.cuda.is_available():
            self.total_vram = torch.cuda.get_device_properties(0).total_memory
            self.logger.info(f"Total VRAM: {self.total_vram / 1024**3:.2f} GB")
        else:
            self.total_vram = 0
            self.logger.info("CUDA not available, using CPU only")
    
    def get_vram_usage(self) -> float:
        """Get current VRAM usage in GB"""
        if not torch.cuda.is_available():
            return 0.0
        
        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        return allocated / 1024**3, reserved / 1024**3
    
    def get_free_vram(self) -> float:
        """Get free VRAM in GB"""
        if not torch.cuda.is_available():
            return 0.0
        
        allocated = torch.cuda.memory_allocated(0)
        total = torch.cuda.get_device_properties(0).total_memory
        return (total - allocated) / 1024**3
    
    def should_offload(self, model_size_gb: float = 0.0) -> bool:
        """Determine if we should offload models to CPU"""
        if not torch.cuda.is_available():
            return False
        
        free_vram = self.get_free_vram()
        total_vram_gb = self.total_vram / 1024**3
        
        # Offload if free VRAM is below threshold
        return free_vram < (total_vram_gb * (1 - self.vram_threshold) + model_size_gb)
    
    def load_model_gpu(self, model, model_name: str = "unknown") -> None:
        """Load a model to GPU, offloading others if necessary"""
        if not torch.cuda.is_available():
            self.logger.info(f"CUDA not available, keeping {model_name} on CPU")
            return
        
        # Estimate model size (rough approximation)
        model_size_gb = self._estimate_model_size(model)
        
        # Check if we need to offload other models
        if self.should_offload(model_size_gb):
            self.logger.info(f"Low VRAM detected, offloading other models before loading {model_name}")
            self._offload_models_to_cpu()
        
        # Move model to GPU
        try:
            if hasattr(model, 'to'):
                model.to(self.device)
                self.logger.info(f"Loaded {model_name} to GPU")
            else:
                # Handle models that might not have .to() method
                self.logger.warning(f"Model {model_name} doesn't have .to() method")
            
            # Track the loaded model
            self.loaded_models[model_name] = {
                'model': model,
                'device': self.device,
                'size_gb': model_size_gb
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load {model_name} to GPU: {e}")
            # Fallback to CPU
            if hasattr(model, 'to'):
                model.to(self.cpu_device)
    
    def unload_model(self, model_name: str) -> None:
        """Unload a specific model to CPU"""
        if model_name not in self.loaded_models:
            return
        
        model_info = self.loaded_models[model_name]
        model = model_info['model']
        
        try:
            if hasattr(model, 'to'):
                model.to(self.cpu_device)
                self.logger.info(f"Unloaded {model_name} to CPU")
                
                # Update tracking
                self.loaded_models[model_name]['device'] = self.cpu_device
                
        except Exception as e:
            self.logger.error(f"Failed to unload {model_name} to CPU: {e}")
    
    def unload_all_models(self) -> None:
        """Unload all models to CPU"""
        for model_name in list(self.loaded_models.keys()):
            self.unload_model(model_name)
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        self.logger.info("All models unloaded to CPU")
    
    def _offload_models_to_cpu(self) -> None:
        """Offload models to CPU to free VRAM"""
        # Sort models by size (largest first) to prioritize offloading
        models_to_offload = []
        for name, info in self.loaded_models.items():
            if info['device'] == self.device:  # Only offload GPU models
                models_to_offload.append((name, info['size_gb']))
        
        # Sort by size (largest first)
        models_to_offload.sort(key=lambda x: x[1], reverse=True)
        
        # Offload models until we have enough free VRAM
        for name, size in models_to_offload:
            if not self.should_offload():
                break
            
            self.unload_model(name)
    
    def _estimate_model_size(self, model) -> float:
        """Estimate model size in GB (rough approximation)"""
        try:
            if hasattr(model, 'parameters'):
                total_params = sum(p.numel() for p in model.parameters())
                # Assume float32 (4 bytes per parameter)
                size_bytes = total_params * 4
                return size_bytes / 1024**3
            else:
                # Fallback estimation
                return 1.0  # Assume 1GB if we can't determine
        except:
            return 1.0  # Default fallback
    
    def cleanup(self) -> None:
        """Clean up resources"""
        self.unload_all_models()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        self.logger.info("Model manager cleanup completed")
    
    def get_status(self) -> dict:
        """Get current status of model manager"""
        status = {
            'total_vram_gb': self.total_vram / 1024**3 if self.total_vram > 0 else 0,
            'free_vram_gb': self.get_free_vram(),
            'loaded_models': len(self.loaded_models),
            'gpu_models': sum(1 for info in self.loaded_models.values() if info['device'] == self.device),
            'cpu_models': sum(1 for info in self.loaded_models.values() if info['device'] == self.cpu_device)
        }
        
        if torch.cuda.is_available():
            allocated, reserved = self.get_vram_usage()
            status.update({
                'allocated_vram_gb': allocated,
                'reserved_vram_gb': reserved
            })
        
        return status
    
    def print_status(self) -> None:
        """Print current status to console"""
        status = self.get_status()
        self.logger.info("=== Model Manager Status ===")
        self.logger.info(f"Total VRAM: {status['total_vram_gb']:.2f} GB")
        self.logger.info(f"Free VRAM: {status['free_vram_gb']:.2f} GB")
        self.logger.info(f"Loaded Models: {status['loaded_models']}")
        self.logger.info(f"GPU Models: {status['gpu_models']}")
        self.logger.info(f"CPU Models: {status['cpu_models']}")
        
        if 'allocated_vram_gb' in status:
            self.logger.info(f"Allocated VRAM: {status['allocated_vram_gb']:.2f} GB")
            self.logger.info(f"Reserved VRAM: {status['reserved_vram_gb']:.2f} GB")
        
        self.logger.info("==========================") 