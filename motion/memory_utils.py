#!/usr/bin/env python3
"""
Memory Management Utilities for Large Model Loading
"""

import torch
import logging
import gc

def clear_cuda_memory():
    """Clear CUDA memory cache and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        logging.info("CUDA memory cache cleared")

def get_memory_info():
    """Get current memory usage information"""
    info = {}
    
    if torch.cuda.is_available():
        info['cuda_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
        info['cuda_reserved'] = torch.cuda.memory_reserved() / 1024**3    # GB
        info['cuda_max_allocated'] = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        # Get device properties
        device_props = torch.cuda.get_device_properties(0)
        info['cuda_total'] = device_props.total_memory / 1024**3  # GB
        info['cuda_free'] = info['cuda_total'] - info['cuda_allocated']
    
    return info

def log_memory_usage(stage=""):
    """Log current memory usage"""
    info = get_memory_info()
    
    if torch.cuda.is_available():
        logging.info(f"Memory Usage {stage}:")
        logging.info(f"  CUDA Allocated: {info['cuda_allocated']:.2f} GB")
        logging.info(f"  CUDA Reserved: {info['cuda_reserved']:.2f} GB")
        logging.info(f"  CUDA Free: {info['cuda_free']:.2f} GB")
        logging.info(f"  CUDA Total: {info['cuda_total']:.2f} GB")
        logging.info(f"  CUDA Max Allocated: {info['cuda_max_allocated']:.2f} GB")
    else:
        logging.info(f"CUDA not available - using CPU only")

def estimate_model_memory(model):
    """Estimate memory usage of a model"""
    total_params = sum(p.numel() for p in model.parameters())
    total_size = sum(p.numel() * p.element_size() for p in model.parameters())
    
    return {
        'parameters': total_params,
        'size_bytes': total_size,
        'size_gb': total_size / 1024**3
    }

def estimate_state_dict_memory(state_dict):
    """Estimate memory usage of a state dict"""
    total_params = 0
    total_size = 0
    
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            total_params += tensor.numel()
            total_size += tensor.numel() * tensor.element_size()
    
    return {
        'parameters': total_params,
        'size_bytes': total_size,
        'size_gb': total_size / 1024**3,
        'keys': len(state_dict)
    }

def safe_model_to_device(model, device, min_free_gb=2.0, state_dict=None):
    """Safely move model to device with memory checking"""
    if device.type == 'cuda' and torch.cuda.is_available():
        # Clear cache first
        clear_cuda_memory()
        
        # Check available memory
        info = get_memory_info()
        
        # Estimate memory requirements
        if state_dict is not None:
            # Use state dict size for accurate estimation
            model_info = estimate_state_dict_memory(state_dict)
            logging.info(f"Attempting to move model to {device}")
            logging.info(f"  State dict size: {model_info['size_gb']:.2f} GB ({model_info['keys']} keys)")
            logging.info(f"  Parameters: {model_info['parameters']:,}")
        else:
            # Fallback to model parameter estimation
            model_info = estimate_model_memory(model)
            logging.info(f"Attempting to move model to {device}")
            logging.info(f"  Model size: {model_info['size_gb']:.2f} GB")
        
        logging.info(f"  Available GPU memory: {info['cuda_free']:.2f} GB")
        logging.info(f"  Required buffer: {min_free_gb:.2f} GB")
        
        total_required = model_info['size_gb'] + min_free_gb
        
        if info['cuda_free'] > total_required:
            try:
                model = model.to(device)
                logging.info(f"✅ Model successfully moved to {device}")
                return model, device
            except torch.cuda.OutOfMemoryError as e:
                logging.warning(f"❌ CUDA OOM when moving model: {e}")
                logging.info("🔄 Falling back to CPU")
                return model, torch.device('cpu')
        else:
            logging.warning(f"❌ Insufficient GPU memory")
            logging.info(f"   Need: {total_required:.2f} GB")
            logging.info(f"   Have: {info['cuda_free']:.2f} GB")
            logging.info("🔄 Using CPU instead")
            return model, torch.device('cpu')
    else:
        logging.info(f"Using CPU device: {device}")
        return model, torch.device('cpu')

def safe_model_to_device_advanced(model, device, min_free_gb=2.0, state_dict=None, enable_partial_loading=True):
    """
    Advanced model loading with partial loading capability inspired by ComfyUI
    
    Args:
        model: PyTorch model to load
        device: Target device
        min_free_gb: Minimum free memory to reserve
        state_dict: Optional state dict for accurate memory estimation
        enable_partial_loading: Enable partial loading for large models
    
    Returns:
        tuple: (model, final_device, loading_info)
    """
    if device.type == 'cuda' and torch.cuda.is_available():
        # Clear cache first
        clear_cuda_memory()
        
        # Get memory information
        info = get_memory_info()
        available_memory_gb = info['cuda_free']
        memory_budget_gb = available_memory_gb - min_free_gb
        
        logging.info(f"🚀 Advanced model loading to {device}")
        logging.info(f"  Available memory: {available_memory_gb:.2f} GB")
        logging.info(f"  Memory budget: {memory_budget_gb:.2f} GB")
        
        # Estimate model size
        if state_dict is not None:
            model_info = estimate_state_dict_memory(state_dict)
            total_model_size_gb = model_info['size_gb']
        else:
            model_info = estimate_model_memory(model)
            total_model_size_gb = model_info['size_gb']
        
        logging.info(f"  Model size: {total_model_size_gb:.2f} GB")
        
        # Check if we can load the entire model
        if total_model_size_gb <= memory_budget_gb:
            try:
                model = model.to(device)
                logging.info(f"✅ Full model loaded to {device}")
                return model, device, {
                    'loading_type': 'full',
                    'modules_loaded': 'all',
                    'memory_used_gb': total_model_size_gb,
                    'memory_budget_gb': memory_budget_gb
                }
            except torch.cuda.OutOfMemoryError as e:
                logging.warning(f"❌ CUDA OOM during full loading: {e}")
                if enable_partial_loading:
                    logging.info("🔄 Attempting partial loading...")
                    return _load_model_partially(model, device, memory_budget_gb, state_dict)
                else:
                    logging.info("🔄 Falling back to CPU")
                    return model, torch.device('cpu'), {
                        'loading_type': 'cpu_fallback',
                        'reason': 'oom_during_full_loading'
                    }
        else:
            # Model is too large for full loading
            if enable_partial_loading:
                logging.info(f"📊 Model too large for full loading ({total_model_size_gb:.2f} GB > {memory_budget_gb:.2f} GB)")
                logging.info("🔄 Attempting partial loading...")
                return _load_model_partially(model, device, memory_budget_gb, state_dict)
            else:
                logging.info("🔄 Partial loading disabled, using CPU")
                return model, torch.device('cpu'), {
                    'loading_type': 'cpu_fallback',
                    'reason': 'model_too_large'
                }
    else:
        logging.info(f"Using CPU device: {device}")
        return model, torch.device('cpu'), {
            'loading_type': 'cpu_only',
            'reason': 'cuda_not_available'
        }

def _load_model_partially(model, device, memory_budget_gb, state_dict=None):
    """
    Load model partially by moving modules selectively to GPU
    
    Args:
        model: PyTorch model
        device: Target GPU device
        memory_budget_gb: Available memory budget in GB
        state_dict: Optional state dict for accurate estimation
    
    Returns:
        tuple: (model, device, loading_info)
    """
    logging.info("🔧 Starting partial model loading...")
    
    # Analyze model structure
    modules_info = _analyze_model_modules(model, state_dict)
    
    # Sort modules by size (largest first)
    modules_info.sort(key=lambda x: x['size_gb'], reverse=True)
    
    # Load modules within memory budget
    loaded_modules = []
    remaining_memory_gb = memory_budget_gb
    total_loaded_gb = 0
    
    for module_info in modules_info:
        module_size_gb = module_info['size_gb']
        module_name = module_info['name']
        module_obj = module_info['module']
        
        if module_size_gb <= remaining_memory_gb:
            try:
                # Move module to GPU
                module_obj.to(device)
                loaded_modules.append(module_info)
                remaining_memory_gb -= module_size_gb
                total_loaded_gb += module_size_gb
                logging.info(f"  ✅ Loaded {module_name}: {module_size_gb:.3f} GB")
            except torch.cuda.OutOfMemoryError:
                logging.warning(f"  ⚠️  OOM loading {module_name}, skipping")
                break
        else:
            logging.info(f"  📊 Skipping {module_name}: {module_size_gb:.3f} GB (too large)")
    
    # Set up dynamic loading for remaining modules
    dynamic_modules = []
    for module_info in modules_info:
        if module_info not in loaded_modules:
            _setup_dynamic_loading(module_info['module'], device)
            dynamic_modules.append(module_info)
    
    logging.info(f"🎉 Partial loading complete:")
    logging.info(f"  Modules loaded to GPU: {len(loaded_modules)}")
    logging.info(f"  Modules with dynamic loading: {len(dynamic_modules)}")
    logging.info(f"  Total GPU memory used: {total_loaded_gb:.3f} GB")
    logging.info(f"  Remaining budget: {remaining_memory_gb:.3f} GB")
    
    return model, device, {
        'loading_type': 'partial',
        'modules_loaded': len(loaded_modules),
        'modules_dynamic': len(dynamic_modules),
        'memory_used_gb': total_loaded_gb,
        'memory_budget_gb': memory_budget_gb,
        'loaded_modules': [m['name'] for m in loaded_modules],
        'dynamic_modules': [m['name'] for m in dynamic_modules]
    }

def _analyze_model_modules(model, state_dict=None):
    """
    Analyze model modules and estimate their memory usage
    
    Args:
        model: PyTorch model
        state_dict: Optional state dict for accurate estimation
    
    Returns:
        list: List of module information dictionaries
    """
    modules_info = []
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            # Estimate module size
            if state_dict is not None:
                # Use state dict for accurate estimation
                module_params = 0
                module_size_bytes = 0
                
                for param_name, param in module.named_parameters():
                    state_key = f"{name}.{param_name}" if name else param_name
                    if state_key in state_dict:
                        module_params += state_dict[state_key].numel()
                        module_size_bytes += state_dict[state_key].numel() * state_dict[state_key].element_size()
                
                if module_size_bytes > 0:
                    module_size_gb = module_size_bytes / (1024**3)
                else:
                    # Fallback to parameter estimation
                    module_params = sum(p.numel() for p in module.parameters())
                    module_size_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
                    module_size_gb = module_size_bytes / (1024**3)
            else:
                # Use parameter estimation
                module_params = sum(p.numel() for p in module.parameters())
                module_size_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
                module_size_gb = module_size_bytes / (1024**3)
            
            if module_size_gb > 0.001:  # Only include modules > 1MB
                modules_info.append({
                    'name': name or 'root',
                    'module': module,
                    'size_gb': module_size_gb,
                    'parameters': module_params,
                    'size_bytes': module_size_bytes
                })
    
    return modules_info

def _setup_dynamic_loading(module, device):
    """
    Set up dynamic loading for a module (placeholder for now)
    
    Args:
        module: PyTorch module
        device: Target device
    """
    # For now, we'll just mark the module for dynamic loading
    # In a full implementation, this would set up weight functions
    # similar to ComfyUI's LowVramPatch system
    
    if not hasattr(module, '_dynamic_loading_setup'):
        module._dynamic_loading_setup = True
        module._target_device = device
        logging.debug(f"  🔄 Dynamic loading setup for {type(module).__name__}")

if __name__ == "__main__":
    print("Memory Management Utilities")
    log_memory_usage("Startup")
