#!/usr/bin/env python3
"""
Memory Management Utilities for Large Model Loading
"""

import torch
import logging
import gc

def clear_cuda_memory():
    """Clear CUDA memory cache and run garbage collection aggressively"""
    if torch.cuda.is_available():
        # Multiple rounds of cleanup for stubborn memory
        for i in range(3):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
        
        # Final cleanup
        torch.cuda.empty_cache()
        logging.info("CUDA memory cache cleared aggressively")

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
        
        # Get more accurate free memory using memory stats
        try:
            memory_stats = torch.cuda.memory_stats()
            # Use bytes_free from memory stats if available
            if 'bytes_free' in memory_stats:
                info['cuda_free'] = memory_stats['bytes_free'] / 1024**3
            else:
                # Fallback to total - reserved
                info['cuda_free'] = info['cuda_total'] - info['cuda_reserved']
        except:
            # Fallback to total - reserved
            info['cuda_free'] = info['cuda_total'] - info['cuda_reserved']
    
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
    """Estimate memory usage of a state dict - ComfyUI approach (no multipliers)"""
    total_params = 0
    total_size = 0
    
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            total_params += tensor.numel()
            total_size += tensor.numel() * tensor.element_size()
    
    # ComfyUI approach: No multipliers, just raw model size
    # The system will handle memory management dynamically
    return {
        'parameters': total_params,
        'size_bytes': total_size,
        'size_gb': total_size / 1024**3,
        'size_gb_gpu': total_size / 1024**3,  # Same as raw size
        'size_gb_cpu': total_size / 1024**3,   # Same as raw size
        'keys': len(state_dict),
        'gpu_multiplier': 1.0,  # No multiplier
        'cpu_multiplier': 1.0   # No multiplier
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
    ComfyUI-style model loading with CPU-first loading and patcher assignment
    
    Args:
        model: PyTorch model
        device: Target device
        min_free_gb: Minimum free memory to reserve
        state_dict: Optional state dict for memory estimation
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
        total_vram_gb = info['cuda_total']
        
        logging.info(f"🚀 ComfyUI-style model loading to {device}")
        logging.info(f"  Available GPU memory: {available_memory_gb:.2f} GB")
        logging.info(f"  Total VRAM: {total_vram_gb:.2f} GB")
        
        # ComfyUI VRAM state detection
        if total_vram_gb < 4:
            vram_state = "NO_VRAM"
        elif total_vram_gb < 8:
            vram_state = "LOW_VRAM"
        elif total_vram_gb < 16:
            vram_state = "NORMAL_VRAM"
        else:
            vram_state = "HIGH_VRAM"
            
        logging.info(f"  Detected VRAM state: {vram_state}")
        
        # ComfyUI approach: Calculate model size directly
        if state_dict is not None:
            model_info = estimate_state_dict_memory(state_dict)
            model_size_gb = model_info['size_gb']
            logging.info(f"  Model size: {model_size_gb:.2f} GB")
        else:
            model_info = estimate_model_memory(model)
            model_size_gb = model_info['size_gb']
            logging.info(f"  Model size: {model_size_gb:.2f} GB")
        
        # ComfyUI decision logic with CPU-first loading
        if vram_state == "HIGH_VRAM":
            # High VRAM - try complete loading first
            if model_size_gb < available_memory_gb - 1.0:  # Reserve 1GB
                try:
                    model = model.to(device)
                    logging.info(f"✅ Model loaded to GPU (HIGH_VRAM mode)")
                    return model, device, {
                        'loading_type': 'full_gpu',
                        'vram_state': vram_state,
                        'memory_used_gb': model_size_gb,
                        'available_memory_gb': available_memory_gb,
                        'patcher_type': 'complete'
                    }
                except torch.cuda.OutOfMemoryError as e:
                    logging.warning(f"❌ CUDA OOM during GPU loading: {e}")
                    logging.info("🔄 Falling back to CPU with dynamic loading...")
                    clear_cuda_memory()
                    return _setup_dynamic_model_loading(model, device, state_dict)
            else:
                # Model too large for complete loading, use partial loading
                logging.info(f"📊 Model too large for complete loading ({model_size_gb:.2f} GB > {available_memory_gb - 1.0:.2f} GB)")
                logging.info("🔄 Setting up CPU-first loading with partial patcher...")
                return _setup_cpu_first_partial_loading(model, device, state_dict, available_memory_gb - 1.0)
        else:
            # Low/Normal VRAM - use CPU-first loading with patcher
            reserved_memory_gb = 1.0
            usable_memory_gb = available_memory_gb - reserved_memory_gb
            
            logging.info(f"  Usable memory: {usable_memory_gb:.2f} GB")
            logging.info(f"  Model size: {model_size_gb:.2f} GB")
            
            if model_size_gb < usable_memory_gb:
                try:
                    model = model.to(device)
                    logging.info(f"✅ Model loaded to GPU (fits in {usable_memory_gb:.2f} GB)")
                    return model, device, {
                        'loading_type': 'full_gpu',
                        'vram_state': vram_state,
                        'memory_used_gb': model_size_gb,
                        'available_memory_gb': available_memory_gb,
                        'usable_memory_gb': usable_memory_gb,
                        'patcher_type': 'complete'
                    }
                except torch.cuda.OutOfMemoryError as e:
                    logging.warning(f"❌ CUDA OOM during GPU loading: {e}")
                    logging.info("🔄 Falling back to CPU with dynamic loading...")
                    clear_cuda_memory()
                    return _setup_dynamic_model_loading(model, device, state_dict)
            else:
                # Model too large for GPU, use CPU-first loading with partial patcher
                logging.info(f"📊 Model too large for GPU ({model_size_gb:.2f} GB > {usable_memory_gb:.2f} GB)")
                logging.info("🔄 Setting up CPU-first loading with partial patcher...")
                return _setup_cpu_first_partial_loading(model, device, state_dict, usable_memory_gb)
    else:
        logging.info(f"Using CPU device: {device}")
        return model, torch.device('cpu'), {
            'loading_type': 'cpu_only',
            'reason': 'cuda_not_available',
            'patcher_type': 'cpu_only'
        }

def _setup_cpu_first_partial_loading(model, device, state_dict=None, memory_budget_gb=0):
    """
    Set up CPU-first loading with partial patcher (ComfyUI approach)
    
    Args:
        model: PyTorch model
        device: Target GPU device for partial loading
        state_dict: Optional state dict for module analysis
        memory_budget_gb: Memory budget for partial loading
    
    Returns:
        tuple: (model, cpu_device, loading_info)
    """
    logging.info("🔧 Setting up CPU-first loading with partial patcher...")
    
    # Load entire model to CPU first (ComfyUI approach)
    cpu_device = torch.device('cpu')
    model = model.to(cpu_device)
    
    # Aggressively clear CUDA memory after moving to CPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        logging.info("🧹 Aggressively cleared CUDA memory after CPU transfer")
    
    # Analyze model structure for partial loading
    modules_info = _analyze_model_modules(model, state_dict)
    
    logging.info(f"📊 Found {len(modules_info)} leaf modules for partial loading")
    if modules_info:
        total_modules_size = sum(m['size_gb'] for m in modules_info)
        logging.info(f"📊 Total modules size: {total_modules_size:.3f} GB")
    
    # Store module info for partial loading
    model._partial_loading_info = {
        'modules_info': modules_info,
        'target_device': device,
        'loaded_modules': set(),
        'memory_budget_gb': memory_budget_gb,
        'patcher_type': 'partial'
    }
    
    logging.info(f"✅ Model loaded to CPU with partial patcher setup")
    logging.info(f"   Target GPU device: {device}")
    logging.info(f"   Modules available for partial loading: {len(modules_info)}")
    logging.info(f"   Memory budget: {memory_budget_gb:.2f} GB")
    
    return model, cpu_device, {
        'loading_type': 'cpu_first_partial',
        'modules_available': len(modules_info),
        'target_gpu_device': str(device),
        'total_size_gb': total_modules_size if modules_info else 0,
        'memory_budget_gb': memory_budget_gb,
        'patcher_type': 'partial'
    }

def _setup_dynamic_model_loading(model, device, state_dict=None):
    """
    Set up dynamic loading for large models - load entire model to CPU and prepare for dynamic GPU loading
    
    Args:
        model: PyTorch model
        device: Target GPU device for dynamic loading
        state_dict: Optional state dict for module analysis
    
    Returns:
        tuple: (model, cpu_device, loading_info)
    """
    logging.info("🔧 Setting up dynamic model loading...")
    
    # Load entire model to CPU
    cpu_device = torch.device('cpu')
    model = model.to(cpu_device)
    
    # Aggressively clear CUDA memory after moving to CPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Wait for all operations to complete
        # Force garbage collection
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        logging.info("🧹 Aggressively cleared CUDA memory after CPU transfer")
    
    # Analyze model structure for dynamic loading
    modules_info = _analyze_model_modules(model, state_dict)
    
    logging.info(f"📊 Found {len(modules_info)} leaf modules for dynamic loading")
    if modules_info:
        total_modules_size = sum(m['size_gb'] for m in modules_info)
        logging.info(f"📊 Total modules size: {total_modules_size:.3f} GB")
    
    # Store module info for dynamic loading
    model._dynamic_loading_info = {
        'modules_info': modules_info,
        'target_device': device,
        'loaded_modules': set(),
        'memory_budget_gb': 10.0  # Reserve 10GB for dynamic loading
    }
    
    logging.info(f"✅ Model loaded to CPU with dynamic loading setup")
    logging.info(f"   Target GPU device: {device}")
    logging.info(f"   Modules available for dynamic loading: {len(modules_info)}")
    
    return model, cpu_device, {
        'loading_type': 'dynamic_cpu',
        'modules_available': len(modules_info),
        'target_gpu_device': str(device),
        'total_size_gb': total_modules_size if modules_info else 0
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
    
    logging.info(f"📊 Found {len(modules_info)} leaf modules")
    if modules_info:
        total_modules_size = sum(m['size_gb'] for m in modules_info)
        logging.info(f"📊 Total modules size: {total_modules_size:.3f} GB")
        logging.info(f"📊 Memory budget: {memory_budget_gb:.3f} GB")
    
    # Sort modules by size (largest first)
    modules_info.sort(key=lambda x: x['size_gb'], reverse=True)
    
    # Load modules within memory budget
    loaded_modules = []
    remaining_memory_gb = memory_budget_gb
    total_loaded_gb = 0
    
    logging.info(f"🔄 Starting to load modules (budget: {memory_budget_gb:.3f} GB)...")
    
    for i, module_info in enumerate(modules_info):
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
                logging.info(f"  ✅ [{i+1}/{len(modules_info)}] Loaded {module_name}: {module_size_gb:.3f} GB (remaining: {remaining_memory_gb:.3f} GB)")
            except torch.cuda.OutOfMemoryError as e:
                logging.warning(f"  ⚠️  [{i+1}/{len(modules_info)}] OOM loading {module_name}: {e}")
                break
        else:
            logging.info(f"  📊 [{i+1}/{len(modules_info)}] Skipping {module_name}: {module_size_gb:.3f} GB (too large, remaining: {remaining_memory_gb:.3f} GB)")
    
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
                    # Apply GPU overhead multiplier for realistic estimation
                    module_size_gb = (module_size_bytes * 2.6) / (1024**3)
                else:
                    # Fallback to parameter estimation
                    module_params = sum(p.numel() for p in module.parameters())
                    module_size_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
                    # Apply GPU overhead multiplier for realistic estimation
                    module_size_gb = (module_size_bytes * 2.6) / (1024**3)
            else:
                # Use parameter estimation
                module_params = sum(p.numel() for p in module.parameters())
                module_size_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
                # Apply GPU overhead multiplier for realistic estimation
                module_size_gb = (module_size_bytes * 2.6) / (1024**3)
            
            if module_size_gb > 0.001:  # Only include modules > 1MB
                modules_info.append({
                    'name': name or 'root',
                    'module': module,
                    'size_gb': module_size_gb,
                    'parameters': module_params,
                    'size_bytes': module_size_bytes
                })
    
    return modules_info

def load_modules_for_inference(model, module_names, device=None):
    """
    Dynamically load specific modules to GPU for inference using ComfyUI-style weight patching
    
    Args:
        model: Model with dynamic loading setup
        module_names: List of module names to load
        device: Target device (uses model's target device if None)
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not hasattr(model, '_dynamic_loading_info'):
        logging.warning("Model doesn't have dynamic loading setup")
        return False
    
    info = model._dynamic_loading_info
    target_device = device or info['target_device']
    modules_info = info['modules_info']
    loaded_modules = info['loaded_modules']
    
    logging.info(f"🔄 Loading {len(module_names)} modules for inference...")
    
    # Find modules to load
    modules_to_load = []
    for module_name in module_names:
        for module_info in modules_info:
            if module_info['name'] == module_name:
                modules_to_load.append(module_info)
                break
    
    # Load modules using ComfyUI-style approach
    for module_info in modules_to_load:
        try:
            module = module_info['module']
            
            # Check if module has ComfyUI-style partial loader
            if hasattr(module, '_partial_loader'):
                # Use ComfyUI-style loading
                loader = module._partial_loader
                loader.load_weights_for_inference()
                logging.info(f"  ✅ ComfyUI-style loaded {module_info['name']}: {module_info['size_gb']:.3f} GB")
            else:
                # Fallback to moving entire module to GPU
                module.to(target_device)
                logging.info(f"  ✅ Standard loaded {module_info['name']}: {module_info['size_gb']:.3f} GB")
            
            loaded_modules.add(module_info['name'])
            
        except torch.cuda.OutOfMemoryError as e:
            logging.warning(f"  ⚠️  OOM loading {module_info['name']}: {e}")
            return False
    
    return True

def unload_modules_after_inference(model, module_names=None):
    """
    Unload modules from GPU after inference using ComfyUI-style approach
    
    Args:
        model: Model with dynamic loading setup
        module_names: Specific modules to unload (None = unload all)
    
    Returns:
        bool: True if successful
    """
    if not hasattr(model, '_dynamic_loading_info'):
        logging.warning("Model doesn't have dynamic loading setup")
        return False
    
    info = model._dynamic_loading_info
    modules_info = info['modules_info']
    loaded_modules = info['loaded_modules']
    cpu_device = torch.device('cpu')
    
    if module_names is None:
        module_names = list(loaded_modules)
    
    logging.info(f"🔄 Unloading {len(module_names)} modules after inference...")
    
    # Find and unload modules
    for module_name in module_names:
        for module_info in modules_info:
            if module_info['name'] == module_name and module_name in loaded_modules:
                try:
                    module = module_info['module']
                    
                    # Check if module has ComfyUI-style partial loader
                    if hasattr(module, '_partial_loader'):
                        # Use ComfyUI-style unloading
                        loader = module._partial_loader
                        loader.evict_weights_after_inference()
                        logging.info(f"  ✅ ComfyUI-style unloaded {module_name}")
                    else:
                        # Fallback to moving entire module back to CPU
                        module.to(cpu_device)
                        logging.info(f"  ✅ Standard unloaded {module_name}")
                    
                    loaded_modules.discard(module_name)
                    
                except Exception as e:
                    logging.warning(f"  ⚠️  Error unloading {module_name}: {e}")
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return True

def _setup_dynamic_loading(module, device):
    """
    Set up dynamic loading for a module using ComfyUI-style weight patching
    
    Args:
        module: PyTorch module
        device: Target device
    """
    # Import our ComfyUI-style partial loading system
    from comfyui_style_partial_loading import ComfyUIStylePartialLoader
    
    if not hasattr(module, '_dynamic_loading_setup'):
        module._dynamic_loading_setup = True
        module._target_device = device
        
        # Set up ComfyUI-style partial loading
        try:
            loader = ComfyUIStylePartialLoader(module, device, memory_budget_gb=2.0)
            module._partial_loader = loader
            logging.info(f"  ✅ ComfyUI-style dynamic loading setup for {type(module).__name__}")
        except Exception as e:
            logging.warning(f"  ⚠️  Failed to setup ComfyUI-style loading: {e}")
            # Fallback to simple marking
            logging.debug(f"  🔄 Basic dynamic loading setup for {type(module).__name__}")

if __name__ == "__main__":
    print("Memory Management Utilities")
    log_memory_usage("Startup")
