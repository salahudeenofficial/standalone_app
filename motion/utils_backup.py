import torch
import safetensors.torch




def load_torch_file(file_path,device=None):
    if device is None:
        device = torch.device("cpu")
    if file_path.lower().endswith(".safetensors"):
        try:
            with safetensors.safe_open(file_path,framework="pt",device=device.type) as f:
                sd = {}
                for k in f.keys():
                    tensor = f.get_tensor(k)
                    tensor = tensor.to(device)
                    sd[k] = tensor
        except Exception as e:
            raise ValueError(f"Error loading safetensors file: {e}")
    else:
        raise ValueError(f"Unsupported file extension: {file_path}")
    
    return sd

def calculate_parameters(sd, prefix=""):
    params = 0
    for k in sd.keys():
        if not k.startswith(prefix):
            continue
        w = sd[k]
        if hasattr(w, "nelement"):  # Only count tensors
            params += w.nelement()
    return params
    return params
def weight_dtype(sd, prefix=""):
    """Determine the weight dtype from state dict"""
    dtypes = {}
    for k in sd.keys():
        if not k.startswith(prefix):
            continue
        w = sd[k]
        if hasattr(w, "dtype"):  # Only process tensors
            dtype = w.dtype
            if dtype not in dtypes:
                dtypes[dtype] = 0
            dtypes[dtype] += w.nelement()
    
    # Return the most common dtype
    if not dtypes:
        return torch.float32
    
    return max(dtypes.items(), key=lambda x: x[1])[0]
    
    # Return the most common dtype
    if not dtypes:
        return torch.float32
    
    return max(dtypes.items(), key=lambda x: x[1])[0]


def state_dict_prefix_replace(sd, replacements, filter_keys=False):
    """Replace prefixes in state dict keys"""
    result = {}
    
    for k, v in sd.items():
        replaced = False
        for old_prefix, new_prefix in replacements.items():
            if k.startswith(old_prefix):
                new_key = k.replace(old_prefix, new_prefix, 1)
                if not filter_keys or new_key:  # Only add if new_key is not empty when filtering
                    result[new_key] = v
                replaced = True
                break
        
        if not replaced and not filter_keys:
            result[k] = v
    
    return result


def load_torch_file(file_path, device=None, return_metadata=False):
    """Load torch file with optional metadata"""
    if device is None:
        device = torch.device("cpu")
    
    if file_path.lower().endswith(".safetensors"):
        try:
            with safetensors.safe_open(file_path, framework="pt", device=device.type) as f:
                sd = {}
                metadata = {}
                
                # Get metadata if requested
                if return_metadata:
                    metadata = f.metadata()
                
                for k in f.keys():
                    tensor = f.get_tensor(k)
                    tensor = tensor.to(device)
                    sd[k] = tensor
                
                if return_metadata:
                    return sd, metadata
                else:
                    return sd
                    
        except Exception as e:
            raise ValueError(f"Error loading safetensors file: {e}")
    else:
        raise ValueError(f"Unsupported file extension: {file_path}")
