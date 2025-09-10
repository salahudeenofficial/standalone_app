"""
Standalone Utils Module - Fixed for proper parameter counting
Replaces comfy.utils functionality
"""

import torch
import safetensors.torch
import os
from typing import Dict, Any, Optional, Union


def load_torch_file(file_path: str, device: Optional[torch.device] = None, return_metadata: bool = False) -> Union[Dict[str, torch.Tensor], tuple]:
    """
    Load PyTorch model file (.safetensors or .ckpt)
    
    Args:
        file_path: Path to the model file
        device: Device to load tensors on
        return_metadata: Whether to return metadata
    
    Returns:
        State dict or (state dict, metadata) tuple
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")
    
    if file_path.endswith('.safetensors'):
        if return_metadata:
            state_dict, metadata = safetensors.torch.load_file(file_path, device=device)
            return state_dict, metadata
        else:
            return safetensors.torch.load_file(file_path, device=device)
    
    elif file_path.endswith('.ckpt'):
        checkpoint = torch.load(file_path, map_location=device)
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        if return_metadata:
            metadata = checkpoint.get('metadata', {})
            return state_dict, metadata
        else:
            return state_dict
    
    else:
        raise ValueError(f"Unsupported file extension: {file_path}")


def calculate_parameters(sd: Dict[str, torch.Tensor], prefix: str = "") -> int:
    """
    Calculate total parameters in state dict, excluding tokenizer data
    
    Args:
        sd: State dictionary
        prefix: Key prefix to filter by
    
    Returns:
        Total number of parameters
    """
    params = 0
    
    # Keys to exclude (tokenizer data, not neural network parameters)
    exclude_keys = {
        'spiece_model',  # SentencePiece tokenizer model
        'tokenizer',     # Generic tokenizer data
        'vocab',         # Vocabulary data
        'merges',        # BPE merges
        'special_tokens', # Special tokens
    }
    
    for k in sd.keys():
        if not k.startswith(prefix):
            continue
        
        # Skip tokenizer data
        if any(exclude_key in k.lower() for exclude_key in exclude_keys):
            continue
            
        w = sd[k]
        if hasattr(w, "nelement"):  # Only count tensors
            params += w.nelement()
    
    return params


def weight_dtype(sd: Dict[str, torch.Tensor], prefix: str = "") -> Optional[torch.dtype]:
    """
    Get the most common dtype in state dict
    
    Args:
        sd: State dictionary
        prefix: Key prefix to filter by
    
    Returns:
        Most common dtype or None
    """
    dtypes = {}
    
    # Keys to exclude (tokenizer data)
    exclude_keys = {
        'spiece_model', 'tokenizer', 'vocab', 'merges', 'special_tokens'
    }
    
    for k in sd.keys():
        if not k.startswith(prefix):
            continue
        
        # Skip tokenizer data
        if any(exclude_key in k.lower() for exclude_key in exclude_keys):
            continue
            
        w = sd[k]
        if hasattr(w, "dtype"):
            dtype = w.dtype
            if dtype not in dtypes:
                dtypes[dtype] = 0
            if hasattr(w, "nelement"):
                dtypes[dtype] += w.nelement()
    
    if len(dtypes) == 0:
        return None
    
    return max(dtypes, key=dtypes.get)


def state_dict_prefix_replace(sd: Dict[str, torch.Tensor], replacements: Dict[str, str], 
                             filter_keys: bool = False) -> Dict[str, torch.Tensor]:
    """
    Replace prefixes in state dict keys
    
    Args:
        sd: State dictionary
        replacements: Dict of old_prefix -> new_prefix
        filter_keys: Whether to filter out keys not matching replacements
    
    Returns:
        Modified state dictionary
    """
    result = {}
    
    for k, v in sd.items():
        new_key = k
        for old_prefix, new_prefix in replacements.items():
            if k.startswith(old_prefix):
                new_key = new_prefix + k[len(old_prefix):]
                break
        
        if filter_keys:
            if any(k.startswith(old_prefix) for old_prefix in replacements.keys()):
                result[new_key] = v
        else:
            result[new_key] = v
    
    return result


if __name__ == "__main__":
    print("Fixed utils.py with proper parameter counting")
    print("✅ Excludes tokenizer data from parameter count")
    print("✅ Proper spiece_model handling")
    print("✅ Accurate parameter calculation")
