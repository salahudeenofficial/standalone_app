"""
Standalone conditioning utilities for motion pipeline
Replaces ComfyUI's node_helpers.conditioning_set_values functionality
"""

import torch
from typing import List, Dict, Any, Optional

class ConditioningData:
    """Represents a conditioning entry with text and additional data"""
    
    def __init__(self, text_embedding: torch.Tensor, additional_data: Dict[str, Any] = None):
        self.text_embedding = text_embedding
        self.additional_data = additional_data or {}
    
    def copy(self):
        """Create a copy of this conditioning data"""
        return ConditioningData(
            text_embedding=self.text_embedding.clone(),
            additional_data=self.additional_data.copy()
        )

def create_empty_conditioning(device='cpu') -> List[ConditioningData]:
    """Create empty conditioning list for initialization"""
    # Create a dummy text embedding (typical CLIP size)
    dummy_embedding = torch.zeros((1, 77, 4096), device=device)
    return [ConditioningData(dummy_embedding)]

def conditioning_set_values(conditioning: List[ConditioningData], 
                          values: Dict[str, Any], 
                          append: bool = True) -> List[ConditioningData]:
    """
    Standalone version of ComfyUI's conditioning_set_values
    
    Args:
        conditioning: List of ConditioningData objects
        values: Dictionary of key-value pairs to add
        append: Whether to append to existing data (True) or replace (False)
    
    Returns:
        Modified conditioning list
    """
    if not conditioning:
        conditioning = create_empty_conditioning()
    
    result = []
    for cond in conditioning:
        new_cond = cond.copy()
        
        if append:
            # Append to existing additional data
            for key, value in values.items():
                if key in new_cond.additional_data:
                    # If key exists and both are lists, extend
                    if isinstance(new_cond.additional_data[key], list) and isinstance(value, list):
                        new_cond.additional_data[key].extend(value)
                    else:
                        # Otherwise replace
                        new_cond.additional_data[key] = value
                else:
                    # New key, just add
                    new_cond.additional_data[key] = value
        else:
            # Replace mode - overwrite existing data
            new_cond.additional_data.update(values)
        
        result.append(new_cond)
    
    return result

def get_conditioning_value(conditioning: List[ConditioningData], key: str) -> Any:
    """Get a value from conditioning data"""
    if not conditioning:
        return None
    
    return conditioning[0].additional_data.get(key, None)

def print_conditioning_info(conditioning: List[ConditioningData], name: str = "Conditioning"):
    """Debug helper to print conditioning information"""
    print(f"\n📋 {name} Info:")
    if not conditioning:
        print("   ❌ Empty conditioning")
        return
    
    for i, cond in enumerate(conditioning):
        print(f"   Entry {i}:")
        print(f"     Text Embedding: {cond.text_embedding.shape}")
        print(f"     Additional Data Keys: {list(cond.additional_data.keys())}")
        
        # Print specific WAN conditioning keys
        for key in ['vace_frames', 'vace_mask', 'vace_strength']:
            if key in cond.additional_data:
                value = cond.additional_data[key]
                if isinstance(value, list) and len(value) > 0:
                    if hasattr(value[0], 'shape'):
                        print(f"     {key}: {[v.shape for v in value]}")
                    else:
                        print(f"     {key}: {value}")
                else:
                    print(f"     {key}: {value}")
