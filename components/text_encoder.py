"""
Text encoding component for the standalone pipeline
Exactly matches ComfyUI's CLIPTextEncode implementation
"""

import torch
import sys
from pathlib import Path

# Add the comfy modules to path
sys.path.insert(0, str(Path(__file__).parent.parent / "comfy"))
sys.path.insert(0, str(Path(__file__).parent.parent / "comfy_extras"))

class CLIPTextEncode:
    """Encode text prompts using CLIP models - matches ComfyUI exactly"""
    
    def encode(self, clip, text):
        """Encode text using CLIP model - matches ComfyUI exactly"""
        if clip is None:
            raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")
        
        tokens = clip.tokenize(text)
        # Return tuple wrapper exactly like ComfyUI does
        return (clip.encode_from_tokens_scheduled(tokens), ) 