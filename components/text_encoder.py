"""
Text encoding component for the standalone pipeline
Adapted from ComfyUI CLIPTextEncode but simplified for direct use
"""

import torch
import sys
from pathlib import Path

# Add the comfy modules to path
sys.path.insert(0, str(Path(__file__).parent.parent / "comfy"))
sys.path.insert(0, str(Path(__file__).parent.parent / "comfy_extras"))

class CLIPTextEncode:
    """Encode text prompts using CLIP models"""
    
    def encode(self, clip, text):
        """Encode text using CLIP model"""
        if clip is None:
            raise RuntimeError("ERROR: clip input is invalid: None")
            
        tokens = clip.tokenize(text)
        return clip.encode_from_tokens_scheduled(tokens) 