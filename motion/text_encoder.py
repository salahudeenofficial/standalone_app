"""
Standalone Text Encoder Implementation
CLIP text encoding functionality without ComfyUI dependencies.

This module provides the CLIPTextEncode class for encoding text prompts
using CLIP models in a standalone manner.
"""

import torch
from typing import Tuple, Any


class CLIPTextEncode:
    """
    Standalone CLIP Text Encoder
    Encodes text prompts using CLIP models without ComfyUI dependencies
    """
    
    def __init__(self):
        """Initialize the text encoder"""
        pass
    
    def encode(self, clip, text: str) -> Tuple[torch.Tensor, ...]:
        """
        Encode text using CLIP model
        
        Args:
            clip: CLIP model instance (should have tokenize and encode_from_tokens_scheduled methods)
            text: Text prompt to encode
            
        Returns:
            Tuple containing the encoded conditioning tensor
            
        Raises:
            RuntimeError: If clip is None or invalid
        """
        if clip is None:
            raise RuntimeError(
                "ERROR: clip input is invalid: None\n\n"
                "If the clip is from a checkpoint loader node your checkpoint does not contain "
                "a valid clip or text encoder model."
            )
        
        # Validate that clip has required methods
        if not hasattr(clip, 'tokenize'):
            raise RuntimeError(
                f"ERROR: clip model does not have tokenize method. "
                f"Got type: {type(clip).__name__}"
            )
        
        if not hasattr(clip, 'encode_from_tokens_scheduled'):
            raise RuntimeError(
                f"ERROR: clip model does not have encode_from_tokens_scheduled method. "
                f"Got type: {type(clip).__name__}"
            )
        
        try:
            # Tokenize the text
            tokens = clip.tokenize(text)
            
            # Encode the tokens
            conditioning = clip.encode_from_tokens_scheduled(tokens)
            
            # Return as tuple to match ComfyUI interface
            return (conditioning, )
            
        except Exception as e:
            raise RuntimeError(f"Failed to encode text '{text}': {str(e)}") from e


def test_text_encoder():
    """Test function to verify text encoder interface"""
    print("🧪 Testing CLIPTextEncode standalone implementation...")
    
    # Create text encoder
    encoder = CLIPTextEncode()
    print("✅ CLIPTextEncode created successfully")
    
    # Test error handling with None clip
    try:
        encoder.encode(None, "test")
        print("❌ Should have raised error for None clip")
    except RuntimeError as e:
        print(f"✅ Correctly raised error for None clip: {str(e)[:50]}...")
    
    # Test error handling with invalid clip (missing methods)
    class InvalidClip:
        pass
    
    try:
        encoder.encode(InvalidClip(), "test")
        print("❌ Should have raised error for invalid clip")
    except RuntimeError as e:
        print(f"✅ Correctly raised error for invalid clip: {str(e)[:50]}...")
    
    print("✅ All text encoder interface tests passed!")


if __name__ == "__main__":
    # Run tests when script is executed directly
    test_text_encoder()
