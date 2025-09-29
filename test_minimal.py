#!/usr/bin/env python3
"""
Minimal test to check basic imports
"""

import sys
from pathlib import Path

# Add the comfy modules to path
sys.path.insert(0, str(Path(__file__).parent / "comfy"))
sys.path.insert(0, str(Path(__file__).parent / "comfy_extras"))

def test_basic_imports():
    """Test basic module imports"""
    try:
        print("Testing basic imports...")
        
        # Test core comfy modules
        import comfy.sd
        print("✓ comfy.sd imported")
        
        import comfy.utils
        print("✓ comfy.utils imported")
        
        import comfy.model_management
        print("✓ comfy.model_management imported")
        
        print("✓ Basic imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    test_basic_imports() 