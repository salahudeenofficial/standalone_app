#!/usr/bin/env python3
"""
Simple test for comfy.sd import
"""

import sys
from pathlib import Path

# Add the comfy modules to path
sys.path.insert(0, str(Path(__file__).parent / "comfy"))

try:
    import comfy.sd
    print("✓ comfy.sd imported successfully!")
except Exception as e:
    print(f"❌ Error importing comfy.sd: {e}")
    import traceback
    traceback.print_exc() 