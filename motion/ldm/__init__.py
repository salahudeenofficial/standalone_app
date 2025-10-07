"""
Motion Pipeline LDM Module
Lightweight Diffusion Model implementation compatible with ComfyUI patterns
"""

# Version info
__version__ = "1.0.0"
__author__ = "Motion Pipeline Team"

# Sub-modules
from . import modules
from . import flux

# Main exports
__all__ = [
    'modules',
    'flux'
]
