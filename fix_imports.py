#!/usr/bin/env python3
"""
Script to fix node_helpers import statements
"""

import os
import glob

def fix_imports():
    """Fix node_helpers imports in all Python files"""
    
    # Fix imports in comfy_extras directory
    comfy_extras_files = glob.glob("comfy_extras/*.py")
    
    for file_path in comfy_extras_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Replace import node_helpers with from comfy import node_helpers
            if "import node_helpers" in content:
                content = content.replace("import node_helpers", "from comfy import node_helpers")
                print(f"Fixed imports in: {file_path}")
            
            with open(file_path, 'w') as f:
                f.write(content)
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    fix_imports()
    print("Import fixes completed!") 