#!/usr/bin/env python3
"""
ComfyUI Step 4 Reference Data Collector - Example Usage
Example script showing how to use the collector in ComfyUI environment
"""

import sys
import os
import torch
import time
from pathlib import Path

# Add motion directory to path (adjust as needed)
sys.path.insert(0, str(Path(__file__).parent))

from comfyui_step4_reference_data_collector import ComfyUIStep4ReferenceCollector

def example_comfyui_reference_collection():
    """
    Example of how to use the ComfyUI reference data collector
    This should be run inside ComfyUI environment
    """
    print("🚀 ComfyUI Step 4 Reference Data Collector - Example")
    print("=" * 60)
    
    # Initialize collector
    collector = ComfyUIStep4ReferenceCollector()
    
    # Example 1: Basic usage with dummy data
    print("\n📊 Example 1: Basic Usage with Dummy Data")
    print("-" * 40)
    
    try:
        # Create dummy data (replace with your actual data)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Dummy initial latent [B, C, T, H, W]
        initial_latent = torch.randn(1, 16, 9, 104, 60, device=device, dtype=torch.float32)
        
        # Dummy conditioning (replace with your actual conditioning)
        positive_conditioning = [
            torch.randn(1, 77, 4096, device=device, dtype=torch.float32),  # Text embedding
            {
                "pooled_output": torch.randn(1, 1024, device=device, dtype=torch.float32),
                "vace_frames": [torch.randn(1, 32, 9, 104, 60, device=device, dtype=torch.float32)],
                "vace_mask": [torch.ones(1, 1, 9, 104, 60, device=device, dtype=torch.float32)],
                "vace_strength": [0.8]
            }
        ]
        
        negative_conditioning = [
            torch.randn(1, 77, 4096, device=device, dtype=torch.float32),  # Text embedding
            {
                "pooled_output": torch.randn(1, 1024, device=device, dtype=torch.float32),
                "vace_frames": [torch.randn(1, 32, 9, 104, 60, device=device, dtype=torch.float32)],
                "vace_mask": [torch.ones(1, 1, 9, 104, 60, device=device, dtype=torch.float32)],
                "vace_strength": [0.8]
            }
        ]
        
        # Dummy UNet model (replace with your actual UNet)
        # Note: In real usage, you would load your actual UNet model
        print("⚠️  Using dummy UNet model - replace with your actual UNet in real usage")
        unet_model = None  # Replace with your actual UNet model
        
        if unet_model is None:
            print("❌ UNet model not provided - skipping example")
            return False
        
        # Collect reference data
        reference_data = collector.collect_all_sections_reference(
            initial_latent=initial_latent,
            positive_conditioning=positive_conditioning,
            negative_conditioning=negative_conditioning,
            unet_model=unet_model,
            seed=42,
            steps=4,
            cfg=7.0,
            sampler_name="euler",
            scheduler="normal",
            denoise=1.0
        )
        
        # Save reference data
        filename = collector.save_reference_data(reference_data)
        print(f"✅ Reference data saved to: {filename}")
        
        return True
        
    except Exception as e:
        print(f"❌ Example 1 failed: {e}")
        return False

def example_individual_sections():
    """
    Example of collecting individual sections
    """
    print("\n📊 Example 2: Individual Sections Collection")
    print("-" * 40)
    
    try:
        collector = ComfyUIStep4ReferenceCollector()
        
        # Create dummy data
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        initial_latent = torch.randn(1, 16, 9, 104, 60, device=device, dtype=torch.float32)
        
        # Dummy UNet model (replace with your actual UNet)
        unet_model = None  # Replace with your actual UNet model
        
        if unet_model is None:
            print("❌ UNet model not provided - skipping example")
            return False
        
        # Section 4.2: Latent Preparation
        print("🔧 Collecting Section 4.2...")
        section_4_2 = collector.collect_section_4_2_reference(
            initial_latent=initial_latent,
            unet_model=unet_model,
            seed=42
        )
        
        if 'error' in section_4_2:
            print(f"❌ Section 4.2 failed: {section_4_2['error']}")
            return False
        
        print("✅ Section 4.2 collected successfully")
        
        # Section 4.3: KSampler Setup
        print("⚙️ Collecting Section 4.3...")
        section_4_3 = collector.collect_section_4_3_reference(
            fixed_latent=section_4_2['outputs']['fixed_latent_tensor'],
            unet_model=unet_model,
            steps=4,
            sampler_name="euler",
            scheduler="normal",
            denoise=1.0
        )
        
        if 'error' in section_4_3:
            print(f"❌ Section 4.3 failed: {section_4_3['error']}")
            return False
        
        print("✅ Section 4.3 collected successfully")
        
        # Section 4.4: Denoising Execution
        print("🎯 Collecting Section 4.4...")
        
        # Dummy conditioning
        positive_conditioning = [torch.randn(1, 77, 4096, device=device, dtype=torch.float32)]
        negative_conditioning = [torch.randn(1, 77, 4096, device=device, dtype=torch.float32)]
        
        section_4_4 = collector.collect_section_4_4_reference(
            ksampler=section_4_3['outputs']['ksampler_instance'],
            noise=section_4_2['outputs']['noise_tensor'],
            positive_conditioning=positive_conditioning,
            negative_conditioning=negative_conditioning,
            fixed_latent=section_4_2['outputs']['fixed_latent_tensor'],
            cfg=7.0,
            seed=42
        )
        
        if 'error' in section_4_4:
            print(f"❌ Section 4.4 failed: {section_4_4['error']}")
            return False
        
        print("✅ Section 4.4 collected successfully")
        
        # Save individual sections
        all_sections = {
            'sections': {
                '4.2': section_4_2,
                '4.3': section_4_3,
                '4.4': section_4_4
            },
            'overall_status': 'success',
            'comfyui_version': 'reference'
        }
        
        filename = collector.save_reference_data(all_sections, "individual_sections_example.json")
        print(f"✅ Individual sections saved to: {filename}")
        
        return True
        
    except Exception as e:
        print(f"❌ Example 2 failed: {e}")
        return False

def example_custom_parameters():
    """
    Example with custom parameters
    """
    print("\n📊 Example 3: Custom Parameters")
    print("-" * 40)
    
    try:
        collector = ComfyUIStep4ReferenceCollector()
        
        # Create dummy data
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        initial_latent = torch.randn(1, 16, 9, 104, 60, device=device, dtype=torch.float32)
        
        # Dummy conditioning
        positive_conditioning = [torch.randn(1, 77, 4096, device=device, dtype=torch.float32)]
        negative_conditioning = [torch.randn(1, 77, 4096, device=device, dtype=torch.float32)]
        
        # Dummy UNet model (replace with your actual UNet)
        unet_model = None  # Replace with your actual UNet model
        
        if unet_model is None:
            print("❌ UNet model not provided - skipping example")
            return False
        
        # Custom parameters for different test scenarios
        custom_params = {
            'seed': 123,                    # Different seed
            'steps': 8,                     # More steps
            'cfg': 12.0,                    # Higher CFG
            'sampler_name': "dpmpp_2m",     # Different sampler
            'scheduler': "karras",          # Different scheduler
            'denoise': 0.8                  # Partial denoising
        }
        
        print(f"Custom parameters: {custom_params}")
        
        # Collect reference data with custom parameters
        reference_data = collector.collect_all_sections_reference(
            initial_latent=initial_latent,
            positive_conditioning=positive_conditioning,
            negative_conditioning=negative_conditioning,
            unet_model=unet_model,
            **custom_params
        )
        
        # Save with descriptive filename
        filename = collector.save_reference_data(
            reference_data, 
            "comfyui_step4_reference_custom_params.json"
        )
        print(f"✅ Custom parameters reference saved to: {filename}")
        
        return True
        
    except Exception as e:
        print(f"❌ Example 3 failed: {e}")
        return False

def main():
    """
    Main function to run all examples
    """
    print("🚀 ComfyUI Step 4 Reference Data Collector - Examples")
    print("=" * 60)
    
    print("This script demonstrates how to use the ComfyUI reference data collector.")
    print("⚠️  Note: These examples use dummy data. In real usage, replace with your actual data.")
    print()
    
    # Check if running in ComfyUI environment
    try:
        import comfy.sample
        import comfy.samplers
        print("✅ ComfyUI environment detected")
    except ImportError:
        print("❌ ComfyUI environment not detected")
        print("This script should be run inside ComfyUI environment")
        return False
    
    # Run examples
    examples = [
        ("Basic Usage", example_comfyui_reference_collection),
        ("Individual Sections", example_individual_sections),
        ("Custom Parameters", example_custom_parameters)
    ]
    
    results = {}
    for name, example_func in examples:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print('='*60)
        
        try:
            success = example_func()
            results[name] = success
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"\n{name}: {status}")
        except Exception as e:
            print(f"\n{name}: ❌ FAILED - {e}")
            results[name] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    
    for name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{name}: {status}")
    
    total_success = sum(results.values())
    total_examples = len(results)
    
    print(f"\nTotal: {total_success}/{total_examples} examples succeeded")
    
    if total_success == total_examples:
        print("🎉 All examples completed successfully!")
        return True
    else:
        print("⚠️  Some examples failed - check the output above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
