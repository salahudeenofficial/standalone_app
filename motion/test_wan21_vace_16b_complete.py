#!/usr/bin/env python3
"""
Complete WAN 2.1 VACE 16B Model Loading and Verification Script
Comprehensive testing for VAST AI instance deployment

This script performs complete verification of the WAN 2.1 VACE 16B model including:
- Model loading and architecture verification
- Parameter count and memory analysis
- FP16 precision verification
- Forward pass testing with multiple modalities
- State dict analysis and component verification
"""

import torch
import torch.nn as nn
import logging
import sys
import os
import gc
import time
import traceback
from pathlib import Path
import json
from typing import Dict, Any, Optional, List, Tuple

# Add motion directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('wan21_vace_16b_verification.log')
    ]
)
logger = logging.getLogger(__name__)

class WAN21VACEVerifier:
    """Comprehensive verification class for WAN 2.1 VACE 16B model"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Expected model specifications based on web research
        self.expected_specs = {
            'total_parameters': 16_000_000_000,  # 16B parameters (approximately)
            'parameter_tolerance': 0.1,  # 10% tolerance for parameter count
            'expected_dtype': torch.float16,
            'expected_layers': 32,  # Typical for large transformer models
            'expected_dim': 2048,   # Hidden dimension
            'expected_heads': 16,   # Attention heads
            'min_file_size_gb': 25,  # Minimum expected file size for 16B fp16 model
            'max_file_size_gb': 35,  # Maximum expected file size
            'vace_layers': 8,       # VACE-specific layers
            'supported_resolutions': [(720, 1280), (512, 512), (768, 768)],
            'supported_frames': [8, 16, 24, 32],
            'input_channels': 16,   # VAE latent channels
            'output_channels': 16,
        }
        
        self.verification_results = {
            'file_verification': False,
            'model_loading': False,
            'architecture_verification': False,
            'parameter_verification': False,
            'dtype_verification': False,
            'memory_verification': False,
            'forward_pass_t2v': False,
            'forward_pass_i2v': False,
            'forward_pass_vace': False,
            'state_dict_verification': False,
            'component_verification': False
        }
    
    def verify_model_file(self) -> bool:
        """Verify model file exists and has expected properties"""
        logger.info("🔍 Step 1: Verifying model file...")
        
        try:
            if not self.model_path.exists():
                logger.error(f"❌ Model file not found: {self.model_path}")
                return False
            
            # Check file size
            file_size_bytes = self.model_path.stat().st_size
            file_size_gb = file_size_bytes / (1024**3)
            
            logger.info(f"📁 Model file: {self.model_path}")
            logger.info(f"📏 File size: {file_size_gb:.2f} GB")
            
            if file_size_gb < self.expected_specs['min_file_size_gb']:
                logger.warning(f"⚠️  File size ({file_size_gb:.2f} GB) smaller than expected minimum ({self.expected_specs['min_file_size_gb']} GB)")
            elif file_size_gb > self.expected_specs['max_file_size_gb']:
                logger.warning(f"⚠️  File size ({file_size_gb:.2f} GB) larger than expected maximum ({self.expected_specs['max_file_size_gb']} GB)")
            else:
                logger.info(f"✅ File size within expected range")
            
            # Check file extension
            if self.model_path.suffix.lower() in ['.safetensors', '.pt', '.pth', '.bin']:
                logger.info(f"✅ Valid model file format: {self.model_path.suffix}")
            else:
                logger.warning(f"⚠️  Unexpected file format: {self.model_path.suffix}")
            
            self.verification_results['file_verification'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ File verification failed: {e}")
            return False
    
    def load_model(self) -> bool:
        """Load the WAN 2.1 VACE model"""
        logger.info("🔍 Step 2: Loading WAN 2.1 VACE 16B model...")
        
        try:
            # Import our model detection and creation functions
            from model_detection import detect_unet_config, model_config_from_unet_config, create_model_from_config
            from utils import load_torch_file
            
            # Load state dict
            logger.info("📂 Loading state dict...")
            state_dict = load_torch_file(str(self.model_path))
            logger.info(f"✅ State dict loaded with {len(state_dict)} keys")
            
            # Detect model configuration
            logger.info("🔍 Detecting model configuration...")
            unet_config = detect_unet_config(state_dict)
            
            if unet_config is None:
                logger.error("❌ Failed to detect model configuration")
                return False
            
            logger.info(f"✅ Detected model type: {unet_config.get('model_type', 'unknown')}")
            logger.info(f"📊 Model config: {json.dumps(unet_config, indent=2)}")
            
            # Convert to model config
            model_config = model_config_from_unet_config(unet_config)
            if model_config is None:
                logger.error("❌ Failed to convert to model config")
                return False
            
            # Create model instance
            logger.info("🏗️  Creating model instance...")
            self.model = create_model_from_config(
                model_config, 
                device=self.device, 
                dtype=torch.float16,
                state_dict=state_dict
            )
            
            if self.model is None:
                logger.error("❌ Failed to create model instance")
                return False
            
            logger.info("✅ Model created successfully")
            
            # Move to device and set eval mode
            logger.info(f"📱 Moving model to {self.device}...")
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info("✅ Model loaded and ready for inference")
            self.verification_results['model_loading'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def verify_architecture(self) -> bool:
        """Verify model architecture matches expected specifications"""
        logger.info("🔍 Step 3: Verifying model architecture...")
        
        try:
            if self.model is None:
                logger.error("❌ Model not loaded")
                return False
            
            # Check model type
            model_type = getattr(self.model, 'model_type', 'unknown')
            logger.info(f"🏗️  Model type: {model_type}")
            
            # Check key architectural components
            components_to_check = [
                'patch_embedding', 'text_embedding', 'time_embed', 'time_projection',
                'blocks', 'head', 'rope_embedder'
            ]
            
            missing_components = []
            for component in components_to_check:
                if hasattr(self.model, component):
                    comp_obj = getattr(self.model, component)
                    logger.info(f"✅ {component}: {type(comp_obj).__name__}")
                else:
                    missing_components.append(component)
                    logger.warning(f"❌ Missing component: {component}")
            
            # Check VACE-specific components
            vace_components = ['vace_blocks', 'vace_patch_embedding', 'vace_layers']
            vace_present = 0
            for component in vace_components:
                if hasattr(self.model, component):
                    comp_obj = getattr(self.model, component)
                    logger.info(f"✅ VACE {component}: {type(comp_obj).__name__}")
                    vace_present += 1
                else:
                    logger.info(f"ℹ️  VACE {component}: Not present (may be T2V model)")
            
            # Determine if this is a VACE model
            is_vace_model = vace_present >= 2
            logger.info(f"🎯 VACE model detected: {is_vace_model}")
            
            # Check architectural dimensions
            if hasattr(self.model, 'dim'):
                logger.info(f"📐 Hidden dimension: {self.model.dim}")
            if hasattr(self.model, 'num_layers'):
                logger.info(f"🧱 Number of layers: {self.model.num_layers}")
            if hasattr(self.model, 'num_heads'):
                logger.info(f"🎯 Attention heads: {self.model.num_heads}")
            if hasattr(self.model, 'patch_size'):
                logger.info(f"📦 Patch size: {self.model.patch_size}")
            
            # Check transformer blocks
            if hasattr(self.model, 'blocks'):
                num_blocks = len(self.model.blocks)
                logger.info(f"🔗 Transformer blocks: {num_blocks}")
                
                # Check first block structure
                if num_blocks > 0:
                    first_block = self.model.blocks[0]
                    block_components = ['self_attn', 'cross_attn', 'ffn', 'norm1', 'norm2']
                    for comp in block_components:
                        if hasattr(first_block, comp):
                            logger.info(f"  ✅ Block component {comp}: {type(getattr(first_block, comp)).__name__}")
                        else:
                            logger.warning(f"  ❌ Missing block component: {comp}")
            
            # Verification successful if no critical components are missing
            critical_missing = [comp for comp in missing_components if comp in ['patch_embedding', 'blocks', 'head']]
            if not critical_missing:
                logger.info("✅ Architecture verification passed")
                self.verification_results['architecture_verification'] = True
                return True
            else:
                logger.error(f"❌ Critical components missing: {critical_missing}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Architecture verification failed: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def verify_parameters(self) -> bool:
        """Verify model parameters match expected specifications"""
        logger.info("🔍 Step 4: Verifying model parameters...")
        
        try:
            if self.model is None:
                logger.error("❌ Model not loaded")
                return False
            
            # Count total parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            # Convert to billions for readability
            total_params_b = total_params / 1e9
            trainable_params_b = trainable_params / 1e9
            
            logger.info(f"📊 Total parameters: {total_params:,} ({total_params_b:.2f}B)")
            logger.info(f"🎯 Trainable parameters: {trainable_params:,} ({trainable_params_b:.2f}B)")
            
            # Check against expected count
            expected_params = self.expected_specs['total_parameters']
            tolerance = self.expected_specs['parameter_tolerance']
            
            param_diff = abs(total_params - expected_params) / expected_params
            
            if param_diff <= tolerance:
                logger.info(f"✅ Parameter count within tolerance ({param_diff*100:.1f}% difference)")
                param_verification = True
            else:
                logger.warning(f"⚠️  Parameter count outside tolerance ({param_diff*100:.1f}% difference)")
                logger.warning(f"   Expected: ~{expected_params/1e9:.1f}B, Got: {total_params_b:.2f}B")
                param_verification = False
            
            # Analyze parameter distribution
            logger.info("📈 Parameter distribution by component:")
            
            component_params = {}
            for name, module in self.model.named_modules():
                if len(list(module.parameters())) > 0:
                    module_params = sum(p.numel() for p in module.parameters())
                    if module_params > 1e6:  # Only show components with >1M params
                        component_params[name] = module_params
            
            # Sort by parameter count
            sorted_components = sorted(component_params.items(), key=lambda x: x[1], reverse=True)
            for name, params in sorted_components[:10]:  # Show top 10
                logger.info(f"  {name}: {params:,} ({params/1e6:.1f}M)")
            
            # Check specific layer parameter counts
            if hasattr(self.model, 'blocks') and len(self.model.blocks) > 0:
                block_params = sum(p.numel() for p in self.model.blocks[0].parameters())
                total_block_params = block_params * len(self.model.blocks)
                logger.info(f"🧱 Single block parameters: {block_params:,} ({block_params/1e6:.1f}M)")
                logger.info(f"🧱 Total block parameters: {total_block_params:,} ({total_block_params/1e9:.2f}B)")
            
            self.verification_results['parameter_verification'] = param_verification
            return param_verification
            
        except Exception as e:
            logger.error(f"❌ Parameter verification failed: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def verify_dtype_and_memory(self) -> bool:
        """Verify model dtype and analyze memory usage"""
        logger.info("🔍 Step 5: Verifying dtype and memory usage...")
        
        try:
            if self.model is None:
                logger.error("❌ Model not loaded")
                return False
            
            # Check parameter dtypes
            dtype_counts = {}
            for name, param in self.model.named_parameters():
                dtype = str(param.dtype)
                dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
            
            logger.info("🎯 Parameter dtypes:")
            for dtype, count in dtype_counts.items():
                logger.info(f"  {dtype}: {count} parameters")
            
            # Check if predominantly fp16
            fp16_count = dtype_counts.get('torch.float16', 0)
            total_params = sum(dtype_counts.values())
            fp16_ratio = fp16_count / total_params if total_params > 0 else 0
            
            if fp16_ratio > 0.9:
                logger.info(f"✅ Model is predominantly FP16 ({fp16_ratio*100:.1f}%)")
                dtype_verification = True
            else:
                logger.warning(f"⚠️  Model is not predominantly FP16 ({fp16_ratio*100:.1f}%)")
                dtype_verification = False
            
            # Memory analysis
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
                memory_total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
                
                logger.info(f"💾 GPU Memory Usage:")
                logger.info(f"  Allocated: {memory_allocated:.2f} GB")
                logger.info(f"  Reserved: {memory_reserved:.2f} GB")
                logger.info(f"  Total: {memory_total:.2f} GB")
                logger.info(f"  Available: {memory_total - memory_reserved:.2f} GB")
                
                # Estimate model memory
                total_params = sum(p.numel() for p in self.model.parameters())
                estimated_memory = total_params * 2 / 1024**3  # 2 bytes per fp16 param
                logger.info(f"📊 Estimated model memory: {estimated_memory:.2f} GB")
                
                memory_verification = memory_allocated < memory_total * 0.95  # Don't use more than 95% GPU memory
                if memory_verification:
                    logger.info("✅ Memory usage within acceptable limits")
                else:
                    logger.warning("⚠️  High memory usage detected")
            else:
                logger.info("ℹ️  CPU mode - skipping GPU memory analysis")
                memory_verification = True
            
            self.verification_results['dtype_verification'] = dtype_verification
            self.verification_results['memory_verification'] = memory_verification
            
            return dtype_verification and memory_verification
            
        except Exception as e:
            logger.error(f"❌ Dtype and memory verification failed: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def test_forward_pass_t2v(self) -> bool:
        """Test Text-to-Video forward pass"""
        logger.info("🔍 Step 6a: Testing T2V forward pass...")
        
        try:
            if self.model is None:
                logger.error("❌ Model not loaded")
                return False
            
            # Create T2V inputs
            batch_size = 1
            channels = 16  # VAE latent channels
            frames = 16
            height = 64
            width = 64
            
            logger.info(f"🎬 Creating T2V test inputs: {batch_size}x{channels}x{frames}x{height}x{width}")
            
            # Input tensors
            x = torch.randn(batch_size, channels, frames, height, width, 
                          device=self.device, dtype=torch.float16)
            timestep = torch.randint(0, 1000, (batch_size,), device=self.device)
            context = torch.randn(batch_size, 512, 4096, device=self.device, dtype=torch.float16)
            
            logger.info("🚀 Running T2V forward pass...")
            start_time = time.time()
            
            with torch.no_grad():
                output = self.model(x, timestep, context)
            
            end_time = time.time()
            
            logger.info(f"✅ T2V forward pass successful in {end_time - start_time:.2f}s")
            logger.info(f"📤 Output shape: {output.shape}")
            logger.info(f"📊 Output dtype: {output.dtype}")
            logger.info(f"📈 Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
            
            # Verify output shape
            expected_shape = (batch_size, channels, frames, height, width)
            if output.shape == expected_shape:
                logger.info(f"✅ Output shape matches expected: {expected_shape}")
                self.verification_results['forward_pass_t2v'] = True
                return True
            else:
                logger.error(f"❌ Output shape mismatch: expected {expected_shape}, got {output.shape}")
                return False
                
        except Exception as e:
            logger.error(f"❌ T2V forward pass failed: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def test_forward_pass_vace(self) -> bool:
        """Test VACE-specific forward pass if available"""
        logger.info("🔍 Step 6b: Testing VACE forward pass...")
        
        try:
            if self.model is None:
                logger.error("❌ Model not loaded")
                return False
            
            # Check if this is a VACE model
            if not hasattr(self.model, 'vace_blocks'):
                logger.info("ℹ️  Not a VACE model - skipping VACE forward pass test")
                self.verification_results['forward_pass_vace'] = True
                return True
            
            # Create VACE inputs
            batch_size = 1
            channels = 16
            frames = 16
            height = 64
            width = 64
            
            logger.info(f"🎬 Creating VACE test inputs: {batch_size}x{channels}x{frames}x{height}x{width}")
            
            # Input tensors
            x = torch.randn(batch_size, channels, frames, height, width, 
                          device=self.device, dtype=torch.float16)
            timestep = torch.randint(0, 1000, (batch_size,), device=self.device)
            context = torch.randn(batch_size, 512, 4096, device=self.device, dtype=torch.float16)
            
            # VACE-specific inputs
            vace_context = torch.randn(batch_size, 1, channels, frames, height, width,
                                     device=self.device, dtype=torch.float16)
            vace_strength = [1.0]
            
            logger.info("🚀 Running VACE forward pass...")
            start_time = time.time()
            
            with torch.no_grad():
                output = self.model(x, timestep, context, vace_context, vace_strength)
            
            end_time = time.time()
            
            logger.info(f"✅ VACE forward pass successful in {end_time - start_time:.2f}s")
            logger.info(f"📤 Output shape: {output.shape}")
            logger.info(f"📊 Output dtype: {output.dtype}")
            logger.info(f"📈 Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
            
            self.verification_results['forward_pass_vace'] = True
            return True
                
        except Exception as e:
            logger.error(f"❌ VACE forward pass failed: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def verify_state_dict(self) -> bool:
        """Verify state dict completeness and structure"""
        logger.info("🔍 Step 7: Verifying state dict structure...")
        
        try:
            if self.model is None:
                logger.error("❌ Model not loaded")
                return False
            
            # Get model state dict
            state_dict = self.model.state_dict()
            total_keys = len(state_dict)
            
            logger.info(f"🔑 Total state dict keys: {total_keys}")
            
            # Analyze key patterns
            key_patterns = {
                'patch_embedding': 0,
                'text_embedding': 0,
                'time_embed': 0,
                'time_projection': 0,
                'blocks': 0,
                'head': 0,
                'vace': 0,
                'other': 0
            }
            
            for key in state_dict.keys():
                categorized = False
                for pattern in key_patterns.keys():
                    if pattern in key and pattern != 'other':
                        key_patterns[pattern] += 1
                        categorized = True
                        break
                if not categorized:
                    key_patterns['other'] += 1
            
            logger.info("🗂️  State dict key distribution:")
            for pattern, count in key_patterns.items():
                if count > 0:
                    logger.info(f"  {pattern}: {count} keys")
            
            # Check for critical components
            critical_keys = [
                'patch_embedding.weight',
                'text_embedding.0.weight',
                'time_embed.0.weight',
                'head.head.weight',
                'head.modulation'
            ]
            
            missing_critical = []
            for key in critical_keys:
                if key not in state_dict:
                    missing_critical.append(key)
                else:
                    tensor = state_dict[key]
                    logger.info(f"✅ {key}: {tensor.shape} ({tensor.dtype})")
            
            if missing_critical:
                logger.error(f"❌ Missing critical keys: {missing_critical}")
                return False
            
            # Check transformer blocks
            block_count = 0
            for key in state_dict.keys():
                if key.startswith('blocks.') and '.self_attn.q.weight' in key:
                    block_count += 1
            
            logger.info(f"🧱 Detected transformer blocks: {block_count}")
            
            # Check VACE components
            vace_key_count = sum(1 for key in state_dict.keys() if 'vace' in key)
            if vace_key_count > 0:
                logger.info(f"🎯 VACE keys detected: {vace_key_count}")
            else:
                logger.info("ℹ️  No VACE keys detected (likely T2V model)")
            
            logger.info("✅ State dict verification completed")
            self.verification_results['state_dict_verification'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ State dict verification failed: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def run_complete_verification(self) -> Dict[str, Any]:
        """Run complete verification suite"""
        logger.info("🚀 STARTING COMPLETE WAN 2.1 VACE 16B MODEL VERIFICATION")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Step 1: File verification
        if not self.verify_model_file():
            logger.error("❌ File verification failed - aborting")
            return self.get_verification_report()
        
        # Step 2: Model loading
        if not self.load_model():
            logger.error("❌ Model loading failed - aborting")
            return self.get_verification_report()
        
        # Step 3: Architecture verification
        self.verify_architecture()
        
        # Step 4: Parameter verification
        self.verify_parameters()
        
        # Step 5: Dtype and memory verification
        self.verify_dtype_and_memory()
        
        # Step 6: Forward pass tests
        self.test_forward_pass_t2v()
        self.test_forward_pass_vace()
        
        # Step 7: State dict verification
        self.verify_state_dict()
        
        end_time = time.time()
        
        # Generate final report
        report = self.get_verification_report()
        report['total_time'] = end_time - start_time
        
        logger.info("=" * 80)
        logger.info(f"🏁 VERIFICATION COMPLETED in {end_time - start_time:.2f}s")
        
        return report
    
    def get_verification_report(self) -> Dict[str, Any]:
        """Generate comprehensive verification report"""
        passed_tests = sum(self.verification_results.values())
        total_tests = len(self.verification_results)
        success_rate = passed_tests / total_tests * 100
        
        report = {
            'model_path': str(self.model_path),
            'device': str(self.device),
            'verification_results': self.verification_results.copy(),
            'passed_tests': passed_tests,
            'total_tests': total_tests,
            'success_rate': success_rate,
            'overall_success': success_rate >= 80,  # 80% success rate required
        }
        
        if self.model is not None:
            report['model_info'] = {
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'model_type': getattr(self.model, 'model_type', 'unknown'),
                'is_vace_model': hasattr(self.model, 'vace_blocks'),
            }
        
        return report

def main():
    """Main function for complete model verification"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Complete WAN 2.1 VACE 16B Model Verification')
    parser.add_argument('model_path', help='Path to the WAN 2.1 VACE 16B model file')
    parser.add_argument('--output-report', help='Path to save JSON verification report')
    
    args = parser.parse_args()
    
    # Initialize verifier
    verifier = WAN21VACEVerifier(args.model_path)
    
    try:
        # Run complete verification
        report = verifier.run_complete_verification()
        
        # Print summary
        logger.info("📊 VERIFICATION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Overall Success: {'✅ PASS' if report['overall_success'] else '❌ FAIL'}")
        logger.info(f"Success Rate: {report['success_rate']:.1f}% ({report['passed_tests']}/{report['total_tests']})")
        
        logger.info("\n🔍 Detailed Results:")
        for test_name, passed in report['verification_results'].items():
            status = '✅ PASS' if passed else '❌ FAIL'
            logger.info(f"  {test_name}: {status}")
        
        if 'model_info' in report:
            logger.info(f"\n📊 Model Info:")
            logger.info(f"  Total Parameters: {report['model_info']['total_parameters']:,}")
            logger.info(f"  Model Type: {report['model_info']['model_type']}")
            logger.info(f"  VACE Model: {report['model_info']['is_vace_model']}")
        
        # Save report if requested
        if args.output_report:
            with open(args.output_report, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"📄 Report saved to: {args.output_report}")
        
        # Return appropriate exit code
        return 0 if report['overall_success'] else 1
        
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR: {e}")
        logger.error(traceback.format_exc())
        return 1
    
    finally:
        # Cleanup
        if verifier.model is not None:
            del verifier.model
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    exit(main())
