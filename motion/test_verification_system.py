#!/usr/bin/env python3
"""
Test script to verify the verification system works correctly
Creates a dummy model checkpoint to test the verification pipeline
"""

import torch
import logging
import sys
import os
from pathlib import Path

# Add motion directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def create_dummy_wan21_vace_model():
    """Create a dummy WAN 2.1 VACE model checkpoint for testing"""
    logger.info("🔧 Creating dummy WAN 2.1 VACE model checkpoint for testing...")
    
    # Model configuration matching expected WAN 2.1 VACE structure
    config = {
        'dim': 2048,
        'num_layers': 4,  # Smaller for testing
        'num_heads': 16,
        'ffn_dim': 8192,
        'freq_dim': 100,  # Test special case
        'text_dim': 4096,
        'in_dim': 16,
        'out_dim': 16,
        'patch_size': (1, 2, 2),
        'vace_layers': 2,  # Smaller for testing
        'vace_in_dim': 16
    }
    
    # Create state dict with correct structure
    state_dict = {}
    
    # Patch embedding
    state_dict['patch_embedding.weight'] = torch.randn(config['dim'], config['in_dim'], *config['patch_size']).to(torch.float16)
    state_dict['patch_embedding.bias'] = torch.randn(config['dim']).to(torch.float16)
    
    # Text embedding
    state_dict['text_embedding.0.weight'] = torch.randn(config['dim'], config['text_dim']).to(torch.float16)
    state_dict['text_embedding.0.bias'] = torch.randn(config['dim']).to(torch.float16)
    state_dict['text_embedding.2.weight'] = torch.randn(config['dim'], config['dim']).to(torch.float16)
    state_dict['text_embedding.2.bias'] = torch.randn(config['dim']).to(torch.float16)
    
    # Time embedding - Special case: freq_dim=100, dim=2048
    state_dict['time_embed.0.weight'] = torch.randn(config['freq_dim'], config['freq_dim']).to(torch.float16)  # [100, 100]
    state_dict['time_embed.0.bias'] = torch.randn(config['freq_dim']).to(torch.float16)
    state_dict['time_embed.2.weight'] = torch.randn(config['dim'], config['freq_dim']).to(torch.float16)  # [2048, 100]
    state_dict['time_embed.2.bias'] = torch.randn(config['dim']).to(torch.float16)
    
    # Time projection
    state_dict['time_projection.1.weight'] = torch.randn(config['dim'] * 6, config['dim']).to(torch.float16)
    state_dict['time_projection.1.bias'] = torch.randn(config['dim'] * 6).to(torch.float16)
    
    # Transformer blocks
    for i in range(config['num_layers']):
        prefix = f'blocks.{i}'
        
        # Self attention
        state_dict[f'{prefix}.self_attn.q.weight'] = torch.randn(config['dim'], config['dim']).to(torch.float16)
        state_dict[f'{prefix}.self_attn.k.weight'] = torch.randn(config['dim'], config['dim']).to(torch.float16)
        state_dict[f'{prefix}.self_attn.v.weight'] = torch.randn(config['dim'], config['dim']).to(torch.float16)
        state_dict[f'{prefix}.self_attn.o.weight'] = torch.randn(config['dim'], config['dim']).to(torch.float16)
        
        # Cross attention
        state_dict[f'{prefix}.cross_attn.q.weight'] = torch.randn(config['dim'], config['dim']).to(torch.float16)
        state_dict[f'{prefix}.cross_attn.k.weight'] = torch.randn(config['dim'], config['dim']).to(torch.float16)
        state_dict[f'{prefix}.cross_attn.v.weight'] = torch.randn(config['dim'], config['dim']).to(torch.float16)
        state_dict[f'{prefix}.cross_attn.o.weight'] = torch.randn(config['dim'], config['dim']).to(torch.float16)
        
        # FFN
        state_dict[f'{prefix}.ffn.0.weight'] = torch.randn(config['ffn_dim'], config['dim']).to(torch.float16)
        state_dict[f'{prefix}.ffn.0.bias'] = torch.randn(config['ffn_dim']).to(torch.float16)
        state_dict[f'{prefix}.ffn.2.weight'] = torch.randn(config['dim'], config['ffn_dim']).to(torch.float16)
        state_dict[f'{prefix}.ffn.2.bias'] = torch.randn(config['dim']).to(torch.float16)
        
        # Normalization
        state_dict[f'{prefix}.norm1.weight'] = torch.ones(config['dim']).to(torch.float16)
        state_dict[f'{prefix}.norm2.weight'] = torch.ones(config['dim']).to(torch.float16)
        
        # Modulation
        state_dict[f'{prefix}.modulation'] = torch.randn(1, 6, config['dim']).to(torch.float16)
    
    # VACE components
    state_dict['vace_patch_embedding.weight'] = torch.randn(config['dim'], config['vace_in_dim'], *config['patch_size']).to(torch.float16)
    state_dict['vace_patch_embedding.bias'] = torch.randn(config['dim']).to(torch.float16)
    
    for i in range(config['vace_layers']):
        prefix = f'vace_blocks.{i}'
        
        # VACE attention components (similar to regular blocks)
        state_dict[f'{prefix}.self_attn.q.weight'] = torch.randn(config['dim'], config['dim']).to(torch.float16)
        state_dict[f'{prefix}.self_attn.k.weight'] = torch.randn(config['dim'], config['dim']).to(torch.float16)
        state_dict[f'{prefix}.self_attn.v.weight'] = torch.randn(config['dim'], config['dim']).to(torch.float16)
        state_dict[f'{prefix}.self_attn.o.weight'] = torch.randn(config['dim'], config['dim']).to(torch.float16)
        
        # VACE-specific projections
        if i == 0:
            state_dict[f'{prefix}.before_proj.weight'] = torch.randn(config['dim'], config['dim']).to(torch.float16)
            state_dict[f'{prefix}.before_proj.bias'] = torch.randn(config['dim']).to(torch.float16)
        
        state_dict[f'{prefix}.after_proj.weight'] = torch.randn(config['dim'], config['dim']).to(torch.float16)
        state_dict[f'{prefix}.after_proj.bias'] = torch.randn(config['dim']).to(torch.float16)
    
    # Head
    output_dim = config['out_dim'] * config['patch_size'][0] * config['patch_size'][1] * config['patch_size'][2]
    state_dict['head.head.weight'] = torch.randn(output_dim, config['dim']).to(torch.float16)
    state_dict['head.head.bias'] = torch.randn(output_dim).to(torch.float16)
    state_dict['head.modulation'] = torch.randn(1, 2, config['dim']).to(torch.float16)
    
    logger.info(f"✅ Created dummy model with {len(state_dict)} keys")
    logger.info(f"📊 Total parameters: {sum(t.numel() for t in state_dict.values()):,}")
    
    return state_dict, config

def test_verification_system():
    """Test the verification system with a dummy model"""
    logger.info("🧪 TESTING WAN 2.1 VACE VERIFICATION SYSTEM")
    logger.info("=" * 60)
    
    try:
        # Create dummy model
        dummy_state_dict, config = create_dummy_wan21_vace_model()
        
        # Save dummy model
        dummy_model_path = "dummy_wan21_vace_test.safetensors"
        logger.info(f"💾 Saving dummy model to {dummy_model_path}...")
        
        try:
            import safetensors.torch
            safetensors.torch.save_file(dummy_state_dict, dummy_model_path)
            logger.info("✅ Dummy model saved as SafeTensors")
        except ImportError:
            torch.save(dummy_state_dict, dummy_model_path.replace('.safetensors', '.pt'))
            dummy_model_path = dummy_model_path.replace('.safetensors', '.pt')
            logger.info("✅ Dummy model saved as PyTorch checkpoint")
        
        # Test verification system
        logger.info("🔍 Testing verification system...")
        from test_wan21_vace_16b_complete import WAN21VACEVerifier
        
        # Initialize verifier with dummy model
        verifier = WAN21VACEVerifier(dummy_model_path)
        
        # Override expected specs for our smaller test model
        verifier.expected_specs['total_parameters'] = sum(t.numel() for t in dummy_state_dict.values())
        verifier.expected_specs['min_file_size_gb'] = 0.1  # Much smaller for test
        verifier.expected_specs['max_file_size_gb'] = 2.0
        verifier.expected_specs['expected_layers'] = config['num_layers']
        
        # Run verification
        report = verifier.run_complete_verification()
        
        # Print results
        logger.info("📊 VERIFICATION TEST RESULTS")
        logger.info("=" * 40)
        logger.info(f"Overall Success: {'✅ PASS' if report['overall_success'] else '❌ FAIL'}")
        logger.info(f"Success Rate: {report['success_rate']:.1f}% ({report['passed_tests']}/{report['total_tests']})")
        
        for test_name, passed in report['verification_results'].items():
            status = '✅ PASS' if passed else '❌ FAIL'
            logger.info(f"  {test_name}: {status}")
        
        # Cleanup
        logger.info("🧹 Cleaning up...")
        if os.path.exists(dummy_model_path):
            os.remove(dummy_model_path)
        
        if report['overall_success']:
            logger.info("🎉 SUCCESS: Verification system is working correctly!")
            logger.info("✅ Ready to test your real WAN 2.1 VACE 16B model")
            return True
        else:
            logger.error("❌ FAILURE: Verification system has issues")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function"""
    success = test_verification_system()
    
    if success:
        logger.info("\n🚀 READY FOR VAST AI!")
        logger.info("You can now run the verification on your actual model:")
        logger.info("./run_model_verification.sh /path/to/your/wan2.1_vace_16b_fp16.safetensors")
        return 0
    else:
        logger.error("\n🔧 Please fix the issues before proceeding")
        return 1

if __name__ == "__main__":
    exit(main())
