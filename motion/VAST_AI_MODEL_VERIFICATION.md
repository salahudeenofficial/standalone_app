# WAN 2.1 VACE 16B Model Verification for VAST AI

This guide helps you completely load and verify your WAN 2.1 VACE 16B model on your VAST AI instance.

## Quick Start

1. **Upload this verification script to your VAST AI instance**
2. **Locate your model file** (usually named something like `wan2.1_vace_16b_fp16.safetensors`)
3. **Run the verification**:

```bash
# Make script executable
chmod +x run_model_verification.sh

# Run verification (replace with your actual model path)
./run_model_verification.sh /path/to/your/wan2.1_vace_16b_fp16.safetensors
```

## Expected Model Specifications

Based on web research, your WAN 2.1 VACE 16B model should have:

### File Specifications
- **File Size**: 25-35 GB (for fp16 precision)
- **Format**: `.safetensors`, `.pt`, `.pth`, or `.bin`
- **Precision**: FP16 (half precision)

### Architecture Specifications
- **Total Parameters**: ~16 billion
- **Model Type**: VACE (Video-to-Video with Advanced Control Enhancement)
- **Transformer Layers**: 32 layers (typical for large models)
- **Hidden Dimension**: 2048
- **Attention Heads**: 16
- **VACE Layers**: 8 (for VACE-specific functionality)

### Capabilities
- **Text-to-Video (T2V)**: Generate videos from text prompts
- **Image-to-Video (I2V)**: Generate videos from input images
- **VACE Features**: Advanced control over video generation
- **Supported Resolutions**: 720p (720x1280), 512x512, 768x768
- **Frame Counts**: 8, 16, 24, 32 frames

## Verification Process

The verification script performs these comprehensive checks:

### 1. File Verification ✅
- Confirms model file exists
- Checks file size is within expected range
- Validates file format

### 2. Model Loading ✅
- Loads the complete state dict
- Detects model configuration automatically
- Creates model instance with proper architecture

### 3. Architecture Verification ✅
- Verifies all required components are present
- Checks transformer block structure
- Confirms VACE-specific components (if VACE model)
- Validates attention mechanism structure

### 4. Parameter Verification ✅
- Counts total parameters (~16B expected)
- Analyzes parameter distribution
- Verifies component parameter counts

### 5. Precision & Memory Verification ✅
- Confirms FP16 precision
- Analyzes GPU memory usage
- Estimates memory requirements

### 6. Forward Pass Testing ✅
- Tests Text-to-Video generation
- Tests VACE-specific functionality
- Validates output shapes and dtypes

### 7. State Dict Analysis ✅
- Analyzes state dict structure
- Verifies critical component keys
- Confirms model completeness

## Common Model Locations on VAST AI

Your model might be located at:
```bash
/workspace/models/wan2.1_vace_16b_fp16.safetensors
/root/models/wan2.1_vace_16b_fp16.safetensors
/home/user/models/wan2.1_vace_16b_fp16.safetensors
/data/models/wan2.1_vace_16b_fp16.safetensors
```

Find your model with:
```bash
find / -name "*wan*" -name "*.safetensors" 2>/dev/null
find / -name "*vace*" -name "*.safetensors" 2>/dev/null
```

## GPU Requirements

For optimal performance, ensure your VAST AI instance has:
- **VRAM**: 24GB+ (RTX 4090, A6000, A100)
- **CUDA**: 11.7+ or 12.x
- **PyTorch**: 2.0+ with CUDA support

Check your GPU:
```bash
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name()}')"
```

## Success Criteria

Your model verification is successful if:
- ✅ All 7 verification steps pass
- ✅ Success rate ≥ 80%
- ✅ Parameter count ~16B (±10% tolerance)
- ✅ Model loads in FP16 precision
- ✅ Forward passes complete successfully
- ✅ Memory usage is reasonable

## Troubleshooting

### Common Issues

**File Not Found**
```bash
# Find your model
find / -name "*wan*" -o -name "*vace*" 2>/dev/null | grep -i safetensors
```

**Out of Memory**
```bash
# Check available memory
nvidia-smi
# Consider using a larger GPU instance
```

**Import Errors**
```bash
# Install required packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install safetensors einops
```

**Model Loading Issues**
- Ensure the model file is not corrupted
- Verify it's a WAN 2.1 VACE model (not a different architecture)
- Check file permissions

### Expected Output

Successful verification should show:
```
🎉 SUCCESS! Model verification completed successfully.
📄 Detailed report saved to: wan21_vace_verification_YYYYMMDD_HHMMSS.json
✅ Your WAN 2.1 VACE 16B model is ready for inference on VAST AI!
```

## Next Steps

After successful verification:

1. **Save the verification report** - Keep the JSON report for reference
2. **Test inference** - Use the model for actual video generation
3. **Monitor performance** - Check inference speed and quality
4. **Optimize settings** - Adjust batch size and resolution for your GPU

## Support

If verification fails:
1. Check the detailed log file: `wan21_vace_16b_verification.log`
2. Review the JSON report for specific issues
3. Ensure your model is a genuine WAN 2.1 VACE 16B checkpoint
4. Verify your VAST AI instance has sufficient resources

The verification script provides detailed logging to help diagnose any issues with your model loading and functionality.
