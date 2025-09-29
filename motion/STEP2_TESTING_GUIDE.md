# Step 2: UNet + CLIP Loading Test Scripts

## 🎯 **Overview**

Created comprehensive test scripts for Step 2 model loading following ComfyUI patterns:

1. **`test_clip_t5_xxl.py`** - Dedicated T5 XXL CLIP testing
2. **`test_step2_unet_clip.py`** - Combined UNet + CLIP testing

## 📊 **T5 XXL FP16 Specifications**

Based on ComfyUI implementation and web research:

```python
T5_XXL_SPECS = {
    "model_name": "UMT5 XXL FP16",
    "model_type": "T5 Text Encoder",
    "architecture": "UMT5 (Unified Multilingual T5)",
    "hidden_size": 4096,           # d_model
    "ffn_dim": 10240,              # d_ff  
    "num_heads": 64,               # num_heads
    "num_layers": 24,              # num_layers (encoder)
    "num_decoder_layers": 24,      # num_decoder_layers
    "vocab_size": 256384,          # vocab_size
    "d_kv": 64,                    # key/value dimension
    "max_length": 99999999,        # Maximum sequence length
    "min_length": 512,             # Minimum sequence length
    "context_dim": 4096,           # Output embedding dimension
    "context_length": 77,           # Standard context length for diffusion
    "dtype": torch.float16,        # FP16 precision
    "dropout_rate": 0.1,
    "layer_norm_epsilon": 1e-06,
    "pad_token_id": 0,
    "eos_token_id": 1,
    "decoder_start_token_id": 0,
    "relative_attention_num_buckets": 32,
    "is_encoder_decoder": True,
    "is_gated_act": True,
    "tie_word_embeddings": False
}
```

## 🧠 **UNet Specifications**

```python
UNET_SPECS = {
    "model_name": "WAN 2.1 VACE 14B",
    "model_size": "14B parameters",
    "input_channels": 16,
    "output_channels": 16,
    "hidden_dim": 5120,
    "ffn_dim": 13824,
    "freq_dim": 256,
    "num_heads": 40,
    "num_layers": 40,
    "context_dim": 4096,
    "context_length": 77,
    "dtype": torch.float16,
    "framework": "Flow Matching",
    "architecture": "Diffusion Transformer (DiT)"
}
```

## 🚀 **Test Scripts**

### 1. T5 XXL CLIP Test (`test_clip_t5_xxl.py`)

**Features:**
- ✅ ComfyUI-style T5 XXL loading with patcher
- ✅ Architecture verification against specifications
- ✅ Text encoding functionality testing
- ✅ Patcher weight loading verification
- ✅ Comprehensive error handling and reporting

**Test Coverage:**
- Model file existence and size verification
- ComfyUI-style patcher loading
- T5 XXL architecture validation
- Text encoding with multiple prompts
- Parameter counting and verification
- Device and dtype validation

### 2. Combined Step 2 Test (`test_step2_unet_clip.py`)

**Features:**
- ✅ Individual UNet loading test
- ✅ Individual CLIP loading test
- ✅ Combined UNet + CLIP loading (Step 2 style)
- ✅ Model interaction verification
- ✅ Comprehensive patcher functionality testing

**Test Coverage:**
- UNet model loading and verification
- CLIP model loading and verification
- Combined loading simulation
- Parameter counting for both models
- Device placement verification
- Patcher functionality validation

## 🔧 **ComfyUI Pattern Compliance**

Both scripts follow ComfyUI patterns:

1. **Loading Method**: Uses `load_state_dict_guess_config()` from ComfyUI
2. **Patcher System**: Implements ComfyUI-style ModelPatcher
3. **Device Management**: Follows ComfyUI's device placement logic
4. **Error Handling**: ComfyUI-style validation and error reporting
5. **Architecture Verification**: Matches ComfyUI's model detection patterns

## 📋 **Usage Instructions**

### Test T5 XXL CLIP Only:
```bash
cd /home/fashionx/v_pipe/standalone_app
python motion/test_clip_t5_xxl.py
```

### Test Combined Step 2 (UNet + CLIP):
```bash
cd /home/fashionx/v_pipe/standalone_app
python motion/test_step2_unet_clip.py
```

## 🎯 **Expected Output**

Both scripts provide:
- ✅ Model loading success/failure status
- ✅ Architecture verification results
- ✅ Parameter counts and model sizes
- ✅ Device and dtype information
- ✅ Patcher functionality verification
- ✅ Comprehensive error reporting

## 💡 **Fallback Behavior**

If model files don't exist:
- Shows warning about missing models
- Tests model class initialization
- Verifies basic functionality without actual weights

## 🔍 **Verification Points**

The scripts verify:
1. **Model Loading**: ComfyUI-style patcher loading
2. **Architecture**: Correct model dimensions and parameters
3. **Functionality**: Text encoding and model interaction
4. **Patcher**: Weight loading and device management
5. **Integration**: Combined model loading (Step 2 style)

## 🎉 **Ready for Testing**

Both test scripts are ready to run with your actual models on VAST AI:
- **UNet**: `models/diffusion_models/wan_2.1_diffusion_model.safetensors`
- **CLIP**: `models/text_encoders/umt5_xxl_fp16.safetensors`

The scripts will provide comprehensive verification that the ComfyUI-style patcher is loading weights correctly and that both models are ready for Step 2 integration.
