#!/bin/bash
# Quick script to run WAN 2.1 VACE 16B model verification on VAST AI
# Usage: ./run_model_verification.sh /path/to/your/model.safetensors

set -e

echo "🚀 WAN 2.1 VACE 16B Model Verification on VAST AI"
echo "=================================================="

# Check if model path is provided
if [ $# -eq 0 ]; then
    echo "❌ Error: Please provide the path to your WAN 2.1 VACE 16B model"
    echo "Usage: $0 /path/to/your/model.safetensors"
    echo ""
    echo "Common model locations on VAST AI:"
    echo "  - /workspace/models/wan2.1_vace_16b_fp16.safetensors"
    echo "  - /root/models/wan2.1_vace_16b_fp16.safetensors"
    echo "  - /home/user/models/wan2.1_vace_16b_fp16.safetensors"
    exit 1
fi

MODEL_PATH="$1"

# Check if model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Error: Model file not found at $MODEL_PATH"
    echo ""
    echo "Please check the path and try again."
    echo "You can use 'find / -name \"*wan*\" -name \"*.safetensors\" 2>/dev/null' to locate your model."
    exit 1
fi

echo "📁 Model file: $MODEL_PATH"
echo "📏 File size: $(du -h "$MODEL_PATH" | cut -f1)"
echo ""

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    echo ""
else
    echo "⚠️  nvidia-smi not found - running in CPU mode"
    echo ""
fi

# Check Python environment
echo "🐍 Python Environment:"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Unknown')"
echo ""

# Run the verification
echo "🔍 Starting model verification..."
echo "This may take several minutes for a 16B parameter model..."
echo ""

REPORT_FILE="wan21_vace_verification_$(date +%Y%m%d_%H%M%S).json"

if python test_wan21_vace_16b_complete.py "$MODEL_PATH" --output-report "$REPORT_FILE"; then
    echo ""
    echo "🎉 SUCCESS! Model verification completed successfully."
    echo "📄 Detailed report saved to: $REPORT_FILE"
    echo ""
    echo "✅ Your WAN 2.1 VACE 16B model is ready for inference on VAST AI!"
else
    echo ""
    echo "❌ FAILURE! Model verification encountered issues."
    echo "📄 Check the detailed report in: $REPORT_FILE"
    echo "📋 Review the log file: wan21_vace_16b_verification.log"
    echo ""
    echo "🔧 Please address the issues before proceeding with inference."
    exit 1
fi
