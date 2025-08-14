# Installation Guide for Standalone Video Pipeline

This guide covers different installation methods for the Standalone Video Pipeline with VRAM optimization features.

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended) or CPU-only setup
- At least 8GB RAM (16GB+ recommended)
- At least 4GB VRAM (8GB+ recommended for optimal performance)

## Installation Methods

### Method 1: Quick Install (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd standalone_app

# Install core dependencies
pip install -r requirements.txt

# Run the pipeline
python pipeline.py
```

### Method 2: Development Install

```bash
# Clone the repository
git clone <your-repo-url>
cd standalone_app

# Install development dependencies
pip install -r requirements-dev.txt

# Install in development mode
pip install -e .

# Run tests
pytest
```

### Method 3: Minimal Install (Production)

```bash
# Clone the repository
git clone <your-repo-url>
cd standalone_app

# Install minimal dependencies
pip install -r requirements-minimal.txt

# Note: Some features may be limited with minimal install
```

### Method 4: Using setup.py

```bash
# Clone the repository
git clone <your-repo-url>
cd standalone_app

# Install with setup.py
pip install .

# Install with GPU monitoring extras
pip install .[gpu-monitoring]

# Install with development extras
pip install .[dev]
```

## Virtual Environment Setup (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## CUDA Setup

### Automatic CUDA Detection
The pipeline automatically detects CUDA availability. If you have CUDA installed, it will be used automatically.

### Manual CUDA Installation
If you need to install CUDA manually:

1. **Install NVIDIA Drivers**: Download from [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
2. **Install CUDA Toolkit**: Download from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
3. **Install PyTorch with CUDA**: 
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## Testing the Installation

### Run Memory Management Tests
```bash
python test_memory_management.py
```

### Run Chunked Processing Tests
```bash
python test_chunked_processing.py
```

### Run Full Pipeline Test
```bash
python pipeline.py
```

## Troubleshooting

### Common Issues

1. **CUDA not available**
   - Ensure NVIDIA drivers are installed
   - Check CUDA version compatibility
   - Verify PyTorch CUDA installation

2. **Memory errors**
   - Reduce batch size or video length
   - Use chunked processing (enabled by default)
   - Check available VRAM

3. **Import errors**
   - Ensure all dependencies are installed
   - Check Python version compatibility
   - Verify virtual environment activation

### Performance Tuning

1. **Adjust chunk sizes** in `components/chunked_processor.py`
2. **Modify memory thresholds** in `components/memory_manager.py`
3. **Change VRAM usage limits** in `components/model_manager.py`

## Configuration

### Environment Variables
```bash
# Set model directory
export COMFY_MODEL_PATH="/path/to/your/models"

# Set CUDA device (optional)
export CUDA_VISIBLE_DEVICES=0
```

### Model Directory Structure
```
models/
├── diffusion_models/     # UNET models
├── text_encoders/        # CLIP models
├── vaes/                 # VAE models
└── loras/                # LoRA models
```

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the test scripts for examples
3. Check GPU memory usage with `nvidia-smi`
4. Enable debug logging in the components

## Performance Expectations

- **Low VRAM (4-6GB)**: Conservative chunking, slower but stable
- **Medium VRAM (8-12GB)**: Balanced performance, good for most use cases
- **High VRAM (16GB+)**: Aggressive chunking, optimal performance

The system automatically adapts to your hardware configuration for the best performance. 