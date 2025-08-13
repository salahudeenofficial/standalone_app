# Standalone Reference Video Pipeline

A standalone application that implements a reference image + control video → output video pipeline using ComfyUI components, stripped of WebSocket, graph execution, and UI dependencies.

## Overview

This application reuses ComfyUI's core functionality to create a streamlined pipeline for generating videos from reference images and control videos. It's designed to be lightweight and focused solely on the video generation process.

## Pipeline Components

The pipeline consists of 9 main steps:

1. **Load Diffusion Model** - Load UNET, CLIP, and VAE models
2. **Apply LoRA** - Apply LoRA weights to modify model behavior
3. **Encode Prompts** - Convert text prompts to CLIP embeddings
4. **Apply ModelSamplingSD3** - Apply SD3 sampling with shift parameter
5. **Generate Initial Latents** - Create initial video latents from reference image and control video
6. **Run KSampler** - Denoise latents using the diffusion model
7. **Trim Video Latent** - Remove reference image frames from output
8. **Decode Frames** - Convert latents back to pixel space
9. **Export Video** - Save frames as MP4 video

## Installation

1. **Clone and setup the standalone app:**
   ```bash
   cd standalone_app
   pip install -r requirements.txt
   ```

2. **Prepare model files:**
   - Place your diffusion models in `models/diffusion_models/`
   - Place your CLIP models in `models/text_encoders/`
   - Place your VAE models in `models/vaes/`
   - Place your LoRA files in `models/loras/`

## Usage

### Basic Usage

```python
from pipeline import ReferenceVideoPipeline

# Initialize pipeline
pipeline = ReferenceVideoPipeline(models_dir="models")

# Run the complete pipeline
output_path = pipeline.run_pipeline(
    unet_model_path="wan_2.1_diffusion_model.safetensors",
    clip_model_path="wan_clip_model.safetensors",
    vae_model_path="wan_vae.safetensors",
    lora_path="Wan21_CausVid_14B_T2V_lora_rank32.safetensors",
    positive_prompt="A beautiful landscape",
    negative_prompt="blurry, low quality",
    control_video_path="control_video.mp4",
    reference_image_path="reference.jpg",
    width=480,
    height=832,
    length=37,
    output_path="generated_video.mp4"
)
```

### Command Line Usage

```bash
python pipeline.py
```

## Model Requirements

- **UNET Model**: WAN 2.1 diffusion model or compatible
- **CLIP Model**: WAN CLIP model or compatible
- **VAE Model**: WAN VAE model or compatible
- **LoRA**: Wan21_CausVid_14B_T2V_lora_rank32 or compatible

## File Structure

```
standalone_app/
├── comfy/                    # Core ComfyUI modules
├── comfy_extras/            # Extended ComfyUI modules
├── components/              # Pipeline components
│   ├── __init__.py
│   ├── model_loader.py     # UNET, CLIP, VAE loaders
│   ├── lora_loader.py      # LoRA loading and application
│   ├── text_encoder.py     # Text prompt encoding
│   ├── model_sampling.py   # SD3 model sampling
│   ├── video_generator.py  # Initial latent generation
│   ├── sampler.py          # K-Sampler for denoising
│   ├── video_processor.py  # Video latent processing
│   ├── vae_decoder.py      # VAE decoding
│   └── video_export.py     # Video export to MP4
├── utils/                   # Utility modules
├── models/                  # Model storage directory
├── pipeline.py              # Main pipeline script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Dependencies

- **PyTorch**: Core deep learning framework
- **OpenCV**: Video processing and export
- **NumPy**: Numerical operations
- **Pillow**: Image processing
- **Safetensors**: Model loading
- **Transformers**: CLIP model support
- **Accelerate**: Model optimization
- **XFormers**: Memory-efficient attention

## Memory Management

The application includes memory management features:
- Automatic device selection (CPU/GPU)
- Memory-efficient sampling
- Batch processing support
- Configurable model precision

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use smaller models
2. **Model Not Found**: Check model paths and file names
3. **Import Errors**: Ensure all ComfyUI modules are properly copied

### Performance Tips

- Use GPU acceleration when available
- Adjust batch size based on available memory
- Use appropriate model precision (fp16/fp32)

## License

This project uses ComfyUI components and follows the same licensing terms.

## Contributing

This is a standalone implementation. For modifications, edit the component files in the `components/` directory. 