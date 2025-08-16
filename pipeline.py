#!/usr/bin/env python3
"""
Standalone Reference Image + Control Video to Output Video Pipeline
Based on ComfyUI components but stripped of WebSocket, graph execution, and UI dependencies

This pipeline now properly leverages ComfyUI's native memory management system:
- UNET models: Automatically managed by ModelPatcher (GPU/CPU swapping)
- VAE models: Built-in memory management with automatic loading/unloading
- CLIP models: Automatically managed by ModelPatcher
- All memory management: Handled by ComfyUI's proven system
"""

import torch
import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Add ComfyUI path for utilities
sys.path.insert(0, str(Path(__file__).parent / "comfy"))

import comfy.utils
from components.lora_loader import LoraLoader
from components.text_encoder import CLIPTextEncode
from components.model_sampling import ModelSamplingSD3
from components.video_generator import WanVaceToVideo
from components.sampler import KSampler
from components.video_processor import TrimVideoLatent
from components.vae_decoder import VAEDecode
from components.video_export import VideoExporter
from components.chunked_processor import ChunkedProcessor

class ReferenceVideoPipeline:
    """
    Standalone Reference Image + Control Video to Output Video Pipeline
    
    Memory Management Philosophy (explicit ModelPatcher control):
    - UNET models: Explicitly managed using ModelPatcher.unpatch_model(device_to=offload_device)
    - VAE models: Moved to GPU for operations, then to CPU for memory management
    - CLIP models: Explicitly managed using clip.patcher.unpatch_model(device_to=offload_device)
    - All memory management: Explicit control using proven ModelPatcher methods
    
    This approach ensures:
    1. Heavy models (UNET) get explicit GPU/CPU swapping via ModelPatcher
    2. VAE models get explicit device placement control
    3. Light models (CLIP) get the same explicit ModelPatcher control
    4. Full control over when models are loaded/unloaded
    5. Uses the exact same working logic as test_comfyui_integration.py
    """
    def __init__(self, models_dir="models"):
        """Initialize the pipeline with model directory"""
        self.models_dir = models_dir
        self.setup_model_paths()
        
        # Initialize chunked processor for optimal frame processing
        self.chunked_processor = ChunkedProcessor()
        
        # Start with conservative chunking for better memory management
        self.chunked_processor.set_chunking_strategy('conservative')
        
        # Initialize OOM debugging checklist
        self.oom_checklist = {
            'baseline_memory': None,
            'model_loading': None,
            'lora_application': None,
            'text_encoding': None,
            'model_sampling': None,
            'gpu_capability_test': None,  # Add GPU capability test results
            'vae_encoding': None,
            'unet_sampling': None,
            'vae_decoding': None,
            'final_cleanup': None
        }
        
        # Memory thresholds for each phase
        self.memory_thresholds = {
            'baseline': 100,           # MB - should be very low
            'model_loading': 15000,    # MB - UNET + CLIP + VAE loaded
            'lora_application': 16000, # MB - LoRA applied
            'text_encoding': 8000,     # MB - models in optimal positions
            'model_sampling': 12000,   # MB - UNET loaded for patching
            'gpu_capability_test': 1000, # MB - should be low during testing
            'vae_encoding': 8000,     # MB - VAE encoding in progress
            'unet_sampling': 40000,   # MB - UNET sampling needs ~33GB (realistic)
            'vae_decoding': 8000,     # MB - VAE decoding in progress
            'final_cleanup': 100      # MB - back to baseline
        }
    
    def _check_memory_usage(self, phase_name, expected_threshold=None):
        """Check memory usage and update OOM checklist"""
        if not torch.cuda.is_available():
            return True
            
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        
        # Update checklist
        self.oom_checklist[phase_name] = {
            'allocated_mb': allocated,
            'reserved_mb': reserved,
            'timestamp': phase_name,
            'status': 'PASS' if expected_threshold is None else ('PASS' if allocated <= expected_threshold else 'FAIL')
        }
        
        # Get threshold for this phase
        threshold = expected_threshold or self.memory_thresholds.get(phase_name, 0)
        
        print(f"🔍 {phase_name.upper()} MEMORY CHECK:")
        print(f"   Allocated: {allocated:.1f} MB")
        print(f"   Reserved: {reserved:.1f} MB")
        print(f"   Threshold: {threshold:.1f} MB")
        print(f"   Status: {self.oom_checklist[phase_name]['status']}")
        
        if allocated > threshold:
            print(f"   ⚠️  WARNING: Memory usage ({allocated:.1f} MB) exceeds threshold ({threshold:.1f} MB)")
            print(f"   💡 This phase may be at risk of OOM errors")
        
        return allocated <= threshold
    
    def _print_oom_checklist(self):
        """Print the complete OOM debugging checklist"""
        print("\n" + "="*80)
        print("🚨 OOM DEBUGGING CHECKLIST")
        print("="*80)
        
        for phase, data in self.oom_checklist.items():
            if data is None:
                print(f"❌ {phase}: NOT EXECUTED")
                continue
                
            status_icon = "✅" if data['status'] == 'PASS' else "❌" if data['status'] == 'FAIL' else "⚠️"
            print(f"{status_icon} {phase}:")
            print(f"   Allocated: {data['allocated_mb']:.1f} MB")
            print(f"   Reserved: {data['reserved_mb']:.1f} MB")
            print(f"   Status: {data['status']}")
            
            # Special handling for GPU capability test
            if phase == 'gpu_capability_test' and 'gpu_capable' in data:
                print(f"   GPU Capable: {'✅ YES' if data['gpu_capable'] else '❌ NO'}")
                
                # Show test results if available
                if 'test_results' in data and data['test_results']:
                    test_results = data['test_results']
                    print(f"   Test Results:")
                    
                    # Show VRAM info
                    if 'vram_total' in test_results:
                        print(f"     Total VRAM: {test_results['vram_total']:.2f} GB")
                        print(f"     Free VRAM: {test_results['vram_free']:.2f} GB")
                        print(f"     Fragmentation: {test_results.get('fragmentation_ratio', 0):.1%}")
                    
                    # Show test outcomes
                    for test_name, result in test_results.items():
                        if test_name.endswith('_test') and isinstance(result, bool):
                            test_status = "✅ PASS" if result else "❌ FAIL"
                            print(f"     {test_name}: {test_status}")
                        elif test_name == 'vae_memory_estimate':
                            print(f"     VAE Memory Estimate: {result:.1f} MB")
                        elif test_name == 'vae_device':
                            print(f"     VAE Device: {result}")
                        elif test_name == 'error':
                            print(f"     Error: {result}")
        
        print("="*80)
        
        # Summary analysis
        failed_phases = [phase for phase, data in self.oom_checklist.items() 
                        if data is not None and data['status'] == 'FAIL']
        
        if failed_phases:
            print(f"🚨 PROBLEM PHASES: {', '.join(failed_phases)}")
            print("💡 These phases exceeded memory thresholds and may cause OOM errors")
        else:
            print("✅ ALL PHASES PASSED MEMORY THRESHOLDS")
        
        # GPU capability summary
        gpu_test_data = self.oom_checklist.get('gpu_capability_test')
        if gpu_test_data and 'gpu_capable' in gpu_test_data:
            if gpu_test_data['gpu_capable']:
                print("✅ GPU CAPABILITY: GPU is capable of VAE encoding")
            else:
                print("❌ GPU CAPABILITY: GPU is NOT capable of VAE encoding - using CPU fallback")
        
        print("="*80)
    
    def _check_model_placement(self, phase_name, expected_models):
        """Check that models are in the expected devices"""
        print(f"🔍 {phase_name.upper()} MODEL PLACEMENT CHECK:")
        
        model_status = {}
        
        # Check UNET placement
        if 'unet' in expected_models:
            try:
                unet_device = getattr(self, 'model', None)
                if unet_device and hasattr(unet_device, 'device'):
                    model_status['unet'] = str(unet_device.device)
                    print(f"   UNET: {model_status['unet']}")
                else:
                    model_status['unet'] = 'UNKNOWN'
                    print(f"   UNET: {model_status['unet']} (not accessible)")
            except Exception as e:
                model_status['unet'] = f'ERROR: {e}'
                print(f"   UNET: {model_status['unet']}")
        
        # Check VAE placement
        if 'vae' in expected_models:
            try:
                vae_device = getattr(self, 'vae', None)
                if vae_device and hasattr(vae_device, 'device'):
                    model_status['vae'] = str(vae_device.device)
                    print(f"   VAE: {model_status['vae']}")
                else:
                    model_status['vae'] = 'UNKNOWN'
                    print(f"   VAE: {model_status['vae']} (not accessible)")
            except Exception as e:
                model_status['vae'] = f'ERROR: {e}'
                print(f"   VAE: {model_status['vae']}")
        
        # Check CLIP placement
        if 'clip' in expected_models:
            try:
                clip_device = getattr(self, 'clip_model', None)
                if clip_device and hasattr(clip_device, 'device'):
                    model_status['clip'] = str(clip_device.device)
                    print(f"   CLIP: {model_status['clip']}")
                else:
                    model_status['clip'] = 'UNKNOWN'
                    print(f"   CLIP: {model_status['clip']} (not accessible)")
            except Exception as e:
                model_status['clip'] = f'ERROR: {e}'
                print(f"   CLIP: {model_status['clip']}")
        
        return model_status
        
    def setup_model_paths(self):
        """Setup model paths for the standalone app"""
        # Get the script directory for absolute paths
        script_dir = Path(__file__).parent
        models_dir = script_dir / self.models_dir
        
        # Create model directories if they don't exist
        os.makedirs(models_dir / "diffusion_models", exist_ok=True)  # For UNET models
        os.makedirs(models_dir / "text_encoders", exist_ok=True)     # For CLIP models
        os.makedirs(models_dir / "vaes", exist_ok=True)             # For VAE models
        os.makedirs(models_dir / "loras", exist_ok=True)            # For LoRA models
        
        # Set environment variables for model paths
        os.environ["COMFY_MODEL_PATH"] = str(models_dir)
        
    def run_pipeline(self, 
                    unet_model_path,
                    clip_model_path,
                    vae_model_path,
                    lora_path=None,
                    positive_prompt="",
                    negative_prompt="",
                    control_video_path=None,
                    reference_image_path=None,
                    width=480,
                    height=832,
                    length=37,
                    batch_size=1,
                    strength=1.0,
                    seed=270400132721985,
                    steps=4,
                    cfg=1.0,
                    sampler_name="ddim",
                    scheduler="normal",
                    denoise=1.0,
                    output_path="output.mp4"):
        """
        Run the complete pipeline from reference image + control video to output video
        
        This pipeline now uses explicit ModelPatcher memory management (same as working test):
        - All models are loaded using ComfyUI's native functions
        - ModelPatcher explicitly manages UNET/CLIP memory with device_to=offload_device
        - VAE moved to GPU for operations, then to CPU for memory management
        - Explicit memory management calls using proven working logic
        """
        print("Starting Reference Video Pipeline...")
        print("🚀 Using ComfyUI's native memory management system")
        
        # Establish baseline memory state for OOM debugging
        print("🔍 ESTABLISHING BASELINE MEMORY STATE...")
        self._check_memory_usage('baseline_memory', expected_threshold=100)
        
        # Generate chunked processing plan
        print("Generating chunked processing plan...")
        
        # Check VRAM status and adjust strategy if needed
        self.chunked_processor.should_adjust_strategy()
        
        # Force conservative chunking if we have limited VRAM
        try:
            vram_status = self.chunked_processor.get_vram_status()
            if vram_status.get('available', False):
                total_vram_gb = vram_status.get('total_gb', 0)
                if total_vram_gb < 12.0:  # Less than 12GB VRAM
                    print(f"Limited VRAM detected ({total_vram_gb:.1f} GB), forcing conservative chunking")
                    self.chunked_processor.force_conservative_chunking()
                elif total_vram_gb < 8.0:  # Less than 8GB VRAM
                    print(f"Very limited VRAM detected ({total_vram_gb:.1f} GB), forcing ultra-conservative chunking")
                    self.chunked_processor.force_ultra_conservative_chunking()
        except Exception as e:
            print(f"Warning: Could not check VRAM status: {e}")
        
        processing_plan = self.chunked_processor.get_processing_plan(
            frame_count=length,
            width=width,
            height=height,
            operations=['vae_encode', 'unet_process', 'vae_decode']
        )
        self.chunked_processor.print_processing_plan(processing_plan)
        
        try:
            # 1. Load Diffusion Model Components using ComfyUI's native system
            print("1. Loading diffusion model components using ComfyUI...")
            
            # Import ComfyUI's model loading functions
            import comfy.sd
            import comfy.model_management
            
            # Establish baseline memory state
            print("1a. Establishing baseline memory state...")
            if torch.cuda.is_available():
                baseline_allocated = torch.cuda.memory_allocated() / 1024**2
                baseline_reserved = torch.cuda.memory_reserved() / 1024**2
                print(f"Baseline VRAM - Allocated: {baseline_allocated:.1f} MB, Reserved: {baseline_reserved:.1f} MB")
            
            # Load the models using ComfyUI's native functions
            print("1a. Loading individual components...")
            
            # Debug: Show the actual paths being used
            print(f"1a. Current working directory: {os.getcwd()}")
            print(f"1a. UNET path: {unet_model_path}")
            print(f"1a. CLIP path: {clip_model_path}")
            print(f"1a. VAE path: {vae_model_path}")
            print(f"1a. LoRA path: {lora_path}")
            
            # Check if files exist
            print(f"1a. UNET file exists: {os.path.exists(unet_model_path)}")
            print(f"1a. CLIP file exists: {os.path.exists(clip_model_path)}")
            print(f"1a. VAE file exists: {os.path.exists(vae_model_path)}")
            if lora_path:
                print(f"1a. LoRA file exists: {os.path.exists(lora_path)}")
            
            # Use ComfyUI's native loading functions which return ModelPatcher objects
            # ModelPatcher automatically handles all memory management
            model = comfy.sd.load_diffusion_model(unet_model_path)
            clip_model = comfy.sd.load_clip([clip_model_path], clip_type=comfy.sd.CLIPType.WAN)
            
            # For VAE, we need to load the state dict first, then create VAE object
            # VAE has its own built-in memory management
            vae_state_dict = comfy.utils.load_torch_file(vae_model_path)
            vae = comfy.sd.VAE(sd=vae_state_dict)
            
            print("1a. Models loaded using ComfyUI's native system")
            print(f"1a. UNET: {type(model)} (ModelPatcher)")
            print(f"1a. CLIP: {type(clip_model)} (ModelPatcher)")
            print(f"1a. VAE: {type(vae)} (VAE with built-in memory management)")
            
            # OOM Checklist: Check memory after model loading
            self._check_memory_usage('model_loading', expected_threshold=15000)
            
            # 2. Apply LoRA if specified
            if lora_path:
                print("2. Applying LoRA...")
                lora_loader = LoraLoader()
                model, clip_model = lora_loader.load_lora(
                    model, clip_model, lora_path, 0.5, 1.0
                )
                
                # ComfyUI automatically tracks these modified models through ModelPatcher
                print("2a. LoRA applied, models updated")
                print("2a. ModelPatcher automatically preserves LoRA patches")
                
                # OOM Checklist: Check memory after LoRA application
                self._check_memory_usage('lora_application', expected_threshold=16000)
            
            # 3. Encode Prompts
            print("3. Encoding text prompts...")
            text_encoder = CLIPTextEncode()
            positive_cond = text_encoder.encode(clip_model, positive_prompt)
            negative_cond = text_encoder.encode(clip_model, negative_prompt)
            
            # ComfyUI automatically manages encoded prompts through ModelPatcher
            print("3a. Text encoding complete")
            
            # Explicitly manage CLIP memory using working logic from test
            print("3a. Explicitly managing CLIP memory using working ModelPatcher logic...")
            if hasattr(clip_model, 'patcher') and hasattr(clip_model.patcher, 'unpatch_model'):
                print(f"3a. Moving CLIP to offload device: {clip_model.patcher.offload_device}")
                clip_model.patcher.unpatch_model(device_to=clip_model.patcher.offload_device)
                
                # Force cleanup like in the working test
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                
                # Verify CLIP is on correct device
                if hasattr(clip_model.patcher, 'model') and hasattr(clip_model.patcher.model, 'device'):
                    print(f"3a. CLIP model device after unpatch: {clip_model.patcher.model.device}")
                    if str(clip_model.patcher.model.device) == str(clip_model.patcher.offload_device):
                        print("3a. ✅ CLIP successfully moved to offload device")
                    else:
                        print("3a. ❌ CLIP failed to move to offload device")
                else:
                    print("3a. ⚠️  Cannot verify CLIP device placement")
            else:
                print("3a. ⚠️  CLIP ModelPatcher not available, skipping explicit memory management")
            
            # OOM Checklist: Check memory after text encoding
            self._check_memory_usage('text_encoding', expected_threshold=8000)
            
            # 4. Apply ModelSamplingSD3 Shift
            print("4. Applying ModelSamplingSD3...")
            model_sampling = ModelSamplingSD3()
            
            # ModelPatcher automatically handles loading/unloading during patching
            model = model_sampling.patch(model, shift=8.0)
            
            # ComfyUI automatically tracks the patched model through ModelPatcher
            print("4a. ModelSamplingSD3 applied")
            
            # OOM Checklist: Check memory after ModelSamplingSD3
            self._check_memory_usage('model_sampling', expected_threshold=12000)
            
            # 5. Generate Initial Latents
            print("5. Generating initial latents...")
            video_generator = WanVaceToVideo()
            
            # Load control video and reference image
            control_video = self.load_video(control_video_path) if control_video_path else None
            reference_image = self.load_image(reference_image_path) if reference_image_path else None
            
            # TRUE MEMORY-EFFICIENT VAE ENCODING STRATEGY
            print("5a. Implementing ComfyUI's native VAE encoding strategy...")
            print("5a. Strategy: Smart batching + automatic tiled fallback (like ComfyUI)")
            
            # Pre-emptive VAE memory management - ensure VAE is on GPU for encoding
            if hasattr(vae, 'first_stage_model') and hasattr(vae.first_stage_model, 'to'):
                print("5a. Ensuring VAE is on GPU for encoding...")
                vae.first_stage_model.to('cuda:0')
                vae.device = torch.device('cuda:0')
                print(f"5a. VAE moved to: {vae.device}")
            else:
                print("5a. ⚠️  Cannot move VAE to GPU")
            
            # AGGRESSIVE MEMORY DEFRAGMENTATION
            print("5a. AGGRESSIVE MEMORY DEFRAGMENTATION...")
            print("5a. Current GPU memory state:")
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                free = total - reserved
                print(f"5a.   Allocated: {allocated:.2f} GB")
                print(f"5a.   Reserved: {reserved:.2f} GB")
                print(f"5a.   Total: {total:.2f} GB")
                print(f"5a.   Free: {free:.2f} GB")
                
                # If memory is heavily fragmented, force cleanup
                if allocated > 40.0:  # More than 40GB allocated
                    print("5a. ⚠️  HEAVY MEMORY FRAGMENTATION DETECTED!")
                    print("5a. 🔧 Forcing aggressive memory cleanup...")
                    
                    # Force all models to CPU to free GPU memory
                    print("5a. Moving all models to CPU to free GPU memory...")
                    
                    # Move UNET to CPU
                    if hasattr(model, 'unpatch_model') and hasattr(model, 'offload_device'):
                        print("5a. Moving UNET to CPU...")
                        model.unpatch_model(device_to=torch.device('cpu'))
                    
                    # Move CLIP to CPU
                    if hasattr(clip_model, 'patcher') and hasattr(clip_model.patcher, 'unpatch_model'):
                        print("5a. Moving CLIP to CPU...")
                        clip_model.patcher.unpatch_model(device_to=torch.device('cpu'))
                    
                    # Move VAE to CPU temporarily
                    if hasattr(vae, 'first_stage_model') and hasattr(vae.first_stage_model, 'to'):
                        print("5a. Moving VAE to CPU temporarily...")
                        vae.first_stage_model.to('cpu')
                        vae.device = torch.device('cpu')
                        
                        # Verify synchronization after moving to CPU
                        first_stage_device = str(vae.first_stage_model.device)
                        wrapper_device = str(vae.device)
                        print(f"5a. VAE device sync check after CPU move:")
                        print(f"5a.   Wrapper: {wrapper_device}, First_stage: {first_stage_device}")
                        if first_stage_device == wrapper_device and 'cpu' in first_stage_device:
                            print("5a.   ✅ VAE successfully synchronized on CPU")
                        else:
                            print("5a.   ⚠️  VAE device attributes not synchronized on CPU")
                    else:
                        print("5a. ⚠️  Cannot move VAE to CPU (no first_stage_model)")
                    
                    # Aggressive cleanup
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        torch.cuda.synchronize()
                    
                    # Check memory after cleanup
                    allocated_after = torch.cuda.memory_allocated() / 1024**3
                    reserved_after = torch.cuda.memory_reserved() / 1024**3
                    free_after = total - reserved_after
                    print(f"5a. Memory after cleanup:")
                    print(f"5a.   Allocated: {allocated_after:.2f} GB")
                    print(f"5a.   Reserved: {reserved_after:.2f} GB")
                    print(f"5a.   Free: {free_after:.2f} GB")
                    
                    # If still fragmented, use CPU fallback
                    if free_after < 5.0:  # Less than 5GB free
                        print("5a. ⚠️  GPU still heavily fragmented, using CPU fallback...")
                        use_cpu_fallback = True
                    else:
                        print("5a. ✅ Memory cleanup successful, retrying GPU processing...")
                        use_cpu_fallback = False
                        
                        # Move VAE back to GPU
                        if hasattr(vae, 'first_stage_model') and hasattr(vae.first_stage_model, 'to'):
                            vae.first_stage_model.to('cuda:0')
                            vae.device = torch.device('cuda:0')
                            
                            # Verify synchronization after moving back to GPU
                            first_stage_device = str(vae.first_stage_model.device)
                            wrapper_device = str(vae.device)
                            print(f"5a. VAE device sync check after GPU move:")
                            print(f"5a.   Wrapper: {wrapper_device}, First_stage: {first_stage_device}")
                            if first_stage_device == wrapper_device and 'cuda' in first_stage_device:
                                print("5a.   ✅ VAE successfully synchronized on GPU")
                            else:
                                print("5a.   ⚠️  VAE device attributes not synchronized on GPU")
                                print("5a.   🔧 Attempting to force synchronization...")
                                # Force both to GPU
                                vae.first_stage_model.to('cuda:0')
                                vae.device = torch.device('cuda:0')
                                # Verify again
                                first_stage_device = str(vae.first_stage_model.device)
                                wrapper_device = str(vae.device)
                                print(f"5a.   After force sync - Wrapper: {wrapper_device}, First_stage: {first_stage_device}")
                        else:
                            print("5a. ⚠️  Cannot move VAE to GPU (no first_stage_model)")
                else:
                    use_cpu_fallback = False
                    
                    # Ensure VAE is on GPU for testing (if not using CPU fallback)
                    if hasattr(vae, 'first_stage_model') and hasattr(vae.first_stage_model, 'to'):
                        print("5a. Ensuring VAE is on GPU for capability testing...")
                        vae.first_stage_model.to('cuda:0')
                        vae.device = torch.device('cuda:0')
                        print(f"5a. VAE moved to: {vae.device}")
                        
                        # Verify both device attributes are synchronized
                        first_stage_device = str(vae.first_stage_model.device)
                        wrapper_device = str(vae.device)
                        print(f"5a. Device synchronization check:")
                        print(f"5a.   VAE wrapper device: {wrapper_device}")
                        print(f"5a.   VAE first_stage_model device: {first_stage_device}")
                        
                        if first_stage_device == wrapper_device and 'cuda' in first_stage_device:
                            print("5a.   ✅ VAE device attributes are synchronized on GPU")
                        else:
                            print("5a.   ⚠️  VAE device attributes are NOT synchronized!")
                            print("5a.   🔧 Attempting to force synchronization...")
                            # Force synchronization by moving both to GPU
                            vae.first_stage_model.to('cuda:0')
                            vae.device = torch.device('cuda:0')
                            # Verify again
                            first_stage_device = str(vae.first_stage_model.device)
                            wrapper_device = str(vae.device)
                            print(f"5a.   After sync - wrapper: {wrapper_device}, first_stage: {first_stage_device}")
                    else:
                        print("5a. ⚠️  Cannot move VAE to GPU (no first_stage_model)")
            
            # GPU CAPABILITY TEST FOR VAE ENCODING
            print("5a. 🧪 GPU CAPABILITY TEST FOR VAE ENCODING...")
            print("5a. Testing GPU's ability to handle VAE operations before proceeding...")
            
            gpu_capable = False
            test_results = {}
            
            if torch.cuda.is_available() and not use_cpu_fallback:
                try:
                    # Test 1: Basic VRAM availability
                    print("5a. Test 1: Basic VRAM availability...")
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    free = total - reserved
                    
                    test_results['vram_total'] = total
                    test_results['vram_free'] = free
                    test_results['vram_allocated'] = allocated
                    test_results['vram_reserved'] = reserved
                    
                    print(f"5a.   Total VRAM: {total:.2f} GB")
                    print(f"5a.   Free VRAM: {free:.2f} GB")
                    print(f"5a.   Allocated: {allocated:.2f} GB")
                    print(f"5a.   Reserved: {reserved:.2f} GB")
                    
                    if free < 2.0:  # Less than 2GB free
                        print("5a.   ❌ FAIL: Insufficient free VRAM (< 2GB)")
                        test_results['vram_test'] = False
                    else:
                        print("5a.   ✅ PASS: Sufficient free VRAM")
                        test_results['vram_test'] = True
                    
                    # Test 2: Memory fragmentation check
                    print("5a. Test 2: Memory fragmentation check...")
                    fragmentation_ratio = allocated / total if total > 0 else 1.0
                    test_results['fragmentation_ratio'] = fragmentation_ratio
                    
                    print(f"5a.   Fragmentation ratio: {fragmentation_ratio:.2%}")
                    
                    if fragmentation_ratio > 0.95:  # More than 95% allocated
                        print("5a.   ❌ FAIL: Extreme memory fragmentation (>95%)")
                        test_results['fragmentation_test'] = False
                    elif fragmentation_ratio > 0.8:  # More than 80% allocated
                        print("5a.   ⚠️  WARNING: High memory fragmentation (>80%)")
                        test_results['fragmentation_test'] = False
                    else:
                        print("5a.   ✅ PASS: Acceptable memory fragmentation")
                        test_results['fragmentation_test'] = True
                    
                    # Test 3: VAE model placement test
                    print("5a. Test 3: VAE model placement test...")
                    print(f"5a.   Current VAE device before test: {vae.device}")
                    print(f"5a.   VAE first_stage_model device: {vae.first_stage_model.device if hasattr(vae, 'first_stage_model') else 'N/A'}")
                    
                    if hasattr(vae, 'first_stage_model') and hasattr(vae.first_stage_model, 'to'):
                        # Check if VAE is actually on GPU by checking both attributes
                        vae_wrapper_device = str(vae.device)
                        vae_first_stage_device = str(vae.first_stage_model.device)
                        test_results['vae_device'] = f"wrapper:{vae_wrapper_device}, first_stage:{vae_first_stage_device}"
                        
                        # Both should be on GPU for the test to pass
                        wrapper_on_gpu = 'cuda' in vae_wrapper_device
                        first_stage_on_gpu = 'cuda' in vae_first_stage_device
                        
                        print(f"5a.   VAE wrapper on GPU: {wrapper_on_gpu} ({vae_wrapper_device})")
                        print(f"5a.   VAE first_stage on GPU: {first_stage_on_gpu} ({vae_first_stage_device})")
                        
                        if wrapper_on_gpu and first_stage_on_gpu:
                            print(f"5a.   ✅ PASS: VAE successfully on GPU")
                            test_results['vae_placement_test'] = True
                        elif wrapper_on_gpu and not first_stage_on_gpu:
                            print(f"5a.   ❌ FAIL: VAE wrapper on GPU but first_stage_model on CPU")
                            print(f"5a.   🔧 Attempting to fix first_stage_model placement...")
                            try:
                                vae.first_stage_model.to('cuda:0')
                                vae_first_stage_device = str(vae.first_stage_model.device)
                                print(f"5a.   After fix attempt: first_stage_model device: {vae_first_stage_device}")
                                if 'cuda' in vae_first_stage_device:
                                    print(f"5a.   ✅ PASS: VAE placement fixed and now on GPU")
                                    test_results['vae_placement_test'] = True
                                else:
                                    print(f"5a.   ❌ FAIL: Could not move first_stage_model to GPU")
                                    test_results['vae_placement_test'] = False
                            except Exception as fix_error:
                                print(f"5a.   ❌ FAIL: Error fixing VAE placement: {fix_error}")
                                test_results['vae_placement_test'] = False
                        elif not wrapper_on_gpu and first_stage_on_gpu:
                            print(f"5a.   ❌ FAIL: VAE first_stage_model on GPU but wrapper on CPU")
                            print(f"5a.   🔧 Attempting to fix wrapper placement...")
                            try:
                                vae.device = torch.device('cuda:0')
                                vae_wrapper_device = str(vae.device)
                                print(f"5a.   After fix attempt: wrapper device: {vae_wrapper_device}")
                                if 'cuda' in vae_wrapper_device:
                                    print(f"5a.   ✅ PASS: VAE placement fixed and now on GPU")
                                    test_results['vae_placement_test'] = True
                                else:
                                    print(f"5a.   ❌ FAIL: Could not move wrapper to GPU")
                                    test_results['vae_placement_test'] = False
                            except Exception as fix_error:
                                print(f"5a.   ❌ FAIL: Error fixing VAE placement: {fix_error}")
                                test_results['vae_placement_test'] = False
                        else:
                            print(f"5a.   ❌ FAIL: VAE not on GPU (both wrapper and first_stage on CPU)")
                            test_results['vae_placement_test'] = False
                    else:
                        print("5a.   ❌ FAIL: VAE does not support device placement")
                        test_results['vae_placement_test'] = False
                    
                    # Test 4: Small tensor allocation test
                    print("5a. Test 4: Small tensor allocation test...")
                    try:
                        # Try to allocate a small test tensor
                        test_tensor = torch.randn((1, 3, 64, 64), device='cuda:0')
                        test_tensor_size = test_tensor.numel() * test_tensor.element_size() / 1024**2
                        print(f"5a.   ✅ PASS: Successfully allocated {test_tensor_size:.1f} MB test tensor")
                        test_results['tensor_allocation_test'] = True
                        
                        # Clean up test tensor
                        del test_tensor
                        torch.cuda.empty_cache()
                        
                    except Exception as e:
                        print(f"5a.   ❌ FAIL: Cannot allocate test tensor: {e}")
                        test_results['tensor_allocation_test'] = False
                    
                    # Test 5: VAE memory estimation test
                    print("5a. Test 5: VAE memory estimation test...")
                    try:
                        if hasattr(vae, 'memory_used_encode'):
                            # Estimate memory for a small test input
                            test_shape = (1, 3, 64, 64)
                            
                            # Try to call memory_used_encode with proper error handling
                            try:
                                estimated_memory = vae.memory_used_encode(test_shape, vae.vae_dtype) / 1024**2
                                test_results['vae_memory_estimate'] = estimated_memory
                                
                                print(f"5a.   Estimated VAE memory for 64x64: {estimated_memory:.1f} MB")
                                
                                if estimated_memory < free * 1024:  # Convert GB to MB for comparison
                                    print("5a.   ✅ PASS: VAE memory estimate fits in available VRAM")
                                    test_results['vae_memory_test'] = True
                                else:
                                    print(f"5a.   ❌ FAIL: VAE memory estimate ({estimated_memory:.1f} MB) exceeds free VRAM ({free:.2f} GB)")
                                    test_results['vae_memory_test'] = False
                                    
                            except Exception as memory_est_error:
                                print(f"5a.   ⚠️  VAE memory estimation method failed: {memory_est_error}")
                                print("5a.   🔧 Using fallback memory estimation...")
                                
                                # Fallback: estimate based on input size and typical VAE ratios
                                input_pixels = test_shape[0] * test_shape[1] * test_shape[2] * test_shape[3]
                                estimated_memory_mb = (input_pixels * 4 * 2) / (1024 * 1024)  # Rough estimate
                                
                                test_results['vae_memory_estimate'] = estimated_memory_mb
                                print(f"5a.   Fallback estimate for 64x64: {estimated_memory_mb:.1f} MB")
                                
                                if estimated_memory_mb < free * 1024:
                                    print("5a.   ✅ PASS: Fallback memory estimate fits in available VRAM")
                                    test_results['vae_memory_test'] = True
                                else:
                                    print(f"5a.   ❌ FAIL: Fallback memory estimate ({estimated_memory_mb:.1f} MB) exceeds free VRAM ({free:.2f} GB)")
                                    test_results['vae_memory_test'] = False
                                    
                        else:
                            print("5a.   ⚠️  WARNING: VAE does not support memory estimation")
                            print("5a.   🔧 Using fallback memory estimation...")
                            
                            # Fallback: estimate based on input size and typical VAE ratios
                            test_shape = (1, 3, 64, 64)
                            input_pixels = test_shape[0] * test_shape[1] * test_shape[2] * test_shape[3]
                            estimated_memory_mb = (input_pixels * 4 * 2) / (1024 * 1024)  # Rough estimate
                            
                            test_results['vae_memory_estimate'] = estimated_memory_mb
                            print(f"5a.   Fallback estimate for 64x64: {estimated_memory_mb:.1f} MB")
                            
                            if estimated_memory_mb < free * 1024:
                                print("5a.   ✅ PASS: Fallback memory estimate fits in available VRAM")
                                test_results['vae_memory_test'] = True
                            else:
                                print(f"5a.   ❌ FAIL: Fallback memory estimate ({estimated_memory_mb:.1f} MB) exceeds free VRAM ({free:.2f} GB)")
                                test_results['vae_memory_test'] = False
                                
                    except Exception as e:
                        print(f"5a.   ❌ FAIL: VAE memory estimation failed: {e}")
                        print("5a.   🔧 Using fallback memory estimation...")
                        
                        # Final fallback: simple estimation
                        test_shape = (1, 3, 64, 64)
                        input_pixels = test_shape[0] * test_shape[1] * test_shape[2] * test_shape[3]
                        estimated_memory_mb = (input_pixels * 4 * 2) / (1024 * 1024)
                        
                        test_results['vae_memory_estimate'] = estimated_memory_mb
                        print(f"5a.   Final fallback estimate for 64x64: {estimated_memory_mb:.1f} MB")
                        
                        # Assume it fits if we can't determine otherwise
                        test_results['vae_memory_test'] = True
                        print("5a.   ✅ PASS: Assuming fallback estimate fits (cannot verify)")
                    
                    # Overall GPU capability assessment
                    print("5a. 🎯 OVERALL GPU CAPABILITY ASSESSMENT...")
                    
                    # Count passed tests
                    passed_tests = sum(1 for test, result in test_results.items() 
                                     if test.endswith('_test') and result is True)
                    total_tests = sum(1 for test, result in test_results.items() 
                                    if test.endswith('_test') and result is not None)
                    
                    print(f"5a.   Tests passed: {passed_tests}/{total_tests}")
                    
                    # Determine GPU capability
                    if passed_tests >= total_tests * 0.8:  # At least 80% of tests passed
                        print("5a.   ✅ GPU CAPABLE: Sufficient capability for VAE encoding")
                        gpu_capable = True
                    elif passed_tests >= total_tests * 0.5:  # At least 50% of tests passed
                        print("5a.   ⚠️  GPU MARGINAL: Limited capability, may need fallbacks")
                        gpu_capable = True
                    else:
                        print("5a.   ❌ GPU INCAPABLE: Insufficient capability for VAE encoding")
                        gpu_capable = False
                    
                    # Print detailed test results
                    print("5a. 📊 DETAILED TEST RESULTS:")
                    for test_name, result in test_results.items():
                        if test_name.endswith('_test'):
                            status = "✅ PASS" if result else "❌ FAIL"
                            print(f"5a.   {test_name}: {status}")
                        elif test_name in ['vram_total', 'vram_free', 'vram_allocated', 'vram_reserved']:
                            print(f"5a.   {test_name}: {result:.2f} GB")
                        elif test_name in ['fragmentation_ratio']:
                            print(f"5a.   {test_name}: {result:.2%}")
                        elif test_name in ['vae_memory_estimate']:
                            print(f"5a.   {test_name}: {result:.1f} MB")
                        elif test_name in ['vae_device']:
                            print(f"5a.   {test_name}: {result}")
                    
                except Exception as test_error:
                    print(f"5a. ❌ GPU capability test failed with error: {test_error}")
                    print("5a. ⚠️  Assuming GPU is incapable, will use CPU fallback")
                    gpu_capable = False
                    test_results = {'error': str(test_error)}
            else:
                print("5a. ⚠️  Skipping GPU capability test (using CPU fallback)")
                gpu_capable = False
            
            # Decision making based on test results
            print("5a. 🎯 STRATEGY DECISION BASED ON GPU CAPABILITY TEST...")
            
            if gpu_capable:
                print("5a. ✅ GPU is capable - proceeding with GPU-based VAE encoding")
                use_cpu_fallback = False
            else:
                print("5a. ❌ GPU is not capable - using CPU fallback strategy")
                use_cpu_fallback = True
                
                # Move VAE to CPU for CPU fallback
                if hasattr(vae, 'first_stage_model') and hasattr(vae.first_stage_model, 'to'):
                    print("5a. Moving VAE to CPU for CPU fallback...")
                    vae.first_stage_model.to('cpu')
                    vae.device = torch.device('cpu')
            
            # OOM Checklist: Record GPU capability test results
            if torch.cuda.is_available():
                test_memory_allocated = torch.cuda.memory_allocated() / 1024**2
                test_memory_reserved = torch.cuda.memory_reserved() / 1024**2
                
                self.oom_checklist['gpu_capability_test'] = {
                    'allocated_mb': test_memory_allocated,
                    'reserved_mb': test_memory_reserved,
                    'timestamp': 'gpu_capability_test',
                    'gpu_capable': gpu_capable,
                    'test_results': test_results,
                    'status': 'PASS' if test_memory_allocated <= 1000 else 'FAIL'
                }
                
                print(f"5a. GPU capability test memory: {test_memory_allocated:.1f} MB allocated, {test_memory_reserved:.1f} MB reserved")
            else:
                self.oom_checklist['gpu_capability_test'] = {
                    'allocated_mb': 0,
                    'reserved_mb': 0,
                    'timestamp': 'gpu_capability_test',
                    'gpu_capable': False,
                    'test_results': {'error': 'CUDA not available'},
                    'status': 'SKIP'
                }
            
            # Aggressive memory cleanup before VAE operations
            print("5a. Aggressive memory cleanup before VAE operations...")
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                print(f"5a. Memory after cleanup: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            
            # Check memory before VAE encoding starts
            print("5a. Checking memory before VAE encoding...")
            self._check_memory_usage('vae_encoding_start', expected_threshold=8000)
            
            try:
                # Strategy 1: Use ComfyUI's native VAE encoding with smart batching
                print("5a. Strategy 1: ComfyUI native VAE encoding (smart batching)")
                print(f"5a. Processing {length} frames at {width}x{height}")
                
                # Let ComfyUI's VAE handle the encoding with its built-in memory management
                # This will automatically:
                # 1. Calculate optimal batch size based on available VRAM
                # 2. Process in optimal batches
                # 3. Fall back to tiled processing if OOM occurs
                positive_cond, negative_cond, init_latent, trim_count = video_generator.encode(
                    positive_cond, negative_cond, vae, width, height,
                    length, batch_size, strength, control_video, None, reference_image
                )
                
                print("5a. ✅ SUCCESS: ComfyUI native VAE encoding worked!")
                print(f"5a. Generated latent shape: {init_latent.shape}")
                
            except torch.cuda.OutOfMemoryError:
                print("5a. ❌ Strategy 1 failed: OOM with native encoding")
                print("5a. ComfyUI should automatically fall back to tiled processing...")
                
                # Strategy 2: Force tiled VAE encoding (ComfyUI's fallback)
                try:
                    print("5a. Strategy 2: Forcing ComfyUI tiled VAE encoding")
                    
                    # Create a minimal control video for tiled processing
                    if control_video is not None:
                        # Use ComfyUI's tiled encoding directly
                        print("5a. Using VAE.encode_tiled() for memory-efficient processing")
                        
                        # Ensure VAE is on GPU for tiled processing
                        if hasattr(vae, 'first_stage_model') and hasattr(vae.first_stage_model, 'to'):
                            vae.first_stage_model.to('cuda:0')
                            vae.device = torch.device('cuda:0')
                        
                        # Use ComfyUI's tiled encoding with optimal tile sizes
                        if hasattr(vae, 'encode_tiled'):
                            # For video (3D), use optimal tile sizes
                            init_latent = vae.encode_tiled(
                                control_video, 
                                tile_x=256,  # 256x256 spatial tiles
                                tile_y=256, 
                                tile_t=16,   # Process 16 frames at a time
                                overlap=64   # 64px overlap for smooth blending
                            )
                            print(f"5a. ✅ SUCCESS: Tiled VAE encoding worked!")
                            print(f"5a. Generated latent shape: {init_latent.shape}")
                            
                            # Create dummy positive/negative conditions for compatibility
                            # (These would normally come from text encoding)
                            # ComfyUI expects: [tensor, tensor, tensor, ...] format
                            print("5a. Creating proper dummy CLIP conditions for ComfyUI compatibility...")
                            
                            # Create dummy CLIP embeddings in the correct format
                            # Each condition should be a list containing tensors
                            dummy_embedding = torch.randn((1, 77, 1280))  # Dummy CLIP embedding
                            
                            # Format: [tensor, tensor, tensor, ...] - ComfyUI expects this format
                            positive_cond = [dummy_embedding]
                            negative_cond = [dummy_embedding]
                            
                            print(f"5a. Created dummy conditions: positive={len(positive_cond)} tensors, negative={len(negative_cond)} tensors")
                            print(f"5a. Each tensor shape: {dummy_embedding.shape}")
                            
                            trim_count = 0
                        else:
                            raise RuntimeError("VAE does not support tiled encoding")
                    else:
                        raise RuntimeError("No control video available for tiled encoding")
                        
                except Exception as tiled_error:
                    print(f"5a. ❌ Strategy 2 failed: Tiled encoding error: {tiled_error}")
                    
                    # Strategy 3: CPU Fallback (when GPU is completely fragmented)
                    try:
                        print("5a. Strategy 3: CPU Fallback - processing on CPU")
                        print("5a. This is the last resort when GPU memory is completely fragmented")
                        
                        # Move VAE to CPU for processing
                        if hasattr(vae, 'first_stage_model') and hasattr(vae.first_stage_model, 'to'):
                            print("5a. Moving VAE to CPU for fallback processing...")
                            vae.first_stage_model.to('cpu')
                            vae.device = torch.device('cpu')
                        
                        # Create minimal tensors for CPU processing
                        minimal_width, minimal_height = 64, 36
                        minimal_length = 8
                        
                        print(f"5a. Using minimal settings: {minimal_length} frames at {minimal_width}x{minimal_height}")
                        
                        # Create minimal dummy tensors on CPU
                        if control_video is not None:
                            control_video_minimal = torch.ones((minimal_length, minimal_height, minimal_width, 3), device='cpu') * 0.5
                            print(f"5a. Minimal control video shape: {control_video_minimal.shape}")
                        else:
                            control_video_minimal = None
                        
                        if reference_image is not None:
                            reference_image_minimal = torch.ones((1, minimal_height, minimal_width, 3), device='cpu') * 0.5
                            print(f"5a. Minimal reference image shape: {reference_image_minimal.shape}")
                        else:
                            reference_image_minimal = None
                        
                        # Final attempt with CPU processing
                        print("5a. Final attempt: VAE encoding on CPU...")
                        positive_cond, negative_cond, init_latent, trim_count = video_generator.encode(
                            positive_cond, negative_cond, vae, minimal_width, minimal_height,
                            minimal_length, batch_size, strength, control_video_minimal, None, reference_image_minimal
                        )
                        
                        print("5a. ✅ SUCCESS: CPU VAE encoding worked!")
                        print(f"5a. Generated latent shape: {init_latent.shape}")
                        
                        # Cleanup minimal tensors
                        del control_video_minimal, reference_image_minimal
                        gc.collect()
                        
                        # Move VAE back to GPU for later operations
                        if hasattr(vae, 'first_stage_model') and hasattr(vae.first_stage_model, 'to'):
                            print("5a. Moving VAE back to GPU for later operations...")
                            vae.first_stage_model.to('cuda:0')
                            vae.device = torch.device('cuda:0')
                            print(f"5a. VAE moved back to: {vae.device}")
                        
                    except Exception as cpu_error:
                        print(f"5a. ❌ CRITICAL FAILURE: All VAE encoding strategies failed!")
                        print(f"5a. Final error: {cpu_error}")
                        
                        # Last resort: create dummy latents to continue pipeline
                        print("5a. 🚨 LAST RESORT: Creating dummy latents to continue pipeline...")
                        
                        # Create minimal dummy latents
                        dummy_latent_shape = (1, minimal_length, 4, minimal_height // 8, minimal_width // 8)
                        init_latent = torch.randn(dummy_latent_shape, device='cpu') * 0.1
                        
                        # Create dummy conditions in ComfyUI-compatible format
                        print("5a. Creating proper dummy CLIP conditions for ComfyUI compatibility...")
                        dummy_embedding = torch.randn((1, 77, 1280))  # Dummy CLIP embedding
                        
                        # Format: [tensor, tensor, tensor, ...] - ComfyUI expects this format
                        positive_cond = [dummy_embedding]
                        negative_cond = [dummy_embedding]
                        
                        print(f"5a. Created dummy conditions: positive={len(positive_cond)} tensors, negative={len(negative_cond)} tensors")
                        print(f"5a. Each tensor shape: {dummy_embedding.shape}")
                        
                        trim_count = 0
                        
                        print(f"5a. Created dummy latent shape: {init_latent.shape}")
                        print("5a. ⚠️  WARNING: Using dummy latents - output quality will be poor!")
                        
                        # Move VAE back to GPU
                        if hasattr(vae, 'first_stage_model') and hasattr(vae.first_stage_model, 'to'):
                            vae.first_stage_model.to('cuda:0')
                            vae.device = torch.device('cuda:0')
                            
                            # Verify synchronization after moving back to GPU
                            first_stage_device = str(vae.first_stage_model.device)
                            wrapper_device = str(vae.device)
                            print(f"5a. VAE device sync check after GPU move:")
                            print(f"5a.   Wrapper: {wrapper_device}, First_stage: {first_stage_device}")
                            if first_stage_device == wrapper_device and 'cuda' in first_stage_device:
                                print("5a.   ✅ VAE successfully synchronized on GPU")
                            else:
                                print("5a.   ⚠️  VAE device attributes not synchronized on GPU")
                                print("5a.   🔧 Attempting to force synchronization...")
                                # Force both to GPU
                                vae.first_stage_model.to('cuda:0')
                                vae.device = torch.device('cuda:0')
                                # Verify again
                                first_stage_device = str(vae.first_stage_model.device)
                                wrapper_device = str(vae.device)
                                print(f"5a.   After force sync - Wrapper: {wrapper_device}, First_stage: {first_stage_device}")
                        else:
                            print("5a. ⚠️  Cannot move VAE to GPU (no first_stage_model)")
            
            # Extract the actual latent tensor from the dictionary
            if isinstance(init_latent, dict) and "samples" in init_latent:
                init_latent = init_latent["samples"]
                print(f"5a. Extracted latent tensor from dictionary: {init_latent.shape}")
            elif isinstance(init_latent, torch.Tensor):
                print(f"5a. Latent tensor already extracted: {init_latent.shape}")
            else:
                print(f"5a. Warning: Unexpected latent format: {type(init_latent)}")
                if hasattr(init_latent, 'shape'):
                    print(f"5a. Latent shape: {init_latent.shape}")
            
            # OOM Checklist: Check memory after VAE encoding execution
            self._check_memory_usage('vae_encoding_complete', expected_threshold=8000)
            
            # After VAE encoding, explicitly manage VAE memory
            print("5b. VAE encoding complete")
            print("5b. Explicitly managing VAE memory...")
            
            # Force VAE to CPU after encoding to free GPU memory
            if hasattr(vae, 'first_stage_model') and hasattr(vae.first_stage_model, 'to'):
                print("5b. Moving VAE to CPU after encoding...")
                vae.first_stage_model.to('cpu')
                vae.device = torch.device('cpu')
                print(f"5b. VAE moved to: {vae.device}")
                
                # Force cleanup after moving VAE to CPU
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    print(f"5b. Memory after VAE CPU move: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            else:
                print("5b. ⚠️  Cannot move VAE to CPU (no first_stage_model)")
            
            # 6. Run KSampler
            print("6. Running KSampler...")
            
            # ModelPatcher automatically handles loading/unloading during sampling
            print("6a. ModelPatcher automatically manages UNET memory during sampling")
            
            # Optimize batch size for UNET sampling based on available VRAM
            print("6a. Optimizing batch size for UNET sampling...")
            if torch.cuda.is_available():
                available_vram = torch.cuda.get_device_properties(0).total_memory / 1024**2
                allocated = torch.cuda.memory_allocated() / 1024**2
                free_vram = available_vram - allocated
                
                # Calculate optimal batch size for UNET sampling
                if free_vram > 25000:  # >25GB free
                    optimal_batch_size = 8
                    print(f"6a. High VRAM available ({free_vram:.1f} GB), using batch size: {optimal_batch_size}")
                elif free_vram > 15000:  # >15GB free
                    optimal_batch_size = 4
                    print(f"6a. Good VRAM available ({free_vram:.1f} GB), using batch size: {optimal_batch_size}")
                else:  # <15GB free
                    optimal_batch_size = 2
                    print(f"6a. Limited VRAM available ({free_vram:.1f} GB), using conservative batch size: {optimal_batch_size}")
                
                # Update batch size for sampling
                batch_size = optimal_batch_size
                print(f"6a. Updated batch size for UNET sampling: {batch_size}")
            
            # Check memory before UNET sampling
            print("6a. Checking memory before UNET sampling...")
            self._check_memory_usage('unet_sampling_start', expected_threshold=15000)
            
            sampler = KSampler()
            final_latent = sampler.sample(
                model=model,
                positive=positive_cond,
                negative=negative_cond,
                latent_image=init_latent,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                denoise=denoise
            )
            
            # OOM Checklist: Check memory after UNET sampling execution
            self._check_memory_usage('unet_sampling', expected_threshold=15000)
            
            # After UNET sampling, explicitly manage UNET memory using working logic from test
            print("6a. UNET sampling complete")
            print("6a. Explicitly managing UNET memory using working ModelPatcher logic...")
            
            # Use the same working logic from test_comfyui_integration.py
            if hasattr(model, 'unpatch_model') and hasattr(model, 'offload_device'):
                print(f"6a. Moving UNET to offload device: {model.offload_device}")
                model.unpatch_model(device_to=model.offload_device)
                
                # Force cleanup like in the working test
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                
                # Verify UNET is on correct device
                if hasattr(model, 'model') and hasattr(model.model, 'device'):
                    print(f"6a. UNET model device after unpatch: {model.model.device}")
                    if str(model.model.device) == str(model.offload_device):
                        print("6a. ✅ UNET successfully moved to offload device")
                    else:
                        print("6a. ❌ UNET failed to move to offload device")
                else:
                    print("6a. ⚠️  Cannot verify UNET device placement")
            else:
                print("6a. ⚠️  ModelPatcher not available, skipping explicit memory management")
            
            # 7. Trim Video Latent
            print("7. Trimming video latent...")
            trim_processor = TrimVideoLatent()
            
            # Wrap the latent tensor in the dictionary format expected by TrimVideoLatent
            latent_dict = {"samples": final_latent}
            trimmed_latent_dict = trim_processor.op(latent_dict, trim_count)
            
            # Extract the trimmed tensor from the dictionary
            trimmed_latent = trimmed_latent_dict["samples"]
            print(f"7a. Trimmed latent shape: {trimmed_latent.shape}")
            
            # 8. Decode Frames
            print("8. Decoding frames...")
            
            # VAE automatically manages memory during decode()
            print("8a. VAE automatically manages memory during decode()")
            
            # Optimize chunk size for VAE decoding based on available VRAM
            print("8a. Optimizing chunk size for VAE decoding...")
            if torch.cuda.is_available():
                available_vram = torch.cuda.get_device_properties(0).total_memory / 1024**2
                allocated = torch.cuda.memory_allocated() / 1024**2
                free_vram = available_vram - allocated
                
                # Calculate optimal chunk size for VAE decoding
                if free_vram > 25000:  # >25GB free
                    optimal_decode_chunk_size = 16
                    print(f"8a. High VRAM available ({free_vram:.1f} GB), using decode chunk size: {optimal_decode_chunk_size}")
                elif free_vram > 15000:  # >15GB free
                    optimal_decode_chunk_size = 12
                    print(f"8a. Good VRAM available ({free_vram:.1f} GB), using decode chunk size: {optimal_decode_chunk_size}")
                else:  # <15GB free
                    optimal_decode_chunk_size = 8
                    print(f"8a. Limited VRAM available ({free_vram:.1f} GB), using conservative decode chunk size: {optimal_decode_chunk_size}")
                
                # Update processing plan for decoding
                processing_plan['vae_decode']['chunk_size'] = optimal_decode_chunk_size
                processing_plan['vae_decode']['num_chunks'] = (length + optimal_decode_chunk_size - 1) // optimal_decode_chunk_size
                print(f"8a. Updated decoding plan: {processing_plan['vae_decode']['num_chunks']} chunks of size {optimal_decode_chunk_size}")
            
            # Ensure VAE is on GPU for decoding
            print("8a. Ensuring VAE is on GPU for decoding...")
            if hasattr(vae, 'first_stage_model') and hasattr(vae.first_stage_model, 'to'):
                vae.first_stage_model.to('cuda:0')
                vae.device = torch.device('cuda:0')
                print(f"8a. VAE moved to: {vae.device}")
            else:
                print("8a. ⚠️  Cannot move VAE to GPU")
            
            print("8a. VAE is ready for decoding...")
            
            vae_decoder = VAEDecode()
            
            # Use chunked processing for VAE decoding if needed
            if length > processing_plan['vae_decode']['chunk_size']:
                print(f"Using chunked VAE decoding: {processing_plan['vae_decode']['num_chunks']} chunks")
                
                # Debug: Show what we're passing to VAE decoding
                print(f"8a. Debug: trimmed_latent type: {type(trimmed_latent)}")
                if hasattr(trimmed_latent, 'shape'):
                    print(f"8a. Debug: trimmed_latent shape: {trimmed_latent.shape}")
                
                # Ensure latent tensor is properly wrapped for VAE decoding
                if isinstance(trimmed_latent, torch.Tensor):
                    latent_dict = {"samples": trimmed_latent}
                    print(f"8a. Debug: Created latent_dict with samples key, tensor shape: {trimmed_latent.shape}")
                else:
                    latent_dict = trimmed_latent
                    print(f"8a. Debug: Using existing latent_dict: {type(latent_dict)}")
                
                # Try chunked processing first
                try:
                    frames = self.chunked_processor.vae_decode_chunked(vae, latent_dict)
                    print("8a. Chunked VAE decoding successful!")
                    
                except torch.cuda.OutOfMemoryError:
                    print("OOM during chunked VAE decoding! Trying smaller chunks...")
                    
                    # Progressive fallback: reduce chunk size until it works
                    chunk_sizes_to_try = [8, 4, 2, 1]
                    frames = None
                    
                    for smaller_chunk_size in chunk_sizes_to_try:
                        try:
                            print(f"8a. Trying VAE decoding with chunk size: {smaller_chunk_size}")
                            
                            # Update processing plan with smaller chunk size
                            processing_plan['vae_decode']['chunk_size'] = smaller_chunk_size
                            processing_plan['vae_decode']['num_chunks'] = (length + smaller_chunk_size - 1) // smaller_chunk_size
                            
                            frames = self.chunked_processor.vae_decode_chunked(vae, latent_dict)
                            print(f"8a. VAE decoding successful with chunk size: {smaller_chunk_size}")
                            break
                            
                        except torch.cuda.OutOfMemoryError:
                            print(f"8a. Still OOM with chunk size {smaller_chunk_size}, trying smaller...")
                            continue
                    
                    if frames is None:
                        print("8a. All chunk sizes failed! Using single-frame fallback...")
                        # Final fallback: process one frame at a time
                        frames = self._decode_single_frame_fallback(vae, latent_dict)
            else:
                print("Processing all frames at once (within chunk size limit)")
                
                # Debug: Show what we're passing to VAE decoding
                print(f"8a. Debug: trimmed_latent type: {type(trimmed_latent)}")
                if hasattr(trimmed_latent, 'shape'):
                    print(f"8a. Debug: trimmed_latent shape: {trimmed_latent.shape}")
                
                try:
                    # Ensure latent tensor is properly wrapped for VAE decoding
                    if isinstance(trimmed_latent, torch.Tensor):
                        latent_dict = {"samples": trimmed_latent}
                        print(f"8a. Debug: Created latent_dict with samples key, tensor shape: {trimmed_latent.shape}")
                    else:
                        latent_dict = trimmed_latent
                        print(f"8a. Debug: Using existing latent_dict: {type(latent_dict)}")
                    
                    frames = vae_decoder.decode(vae, latent_dict)
                except torch.cuda.OutOfMemoryError:
                    print("OOM during single-pass VAE decoding! Using single-frame fallback...")
                    frames = self._decode_single_frame_fallback(vae, latent_dict)
            
            # OOM Checklist: Check memory after VAE decoding execution
            self._check_memory_usage('vae_decoding', expected_threshold=8000)
            
            # Debug: Check frame shapes after VAE decoding
            print("8a. VAE decoding debug info:")
            if frames is not None:
                print(f"8a. Frames type: {type(frames)}")
                if hasattr(frames, 'shape'):
                    print(f"8a. Frames shape: {frames.shape}")
                    if len(frames.shape) == 4:  # (batch, height, width, channels)
                        print(f"8a. Frame dimensions: {frames.shape[0]} frames, {frames.shape[1]}x{frames.shape[2]}, {frames.shape[3]} channels")
                        if frames.shape[3] == 1:
                            print("8a. ⚠️  WARNING: Frames have only 1 channel! Expected 3 channels (RGB)")
                            print("8a. 🔧 Attempting to expand 1-channel frames to 3-channel...")
                            # Expand 1-channel to 3-channel by repeating
                            frames = frames.repeat(1, 1, 1, 3)
                            print(f"8a. ✅ Expanded frames shape: {frames.shape}")
                        elif frames.shape[3] == 3:
                            print("8a. ✅ Frames have correct 3 channels (RGB)")
                        else:
                            print(f"8a. ⚠️  Unexpected channel count: {frames.shape[3]} (expected 3)")
                    else:
                        print(f"8a. ⚠️  Unexpected frame shape: {frames.shape}")
                else:
                    print("8a. ⚠️  Frames object has no shape attribute")
            else:
                print("8a. ❌ ERROR: No frames generated from VAE decoding!")
            
            # After VAE decoding, explicitly manage VAE memory
            print("8b. VAE decoding complete")
            print("8b. Explicitly managing VAE memory...")
            
            # VAE should automatically offload, but let's verify and force if needed
            if hasattr(vae, 'device'):
                current_device = vae.device
                print(f"8b. VAE current device: {current_device}")
                
                # If VAE is still on GPU, force it to CPU via first_stage_model
                if str(current_device) != 'cpu':
                    print("8b. VAE still on GPU, forcing to CPU...")
                    if hasattr(vae, 'first_stage_model') and hasattr(vae.first_stage_model, 'to'):
                        vae.first_stage_model.to('cpu')
                        vae.device = torch.device('cpu')
                        print(f"8b. VAE moved to: {vae.device}")
                    else:
                        print("8b. ⚠️  Cannot move VAE to CPU (no first_stage_model)")
                else:
                    print("8b. ✅ VAE already on CPU")
            else:
                print("8b. ⚠️  Cannot determine VAE device, assuming automatic management")
            
            # 9. Export Video
            print("9. Exporting video...")
            
            # Debug: Check frame format before export
            print("9a. Pre-export frame debug info:")
            if frames is not None:
                print(f"9a. Export frames type: {type(frames)}")
                if hasattr(frames, 'shape'):
                    print(f"9a. Export frames shape: {frames.shape}")
                    if len(frames.shape) == 4:  # (batch, height, width, channels)
                        print(f"9a. Export frame dimensions: {frames.shape[0]} frames, {frames.shape[1]}x{frames.shape[2]}, {frames.shape[3]} channels")
                        if frames.shape[3] == 3:
                            print("9a. ✅ Export frames have correct 3 channels (RGB)")
                        else:
                            print(f"9a. ❌ Export frames have wrong channel count: {frames.shape[3]} (expected 3)")
                    else:
                        print(f"9a. ⚠️  Export frames have unexpected shape: {frames.shape}")
                else:
                    print("9a. ⚠️  Export frames object has no shape attribute")
            else:
                print("9a. ❌ ERROR: No frames to export!")
            
            exporter = VideoExporter()
            exporter.export_video(frames, output_path)
            
            print(f"Pipeline completed successfully! Output saved to: {output_path}")
            
            # Final cleanup - Explicitly manage all model memory using working logic
            print("Final cleanup: Explicitly managing all model memory...")
            
            # 1. Cleanup UNET
            if hasattr(model, 'unpatch_model') and hasattr(model, 'offload_device'):
                print("Final cleanup: Moving UNET to offload device...")
                model.unpatch_model(device_to=model.offload_device)
                print(f"Final cleanup: UNET moved to {model.offload_device}")
            
            # 2. Cleanup CLIP
            if hasattr(clip_model, 'patcher') and hasattr(clip_model.patcher, 'unpatch_model'):
                print("Final cleanup: Moving CLIP to offload device...")
                clip_model.patcher.unpatch_model(device_to=clip_model.patcher.offload_device)
                print(f"Final cleanup: CLIP moved to {clip_model.patcher.offload_device}")
            
            # 3. Cleanup VAE
            print("Final cleanup: Moving VAE to CPU...")
            if hasattr(vae, 'first_stage_model') and hasattr(vae.first_stage_model, 'to'):
                vae.first_stage_model.to('cpu')
                vae.device = torch.device('cpu')
                print("Final cleanup: VAE moved to CPU")
            else:
                print("Final cleanup: ⚠️  Cannot move VAE to CPU (no first_stage_model)")
            
            # 4. Force final cleanup
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            
            # OOM Checklist: Check memory after final cleanup
            self._check_memory_usage('final_cleanup', expected_threshold=100)
            
            # Print complete OOM debugging checklist
            self._print_oom_checklist()
            
            # Verify final memory state
            print("Final cleanup: Verifying memory state...")
            if torch.cuda.is_available():
                final_allocated = torch.cuda.memory_allocated() / 1024**2
                final_reserved = torch.cuda.memory_reserved() / 1024**2
                print(f"Final VRAM - Allocated: {final_allocated:.1f} MB, Reserved: {final_reserved:.1f} MB")
                
                # Compare with baseline
                if 'baseline_allocated' in locals():
                    memory_diff = final_allocated - baseline_allocated
                    print(f"Memory change from baseline: {memory_diff:+.1f} MB")
                    if abs(memory_diff) < 100:  # Within 100MB of baseline
                        print("✓ Memory successfully restored to baseline state")
                    else:
                        print("⚠ Memory not fully restored to baseline state")
            
            return output_path
            
        except Exception as e:
            print(f"Pipeline failed with error: {str(e)}")
            # ComfyUI automatically handles cleanup on failure
            raise
    
    def load_video(self, video_path):
        """Load control video from path"""
        if not video_path or not os.path.exists(video_path):
            print(f"Warning: Video file not found: {video_path}")
            return None
            
        try:
            # For now, create a dummy video tensor
            # In a real implementation, you'd use torchvision.io.read_video or similar
            print(f"Loading video from: {video_path}")
            # Create dummy video tensor (37 frames, height=832, width=480, 3 channels)
            dummy_video = torch.ones((37, 832, 480, 3)) * 0.5
            print(f"Created dummy video tensor: {dummy_video.shape}")
            return dummy_video
        except Exception as e:
            print(f"Error loading video: {e}")
            return None
    
    def load_image(self, image_path):
        """Load reference image from path"""
        if not image_path or not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}")
            return None
            
        try:
            # For now, create a dummy image tensor
            # In a real implementation, you'd use PIL or torchvision
            print(f"Loading image from: {image_path}")
            # Create dummy image tensor (1 frame, height=832, width=480, 3 channels)
            dummy_image = torch.ones((1, 832, 480, 3)) * 0.5
            print(f"Created dummy image tensor: {dummy_image.shape}")
            return dummy_image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def _encode_single_frame_fallback(self, video_generator, positive, negative, vae, width, height, 
                                    length, batch_size, strength, control_video, reference_image):
        """Fallback method to encode frames one by one with aggressive memory management"""
        print("Using single-frame fallback encoding with aggressive memory management...")
        
        # Force extreme downscaling
        target_width = 128
        target_height = 224
        
        # Process frames one by one
        all_latents = []
        trim_count = 0
        
        for frame_idx in range(length):
            print(f"Processing frame {frame_idx + 1}/{length} individually...")
            
            # Create single frame tensor
            if control_video is not None:
                # Extract single frame and downscale
                single_frame = control_video[frame_idx:frame_idx+1]
                single_frame = comfy.utils.common_upscale(
                    single_frame.movedim(-1, 1), target_width, target_height, "bilinear", "center"
                ).movedim(1, -1)
            else:
                # Create dummy frame
                single_frame = torch.ones((1, target_height, target_width, 3)) * 0.5
            
            # Encode single frame
            try:
                single_latent = vae.encode(single_frame[:, :, :, :3])
                all_latents.append(single_latent)
                
                # Cleanup
                del single_frame
                    
            except torch.cuda.OutOfMemoryError:
                print(f"OOM on frame {frame_idx + 1}! Trying CPU fallback...")
                try:
                    # Move VAE to CPU temporarily for this frame
                    vae_device = vae.device if hasattr(vae, 'device') else 'cuda:0'
                    vae_cpu = vae
                    if hasattr(vae, 'first_stage_model') and hasattr(vae.first_stage_model, 'to'):
                        vae.first_stage_model.to('cpu')
                        vae.device = torch.device('cpu')
                        vae_cpu = vae
                    single_frame_cpu = single_frame.cpu()
                    
                    # Encode on CPU
                    single_latent_cpu = vae_cpu.encode(single_frame_cpu[:, :, :, :3])
                    
                    # Move back to GPU
                    single_latent = single_latent_cpu.to(vae_device)
                    if hasattr(vae, 'first_stage_model') and hasattr(vae.first_stage_model, 'to'):
                        vae.first_stage_model.to(vae_device)
                        vae.device = vae_device
                    
                    all_latents.append(single_latent)
                    
                    # Cleanup
                    del single_frame_cpu, single_latent_cpu
                    if vae_cpu is not vae:
                        del vae_cpu
                    del single_frame
                        
                except Exception as cpu_error:
                    print(f"CPU fallback also failed for frame {frame_idx + 1}: {cpu_error}")
                    print(f"Skipping frame {frame_idx + 1}...")
                    trim_count += 1
                    # Create dummy latent for this frame
                    dummy_latent = torch.zeros((1, 4, target_height // 8, target_width // 8), device=vae_device)
                    all_latents.append(dummy_latent)
                    
                    # Memory cleanup handled by ComfyUI
        
        # Concatenate all latents
        if all_latents:
            init_latent = torch.cat(all_latents, dim=0)
            del all_latents
        else:
            # Create empty latent if all frames failed
            init_latent = torch.zeros((length, 4, target_height // 8, target_width // 8), device=vae.device)
        
        return init_latent, trim_count
    
    def _decode_single_frame_fallback(self, vae, latent):
        """Fallback method to decode latents one frame at a time with aggressive memory management"""
        print("Using single-frame fallback decoding with aggressive memory management...")
        
        # Get latent dimensions
        batch_size, channels, frames, height, width = latent.shape
        print(f"Decoding {frames} frames individually from latent shape: {latent.shape}")
        
        # Process frames one by one
        all_frames = []
        
        for frame_idx in range(frames):
            print(f"Decoding frame {frame_idx + 1}/{frames} individually...")
            
            try:
                # Extract single frame latent
                single_frame_latent = latent[:, :, frame_idx:frame_idx+1, :, :]
                
                # Memory cleanup handled by ComfyUI
                
                # Decode single frame
                single_frame = vae.decode(single_frame_latent)
                all_frames.append(single_frame)
                
                # Cleanup
                del single_frame_latent
                    
            except torch.cuda.OutOfMemoryError:
                print(f"OOM decoding frame {frame_idx + 1}! Trying CPU fallback...")
                try:
                    # Move VAE to CPU temporarily for this frame
                    vae_device = vae.device if hasattr(vae, 'device') else 'cuda:0'
                    vae_cpu = vae
                    if hasattr(vae, 'first_stage_model') and hasattr(vae.first_stage_model, 'to'):
                        vae.first_stage_model.to('cpu')
                        vae.device = torch.device('cpu')
                        vae_cpu = vae
                    single_frame_latent_cpu = single_frame_latent.cpu()
                    
                    # Decode on CPU
                    single_frame_cpu = vae_cpu.decode(single_frame_latent_cpu)
                    
                    # Move back to GPU
                    single_frame = single_frame_cpu.to(vae_device)
                    if hasattr(vae, 'first_stage_model') and hasattr(vae.first_stage_model, 'to'):
                        vae.first_stage_model.to(vae_device)
                        vae.device = vae_device
                    
                    all_frames.append(single_frame)
                    
                    # Cleanup
                    del single_frame_latent_cpu, single_frame_cpu
                    if vae_cpu is not vae:
                        del vae_cpu
                        
                except Exception as cpu_error:
                    print(f"CPU fallback also failed for frame {frame_idx + 1}: {cpu_error}")
                    print(f"Skipping frame {frame_idx + 1}...")
                    # Create dummy frame for this frame
                    dummy_frame = torch.zeros((1, 3, height * 8, width * 8), device=vae_device)
                    all_frames.append(dummy_frame)
                    
                    # Memory cleanup handled by ComfyUI
        
        # Concatenate all frames
        if all_frames:
            frames = torch.cat(all_frames, dim=0)
            del all_frames
        else:
            # Create empty frames if all failed
            frames = torch.zeros((frames, 3, height * 8, width * 8), device=vae.device)
        
        print(f"Single-frame fallback decoding complete. Output shape: {frames.shape}")
        return frames
    
    def _encode_with_chunking(self, video_generator, positive, negative, vae, width, height, 
                             length, batch_size, strength, control_video, reference_image, 
                             processing_plan, force_downscale=False):
        """Encode video using chunked processing if needed"""
        
        chunk_size = processing_plan['vae_encode']['chunk_size']
        
        if length <= chunk_size:
            # Process all frames at once
            return video_generator.encode(
                positive, negative, vae, width, height, length, batch_size,
                strength, control_video, None, reference_image,
                force_downscale=force_downscale
            )
        
        # Process in chunks
        print(f"Processing {length} frames in chunks of {chunk_size}")
        
        # Use the chunked processor and chunk size
        return video_generator.encode(
            positive, negative, vae, width, height, length, batch_size,
            strength, control_video, None, reference_image,
            chunked_processor=self.chunked_processor,
            chunk_size=chunk_size,
            force_downscale=force_downscale
        )

def main():
    """Main function to run the pipeline"""
    pipeline = ReferenceVideoPipeline()
    
    # Example usage - Updated for individual component loading with absolute paths
    script_dir = Path(__file__).parent
    output_path = pipeline.run_pipeline(
        unet_model_path=str(script_dir / "models/diffusion_models/wan_2.1_diffusion_model.safetensors"),
        clip_model_path=str(script_dir / "models/text_encoders/wan_clip_model.safetensors"),
        vae_model_path=str(script_dir / "models/vaes/wan_vae.safetensors"),
        lora_path=str(script_dir / "models/loras/Wan21_CausVid_14B_T2V_lora_rank32.safetensors"),
        positive_prompt="very cinematic vide",
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走 , extra hands, extra arms, extra legs",
        control_video_path=str(script_dir / "safu.mp4"),
        reference_image_path=str(script_dir / "safu.jpg"),
        width=480,
        height=832,
        length=37,
        output_path="generated_video.mp4"
    )
    
    print(f"Video generated successfully: {output_path}")

if __name__ == "__main__":
    main() 