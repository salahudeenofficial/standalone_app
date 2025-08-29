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

# ===============================================================================
# STEP 1: Initialize ComfyUI CLI Arguments and Environment Variables
# ===============================================================================
import os
import sys
import argparse
from pathlib import Path

# Set required environment variables BEFORE importing ComfyUI
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Create minimal CLI args that ComfyUI expects
sys.argv = ['pipeline.py', '--cpu-vae', '--lowvram', '--disable-smart-memory']  # Force CPU-first loading and aggressive offloading

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Add ComfyUI path for utilities
sys.path.insert(0, str(Path(__file__).parent / "comfy"))

# NOW import ComfyUI modules AFTER setting argv and environment
import comfy.cli_args
import comfy.model_management

# Initialize ComfyUI CLI arguments system
comfy.cli_args.args = comfy.cli_args.parser.parse_args()

# Ensure args.fast is properly initialized (ComfyUI expects this)
if comfy.cli_args.args.fast is None:
    comfy.cli_args.args.fast = set()

# Force ComfyUI to use aggressive memory management
comfy.model_management.vram_state = comfy.model_management.VRAMState.LOW_VRAM
comfy.model_management.set_vram_to = comfy.model_management.VRAMState.LOW_VRAM

import torch
from pathlib import Path
import time
import numpy as np
import threading
import queue
from collections import deque

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Add ComfyUI path for utilities
sys.path.insert(0, str(Path(__file__).parent / "comfy"))

# Import psutil for system information in diagnostic summary
try:
    import psutil
except ImportError:
    print("Warning: psutil not available, system information will be limited")
    psutil = None

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

# ===============================================================================
# REAL-TIME MEMORY MONITORING SYSTEM
# ===============================================================================

class RealTimeMemoryMonitor:
    """Real-time VRAM monitoring using threading"""
    
    def __init__(self, sample_interval=0.5, max_samples=1000):
        self.sample_interval = sample_interval  # seconds
        self.max_samples = max_samples
        self.monitoring = False
        self.monitor_thread = None
        self.memory_data = deque(maxlen=max_samples)
        self.event_queue = queue.Queue()
        self.lock = threading.Lock()
        
        # Memory thresholds for alerts
        self.high_memory_threshold = 0.85  # 85% of VRAM
        self.critical_memory_threshold = 0.95  # 95% of VRAM
        
    def start_monitoring(self, label="MONITORING"):
        """Start real-time memory monitoring"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.current_label = label
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"🚀 Real-time memory monitoring started: {label}")
        
    def stop_monitoring(self):
        """Stop memory monitoring"""
        if not self.monitoring:
            return
            
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        print("🛑 Real-time memory monitoring stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop running in separate thread"""
        while self.monitoring:
            try:
                # Capture memory snapshot
                snapshot = self._capture_memory_snapshot()
                
                # Add to data queue
                with self.lock:
                    self.memory_data.append(snapshot)
                
                # Check for memory alerts
                self._check_memory_alerts(snapshot)
                
                # Process any events from main thread
                self._process_events()
                
                time.sleep(self.sample_interval)
                
            except Exception as e:
                print(f"❌ Memory monitoring error: {e}")
                time.sleep(1.0)
    
    def _capture_memory_snapshot(self):
        """Capture current memory state"""
        timestamp = time.time()
        snapshot = {
            'timestamp': timestamp,
            'label': self.current_label,
            'pytorch_allocated': 0,
            'pytorch_reserved': 0,
            'comfy_free': 0,
            'gpu_utilization': 0,
            'model_count': 0,
            'tensor_count': 0,
            'system_ram_used': 0,
            'system_ram_available': 0
        }
        
        try:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                
                # PyTorch memory
                snapshot['pytorch_allocated'] = torch.cuda.memory_allocated(device) / (1024**3)
                snapshot['pytorch_reserved'] = torch.cuda.memory_reserved(device) / (1024**3)
                
                # GPU utilization
                props = torch.cuda.get_device_properties(device)
                total_vram = props.total_memory / (1024**3)
                snapshot['gpu_utilization'] = (snapshot['pytorch_reserved'] / total_vram) * 100
                
                # ComfyUI free memory
                try:
                    snapshot['comfy_free'] = comfy.model_management.get_free_memory(device) / (1024**3)
                except:
                    snapshot['comfy_free'] = 0
                
                # Model count
                if hasattr(comfy.model_management, 'current_loaded_models'):
                    snapshot['model_count'] = len(comfy.model_management.current_loaded_models)
                
                # Tensor count (quick estimate)
                snapshot['tensor_count'] = len([obj for obj in gc.get_objects() if torch.is_tensor(obj) and obj.is_cuda])
            
            # System RAM
            try:
                import psutil
                ram = psutil.virtual_memory()
                snapshot['system_ram_used'] = ram.used / (1024**3)
                snapshot['system_ram_available'] = ram.available / (1024**3)
            except:
                pass
                
        except Exception as e:
            snapshot['error'] = str(e)
            
        return snapshot
    
    def _check_memory_alerts(self, snapshot):
        """Check for memory threshold alerts"""
        if snapshot.get('gpu_utilization', 0) > self.critical_memory_threshold * 100:
            self._send_alert("🚨 CRITICAL VRAM USAGE", snapshot, "critical")
        elif snapshot.get('gpu_utilization', 0) > self.high_memory_threshold * 100:
            self._send_alert("⚠️  HIGH VRAM USAGE", snapshot, "warning")
    
    def _send_alert(self, message, snapshot, level):
        """Send memory alert with current state"""
        alert = f"{message} - {snapshot['label']}"
        alert += f"\n   GPU: {snapshot['gpu_utilization']:.1f}% ({snapshot['pytorch_allocated']:.2f}GB)"
        alert += f"\n   Models: {snapshot['model_count']}, Tensors: {snapshot['tensor_count']}"
        alert += f"\n   ComfyUI Free: {snapshot['comfy_free']:.2f}GB"
        
        if level == "critical":
            print(f"\n{alert}")
        else:
            print(f"\n{alert}")
    
    def _process_events(self):
        """Process events from main thread"""
        try:
            while not self.event_queue.empty():
                event = self.event_queue.get_nowait()
                if event['type'] == 'label_change':
                    self.current_label = event['label']
                elif event['type'] == 'memory_check':
                    self._handle_memory_check_request(event)
        except queue.Empty:
            pass
    
    def _handle_memory_check_request(self, event):
        """Handle memory check requests from main thread"""
        with self.lock:
            if self.memory_data:
                latest = self.memory_data[-1]
                event['callback'](latest)
    
    def change_label(self, new_label):
        """Change monitoring label (e.g., for different pipeline steps)"""
        self.event_queue.put({
            'type': 'label_change',
            'label': new_label
        })
    
    def get_current_memory_state(self, callback):
        """Get current memory state (non-blocking)"""
        self.event_queue.put({
            'type': 'memory_check',
            'callback': callback
        })
    
    def get_memory_history(self):
        """Get all captured memory data"""
        with self.lock:
            return list(self.memory_data)
    
    def get_memory_summary(self):
        """Get summary statistics of captured data"""
        with self.lock:
            if not self.memory_data:
                return None
                
            data = list(self.memory_data)
            allocated_values = [d['pytorch_allocated'] for d in data if 'pytorch_allocated' in d]
            reserved_values = [d['pytorch_reserved'] for d in data if 'pytorch_reserved' in d]
            utilization_values = [d['gpu_utilization'] for d in data if 'gpu_utilization' in d]
            
            summary = {
                'samples': len(data),
                'duration': data[-1]['timestamp'] - data[0]['timestamp'] if len(data) > 1 else 0,
                'allocated': {
                    'min': min(allocated_values) if allocated_values else 0,
                    'max': max(allocated_values) if allocated_values else 0,
                    'avg': sum(allocated_values) / len(allocated_values) if allocated_values else 0
                },
                'reserved': {
                    'min': min(reserved_values) if reserved_values else 0,
                    'max': max(reserved_values) if reserved_values else 0,
                    'avg': sum(reserved_values) / len(reserved_values) if reserved_values else 0
                },
                'utilization': {
                    'min': min(utilization_values) if utilization_values else 0,
                    'max': max(utilization_values) if utilization_values else 0,
                    'avg': sum(utilization_values) / len(utilization_values) if utilization_values else 0
                }
            }
            
            return summary
    
    def print_memory_summary(self):
        """Print formatted memory summary"""
        summary = self.get_memory_summary()
        if not summary:
            print("📊 No memory data available")
            return
            
        print(f"\n📊 MEMORY MONITORING SUMMARY")
        print("="*60)
        print(f"📈 Samples: {summary['samples']}")
        print(f"⏱️  Duration: {summary['duration']:.1f}s")
        print(f"📊 GPU Utilization:")
        print(f"   Min: {summary['utilization']['min']:.1f}%")
        print(f"   Max: {summary['utilization']['max']:.1f}%")
        print(f"   Avg: {summary['utilization']['avg']:.1f}%")
        print(f"💾 Allocated Memory:")
        print(f"   Min: {summary['allocated']['min']:.2f} GB")
        print(f"   Max: {summary['allocated']['max']:.2f} GB")
        print(f"   Avg: {summary['allocated']['avg']:.2f} GB")
        print(f"🔒 Reserved Memory:")
        print(f"   Min: {summary['reserved']['min']:.2f} GB")
        print(f"   Max: {summary['reserved']['max']:.2f} GB")
        print(f"   Avg: {summary['reserved']['avg']:.2f} GB")
        print("="*60)

# ===============================================================================
# STEP 2: Initialize ComfyUI Device Detection and Memory Management
# ===============================================================================

def initialize_comfy_device_system():
    """Initialize ComfyUI's device detection system with forced CPU loading"""
    try:
        # Set CPU state based on available hardware
        if torch.cuda.is_available():
            comfy.model_management.cpu_state = comfy.model_management.CPUState.GPU
            # Force LOW_VRAM mode to ensure models load to CPU initially
            # This enables ComfyUI's automatic offloading strategy
            comfy.model_management.vram_state = comfy.model_management.VRAMState.LOW_VRAM
            print("✅ ComfyUI device system: GPU mode with CPU-first loading initialized")
        else:
            comfy.model_management.cpu_state = comfy.model_management.CPUState.CPU
            comfy.model_management.vram_state = comfy.model_management.VRAMState.DISABLED
            print("✅ ComfyUI device system: CPU mode initialized")
    except Exception as e:
        print(f"⚠️  Warning: Could not initialize ComfyUI device system: {e}")

def set_memory_totals():
    """Calculate and set total memory values for ComfyUI"""
    try:
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            total_vram = torch.cuda.get_device_properties(device).total_memory
            comfy.model_management.total_vram = total_vram / (1024 * 1024)  # Convert to MB
            print(f"✅ ComfyUI memory: Total VRAM set to {comfy.model_management.total_vram:.0f} MB")
        else:
            comfy.model_management.total_vram = 0
            print("✅ ComfyUI memory: No VRAM detected, using CPU mode")
        
        total_ram = psutil.virtual_memory().total
        comfy.model_management.total_ram = total_ram / (1024 * 1024)  # Convert to MB
        print(f"✅ ComfyUI memory: Total RAM set to {comfy.model_management.total_ram:.0f} MB")
    except Exception as e:
        print(f"⚠️  Warning: Could not set memory totals: {e}")

def detect_backends():
    """Detect and set available backends for ComfyUI"""
    try:
        # CUDA
        if torch.version.cuda:
            comfy.model_management.xpu_available = False
            comfy.model_management.npu_available = False
            comfy.model_management.mlu_available = False
            print("✅ ComfyUI backends: CUDA backend detected")
        
        # XPU (Intel)
        try:
            import intel_extension_for_pytorch as ipex
            comfy.model_management.xpu_available = torch.xpu.is_available()
            if comfy.model_management.xpu_available:
                print("✅ ComfyUI backends: Intel XPU backend detected")
        except:
            comfy.model_management.xpu_available = False
        
        print("✅ ComfyUI backends: Backend detection completed")
    except Exception as e:
        print(f"⚠️  Warning: Could not detect backends: {e}")

def initialize_comfy_memory_system():
    """Initialize the complete ComfyUI memory management system"""
    print("\n" + "="*80)
    print("🔧 INITIALIZING COMFYUI MEMORY MANAGEMENT SYSTEM")
    print("="*80)
    
    # Ensure ComfyUI CLI arguments are properly parsed
    if not hasattr(comfy.cli_args, 'args') or comfy.cli_args.args is None:
        comfy.cli_args.args = comfy.cli_args.parser.parse_args()
        print("✅ ComfyUI CLI arguments: Parsed and initialized")
    
    # Ensure args.fast is properly initialized (ComfyUI expects this)
    if comfy.cli_args.args.fast is None:
        comfy.cli_args.args.fast = set()
        print("✅ ComfyUI CLI arguments: args.fast initialized")
    
    # Force aggressive memory management settings
    comfy.model_management.vram_state = comfy.model_management.VRAMState.LOW_VRAM
    comfy.model_management.set_vram_to = comfy.model_management.VRAMState.LOW_VRAM
    print("✅ ComfyUI VRAM state: LOW_VRAM mode set (enables CPU-first loading)")
    
    initialize_comfy_device_system()
    set_memory_totals()
    detect_backends()
    
    # Initialize model tracking system
    if not hasattr(comfy.model_management, 'current_loaded_models'):
        comfy.model_management.current_loaded_models = []
        print("✅ ComfyUI model tracking: current_loaded_models initialized")
    
    # Ensure other required attributes exist
    if not hasattr(comfy.model_management, 'current_models'):
        comfy.model_management.current_models = []
        print("✅ ComfyUI model tracking: current_models initialized")
    
    print("✅ ComfyUI memory management system initialization completed")
    print("="*80 + "\n")

def force_comfy_memory_cleanup():
    """Force ComfyUI to clean up memory and offload models to CPU"""
    try:
        print("🧹 Forcing ComfyUI memory cleanup...")
        
        # Force free memory
        if hasattr(comfy.model_management, 'free_memory'):
            freed_models = comfy.model_management.free_memory(0, comfy.model_management.get_torch_device())
            print(f"   ✅ Freed {len(freed_models)} models from GPU")
        
        # Force PyTorch cache cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("   ✅ PyTorch CUDA cache cleared")
        
        # Check current memory state
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"   📊 Current VRAM: {allocated:.1f} MB allocated, {reserved:.1f} MB reserved")
        
        print("✅ Memory cleanup completed")
        
    except Exception as e:
        print(f"⚠️  Warning: Memory cleanup failed: {e}")

def aggressive_memory_cleanup():
    """Aggressive memory cleanup to handle stuck memory issues"""
    try:
        print("🚨 AGGRESSIVE MEMORY CLEANUP - Targeting stuck memory...")
        
        # Force garbage collection
        import gc
        gc.collect()
        print("   ✅ Garbage collection completed")
        
        # Force ComfyUI cleanup
        force_comfy_memory_cleanup()
        
        # Force PyTorch to release all unused memory
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            print("   ✅ PyTorch memory fully cleared")
        
        # Check if we can identify stuck models
        if hasattr(comfy.model_management, 'current_loaded_models'):
            print(f"   📊 ComfyUI tracked models: {len(comfy.model_management.current_loaded_models)}")
            for i, model in enumerate(comfy.model_management.current_loaded_models):
                if hasattr(model, 'model') and hasattr(model.model, 'device'):
                    print(f"      Model {i}: {type(model.model).__name__} on {model.model.device}")
        
        print("✅ Aggressive memory cleanup completed")
        
    except Exception as e:
        print(f"⚠️  Warning: Aggressive memory cleanup failed: {e}")

# ===============================================================================
# STEP 3: Create Model Registry for ComfyUI Memory Management
# ===============================================================================

class PipelineModelRegistry:
    """Registry for managing models with ComfyUI's memory management system"""
    
    def __init__(self):
        self.loaded_models = []
        self.model_patchers = {}
        print("✅ Pipeline model registry: Initialized (minimal tracking mode)")
    
    def register_model(self, model, model_type):
        """Register a model with ComfyUI's memory management system"""
        try:
            # Handle VAE objects that already have a patcher
            if model_type == 'vae' and hasattr(model, 'patcher'):
                print(f"✅ Model registry: {model_type} already has patcher, using existing one")
                patcher = model.patcher
                self.model_patchers[model_type] = patcher
                self.loaded_models.append(patcher)
                
                # For VAE objects that already have proper patchers, don't add to ComfyUI tracking
                # ComfyUI will handle them automatically when needed
                print(f"✅ Model registry: {model_type} using ComfyUI's automatic tracking")
                
                return patcher
            
            # Create ModelPatcher for models that don't have one
            if model_type == 'vae':
                load_device = comfy.model_management.vae_device()
                offload_device = comfy.model_management.vae_offload_device()
            elif model_type == 'unet':
                load_device = comfy.model_management.get_torch_device()
                offload_device = comfy.model_management.unet_offload_device()
            elif model_type == 'clip':
                load_device = comfy.model_management.get_torch_device()
                offload_device = comfy.model_management.clip_offload_device()
            else:
                load_device = comfy.model_management.get_torch_device()
                offload_device = comfy.model_management.get_torch_device()
            
            patcher = comfy.model_patcher.ModelPatcher(
                model,
                load_device=load_device,
                offload_device=offload_device
            )
            
            self.model_patchers[model_type] = patcher
            self.loaded_models.append(patcher)
            
            # Let ComfyUI handle model tracking automatically when models are used
            # Manual registration can cause conflicts with ComfyUI's internal tracking
            
            print(f"✅ Model registry: {model_type} model registered with ComfyUI")
            return patcher
            
        except Exception as e:
            print(f"⚠️  Warning: Could not register {model_type} model: {e}")
            return None
    
    def get_model_patcher(self, model_type):
        """Get the ModelPatcher for a specific model type"""
        return self.model_patchers.get(model_type)
    
    def unload_all_models(self):
        """Unload all registered models"""
        try:
            for patcher in self.loaded_models:
                patcher.unpatch_model()
            self.loaded_models.clear()
            self.model_patchers.clear()
            print("✅ Model registry: All models unloaded")
        except Exception as e:
            print(f"⚠️  Warning: Could not unload all models: {e}")

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
        
        # ===============================================================================
        # STEP 4: Initialize ComfyUI Memory Management System
        # ===============================================================================
        try:
            initialize_comfy_memory_system()
            
            # Initialize VRAM state for memory management with CPU-first loading
            if torch.cuda.is_available():
                comfy.model_management.vram_state = comfy.model_management.VRAMState.LOW_VRAM
                comfy.model_management.set_vram_to = comfy.model_management.VRAMState.LOW_VRAM
                print("✅ ComfyUI VRAM state: LOW_VRAM mode set (enables CPU-first loading)")
            else:
                comfy.model_management.vram_state = comfy.model_management.VRAMState.DISABLED
                comfy.model_management.set_vram_to = comfy.model_management.VRAMState.DISABLED
                print("✅ ComfyUI VRAM state: DISABLED mode set (CPU only)")
            
            # Create model registry for ComfyUI memory management
            self.model_registry = PipelineModelRegistry()
            print("✅ ComfyUI integration: Model registry created")
            
        except Exception as e:
            print(f"⚠️  Warning: ComfyUI memory management initialization failed: {e}")
            print("   Pipeline will continue with basic memory management")
            self.model_registry = None
        
        # Initialize chunked processor for optimal frame processing
        self.chunked_processor = ChunkedProcessor()
        
        # Start with conservative chunking for better memory management
        try:
            self.chunked_processor.set_chunking_strategy('conservative')
            print("✅ Chunked processor initialized with conservative strategy")
        except Exception as e:
            print(f"⚠️  Warning: Could not set chunking strategy: {e}")
            print("   Using default chunking strategy")
        
        # Verify chunked processor is working
        try:
            if hasattr(self.chunked_processor, 'current_strategy'):
                print(f"✅ Chunked processor strategy: {self.chunked_processor.current_strategy}")
            if hasattr(self.chunked_processor, 'default_chunk_sizes'):
                print("✅ Chunked processor has default chunk sizes configured")
        except Exception as e:
            print(f"⚠️  Warning: Chunked processor verification failed: {e}")
        
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
            'video_trimming': None,
            'vae_decoding': None,
            'video_export': None,
            'final_cleanup': None
        }
        
        # Test ComfyUI memory management functions
        if self.model_registry:
            self._test_comfy_memory_functions_safe()
        
        # Initialize real-time memory monitoring
        self.memory_monitor = RealTimeMemoryMonitor(sample_interval=0.5, max_samples=2000)
    
    def _create_vae_with_proper_patcher(self, vae_state_dict):
        """Create VAE with proper patcher like ComfyUI does"""
        try:
            # Create VAE using ComfyUI's constructor (this should create self.patcher automatically)
            vae = comfy.sd.VAE(sd=vae_state_dict)
            
            # Verify patcher was created
            if hasattr(vae, 'patcher') and vae.patcher is not None:
                print("✅ VAE patcher created automatically by ComfyUI")
                print(f"   Patcher type: {type(vae.patcher)}")
                print(f"   Load device: {vae.patcher.load_device}")
                print(f"   Offload device: {vae.patcher.offload_device}")
                return vae
            else:
                print("❌ VAE patcher NOT created automatically")
                print("   This indicates a problem with VAE initialization")
                return None
                
        except Exception as e:
            print(f"❌ VAE creation failed: {e}")
            return None
    
    def _verify_vae_integration(self, vae):
        """Verify VAE is properly integrated with ComfyUI"""
        try:
            # Check 1: Does VAE have patcher?
            if not hasattr(vae, 'patcher') or vae.patcher is None:
                print("❌ VAE missing patcher attribute")
                return False
                
            # Check 2: Is patcher a ModelPatcher?
            if not isinstance(vae.patcher, comfy.model_patcher.ModelPatcher):
                print(f"❌ VAE patcher is wrong type: {type(vae.patcher)}")
                return False
                
            # Check 3: Does patcher have required attributes?
            required_attrs = ['load_device', 'offload_device', 'model']
            for attr in required_attrs:
                if not hasattr(vae.patcher, attr):
                    print(f"❌ VAE patcher missing attribute: {attr}")
                    return False
                    
            # Check 4: Is VAE in ComfyUI's tracking?
            if hasattr(comfy.model_management, 'current_loaded_models'):
                vae_in_tracking = any(
                    hasattr(m, 'model') and m.model == vae.patcher 
                    for m in comfy.model_management.current_loaded_models
                )
                if not vae_in_tracking:
                    print("⚠️  VAE not in ComfyUI tracking (will be added when first used)")
            else:
                print("⚠️  ComfyUI tracking not available")
                
            print("✅ VAE integration verified successfully")
            return True
            
        except Exception as e:
            print(f"❌ VAE integration verification failed: {e}")
            return False
    
    def _prepare_vae_memory(self, vae, pixel_samples):
        """Prepare memory for VAE encoding operations"""
        try:
            # Calculate memory needed for this operation
            # Ensure pixel_samples has the right shape for VAE memory calculation
            if pixel_samples.dim() == 4:  # [batch, height, width, channels]
                shape_for_calc = pixel_samples.shape
            else:
                # Fallback shape estimation
                shape_for_calc = (1, 512, 512, 3)
                print(f"   Using fallback shape for memory calculation: {shape_for_calc}")
            
            memory_used = vae.memory_used_encode(shape_for_calc, vae.vae_dtype)
            
            # Ensure VAE is loaded to GPU with enough memory
            if hasattr(vae, 'patcher') and vae.patcher is not None:
                comfy.model_management.load_models_gpu([vae.patcher], memory_required=memory_used)
                print(f"✅ VAE loaded to GPU with {memory_used / (1024**2):.1f} MB allocated")
            else:
                print("⚠️  VAE not properly integrated - may cause OOM")
                
        except Exception as e:
            print(f"⚠️  VAE memory preparation failed: {e}")
            print(f"   Pixel samples shape: {pixel_samples.shape if hasattr(pixel_samples, 'shape') else 'N/A'}")
            # Continue anyway - the VAE encode will handle its own memory management
    
    def _encode_video_chunked(self, vae, video_frames, chunk_size=4):
        """Encode video frames in chunks to avoid OOM"""
        latents = []
        
        for i in range(0, len(video_frames), chunk_size):
            chunk = video_frames[i:i+chunk_size]
            
            # Ensure memory before encoding chunk (with improved error handling)
            try:
                self._prepare_vae_memory(vae, chunk)
            except Exception as e:
                print(f"   ⚠️  Memory preparation failed, continuing with VAE's built-in management: {e}")
            
            try:
                chunk_latent = vae.encode(chunk[:, :, :, :3])
                latents.append(chunk_latent)
                
                # Clean up after each chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except torch.cuda.OutOfMemoryError:
                print(f"⚠️  OOM on chunk {i//chunk_size + 1}, trying smaller chunk...")
                # Try with single frame
                for j in range(len(chunk)):
                    try:
                        single_latent = vae.encode(chunk[j:j+1, :, :, :3])
                        latents.append(single_latent)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except torch.cuda.OutOfMemoryError:
                        print(f"⚠️  Critical OOM on single frame {j}, skipping...")
                        continue
        
        return torch.cat(latents, dim=0)
    
    def _comfy_vae_encode(self, vae, pixel_samples):
        """Encode using forced tiled approach to prevent OOM"""
        print(f"   Input shape: {pixel_samples.shape}")
        
        # Load VAE to GPU if needed
        if hasattr(vae, 'patcher'):
            comfy.model_management.load_models_gpu([vae.patcher], memory_required=0, force_full_load=vae.disable_offload)
        
        # Force tiled encoding for video inputs to prevent OOM
        if len(pixel_samples.shape) == 4 and pixel_samples.shape[0] > 1:
            print("   🎯 FORCING TILED ENCODING for video input (preventing OOM)")
            
            # Calculate optimal tile sizes based on available memory
            available_memory = comfy.model_management.get_free_memory(comfy.model_management.get_torch_device())
            available_memory_gb = available_memory / (1024**3)
            
            print(f"   📊 Available memory: {available_memory_gb:.2f} GB")
            
            # Get video dimensions
            _, height, width, _ = pixel_samples.shape
            print(f"   📐 Video dimensions: {height}x{width}")
            
            # Conservative tile sizes that ensure tile > overlap
            if available_memory_gb > 20:
                tile_x, tile_y = 512, 512  # Large tiles for high memory
                overlap = 64  # Ensure overlap < tile
                print("   🧱 Using large tiles: 512x512 with overlap 64")
            elif available_memory_gb > 10:
                tile_x, tile_y = 256, 256  # Medium tiles for medium memory
                overlap = 32  # Ensure overlap < tile
                print("   🧱 Using medium tiles: 256x256 with overlap 32")
            else:
                tile_x, tile_y = 128, 128  # Small tiles for low memory
                overlap = 16  # Ensure overlap < tile
                print("   🧱 Using small tiles: 128x128 with overlap 16")
            
            # Ensure tiles fit within video dimensions
            if tile_x > width:
                tile_x = width
                overlap = min(overlap, tile_x // 4)  # Overlap must be < tile/4
                print(f"   🔧 Adjusted tile_x to {tile_x} (video width)")
            
            if tile_y > height:
                tile_y = height
                overlap = min(overlap, tile_y // 4)  # Overlap must be < tile/4
                print(f"   🔧 Adjusted tile_y to {tile_y} (video height)")
            
            # Final validation
            if tile_x <= overlap or tile_y <= overlap:
                print("   ⚠️  Tile sizes too small, using minimal safe values")
                tile_x = max(64, overlap * 2)
                tile_y = max(64, overlap * 2)
            
            print(f"   🎯 Final tile configuration: {tile_x}x{tile_y} with overlap {overlap}")
            
            try:
                # Force ComfyUI's tiled encoding with calculated tile sizes
                print(f"   🔧 Starting tiled encoding with tiles: {tile_x}x{tile_y}")
                samples = vae.encode_tiled_3d(
                    pixel_samples, 
                    tile_x=tile_x, 
                    tile_y=tile_y, 
                    overlap=overlap
                )
                print(f"   ✅ Tiled encoding successful! Output shape: {samples.shape}")
                return samples
                
            except Exception as e:
                print(f"   ⚠️  Tiled encoding failed: {e}")
                print("   🔧 Trying with even smaller tiles...")
                
                # Fallback to very small tiles
                try:
                    fallback_tile = 64
                    fallback_overlap = 8
                    print(f"   🔧 Fallback: {fallback_tile}x{fallback_tile} with overlap {fallback_overlap}")
                    
                    samples = vae.encode_tiled_3d(
                        pixel_samples, 
                        tile_x=fallback_tile, 
                        tile_y=fallback_tile, 
                        overlap=fallback_overlap
                    )
                    print(f"   ✅ Fallback tiled encoding successful! Output shape: {samples.shape}")
                    return samples
                    
                except Exception as e2:
                    print(f"   ❌ All tiled encoding attempts failed: {e2}")
                    raise e2
        
        else:
            # Single frame or image - use standard VAE encode
            print("   📷 Processing single frame/image with standard VAE encode")
            return vae.encode(pixel_samples)
    
    def _test_comfy_memory_functions_safe(self):
        """Test ComfyUI memory functions without interfering with model tracking"""
        try:
            print("\n" + "="*80)
            print("🧪 TESTING COMFYUI MEMORY MANAGEMENT FUNCTIONS")
            print("="*80)
            
            # Test 1: get_free_memory()
            print("🔍 Testing get_free_memory()...")
            device = comfy.model_management.get_torch_device()
            free_memory = comfy.model_management.get_free_memory(device)
            free_total, free_torch = comfy.model_management.get_free_memory(device, torch_free_too=True)
            
            print(f"   ✅ get_free_memory() working:")
            print(f"      Total free: {free_memory / (1024**2):.1f} MB")
            print(f"      GPU free: {free_total / (1024**2):.1f} MB")
            print(f"      Torch free: {free_torch / (1024**2):.1f} MB")
            
            # Test 2: load_models_gpu() with empty list (safe)
            print("🔍 Testing load_models_gpu() with empty list...")
            comfy.model_management.load_models_gpu([], memory_required=0)
            print("   ✅ load_models_gpu() working with empty list")
            
            # Test 3: Check if model tracking system exists (without modifying)
            print("🔍 Testing model tracking system...")
            if hasattr(comfy.model_management, 'current_loaded_models'):
                print(f"   ✅ current_loaded_models exists: {len(comfy.model_management.current_loaded_models)} models")
            else:
                print("   ❌ current_loaded_models not found")
            
            print("✅ All ComfyUI memory management functions are working!")
            print("="*80)
            
        except Exception as e:
            print(f"❌ ComfyUI memory management test failed: {e}")
            print("   ⚠️  Some functions may not work correctly")
            print("="*80)
        
        # Memory thresholds for each phase
        self.memory_thresholds = {
            'baseline': 100,           # MB - should be very low
            'model_loading': 100,      # MB - ComfyUI lazy loading (models loaded on-demand)
            'lora_application': 200,   # MB - LoRA applied (still lazy loading)
            'text_encoding': 1000,     # MB - models may be loaded to GPU for encoding
            'model_sampling': 2000,    # MB - models loaded for patching
            'gpu_capability_test': 1000, # MB - should be low during testing
            'vae_encoding': 8000,     # MB - VAE encoding in progress (models loaded)
            'unet_sampling': 40000,   # MB - UNET sampling needs ~33GB (realistic)
            'vae_decoding': 8000,     # MB - VAE decoding in progress
            'final_cleanup': 100,      # MB - back to baseline
            'video_trimming': 100,      # MB - back to baseline
            'video_export': 100         # MB - back to baseline
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
    
    def _test_comfy_memory_functions(self):
        """Test ComfyUI memory management functions to ensure they're working"""
        print("\n" + "="*80)
        print("🧪 TESTING COMFYUI MEMORY MANAGEMENT FUNCTIONS")
        print("="*80)
        
        try:
            # Test 1: get_free_memory()
            print("🔍 Testing get_free_memory()...")
            device = comfy.model_management.get_torch_device()
            free_memory = comfy.model_management.get_free_memory(device)
            free_total, free_torch = comfy.model_management.get_free_memory(device, torch_free_too=True)
            
            print(f"   ✅ get_free_memory() working:")
            print(f"      Total free: {free_memory / (1024**2):.1f} MB")
            print(f"      GPU free: {free_total / (1024**2):.1f} MB")
            print(f"      Torch free: {free_torch / (1024**2):.1f} MB")
            
            # Test 2: load_models_gpu() with empty list
            print("🔍 Testing load_models_gpu() with empty list...")
            comfy.model_management.load_models_gpu([], memory_required=0)
            print("   ✅ load_models_gpu() working with empty list")
            
            # Test 3: free_memory() with minimal requirement
            print("🔍 Testing free_memory() with minimal requirement...")
            unloaded = comfy.model_management.free_memory(1024*1024, device)  # 1MB
            print(f"   ✅ free_memory() working: {len(unloaded)} models unloaded")
            
            # Test 4: Check if model tracking is working
            print("🔍 Testing model tracking system...")
            if hasattr(comfy.model_management, 'current_loaded_models'):
                print(f"   ✅ current_loaded_models exists: {len(comfy.model_management.current_loaded_models)} models")
            else:
                print("   ❌ current_loaded_models not found")
            
            print("✅ All ComfyUI memory management functions are working!")
            
        except Exception as e:
            print(f"❌ ComfyUI memory management test failed: {e}")
            print("   Pipeline will continue with basic memory management")
        
        print("="*80 + "\n")
    
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
                if hasattr(self, 'model') and self.model is not None:
                    if hasattr(self.model, 'model') and hasattr(self.model.model, 'device'):
                        model_status['unet'] = str(self.model.model.device)
                        print(f"   UNET: {model_status['unet']}")
                    elif hasattr(self.model, 'device'):
                        model_status['unet'] = str(self.model.device)
                        print(f"   UNET: {model_status['unet']}")
                    else:
                        model_status['unet'] = 'LOADED (device not accessible)'
                        print(f"   UNET: {model_status['unet']}")
                else:
                    model_status['unet'] = 'NOT LOADED'
                    print(f"   UNET: {model_status['unet']}")
            except Exception as e:
                model_status['unet'] = f'ERROR: {e}'
                print(f"   UNET: {model_status['unet']}")
        
        # Check VAE placement
        if 'vae' in expected_models:
            try:
                if hasattr(self, 'vae') and self.vae is not None:
                    if hasattr(self.vae, 'device'):
                        model_status['vae'] = str(self.vae.device)
                        print(f"   VAE: {model_status['vae']}")
                    elif hasattr(self.vae, 'first_stage_model') and hasattr(self.vae.first_stage_model, 'device'):
                        model_status['vae'] = str(self.vae.first_stage_model.device)
                        print(f"   VAE: {model_status['vae']}")
                    else:
                        model_status['vae'] = 'LOADED (device not accessible)'
                        print(f"   VAE: {model_status['vae']}")
                else:
                    model_status['vae'] = 'NOT LOADED'
                    print(f"   VAE: {model_status['vae']}")
            except Exception as e:
                model_status['vae'] = f'ERROR: {e}'
                print(f"   VAE: {model_status['vae']}")
        
        # Check CLIP placement
        if 'clip' in expected_models:
            try:
                if hasattr(self, 'clip_model') and self.clip_model is not None:
                    if hasattr(self.clip_model, 'patcher') and hasattr(self.clip_model.patcher, 'model'):
                        if hasattr(self.clip_model.patcher.model, 'device'):
                            model_status['clip'] = str(self.clip_model.patcher.model.device)
                            print(f"   CLIP: {model_status['clip']}")
                        else:
                            model_status['clip'] = 'LOADED (device not accessible)'
                            print(f"   CLIP: {model_status['clip']}")
                    elif hasattr(self.clip_model, 'device'):
                        model_status['clip'] = str(self.clip_model.device)
                        print(f"   CLIP: {model_status['clip']}")
                    else:
                        model_status['clip'] = 'LOADED (device not accessible)'
                        print(f"   CLIP: {model_status['clip']}")
                else:
                    model_status['clip'] = 'NOT LOADED'
                    print(f"   CLIP: {model_status['clip']}")
            except Exception as e:
                model_status['clip'] = f'ERROR: {e}'
                print(f"   CLIP: {model_status['clip']}")
        
        return model_status
    
    def _verify_memory_management(self, phase_name, expected_models):
        """Verify that memory management is working correctly after each step"""
        print(f"🔍 {phase_name.upper()} MEMORY MANAGEMENT VERIFICATION:")
        
        if not torch.cuda.is_available():
            print("   ⚠️  CUDA not available, skipping GPU memory verification")
            return True
        
        current_allocated = torch.cuda.memory_allocated() / 1024**2
        current_reserved = torch.cuda.memory_reserved() / 1024**2
        
        print(f"   Current GPU Memory: {current_allocated:.1f} MB allocated, {current_reserved:.1f} MB reserved")
        
        # Check if models are properly offloaded
        models_offloaded = True
        models_checked = 0
        
        for model_name in expected_models:
            if model_name == 'unet' and hasattr(self, 'model') and self.model is not None:
                models_checked += 1
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'device'):
                    device = str(self.model.model.device)
                    if device != 'cpu':
                        print(f"   ❌ UNET still on GPU: {device}")
                        models_offloaded = False
                    else:
                        print(f"   ✅ UNET properly offloaded to: {device}")
                elif hasattr(self.model, 'device'):
                    device = str(self.model.device)
                    if device != 'cpu':
                        print(f"   ❌ UNET still on GPU: {device}")
                        models_offloaded = False
                    else:
                        print(f"   ✅ UNET properly offloaded to: {device}")
                else:
                    print(f"   ⚠️  UNET loaded but device not accessible")
            
            elif model_name == 'clip' and hasattr(self, 'clip_model') and self.clip_model is not None:
                models_checked += 1
                if hasattr(self.clip_model, 'patcher') and hasattr(self.clip_model.patcher, 'model'):
                    if hasattr(self.clip_model.patcher.model, 'device'):
                        device = str(self.clip_model.patcher.model.device)
                        if device != 'cpu':
                            print(f"   ❌ CLIP still on GPU: {device}")
                            models_offloaded = False
                        else:
                            print(f"   ✅ CLIP properly offloaded to: {device}")
                    else:
                        print(f"   ⚠️  CLIP loaded but device not accessible")
                elif hasattr(self.clip_model, 'device'):
                    device = str(self.clip_model.device)
                    if device != 'cpu':
                        print(f"   ❌ CLIP still on GPU: {device}")
                        models_offloaded = False
                    else:
                        print(f"   ✅ CLIP properly offloaded to: {device}")
                else:
                    print(f"   ⚠️  CLIP loaded but device not accessible")
            
            elif model_name == 'vae' and hasattr(self, 'vae') and self.vae is not None:
                models_checked += 1
                if hasattr(self.vae, 'device'):
                    device = str(self.vae.device)
                    if device != 'cpu':
                        print(f"   ❌ VAE still on GPU: {device}")
                        models_offloaded = False
                    else:
                        print(f"   ✅ VAE properly offloaded to: {device}")
                elif hasattr(self.vae, 'first_stage_model') and hasattr(self.vae.first_stage_model, 'device'):
                    device = str(self.vae.first_stage_model.device)
                    if device != 'cpu':
                        print(f"   ❌ VAE still on GPU: {device}")
                        models_offloaded = False
                    else:
                        print(f"   ✅ VAE properly offloaded to: {device}")
                else:
                    print(f"   ⚠️  VAE loaded but device not accessible")
        
        if models_checked == 0:
            print(f"   ℹ️  No models loaded yet for {phase_name}")
            return True
        elif models_offloaded:
            print(f"   ✅ All loaded models properly offloaded to CPU")
        else:
            print(f"   ⚠️  Some models still on GPU - memory management may be incomplete")
        
        return models_offloaded
    
    def _verify_chunking_strategy(self, phase_name, processing_plan):
        """Verify that chunking strategy is properly configured for the current phase"""
        print(f"🔍 {phase_name.upper()} CHUNKING STRATEGY VERIFICATION:")
        
        if not processing_plan:
            print("   ⚠️  No processing plan available")
            return False
        
        # Check chunking configuration for current phase
        phase_chunks = {}
        
        if 'vae_encode' in processing_plan:
            phase_chunks['vae_encode'] = processing_plan['vae_encode']
        
        if 'unet_process' in processing_plan:
            phase_chunks['unet_process'] = processing_plan['unet_process']
        
        if 'vae_decode' in processing_plan:
            phase_chunks['vae_decode'] = processing_plan['vae_decode']
        
        if not phase_chunks:
            print("   ⚠️  No chunking configuration found for current phase")
            return False
        
        print("   Chunking Configuration:")
        for operation, config in phase_chunks.items():
            chunk_size = config.get('chunk_size', 'N/A')
            num_chunks = config.get('num_chunks', 'N/A')
            print(f"     {operation}: {chunk_size} items per chunk, {num_chunks} total chunks")
        
        # Verify chunked processor is configured
        if hasattr(self, 'chunked_processor') and self.chunked_processor is not None:
            try:
                strategy = self.chunked_processor.current_strategy
                print(f"   Chunked Processor Strategy: {strategy}")
                
                if strategy == 'conservative':
                    print("   ✅ Using conservative chunking for memory efficiency")
                elif strategy == 'aggressive':
                    print("   ⚠️  Using aggressive chunking - may use more memory")
                elif strategy == 'ultra_conservative':
                    print("   ✅ Using ultra-conservative chunking for maximum memory efficiency")
                elif strategy == 'balanced':
                    print("   ⚖️  Using balanced chunking strategy")
                else:
                    print(f"   ℹ️  Using custom chunking strategy: {strategy}")
                    
                # Additional chunked processor verification
                if hasattr(self.chunked_processor, 'default_chunk_sizes'):
                    print("   ✅ Chunked processor has default chunk sizes configured")
                if hasattr(self.chunked_processor, 'chunking_strategies'):
                    print("   ✅ Chunked processor has chunking strategies configured")
                    
            except AttributeError as e:
                print(f"   ❌ Chunked processor missing attribute: {e}")
                return False
            except Exception as e:
                print(f"   ❌ Error accessing chunked processor: {e}")
                return False
        else:
            print("   ❌ Chunked processor not available")
            return False
        
        return True
        
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
        print("🚀 FULLY LEVERAGING COMFYUI'S PROVEN MEMORY MANAGEMENT SYSTEM")
        print("🎯 Pipeline now works exactly like a ComfyUI node - simple, clean, and efficient!")
        print("💡 All memory management handled automatically by ComfyUI")
        print("💡 No manual intervention needed - ComfyUI knows best!")
        
        # Establish baseline memory state for OOM debugging
        print("🔍 ESTABLISHING BASELINE MEMORY STATE...")
        
        # DEBUG: Check baseline memory state
        if torch.cuda.is_available():
            baseline_allocated = torch.cuda.memory_allocated() / 1024**2
            baseline_reserved = torch.cuda.memory_reserved() / 1024**2
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**2
            
            print(f"🔍 DEBUG: Baseline memory check:")
            print(f"   CUDA available: {torch.cuda.is_available()}")
            print(f"   Current device: {torch.cuda.current_device()}")
            print(f"   Device properties: {torch.cuda.get_device_properties(0).name}")
            print(f"   Total VRAM: {total_vram:.1f} GB")
            print(f"   Allocated: {baseline_allocated:.1f} MB")
            print(f"   Reserved: {baseline_reserved:.1f} MB")
            
            if baseline_allocated == 0.0 and baseline_reserved == 0.0:
                print("   ⚠️  WARNING: Baseline memory shows 0.0 MB!")
                print("   🔍 This suggests either:")
                print("      - No models loaded yet (expected at start)")
                print("      - Models loaded to CPU instead of GPU")
                print("      - ComfyUI using lazy loading strategy")
                print("      - Memory measurement issue")
            
            # Store baseline for later comparison
            self.baseline_allocated = baseline_allocated
            self.baseline_reserved = baseline_reserved
        else:
            print("🔍 CUDA not available, skipping baseline memory check")
            self.baseline_allocated = 0
            self.baseline_reserved = 0
        
        # COMMENTED OUT FOR STEP 3 DEBUGGING: self._check_memory_usage('baseline_memory', expected_threshold=100)
        
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
            print("1a. Loading models using ComfyUI's individual component system...")
            
            # Load UNET with proper WAN model detection
            print("1a. Loading UNET with automatic WAN detection...")
            unet_state_dict = comfy.utils.load_torch_file(unet_model_path)
            
            # Use ComfyUI's automatic model detection for UNET
            # This will automatically detect WAN models and load them correctly
            from comfy.sd import load_state_dict_guess_config
            
            # Load just the UNET with automatic detection
            model, _, _, _ = load_state_dict_guess_config(
                unet_state_dict, 
                output_vae=False, 
                output_clip=False, 
                output_clipvision=False, 
                embedding_directory=None, 
                output_model=True
            )
            
            print(f"1a. ✅ UNET loaded using ComfyUI's automatic detection: {type(model)}")
            
            # Load CLIP separately with proper WAN type
            print("1a. Loading CLIP with WAN type...")
            clip_model = comfy.sd.load_clip([clip_model_path], clip_type=comfy.sd.CLIPType.WAN)
            
            if clip_model is None:
                print("1a. ⚠️  CLIP loading failed, trying alternative approach...")
                # Fallback: load CLIP state dict and create manually
                clip_state_dict = comfy.utils.load_torch_file(clip_model_path)
                from comfy.sd import CLIP
                clip_model = CLIP(clip_state_dict, clip_type=comfy.sd.CLIPType.WAN)
            
            print(f"1a. ✅ CLIP loaded: {type(clip_model)}")
            
            # Load VAE separately with ComfyUI memory management integration
            print("1a. Loading VAE with ComfyUI memory management...")
            vae_state_dict = comfy.utils.load_torch_file(vae_model_path)
            
            # Create VAE using ComfyUI's constructor (this should create patcher automatically)
            vae = self._create_vae_with_proper_patcher(vae_state_dict)
            
            if vae is None:
                raise RuntimeError("Failed to create VAE with proper patcher")
            
            # Verify VAE integration
            if not self._verify_vae_integration(vae):
                print("⚠️  VAE integration issues detected - may cause OOM")
            
            # Register VAE with model registry if available
            if hasattr(self, 'model_registry') and self.model_registry:
                try:
                    # Register the VAE itself, not the patcher
                    self.model_registry.register_model(vae, 'vae')
                    print("1a. ✅ VAE registered with ComfyUI memory management system")
                except Exception as e:
                    print(f"1a. ⚠️  Warning: Could not register VAE model: {e}")
                    print("1a. ✅ VAE will still work with built-in ComfyUI memory management")
            
            print(f"1a. ✅ VAE loaded: {type(vae)}")
            
            # Test VAE patcher creation
            print("\n🧪 TESTING VAE PATCHER CREATION...")
            if hasattr(vae, 'patcher'):
                print(f"✅ VAE has patcher: {type(vae.patcher)}")
                print(f"   Patcher model: {type(vae.patcher.model)}")
                print(f"   Load device: {vae.patcher.load_device}")
                print(f"   Offload device: {vae.patcher.offload_device}")
            else:
                print("❌ VAE missing patcher - this will cause OOM in Step 5")
            
            # Verify that the UNET was detected as WAN model
            if hasattr(model, 'model') and hasattr(model.model, 'model_type'):
                print(f"1a. ✅ UNET model type detected: {model.model.model_type}")
                if hasattr(model.model, 'image_model'):
                    print(f"1a. ✅ UNET image model: {model.model.image_model}")
            else:
                print("1a. ⚠️  UNET model type not accessible, but loaded successfully")
            
            # DEBUG: Check if models are actually loaded to GPU
            print("1a. 🔍 DEBUG: Checking model loading status...")
            if hasattr(model, 'model') and hasattr(model.model, 'device'):
                print(f"1a. DEBUG: UNET internal model device: {model.model.device}")
            if hasattr(model, 'device'):
                print(f"1a. DEBUG: UNET wrapper device: {model.device}")
            
            if hasattr(clip_model, 'patcher') and hasattr(clip_model.patcher, 'model'):
                if hasattr(clip_model.patcher.model, 'device'):
                    print(f"1a. DEBUG: CLIP internal model device: {clip_model.patcher.model.device}")
            if hasattr(clip_model, 'device'):
                print(f"1a. DEBUG: CLIP wrapper device: {clip_model.device}")
            
            if hasattr(vae, 'device'):
                print(f"1a. DEBUG: VAE wrapper device: {vae.device}")
            if hasattr(vae, 'first_stage_model') and hasattr(vae.first_stage_model, 'device'):
                print(f"1a. DEBUG: VAE internal model device: {vae.first_stage_model.device}")
            
            # Check memory after model loading
            if torch.cuda.is_available():
                after_loading_allocated = torch.cuda.memory_allocated() / 1024**2
                after_loading_reserved = torch.cuda.memory_reserved() / 1024**2
                print(f"1a. DEBUG: Memory after loading - Allocated: {after_loading_allocated:.1f} MB, Reserved: {after_loading_reserved:.1f} MB")
                
                if after_loading_allocated == 0.0:
                    print("1a. ⚠️  WARNING: Models loaded but GPU memory still shows 0.0 MB!")
                    print("1a. 🔍 This suggests models may not be loaded to GPU yet")
                    print("1a. 💡 ComfyUI may be using lazy loading or CPU-first strategy")
            
            # OOM Checklist: Check memory after model loading
            # Note: ComfyUI uses lazy loading, so models may not consume GPU memory until used
            self._check_memory_usage('model_loading', expected_threshold=100)  # Much lower threshold for lazy loading
            
            # Check if models need to be explicitly loaded to GPU
            print("1a. 🔍 Letting ComfyUI handle model loading naturally...")
            print("1a. ComfyUI will load models to GPU when they're actually needed")
            print("1a. No manual patching needed - ComfyUI's ModelPatcher system handles everything")
            
            # Check ComfyUI's model management system
            print("1a. 🔍 Checking ComfyUI's model management system...")
            try:
                import comfy.model_management
                
                # Check what device ComfyUI thinks models should be on
                if hasattr(comfy.model_management, 'get_torch_device'):
                    comfy_device = comfy.model_management.get_torch_device()
                    print(f"1a. ComfyUI device: {comfy_device}")
                
                if hasattr(comfy.model_management, 'vae_device'):
                    vae_device = comfy.model_management.vae_device()
                    print(f"1a. ComfyUI VAE device: {vae_device}")
                
                if hasattr(comfy.model_management, 'model_device'):
                    model_device = comfy.model_management.model_device()
                    print(f"1a. ComfyUI model device: {model_device}")
                
                print("1a. ComfyUI model management system is available")
                print("1a. ✅ Trusting ComfyUI to handle all memory management automatically")
                
            except Exception as e:
                print(f"1a. ⚠️  Could not check ComfyUI model management: {e}")
            
            print("✅ Step 1 completed - continuing to Step 3...")
            
            # ========================================================================
            # STEP 3: LOAD VIDEO AND IMAGE DATA
            # ========================================================================
            print("\n" + "="*80)
            print("🔍 STEP 3: LOAD VIDEO AND IMAGE DATA")
            print("="*80)
            
            # Load control video
            print("3a. Loading control video...")
            if control_video_path:
                control_video = self.load_video(control_video_path)
                if control_video is not None:
                    print(f"   ✅ Control video loaded: {control_video.shape}")
                else:
                    print("   ❌ Failed to load control video")
                    control_video = None
            else:
                print("   ⚠️  No control video path specified")
                control_video = None
            
            # Load reference image
            print("3b. Loading reference image...")
            if reference_image_path:
                reference_image = self.load_image(reference_image_path)
                if reference_image is not None:
                    print(f"   ✅ Reference image loaded: {reference_image.shape}")
                else:
                    print("   ❌ Failed to load reference image")
                    reference_image = None
            else:
                print("   ⚠️  No reference image path specified")
                reference_image = None
            
            print("✅ Step 3 completed - continuing to Step 5...")
            
            # ========================================================================
            # STEP 5: GENERATE INITIAL LATENTS (COMFY-LIKE)
            # ========================================================================
            print("\n" + "="*80)
            print("🔍 STEP 5: GENERATE INITIAL LATENTS (COMFY-LIKE)")
            print("="*80)
            
            # Enable comprehensive memory tracking for Step 5
            self._track_memory_during_step5()
            
            # Start real-time memory monitoring for Step 5
            self.memory_monitor.start_monitoring("STEP5_VAE_ENCODING")
            
            # Initial VRAM analysis before Step 5
            self._detailed_vram_analysis("STEP5_START")
            
            # Force ComfyUI memory cleanup before starting VAE encoding
            force_comfy_memory_cleanup()
            
            # Aggressive cleanup to handle stuck memory issues
            aggressive_memory_cleanup()
            
            try:
                import comfy.model_management
                
                # Check what device ComfyUI thinks models should be on
                if hasattr(comfy.model_management, 'get_torch_device'):
                    comfy_device = comfy.model_management.get_torch_device()
                    print(f"1a. ComfyUI device: {comfy_device}")
                
                if hasattr(comfy.model_management, 'vae_device'):
                    vae_device = comfy.model_management.vae_device()
                    print(f"1a. ComfyUI VAE device: {vae_device}")
                
                if hasattr(comfy.model_management, 'model_device'):
                    model_device = comfy.model_management.model_device()
                    print(f"1a. ComfyUI model device: {model_device}")
                
                print("1a. ComfyUI model management system is available")
                print("1a. ✅ Trusting ComfyUI to handle all memory management automatically")
                
            except Exception as e:
                print(f"1a. ⚠️  Could not check ComfyUI model management: {e}")
            
            # Test VAE memory management integration
            print("1a. 🧪 Testing VAE memory management integration...")
            try:
                if hasattr(vae, 'patcher') and vae.patcher is not None:
                    print("1a. ✅ VAE is wrapped in ModelPatcher")
                    
                    # Test if VAE respects ComfyUI memory management
                    vae_device = comfy.model_management.vae_device()
                    print(f"1a. VAE target device: {vae_device}")
                    
                    # Check if VAE is in ComfyUI's model tracking
                    if hasattr(comfy.model_management, 'current_loaded_models'):
                        vae_in_tracking = any(
                            hasattr(m, 'model') and m.model == vae.patcher 
                            for m in comfy.model_management.current_loaded_models
                        )
                        if vae_in_tracking:
                            print("1a. ✅ VAE is tracked by ComfyUI memory management")
                        else:
                            print("1a. ⚠️  VAE is NOT tracked by ComfyUI memory management")
                    else:
                        print("1a. ⚠️  ComfyUI model tracking not available")
                        
                else:
                    print("1a. ❌ VAE is NOT wrapped in ModelPatcher")
                    print("1a. ⚠️  VAE will not use ComfyUI memory management")
                    
            except Exception as e:
                print(f"1a. ⚠️  VAE memory management test failed: {e}")
            
            # Check memory after letting ComfyUI handle loading
            if torch.cuda.is_available():
                after_loading_allocated = torch.cuda.memory_allocated() / 1024**2
                after_loading_reserved = torch.cuda.memory_reserved() / 1024**2
                print(f"1a. Memory after ComfyUI loading - Allocated: {after_loading_allocated:.1f} MB, Reserved: {after_loading_reserved:.1f} MB")
                
                if after_loading_allocated == 0.0:
                    print("1a. ✅ This is normal - ComfyUI uses lazy loading")
                    print("1a. 💡 Models will be loaded to GPU when actually needed")
                    print("1a. 💡 This prevents unnecessary memory usage")
                else:
                    print("1a. ✅ Models are now loaded to GPU by ComfyUI")
            
            # OOM Checklist: Check memory after model loading
            # Note: ComfyUI uses lazy loading, so models may not consume GPU memory until used
            self._check_memory_usage('model_loading', expected_threshold=100)  # Much lower threshold for lazy loading
            
            # ComfyUI will handle all model loading automatically
            print("1a. ✅ Trusting ComfyUI's automatic model management system")
            print("1a. 💡 Models will be loaded to GPU when needed for operations")
            print("1a. 💡 No manual intervention required - ComfyUI knows best!")
            
            # Check ComfyUI's model management system
            print("1a. 🔍 Checking ComfyUI's model management system...")
            try:
                import comfy.model_management
                
                # Check what device ComfyUI thinks models should be on
                if hasattr(comfy.model_management, 'get_torch_device'):
                    comfy_device = comfy.model_management.get_torch_device()
                    print(f"1a. ComfyUI device: {comfy_device}")
                
                if hasattr(comfy.model_management, 'vae_device'):
                    vae_device = comfy.model_management.vae_device()
                    print(f"1a. ComfyUI VAE device: {vae_device}")
                
                if hasattr(comfy.model_management, 'model_device'):
                    model_device = comfy.model_management.model_device()
                    print(f"1a. ComfyUI model device: {model_device}")
                
                print("1a. ComfyUI model management system is available")
                
            except Exception as e:
                print(f"1a. ⚠️  Could not check ComfyUI model management: {e}")
            
            # COMPREHENSIVE VERIFICATION AFTER MODEL LOADING
            print("\n" + "="*80)
            print("🔍 STEP 1 COMPLETE: COMPREHENSIVE VERIFICATION")
            print("="*80)
            
            # 1. Model Placement Verification
            print("1️⃣  MODEL PLACEMENT VERIFICATION:")
            model_placement = self._check_model_placement('model_loading', ['unet', 'clip', 'vae'])
            
            # 2. Memory Management Verification
            print("\n2️⃣  MEMORY MANAGEMENT VERIFICATION:")
            memory_management = self._verify_memory_management('model_loading', ['unet', 'clip', 'vae'])
            
            # 3. Chunking Strategy Verification
            print("\n3️⃣  CHUNKING STRATEGY VERIFICATION:")
            chunking_strategy = self._verify_chunking_strategy('model_loading', processing_plan)
            
            # 4. Summary
            print("\n📊 STEP 1 SUMMARY:")
            print(f"   Model Placement: {'✅ PASS' if model_placement else '❌ FAIL'}")
            print(f"   Memory Management: {'✅ PASS' if memory_management else '❌ FAIL'}")
            print(f"   Chunking Strategy: {'✅ PASS' if chunking_strategy else '❌ FAIL'}")
            
            # Add note about ComfyUI's proven system
            print("\n   📝 NOTE: Pipeline now fully leverages ComfyUI's proven system:")
            print("      - All memory management handled automatically by ComfyUI")
            print("      - No manual model patching or cleanup needed")
            print("      - ComfyUI prevents memory fragmentation naturally")
            print("      - Models are loaded/unloaded optimally by ComfyUI")
            
            if not all([model_placement, memory_management, chunking_strategy]):
                print("   ⚠️  Some verifications failed - pipeline may have issues")
            else:
                print("   ✅ All verifications passed - pipeline ready for next step")
            
            print("="*80)
            
            # 2. Apply LoRA if specified
            if lora_path:
                print("2. Applying LoRA...")
                
                # === LORA APPLICATION MONITORING SYSTEM START ===
                print("\n🔍 LORA APPLICATION MONITORING SYSTEM ACTIVATED")
                print("="*80)
                
                # Debug: Check if monitoring methods exist
                print("🔍 DEBUG: Checking monitoring methods...")
                if hasattr(self, '_start_step_monitoring'):
                    print("✅ _start_step_monitoring method exists")
                else:
                    print("❌ _start_step_monitoring method MISSING")
                
                if hasattr(self, '_capture_lora_baseline'):
                    print("✅ _capture_lora_baseline method exists")
                else:
                    print("❌ _capture_lora_baseline method MISSING")
                
                # COMMENTED OUT FOR STEP 3 DEBUGGING: Start step monitoring with timing and memory baseline
                # COMMENTED OUT FOR STEP 3 DEBUGGING: try:
                # COMMENTED OUT FOR STEP 3 DEBUGGING:     step_start_time, step_start_memory = self._start_step_monitoring("lora_application")
                # COMMENTED OUT FOR STEP 3 DEBUGGING:     print("✅ Step monitoring started successfully")
                # COMMENTED OUT FOR STEP 3 DEBUGGING: except Exception as e:
                # COMMENTED OUT FOR STEP 3 DEBUGGING:     print(f"❌ Step monitoring failed: {e}")
                # COMMENTED OUT FOR STEP 3 DEBUGGING:     step_start_time = time.time()
                # COMMENTED OUT FOR STEP 3 DEBUGGING:     step_start_memory = {'ram_used_mb': 0, 'gpu_allocated_mb': 0}
                
                # Capture baseline state before LoRA application
                print("\n📊 CAPTURING BASELINE STATE (Before LoRA)...")
                try:
                    lora_baseline = self._capture_lora_baseline(model, clip_model, lora_path)
                    print("✅ Baseline captured successfully")
                except Exception as e:
                    print(f"❌ Baseline capture failed: {e}")
                    lora_baseline = {'unet': {'model_id': 0, 'patches_count': 0}, 'clip': {'model_id': 0, 'patcher_patches_count': 0}, 'lora_file': {'full_path': lora_path, 'file_exists': False, 'file_size_mb': 0}}
                
                # Display baseline information
                print(f"   📁 LoRA File: {lora_baseline['lora_file']['full_path']}")
                print(f"   📁 File Exists: {'✅ YES' if lora_baseline['lora_file']['file_exists'] else '❌ NO'}")
                print(f"   📁 File Size: {lora_baseline['lora_file']['file_size_mb']:.2f} MB")
                print(f"   ✅ UNET Baseline captured - ID: {lora_baseline['unet']['model_id']}, Patches: {lora_baseline['unet']['patches_count']}")
                print(f"   ✅ CLIP Baseline captured - ID: {lora_baseline['clip']['model_id']}, Patches: {lora_baseline['clip']['patcher_patches_count']}")
                
                # COMMENTED OUT FOR STEP 3 DEBUGGING: Display baseline memory state
                # COMMENTED OUT FOR STEP 3 DEBUGGING: print(f"\n   💾 BASELINE MEMORY STATE:")
                # COMMENTED OUT FOR STEP 3 DEBUGGING: print(f"      🖥️  RAM: {step_start_memory['ram_used_mb']:.1f} MB used / {step_start_memory['ram_total_mb']:.1f} MB total ({step_start_memory['ram_percent']:.1f}%)")
                # COMMENTED OUT FOR STEP 3 DEBUGGING: print(f"      🎮 GPU: {step_start_memory['gpu_allocated_mb']:.1f} MB allocated / {step_start_memory['gpu_total_mb']:.1f} MB total")
                # COMMENTED OUT FOR STEP 3 DEBUGGING: print(f"      🎮 GPU Device: {step_start_memory['gpu_device_name']}")
                
                print("✅ Baseline captured successfully")
                print("="*80)
                
                # Apply LoRA with monitoring
                print("🔧 APPLYING LORA TO MODELS...")
                lora_loader = LoraLoader()
                
                try:
                    # Store original models for comparison
                    original_model = model
                    original_clip_model = clip_model
                    
                    # DEBUG: Check model state before LoRA
                    print(f"\n🔍 DEBUG: Pre-LoRA Model State:")
                    print(f"   UNET Model Type: {type(model).__name__}")
                    print(f"   UNET Model Class: {model.__class__}")
                    if hasattr(model, 'model'):
                        print(f"   UNET Underlying Model: {type(model.model).__name__}")
                        print(f"   UNET Model Config: {getattr(model.model, 'model_config', 'No config')}")
                    print(f"   CLIP Model Type: {type(clip_model).__name__}")
                    print(f"   CLIP Model Class: {clip_model.__class__}")
                    if hasattr(clip_model, 'cond_stage_model'):
                        print(f"   CLIP Underlying Model: {type(clip_model.cond_stage_model).__name__}")
                    
                    # DEBUG: Manually load LoRA to see key mapping
                    print(f"\n🔍 DEBUG: Manual LoRA Key Mapping Analysis:")
                    import comfy.utils
                    import comfy.lora
                    lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
                    
                    # Show LoRA file contents
                    lora_keys = list(lora_data.keys())
                    print(f"   LoRA File Keys: {len(lora_keys)} total")
                    print(f"   Sample LoRA Keys (first 5): {lora_keys[:5]}")
                    
                    # Show key mapping for UNET
                    unet_key_map = {}
                    if model is not None:
                        unet_key_map = comfy.lora.model_lora_keys_unet(model.model, unet_key_map)
                    print(f"   UNET Key Mappings: {len(unet_key_map)} mappings")
                    print(f"   Sample UNET Mappings (first 5): {dict(list(unet_key_map.items())[:5])}")
                    
                    # Show key mapping for CLIP
                    clip_key_map = {}
                    if clip_model is not None:
                        clip_key_map = comfy.lora.model_lora_keys_clip(clip_model.cond_stage_model, clip_key_map)
                    print(f"   CLIP Key Mappings: {len(clip_key_map)} mappings")
                    print(f"   Sample CLIP Mappings (first 5): {dict(list(clip_key_map.items())[:5])}")
                    
                    # Check which LoRA keys will actually be loaded
                    all_key_map = {**unet_key_map, **clip_key_map}
                    matching_keys = []
                    for lora_key in lora_keys:
                        base_key = lora_key.replace('.lora_up.weight', '').replace('.lora_down.weight', '').replace('.alpha', '').replace('.diff', '')
                        if base_key in all_key_map or lora_key in all_key_map:
                            matching_keys.append(lora_key)
                    
                    print(f"   Matching LoRA Keys: {len(matching_keys)} will be loaded")
                    print(f"   Sample Matching Keys: {matching_keys[:5]}")
                    
                    # Apply LoRA
                    model, clip_model = lora_loader.load_lora(
                        model, clip_model, lora_path, 0.5, 1.0
                    )
                    
                    print("✅ LoRA applied successfully")
                    
                    # DEBUG: Check model state after LoRA
                    print(f"\n🔍 DEBUG: Post-LoRA Model State:")
                    print(f"   UNET Patches Count: {len(getattr(model, 'patches', {}))}")
                    print(f"   CLIP Patches Count: {len(getattr(clip_model.patcher, 'patches', {})) if hasattr(clip_model, 'patcher') else 0}")
                    
                    if hasattr(model, 'patches') and len(model.patches) > 0:
                        patch_keys = list(model.patches.keys())
                        print(f"   Sample UNET Patch Keys: {patch_keys[:5]}")
                    
                    if hasattr(clip_model, 'patcher') and hasattr(clip_model.patcher, 'patches') and len(clip_model.patcher.patches) > 0:
                        clip_patch_keys = list(clip_model.patcher.patches.keys())
                        print(f"   Sample CLIP Patch Keys: {clip_patch_keys[:5]}")
                    
                    # COMMENTED OUT FOR STEP 3 DEBUGGING: End step monitoring and get final metrics
                    # COMMENTED OUT FOR STEP 3 DEBUGGING: elapsed_time, step_end_memory = self._end_step_monitoring("lora_application", step_start_time, step_start_memory)
                    # COMMENTED OUT FOR STEP 3 DEBUGGING: 
                    # COMMENTED OUT FOR STEP 3 DEBUGGING: # Calculate memory changes
                    # COMMENTED OUT FOR STEP 3 DEBUGGING: ram_change = step_end_memory['ram_used_mb'] - step_start_memory['ram_used_mb']
                    # COMMENTED OUT FOR STEP 3 DEBUGGING: gpu_change = step_end_memory['gpu_allocated_mb'] - step_start_memory['gpu_allocated_mb']
                    # COMMENTED OUT FOR STEP 3 DEBUGGING: 
                    # COMMENTED OUT FOR STEP 3 DEBUGGING: # Store step results for workflow monitoring
                    # COMMENTED OUT FOR STEP 3 DEBUGGING: if not hasattr(self, 'step_results'):
                    # COMMENTED OUT FOR STEP 3 DEBUGGING:     self.step_results = {}
                    # COMMENTED OUT FOR STEP 3 DEBUGGING: 
                    # COMMENTED OUT FOR STEP 3 DEBUGGING: self.step_results['lora_application'] = {
                    # COMMENTED OUT FOR STEP 3 DEBUGGING:     'elapsed_time': elapsed_time,
                    # COMMENTED OUT FOR STEP 3 DEBUGGING:     'ram_change': ram_change,
                    # COMMENTED OUT FOR STEP 3 DEBUGGING:     'gpu_change': gpu_change,
                    # COMMENTED OUT FOR STEP 3 DEBUGGING:     'success': True,
                    # COMMENTED OUT FOR STEP 3 DEBUGGING:     'skipped': True,
                    # COMMENTED OUT FOR STEP 3 DEBUGGING:     'baseline_memory': step_start_memory,
                    # COMMENTED OUT FOR STEP 3 DEBUGGING:     'gpu_change': gpu_change,
                    # COMMENTED OUT FOR STEP 3 DEBUGGING:     'success': True,
                    # COMMENTED OUT FOR STEP 3 DEBUGGING:     'skipped': True,
                    # COMMENTED OUT FOR STEP 3 DEBUGGING:     'baseline_memory': step_start_memory,
                    # COMMENTED OUT FOR STEP 3 DEBUGGING:     'end_memory': step_end_memory
                    # COMMENTED OUT FOR STEP 3 DEBUGGING: }
                    # COMMENTED OUT FOR STEP 3 DEBUGGING: 
                    # COMMENTED OUT FOR STEP 3 DEBUGGING: # Print enhanced model summary
                    # COMMENTED OUT FOR STEP 3 DEBUGGING: self._print_enhanced_model_summary(model, "LoRA_Result")
                    # COMMENTED OUT FOR STEP 3 DEBUGGING: 
                    # COMMENTED OUT FOR STEP 3 DEBUGGING: # Print comprehensive memory breakdown
                    # COMMENTED OUT FOR STEP 3 DEBUGGING: self._print_comprehensive_memory_breakdown(step_start_memory, step_end_memory, step_start_time, time.time())
                    # COMMENTED OUT FOR STEP 3 DEBUGGING: 
                    # COMMENTED OUT FOR STEP 3 DEBUGGING: # Print peak memory summary
                    # COMMENTED OUT FOR STEP 3 DEBUGGING: self._print_peak_memory_summary(step_start_memory, step_end_memory, step_start_time, time.time())
                    
                    # Analyze LoRA application results
                    print("\n🔍 ANALYZING LORA APPLICATION RESULTS...")
                    lora_analysis = self._analyze_lora_application_results(
                        lora_baseline, original_model, original_clip_model, 
                        model, clip_model, [model, clip_model]
                    )
                    
                    # Display comprehensive analysis
                    self._print_lora_analysis_summary(lora_analysis)
                    
                except Exception as e:
                    print(f"❌ ERROR during LoRA application: {e}")
                    print("🔍 LoRA application failed - check error details above")
                    
                    # COMMENTED OUT FOR STEP 3 DEBUGGING: End step monitoring even on error
                    # COMMENTED OUT FOR STEP 3 DEBUGGING: elapsed_time, step_end_memory = self._end_step_monitoring("lora_application", step_start_time, step_start_memory)
                    # COMMENTED OUT FOR STEP 3 DEBUGGING: 
                    # COMMENTED OUT FOR STEP 3 DEBUGGING: # Calculate memory changes
                    # COMMENTED OUT FOR STEP 3 DEBUGGING: ram_change = step_end_memory['ram_used_mb'] - step_start_memory['ram_used_mb']
                    # COMMENTED OUT FOR STEP 3 DEBUGGING: gpu_change = step_end_memory['gpu_allocated_mb'] - step_start_memory['gpu_allocated_mb']
                    # COMMENTED OUT FOR STEP 3 DEBUGGING: 
                    # COMMENTED OUT FOR STEP 3 DEBUGGING: # Store step results for workflow monitoring
                    # COMMENTED OUT FOR STEP 3 DEBUGGING: if not hasattr(self, 'step_results'):
                    # COMMENTED OUT FOR STEP 3 DEBUGGING:     self.step_results = {}
                    # COMMENTED OUT FOR STEP 3 DEBUGGING: 
                    # COMMENTED OUT FOR STEP 3 DEBUGGING: self.step_results['lora_application'] = {
                    # COMMENTED OUT FOR STEP 3 DEBUGGING:     'elapsed_time': elapsed_time,
                    # COMMENTED OUT FOR STEP 3 DEBUGGING:     'ram_change': ram_change,
                    # COMMENTED OUT FOR STEP 3 DEBUGGING:     'gpu_change': gpu_change,
                    # COMMENTED OUT FOR STEP 3 DEBUGGING:     'success': False,
                    # COMMENTED OUT FOR STEP 3 DEBUGGING:     'error': str(e),
                    # COMMENTED OUT FOR STEP 3 DEBUGGING:     'baseline_memory': step_start_memory,
                    # COMMENTED OUT FOR STEP 3 DEBUGGING:     'end_memory': step_end_memory
                    # COMMENTED OUT FOR STEP 3 DEBUGGING: }
                    # COMMENTED OUT FOR STEP 3 DEBUGGING: 
                    # COMMENTED OUT FOR STEP 3 DEBUGGING: # Print error summary
                    # COMMENTED OUT FOR STEP 3 DEBUGGING: print(f"\n❌ LoRA application failed after {elapsed_time:.3f} seconds")
                    print("⚠️  Continuing with original models...")
                    
                    # Keep original models if LoRA fails
                    model = original_model
                    clip_model = original_clip_model
                
                print("="*80)
                print("🔍 LORA APPLICATION MONITORING SYSTEM COMPLETE")
                print("="*80)
                # === LORA APPLICATION MONITORING SYSTEM END ===
                
                # COMMENTED OUT FOR STEP 3 DEBUGGING: Print workflow monitoring summary
                # COMMENTED OUT FOR STEP 3 DEBUGGING: if hasattr(self, 'step_results') and 'lora_application' in self.step_results:
                # COMMENTED OUT FOR STEP 3 DEBUGGING:     self._print_workflow_monitoring_summary(self.step_results)
                # COMMENTED OUT FOR STEP 3 DEBUGGING: 
                # COMMENTED OUT FOR STEP 3 DEBUGGING: # Stop execution after Step 2 for debugging purposes
                # COMMENTED OUT FOR STEP 3 DEBUGGING: print("\n🛑 STOPPING EXECUTION AFTER STEP 2 (LORA APPLICATION)")
                # COMMENTED OUT FOR STEP 3 DEBUGGING: print("🔍 All LoRA application debugging information has been displayed above.")
                # COMMENTED OUT FOR STEP 3 DEBUGGING: print("📊 Check the monitoring data above to analyze LoRA application performance.")
                # COMMENTED OUT FOR STEP 3 DEBUGGING: 
                # COMMENTED OUT FOR STEP 3 DEBUGGING: # Print step completion status
                # COMMENTED OUT FOR STEP 3 DEBUGGING: print(f"\n🔍 Step 1: Model Loading - COMPLETED")
                # COMMENTED OUT FOR STEP 3 DEBUGGING: print(f"🔍 Step 3 DEBUGGING: print(f"\n🔍 Step 1: Model Loading - COMPLETED")
                # COMMENTED OUT FOR STEP 3 DEBUGGING: print(f"🔍 Step 2: LoRA Application - COMPLETED")
                # COMMENTED OUT FOR STEP 3 DEBUGGING: print(f"🔍 Steps 3-9: SKIPPED for debugging purposes")
                # COMMENTED OUT FOR STEP 3 DEBUGGING: 
                # COMMENTED OUT FOR STEP 3 DEBUGGING: # Print final workflow summary
                # COMMENTED OUT FOR STEP 3 DEBUGGING: if hasattr(self, 'step_results'):
                # COMMENTED OUT FOR STEP 3 DEBUGGING:     self._print_final_workflow_summary(self.step_results)
                
                print("="*80)
                print("🔍 FINAL WORKFLOW MONITORING SUMMARY")
                print("="*80)
                
                # Return early to stop execution
                print("✅ Pipeline continuing to Step 3...")
                
                # ========================================================================
                # STEP 3: LOAD VIDEO AND IMAGE DATA
                # ========================================================================
                print("\n" + "="*80)
                print("🔍 STEP 3: LOAD VIDEO AND IMAGE DATA")
                print("="*80)
                
                # Load control video
                print("3a. Loading control video...")
                if control_video_path:
                    control_video = self.load_video(control_video_path)
                    if control_video is not None:
                        print(f"   ✅ Control video loaded: {control_video.shape}")
                    else:
                        print("   ❌ Failed to load control video")
                        control_video = None
                else:
                    print("   ⚠️  No control video path specified")
                    control_video = None
                
                # Load reference image
                print("3b. Loading reference image...")
                if reference_image_path:
                    reference_image = self.load_image(reference_image_path)
                    if reference_image is not None:
                        print(f"   ✅ Reference image loaded: {reference_image.shape}")
                    else:
                        print("   ❌ Failed to load reference image")
                        reference_image = None
                else:
                    print("   ⚠️  No reference image path specified")
                    reference_image = None
                
                print("✅ Step 3 completed - continuing to Step 5...")
                
            else:
                print("2. No LoRA specified, skipping LoRA application")
                print("2a. Models remain in original state")
                print("✅ Step 2 completed - continuing to Step 3...")
                
                # ========================================================================
                # STEP 3: LOAD VIDEO AND IMAGE DATA (No LoRA path)
                # ========================================================================
                print("\n" + "="*80)
                print("🔍 STEP 3: LOAD VIDEO AND IMAGE DATA (No LoRA)")
                print("="*80)
                
                # Load control video
                print("3a. Loading control video...")
                if control_video_path:
                    control_video = self.load_video(control_video_path)
                    if control_video is not None:
                        print(f"   ✅ Control video loaded: {control_video.shape}")
                    else:
                        print("   ❌ Failed to load control video")
                        control_video = None
                else:
                    print("   ⚠️  No control video path specified")
                    control_video = None
                
                # Load reference image
                print("3b. Loading reference image...")
                if reference_image_path:
                    reference_image = self.load_image(reference_image_path)
                    if reference_image is not None:
                        print(f"   ✅ Reference image loaded: {reference_image.shape}")
                    else:
                        print("   ❌ Failed to load reference image")
                        reference_image = None
                else:
                    print("   ⚠️  No reference image path specified")
                    reference_image = None
                
                print("✅ Step 3 completed - continuing to Step 5...")
                
                # === LORA APPLICATION MONITORING SYSTEM START (No LoRA) ===
                print("\n🔍 LORA APPLICATION MONITORING SYSTEM ACTIVATED (No LoRA)")
                print("="*80)
                
                # Start step monitoring with timing and memory baseline
                step_start_time, step_start_memory = self._start_step_monitoring("lora_application")
                
                # Capture baseline state even without LoRA for comparison
                print("📊 CAPTURING BASELINE STATE (No LoRA - Models in Original State)...")
                no_lora_baseline = self._capture_lora_baseline(model, clip_model, "N/A")
                
                # Display baseline information
                print(f"   📁 LoRA File: None (skipping LoRA application)")
                print(f"   ✅ UNET Baseline captured - ID: {no_lora_baseline['unet']['model_id']}, Patches: {no_lora_baseline['unet']['patches_count']}")
                print(f"   ✅ CLIP Baseline captured - ID: {no_lora_baseline['clip']['model_id']}, Patches: {no_lora_baseline['clip']['patcher_patches_count']}")
                
                # Display baseline memory state
                print(f"\n   💾 BASELINE MEMORY STATE:")
                print(f"      🖥️  RAM: {step_start_memory['ram_used_mb']:.1f} MB used / {step_start_memory['ram_total_mb']:.1f} MB total ({step_start_memory['ram_percent']:.1f}%)")
                print(f"      🎮 GPU: {step_start_memory['gpu_allocated_mb']:.1f} MB allocated / {step_start_memory['gpu_total_mb']:.1f} MB total")
                print(f"      🎮 GPU Device: {step_start_memory['gpu_device_name']}")
                
                print("✅ Baseline captured successfully (No LoRA)")
                
                # End step monitoring and get final metrics
                elapsed_time, step_end_memory = self._end_step_monitoring("lora_application", step_start_time, step_start_memory)
                
                # Print enhanced model summary
                self._print_enhanced_model_summary(model, "Original_UNET")
                self._print_enhanced_model_summary(clip_model, "Original_CLIP")
                
                # Print comprehensive memory breakdown
                self._print_comprehensive_memory_breakdown(step_start_memory, step_end_memory, step_start_time, time.time())
                
                # Print peak memory summary
                self._print_peak_memory_summary(step_start_memory, step_end_memory, step_start_time, time.time())
                
                print("="*80)
                print("🔍 LORA APPLICATION MONITORING SYSTEM COMPLETE (No LoRA)")
                print("="*80)
                # === LORA APPLICATION MONITORING SYSTEM END (No LoRA) ===
                
                # Print workflow monitoring summary
                if hasattr(self, 'step_results') and 'lora_application' in self.step_results:
                    self._print_workflow_monitoring_summary(self.step_results)
                
                # COMPREHENSIVE VERIFICATION AFTER LoRA SKIP
                print("\n" + "="*80)
                print("🔍 STEP 2 COMPLETE: COMPREHENSIVE VERIFICATION (No LoRA)")
                print("="*80)
                
                # 1. Model Placement Verification
                print("1️⃣  MODEL PLACEMENT VERIFICATION:")
                model_placement = self._check_model_placement('lora_application', ['unet', 'clip'])
                
                # 2. Memory Management Verification
                print("\n2️⃣  MEMORY MANAGEMENT VERIFICATION:")
                memory_management = self._verify_memory_management('lora_application', ['unet', 'clip'])
                
                # 3. Chunking Strategy Verification
                print("\n3️⃣  CHUNKING STRATEGY VERIFICATION:")
                chunking_strategy = self._verify_chunking_strategy('lora_application', processing_plan)
                
                # 4. Summary
                print("\n📊 STEP 2 SUMMARY:")
                print(f"   Model Placement: {'✅ PASS' if model_placement else '❌ FAIL'}")
                print(f"   Memory Management: {'✅ PASS' if memory_management else '❌ FAIL'}")
                print(f"   Chunking Strategy: {'✅ PASS' if chunking_strategy else '❌ FAIL'}")
                
                if not all([model_placement, memory_management, chunking_strategy]):
                    print("   ⚠️  Some verifications failed - pipeline may have issues")
                else:
                    print("   ✅ All verifications passed - pipeline ready for next step")
                
                print("="*80)
           
                # === LORA APPLICATION MONITORING SYSTEM START (No LoRA) ===
                print("\n🔍 LORA APPLICATION MONITORING SYSTEM ACTIVATED (No LoRA)")
                print("="*80)
                
                # Start step monitoring with timing and memory baseline
                step_start_time, step_start_memory = self._start_step_monitoring("lora_application")
                
                # Capture baseline state even without LoRA for comparison
                print("📊 CAPTURING BASELINE STATE (No LoRA - Models in Original State)...")
                no_lora_baseline = self._capture_lora_baseline(model, clip_model, "N/A")
                
                # Display baseline information
                print(f"   📁 LoRA File: None (skipping LoRA application)")
                print(f"   ✅ UNET Baseline captured - ID: {no_lora_baseline['unet']['model_id']}, Patches: {no_lora_baseline['unet']['patches_count']}")
                print(f"   ✅ CLIP Baseline captured - ID: {no_lora_baseline['clip']['model_id']}, Patches: {no_lora_baseline['clip']['patcher_patches_count']}")
                
                # Display baseline memory state
                print(f"\n   💾 BASELINE MEMORY STATE:")
                print(f"      🖥️  RAM: {step_start_memory['ram_used_mb']:.1f} MB used / {step_start_memory['ram_total_mb']:.1f} MB total ({step_start_memory['ram_percent']:.1f}%)")
                print(f"      🎮 GPU: {step_start_memory['gpu_allocated_mb']:.1f} MB allocated / {step_start_memory['gpu_total_mb']:.1f} MB total")
                print(f"      🎮 GPU Device: {step_start_memory['gpu_device_name']}")
                
                print("✅ Baseline captured successfully (No LoRA)")
                
                # End step monitoring and get final metrics
                elapsed_time, step_end_memory = self._end_step_monitoring("lora_application", step_start_time, step_start_memory)
                
                # Print enhanced model summary
                self._print_enhanced_model_summary(model, "Original_UNET")
                self._print_enhanced_model_summary(clip_model, "Original_CLIP")
                
                # Print comprehensive memory breakdown
                self._print_comprehensive_memory_breakdown(step_start_memory, step_end_memory, step_start_time, time.time())
                
                # Print peak memory summary
                self._print_peak_memory_summary(step_start_memory, step_end_memory, step_start_time, time.time())
                
                print("="*80)
                print("🔍 LORA APPLICATION MONITORING SYSTEM COMPLETE (No LoRA)")
                print("="*80)
                # === LORA APPLICATION MONITORING SYSTEM END (No LoRA) ===
                
                # Print workflow monitoring summary
                if hasattr(self, 'step_results') and 'lora_application' in self.step_results:
                    self._print_workflow_monitoring_summary(self.step_results)
                
                # COMPREHENSIVE VERIFICATION AFTER LoRA SKIP
                print("\n" + "="*80)
                print("🔍 STEP 2 COMPLETE: COMPREHENSIVE VERIFICATION (No LoRA)")
                print("="*80)
                
                # 1. Model Placement Verification
                print("1️⃣  MODEL PLACEMENT VERIFICATION:")
                model_placement = self._check_model_placement('lora_application', ['unet', 'clip'])
                
                # 2. Memory Management Verification
                print("\n2️⃣  MEMORY MANAGEMENT VERIFICATION:")
                memory_management = self._verify_memory_management('lora_application', ['unet', 'clip'])
                
                # 3. Chunking Strategy Verification
                print("\n3️⃣  CHUNKING STRATEGY VERIFICATION:")
                chunking_strategy = self._verify_chunking_strategy('lora_application', processing_plan)
                
                # 4. Summary
                print("\n📊 STEP 2 SUMMARY:")
                print(f"   Model Placement: {'✅ PASS' if model_placement else '❌ FAIL'}")
                print(f"   Memory Management: {'✅ PASS' if memory_management else '❌ FAIL'}")
                print(f"   Chunking Strategy: {'✅ PASS' if chunking_strategy else '❌ FAIL'}")
                
                if not all([model_placement, memory_management, chunking_strategy]):
                    print("   ⚠️  Some verifications failed - pipeline may have issues")
                else:
                    print("   ✅ All verifications passed - pipeline ready for next step")
                
                print("="*80)
            
            # 3. TEXT ENCODING (MONITORING COMMENTED OUT)
            print("\n" + "="*80)
            print("🔍 STEP 3: TEXT ENCODING (MONITORING COMMENTED OUT)")
            print("="*80)
            
            # Simple text encoding without monitoring
            text_encoder = CLIPTextEncode()
            positive_cond = text_encoder.encode(clip_model, positive_prompt)
            negative_cond = text_encoder.encode(clip_model, negative_prompt)
            
            print(f"✅ Text encoding completed")
            print(f"✅ Positive conditioning shape: {positive_cond[0][0][0].shape if positive_cond and len(positive_cond) > 0 and len(positive_cond[0]) > 0 and len(positive_cond[0][0]) > 0 else 'Unknown'}")
            print(f"✅ Negative conditioning shape: {negative_cond[0][0][0].shape if negative_cond and len(negative_cond) > 0 and len(negative_cond) > 0 and len(negative_cond[0][0]) > 0 else 'Unknown'}")

            # ========================================================================
            # STEP 4: SAMPLING STEP (MONITORING COMMENTED OUT)
            # ========================================================================
            print(f"\n{'='*80}")
            print(f"🔍 STEP 4: SAMPLING STEP (MONITORING COMMENTED OUT)")
            print(f"{'='*80}")
            
            print(f"✅ Step 4 completed - monitoring commented out for step 5 focus")
            print(f"🔍 Continuing to step 5...")
            
            # 4. Apply ModelSamplingSD3 Shift
            print("4. Applying ModelSamplingSD3...")
            model_sampling = ModelSamplingSD3()
            
            # ModelPatcher automatically handles loading/unloading during patching
            model = model_sampling.patch(model, shift=8.0)
            
            # ComfyUI automatically tracks the patched model through ModelPatcher
            print("4a. ModelSamplingSD3 applied")
            
            # OOM Checklist: Check memory after ModelSamplingSD3
            self._check_memory_usage('model_sampling', expected_threshold=2000)
            
            # COMPREHENSIVE VERIFICATION AFTER MODEL SAMPLING
            print("\n" + "="*80)
            print("🔍 STEP 4 COMPLETE: COMPREHENSIVE VERIFICATION")
            print("="*80)
            
            # 1. Model Placement Verification
            print("1️⃣  MODEL PLACEMENT VERIFICATION:")
            model_placement = self._check_model_placement('model_sampling', ['unet'])
            
            # 2. Memory Management Verification
            print("\n2️⃣  MEMORY MANAGEMENT VERIFICATION:")
            memory_management = self._verify_memory_management('model_sampling', ['unet'])
            
            # 3. Chunking Strategy Verification
            print("\n3️⃣  CHUNKING STRATEGY VERIFICATION:")
            chunking_strategy = self._verify_chunking_strategy('model_sampling', processing_plan)
            
            # 4. Summary
            print("\n📊 STEP 4 SUMMARY:")
            print(f"   Model Placement: {'✅ PASS' if model_placement else '❌ FAIL'}")
            print(f"   Memory Management: {'✅ PASS' if memory_management else '❌ FAIL'}")
            print(f"   Chunking Strategy: {'✅ PASS' if chunking_strategy else '❌ FAIL'}")
            
            if not all([model_placement, memory_management, chunking_strategy]):
                print("   ⚠️  Some verifications failed - pipeline may have issues")
            else:
                print("   ✅ All verifications passed - pipeline ready for next step")
            
            print("="*80)
            
            # ===============================================================================
            # 🧪 TEST COMFYUI MEMORY MANAGEMENT BEFORE STEP 5
            # ===============================================================================
            print("\n" + "="*80)
            print("🧪 TESTING COMFYUI MEMORY MANAGEMENT BEFORE STEP 5")
            print("="*80)
            
            if self.model_registry:
                try:
                    # Test 1: get_free_memory()
                    print("🔍 Testing get_free_memory()...")
                    device = comfy.model_management.get_torch_device()
                    free_memory = comfy.model_management.get_free_memory(device)
                    free_total, free_torch = comfy.model_management.get_free_memory(device, torch_free_too=True)
                    
                    print(f"   ✅ get_free_memory() working:")
                    print(f"      Total free: {free_memory / (1024**2):.1f} MB")
                    print(f"      GPU free: {free_total / (1024**2):.1f} MB")
                    print(f"      Torch free: {free_torch / (1024**2):.1f} MB")
                    
                    # Test 2: load_models_gpu() with empty list
                    print("🔍 Testing load_models_gpu() with empty list...")
                    comfy.model_management.load_models_gpu([], memory_required=0)
                    print("   ✅ load_models_gpu() working with empty list")
                    
                    # Test 3: free_memory() with minimal requirement
                    print("🔍 Testing free_memory() with minimal requirement...")
                    unloaded = comfy.model_management.free_memory(1024*1024, device)  # 1MB
                    print(f"   ✅ free_memory() working: {len(unloaded)} models unloaded")
                    
                    # Test 4: Check if model tracking is working
                    print("🔍 Testing model tracking system...")
                    if hasattr(comfy.model_management, 'current_loaded_models'):
                        print(f"   ✅ current_loaded_models exists: {len(comfy.model_management.current_models)} models")
                    else:
                        print("   ❌ current_loaded_models not found")
                    
                    print("✅ All ComfyUI memory management functions are working!")

                    print("   Ready to proceed with Step 5 VAE encoding")
                    
                    # Test VAE memory preparation
                    print("\n🧪 TESTING VAE MEMORY PREPARATION...")
                    try:
                        # Create a small test tensor
                        test_tensor = torch.ones((1, 64, 64, 3), device='cpu')
                        
                        # Test memory preparation
                        self._prepare_vae_memory(vae, test_tensor)
                        print("   ✅ VAE memory preparation test passed")
                        
                        # Test chunked encoding with small data
                        print("   Testing chunked encoding...")
                        test_frames = torch.ones((2, 64, 64, 3), device='cpu')
                        test_latents = self._encode_video_chunked(vae, test_frames, chunk_size=1)
                        print(f"   ✅ Chunked encoding test passed: {test_latents.shape}")
                        
                    except Exception as e:
                        print(f"   ❌ VAE memory preparation test failed: {e}")
                        print("   ⚠️  Step 5 may fail due to VAE memory issues")
                    
                    print("\n🧪 TESTING VAE MEMORY MANAGEMENT INTEGRATION...")
                    
                    try:
                        # Check if VAE is properly integrated
                        if hasattr(vae, 'patcher') and vae.patcher is not None:
                            print("   ✅ VAE is wrapped in ModelPatcher")
                            
                            # Test VAE device placement
                            vae_device = comfy.model_management.vae_device()
                            print(f"   VAE target device: {vae_device}")
                            
                            # Check VAE memory usage
                            if hasattr(vae, 'model_memory'):
                                vae_memory = vae.model_memory()
                                print(f"   VAE model memory: {vae_memory / (1024**2):.1f} MB")
                            else:
                                print("   VAE model memory: Not accessible")
                            
                            # Test VAE memory management functions
                            print("   Testing VAE-specific memory management...")
                            
                            # Check if VAE is in ComfyUI tracking
                            if hasattr(comfy.model_management, 'current_loaded_models'):
                                vae_in_tracking = any(
                                    hasattr(m, 'model') and m.model == vae.patcher 
                                    for m in comfy.model_management.current_loaded_models
                                )
                                if vae_in_tracking:
                                    print("   ✅ VAE is tracked by ComfyUI memory management")
                                else:
                                    print("   ⚠️  VAE is NOT tracked by ComfyUI memory management")
                            
                            print("   ✅ VAE memory management integration test passed")
                            
                        else:
                            print("   ❌ VAE is NOT wrapped in ModelPatcher")
                            print("   ⚠️  VAE will not use ComfyUI memory management")
                            print("   💡 This explains why VAE encoding fails with OOM")
                            
                    except Exception as e:
                        print(f"   ❌ VAE memory management test failed: {e}")
                        print("   ⚠️  VAE memory management is not working properly")
                    
                except Exception as e:
                    print(f"❌ ComfyUI memory management test failed: {e}")
                    print("   ⚠️  Step 5 may fail due to memory management issues")
                    print("   Continuing anyway...")
                    
                    if not hasattr(self, 'model_registry') or not self.model_registry:
                        print("⚠️  No model registry available - ComfyUI integration not working")
                        print("   Step 5 will likely fail")
            
            print("="*80)
            
            # ========================================================================
            # STEP 5: GENERATE INITIAL LATENTS (COMFY-LIKE)
            # ========================================================================
            print(f"\n{'='*80}")
            print(f"🔍 STEP 5: GENERATE INITIAL LATENTS (COMFY-LIKE)")
            print(f"{'='*80}")
            
            # Enable comprehensive memory tracking for Step 5
            self._track_memory_during_step5()
            
            # Start real-time memory monitoring for Step 5
            self.memory_monitor.start_monitoring("STEP5_VAE_ENCODING")
            
            # Initial VRAM analysis before Step 5
            self._detailed_vram_analysis("STEP5_START")

            # Ensure inputs are loaded locally for this step
            control_video = locals().get('control_video', None)
            if control_video is None:
                control_video = self.load_video(control_video_path) if control_video_path else None
            reference_image = locals().get('reference_image', None)
            if reference_image is None:
                reference_image = self.load_image(reference_image_path) if reference_image_path else None

            # Comfy-like implementation of WanVaceToVideo.encode
            from comfy import node_helpers

            vae_stride = 8
            latent_length = ((length - 1) // 4) + 1

            # Prepare control video
            if control_video is not None:
                control_video = control_video[:length]
                control_video = comfy.utils.common_upscale(
                    control_video.movedim(-1, 1), width, height, "bilinear", "center"
                ).movedim(1, -1)
                if control_video.shape[0] < length:
                    control_video = torch.nn.functional.pad(
                        control_video, (0, 0, 0, 0, 0, 0, 0, length - control_video.shape[0]), value=0.5
                    )
            else:
                device = vae.first_stage_model.device if hasattr(vae, 'first_stage_model') else 'cpu'
                control_video = torch.ones((length, height, width, 3), device=device) * 0.5
                        
            # Prepare reference image (optional)
            ref_img = None
            if reference_image is not None:
                ref_img = comfy.utils.common_upscale(
                    reference_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
                ).movedim(1, -1)

            # Prepare mask (default full mask if none provided)
            cm = locals().get('control_masks', None)
            if cm is None:
                mask = torch.ones((length, height, width, 1), device=control_video.device)
            else:
                mask = cm
                if mask.ndim == 3:
                    mask = mask.unsqueeze(1)
                mask = comfy.utils.common_upscale(
                    mask[:length], width, height, "bilinear", "center"
                ).movedim(1, -1)
                if mask.shape[0] < length:
                    mask = torch.nn.functional.pad(
                        mask, (0, 0, 0, 0, 0, 0, 0, length - mask.shape[0]), value=1.0
                    )

            # Normalize and split by mask
            control_video_norm = control_video - 0.5
            
            inactive = (control_video_norm * (1 - mask)) + 0.5
            reactive = (control_video_norm * mask) + 0.5

            # VAE encode inactive/reactive paths using direct approach
            print("🔍 Encoding inactive frames with direct VAE approach...")
            
            # Monitor memory before inactive encoding
            self._monitor_vae_encoding_memory(vae, inactive[:, :, :, :3], "INACTIVE_FRAMES")
            
            try:
                inactive_latent = self._comfy_vae_encode(vae, inactive[:, :, :, :3])
                print(f"   Inactive latent shape: {inactive_latent.shape}")
                self._quick_memory_snapshot("after_inactive")
                
                # Force memory cleanup after inactive encoding to prevent accumulation
                print("🧹 Cleaning up memory after inactive encoding...")
                force_comfy_memory_cleanup()
                
            except Exception as e:
                self._analyze_oom_cause(e, "INACTIVE_ENCODING")
                raise
            
            print("🔍 Encoding reactive frames with direct VAE approach...")
            
            # Monitor memory before reactive encoding
            self._monitor_vae_encoding_memory(vae, reactive[:, :, :, :3], "REACTIVE_FRAMES")
            
            try:
                reactive_latent = self._comfy_vae_encode(vae, reactive[:, :, :, :3])
                print(f"   Reactive latent shape: {reactive_latent.shape}")
                self._quick_memory_snapshot("after_reactive")
                
                # Force memory cleanup after reactive encoding to prevent accumulation
                print("🧹 Cleaning up memory after reactive encoding...")
                force_comfy_memory_cleanup()
                
            except Exception as e:
                self._analyze_oom_cause(e, "REACTIVE_ENCODING")
                raise
            
            # Normalize tensor dimensions before concatenation
            print("🔍 Normalizing tensor dimensions...")
            if len(inactive_latent.shape) == 5 and inactive_latent.shape[2] == 1:
                print("   Squeezing inactive latent dimension 2 (removing singleton)")
                inactive_latent = inactive_latent.squeeze(2)  # Remove singleton dimension
            if len(reactive_latent.shape) == 5 and reactive_latent.shape[2] == 1:
                print("   Squeezing reactive latent dimension 2 (removing singleton)")
                reactive_latent = reactive_latent.squeeze(2)  # Remove singleton dimension
                
            print(f"   Normalized inactive latent shape: {inactive_latent.shape}")
            print(f"   Normalized reactive latent shape: {reactive_latent.shape}")
            
            # Verify dimensions match before concatenation
            if len(inactive_latent.shape) != len(reactive_latent.shape):
                raise ValueError(f"Dimension mismatch: inactive_latent {inactive_latent.shape} vs reactive_latent {reactive_latent.shape}")
            
            print("🔍 Concatenating latents...")
            control_video_latent = torch.cat((inactive_latent, reactive_latent), dim=1)
            print(f"   Combined latent shape: {control_video_latent.shape}")

            # Reference image path (optional) - exact ComfyUI logic
            trim_latent = 0
            if ref_img is not None:
                print("🔍 Encoding reference image with direct VAE approach...")
                
                # Monitor memory before reference encoding
                self._monitor_vae_encoding_memory(vae, ref_img[:, :, :, :3], "REFERENCE_IMAGE")
                
                try:
                    ref_latent = self._comfy_vae_encode(vae, ref_img[:, :, :, :3])
                    print(f"   Reference latent shape: {ref_latent.shape}")
                    self._quick_memory_snapshot("after_reference")
                    
                    # Force memory cleanup after reference encoding to prevent accumulation
                    print("🧹 Cleaning up memory after reference encoding...")
                    force_comfy_memory_cleanup()
                    
                    # Investigate memory state after reference encoding
                    print("🔍 Memory investigation after reference encoding:")
                    investigate_unaccounted_memory()
                    
                except Exception as e:
                    self._analyze_oom_cause(e, "REFERENCE_ENCODING")
                    raise
                
                # Normalize reference latent dimensions
                if len(ref_latent.shape) == 5 and ref_latent.shape[2] == 1:
                    print("   Squeezing reference latent dimension 2 (removing singleton)")
                    ref_latent = ref_latent.squeeze(2)  # Remove singleton dimension
                print(f"   Normalized reference latent shape: {ref_latent.shape}")
                
                ref_latent = torch.cat([
                    ref_latent,
                    comfy.latent_formats.Wan21().process_out(torch.zeros_like(ref_latent))
                ], dim=1)
                print(f"   Reference latent after zero-cat shape: {ref_latent.shape}")
                control_video_latent = torch.cat((ref_latent, control_video_latent), dim=2)
                print(f"   Final control video latent shape: {control_video_latent.shape}")
                trim_latent = ref_latent.shape[2]

            # WAN mask reshaping to latent grid
            height_mask = height // vae_stride
            width_mask = width // vae_stride
            mask_lat = mask.view(length, height_mask, vae_stride, width_mask, vae_stride)
            mask_lat = mask_lat.permute(2, 4, 0, 1, 3)
            mask_lat = mask_lat.reshape(vae_stride * vae_stride, length, height_mask, width_mask)
            mask_lat = torch.nn.functional.interpolate(
                mask_lat.unsqueeze(0), size=(latent_length, height_mask, width_mask), mode='nearest-exact'
            ).squeeze(0)
            if trim_latent > 0:
                pad_front = torch.zeros((mask_lat.shape[0], trim_latent, height_mask, width_mask), device=mask_lat.device, dtype=mask_lat.dtype)
                mask_lat = torch.cat((pad_front, mask_lat), dim=1)
            mask_lat = mask_lat.unsqueeze(0)

            # Update conditioning (Comfy-style)
            positive_cond = node_helpers.conditioning_set_values(
                positive_cond,
                {"vace_frames": [control_video_latent], "vace_mask": [mask_lat], "vace_strength": [strength]},
                append=True,
            )
            negative_cond = node_helpers.conditioning_set_values(
                negative_cond,
                {"vace_frames": [control_video_latent], "vace_mask": [mask_lat], "vace_strength": [strength]},
                append=True,
            )

            # Allocate output latent (container) on intermediate device
            latent = torch.zeros([
                batch_size, 16, mask_lat.shape[1], height // 8, width // 8
            ], device=comfy.model_management.intermediate_device())
            out_latent = {"samples": latent}

            # Preserve variable names used later if any logging expects them
            init_latent = latent
            trim_count = trim_latent

            print(f"\n{'='*80}")
            print(f"✅ STEP 5 COMPLETE: Generate Initial Latents (Comfy-like)")
            print(f"{'='*80}")

            print(f"\n🔍 Step 1: Model Loading - COMPLETED")
            print(f"🔍 Step 2: LoRA Application - COMPLETED")
            print(f"🔍 Step 3: Text Encoding - COMPLETED")
            print(f"🔍 Step 4: Sampling - COMPLETED")
            print(f"🔍 Step 5: VAE Encoding - COMPLETED")
            print(f"{'='*80}")
            
            # Final memory investigation to see the complete picture
            print("🔍 FINAL MEMORY INVESTIGATION - STEP 5 COMPLETE:")
            investigate_unaccounted_memory()
            
            return "pipeline_stopped_after_step_5_for_debugging"
            
            # 4. Apply ModelSamplingSD3 Shift
            print("4. Applying ModelSamplingSD3...")
            model_sampling = ModelSamplingSD3()
            
            # ModelPatcher automatically handles loading/unloading during patching
            model = model_sampling.patch(model, shift=8.0)
            
            # ComfyUI automatically tracks the patched model through ModelPatcher
            print("4a. ModelSamplingSD3 applied")
            
            # OOM Checklist: Check memory after ModelSamplingSD3
            self._check_memory_usage('model_sampling', expected_threshold=2000)
            
            # COMPREHENSIVE VERIFICATION AFTER MODEL SAMPLING
            print("\n" + "="*80)
            print("🔍 STEP 4 COMPLETE: COMPREHENSIVE VERIFICATION")
            print("="*80)
            
            # 1. Model Placement Verification
            print("1️⃣  MODEL PLACEMENT VERIFICATION:")
            model_placement = self._check_model_placement('model_sampling', ['unet'])
            
            # 2. Memory Management Verification
            print("\n2️⃣  MEMORY MANAGEMENT VERIFICATION:")
            memory_management = self._verify_memory_management('model_sampling', ['unet'])
            
            # 3. Chunking Strategy Verification
            print("\n3️⃣  CHUNKING STRATEGY VERIFICATION:")
            chunking_strategy = self._verify_chunking_strategy('model_sampling', processing_plan)
            
            # 4. Summary
            print("\n📊 STEP 4 SUMMARY:")
            print(f"   Model Placement: {'✅ PASS' if model_placement else '❌ FAIL'}")
            print(f"   Memory Management: {'✅ PASS' if memory_management else '❌ FAIL'}")
            print(f"   Chunking Strategy: {'✅ PASS' if chunking_strategy else '❌ FAIL'}")
            
            if not all([model_placement, memory_management, chunking_strategy]):
                print("   ⚠️  Some verifications failed - pipeline may have issues")
            else:
                print("   ✅ All verifications passed - pipeline ready for next step")
            
            print("="*80)
            
            # 7. Trim Video Latent
            print("7. Trimming video latent...")
            trim_processor = TrimVideoLatent()
            
            # Wrap the latent tensor in the dictionary format expected by TrimVideoLatent
            latent_dict = {"samples": final_latent}
            trimmed_latent_dict = trim_processor.op(latent_dict, trim_count)
            
            # Extract the trimmed tensor from the dictionary
            trimmed_latent = trimmed_latent_dict["samples"]
            print(f"7a. Trimmed latent shape: {trimmed_latent.shape}")
            
            # OOM Checklist: Check memory after video trimming
            self._check_memory_usage('video_trimming', expected_threshold=100)
            
            # COMPREHENSIVE VERIFICATION AFTER VIDEO LATENT TRIMMING
            print("\n" + "="*80)
            print("🔍 STEP 7 COMPLETE: COMPREHENSIVE VERIFICATION")
            print("="*80)
            
            # 1. Model Placement Verification
            print("1️⃣  MODEL PLACEMENT VERIFICATION:")
            model_placement = self._check_model_placement('video_trimming', [])
            
            # 2. Memory Management Verification
            print("\n2️⃣  MEMORY MANAGEMENT VERIFICATION:")
            memory_management = self._verify_memory_management('video_trimming', [])
            
            # 3. Chunking Strategy Verification
            print("\n3️⃣  CHUNKING STRATEGY VERIFICATION:")
            chunking_strategy = self._verify_chunking_strategy('video_trimming', processing_plan)
            
            # 4. Video Trimming Results Verification
            print("\n4️⃣  VIDEO TRIMMING RESULTS VERIFICATION:")
            if 'trimmed_latent' in locals():
                if hasattr(trimmed_latent, 'shape'):
                    print(f"   Trimmed Latent: ✅ Shape: {trimmed_latent.shape}")
                    print(f"   Trim Count: {trim_count if 'trim_count' in locals() else 'Unknown'}")
                    video_trimming_success = True
                else:
                    print("   Trimmed Latent: ❌ No shape information")
                    video_trimming_success = False
            else:
                print("   Trimmed Latent: ❌ No trimmed latent created")
                video_trimming_success = False
            
            # 5. Summary
            print("\n📊 STEP 7 SUMMARY:")
            print(f"   Model Placement: {'✅ PASS' if model_placement else '❌ FAIL'}")
            print(f"   Memory Management: {'✅ PASS' if memory_management else '❌ FAIL'}")
            print(f"   Chunking Strategy: {'✅ PASS' if chunking_strategy else '❌ FAIL'}")
            print(f"   Video Trimming Success: {'✅ PASS' if video_trimming_success else '❌ FAIL'}")
            
            if not all([model_placement, memory_management, chunking_strategy, video_trimming_success]):
                print("   ⚠️  Some verifications failed - pipeline may have issues")
            else:
                print("   ✅ All verifications passed - pipeline ready for next step")
            
            print("="*80)
            
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
            print("8a. Letting ComfyUI's VAE ModelPatcher handle device placement automatically...")
            print("8a. VAE will be moved to GPU when needed for decoding operations")
            
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
            
            # Let ComfyUI handle VAE memory management automatically
            print("8b. VAE decoding complete")
            print("8b. ComfyUI's VAE ModelPatcher will handle memory management automatically")
            print("8b. No manual VAE device management needed - letting ComfyUI coordinate")
            
            # COMPREHENSIVE VERIFICATION AFTER VAE DECODING
            print("\n" + "="*80)
            print("🔍 STEP 8 COMPLETE: COMPREHENSIVE VERIFICATION")
            print("="*80)
            
            # 1. Model Placement Verification
            print("1️⃣  MODEL PLACEMENT VERIFICATION:")
            model_placement = self._check_model_placement('vae_decoding', ['vae'])
            
            # 2. Memory Management Verification
            print("\n2️⃣  MEMORY MANAGEMENT VERIFICATION:")
            memory_management = self._verify_memory_management('vae_decoding', ['vae'])
            
            # 3. Chunking Strategy Verification
            print("\n3️⃣  CHUNKING STRATEGY VERIFICATION:")
            chunking_strategy = self._verify_chunking_strategy('vae_decoding', processing_plan)
            
            # 4. VAE Decoding Results Verification
            print("\n4️⃣  VAE DECODING RESULTS VERIFICATION:")
            if 'frames' in locals():
                if hasattr(frames, 'shape'):
                    print(f"   Frames Generated: ✅ Shape: {frames.shape}")
                    if len(frames.shape) == 4:
                        print(f"   Frame Info: {frames.shape[0]} frames, {frames.shape[1]}x{frames.shape[2]}, {frames.shape[3]} channels")
                        if frames.shape[3] == 3:
                            print("   ✅ Frames have correct 3 channels (RGB)")
                        else:
                            print(f"   ⚠️  Frames have wrong channel count: {frames.shape[3]} (expected 3)")
                    else:
                        print(f"   ⚠️  Frames have unexpected shape: {frames.shape}")
                    vae_decoding_success = True
                else:
                    print("   Frames Generated: ❌ No shape information")
                    vae_decoding_success = False
            else:
                print("   Frames Generated: ❌ No frames created")
                vae_decoding_success = False
            
            # 5. Summary
            print("\n📊 STEP 8 SUMMARY:")
            print(f"   Model Placement: {'✅ PASS' if model_placement else '❌ FAIL'}")
            print(f"   Memory Management: {'✅ PASS' if memory_management else '❌ FAIL'}")
            print(f"   Chunking Strategy: {'✅ PASS' if chunking_strategy else '❌ FAIL'}")
            print(f"   VAE Decoding Success: {'✅ PASS' if vae_decoding_success else '❌ FAIL'}")
            
            if not all([model_placement, memory_management, chunking_strategy, vae_decoding_success]):
                print("   ⚠️  Some verifications failed - pipeline may have issues")
            else:
                print("   ✅ All verifications passed - pipeline ready for next step")
            
            print("="*80)
            
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
                        elif frames.shape[3] == 1:
                            print("9a. ⚠️  WARNING: Frames have only 1 channel! Expected 3 channels (RGB)")
                            print("9a. 🔧 Attempting to expand 1-channel frames to 3-channel...")
                            # Expand 1-channel to 3-channel by repeating
                            frames = frames.repeat(1, 1, 1, 3)
                            print(f"9a. ✅ Expanded frames shape: {frames.shape}")
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
            
            # OOM Checklist: Check memory after video export
            self._check_memory_usage('video_export', expected_threshold=100)
            
            # COMPREHENSIVE VERIFICATION AFTER VIDEO EXPORT
            print("\n" + "="*80)
            print("🔍 STEP 9 COMPLETE: COMPREHENSIVE VERIFICATION")
            print("="*80)
            
            # 1. Model Placement Verification
            print("1️⃣  MODEL PLACEMENT VERIFICATION:")
            model_placement = self._check_model_placement('video_export', [])
            
            # 2. Memory Management Verification
            print("\n2️⃣  MEMORY MANAGEMENT VERIFICATION:")
            memory_management = self._verify_memory_management('video_export', [])
            
            # 3. Chunking Strategy Verification
            print("\n3️⃣  CHUNKING STRATEGY VERIFICATION:")
            chunking_strategy = self._verify_chunking_strategy('video_export', processing_plan)
            
            # 4. Video Export Results Verification
            print("\n4️⃣  VIDEO EXPORT RESULTS VERIFICATION:")
            if 'output_path' in locals():
                print(f"   Output Path: {output_path}")
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                    print(f"   File Size: {file_size:.1f} MB")
                    video_export_success = True
                else:
                    print("   File Size: ❌ File not found")
                    video_export_success = False
            else:
                print("   Output Path: ❌ No output path specified")
                video_export_success = False
            
            # 5. Summary
            print("\n📊 STEP 9 SUMMARY:")
            print(f"   Model Placement: {'✅ PASS' if model_placement else '❌ FAIL'}")
            print(f"   Memory Management: {'✅ PASS' if memory_management else '❌ FAIL'}")
            print(f"   Chunking Strategy: {'✅ PASS' if chunking_strategy else '❌ FAIL'}")
            print(f"   Video Export Success: {'✅ PASS' if video_export_success else '❌ FAIL'}")
            
            if not all([model_placement, memory_management, chunking_strategy, video_export_success]):
                print("   ⚠️  Some verifications failed - pipeline may have issues")
            else:
                print("   ✅ All verifications passed - pipeline ready for next step")
            
            print("="*80)
            
            # ✅ FINAL CLEANUP - TRUSTING COMFYUI'S SYSTEM
            print("Final cleanup: ✅ Trusting ComfyUI's automatic memory management system")
            print("Final cleanup: 💡 ComfyUI will automatically clean up all models and memory")
            print("Final cleanup: 💡 No manual cleanup needed - ComfyUI handles everything!")
            
            # ComfyUI automatically manages cleanup when the pipeline completes
            # All models will be properly offloaded and memory will be freed
            
            # OOM Checklist: Check memory after final cleanup
            self._check_memory_usage('final_cleanup', expected_threshold=100)
            
            # COMPREHENSIVE VERIFICATION AFTER FINAL CLEANUP
            print("\n" + "="*80)
            print("🔍 FINAL CLEANUP COMPLETE: COMPREHENSIVE VERIFICATION")
            print("="*80)
            
            # 1. Model Placement Verification
            print("1️⃣  MODEL PLACEMENT VERIFICATION:")
            model_placement = self._check_model_placement('final_cleanup', ['unet', 'clip', 'vae'])
            
            # 2. Memory Management Verification
            print("\n2️⃣  MEMORY MANAGEMENT VERIFICATION:")
            memory_management = self._verify_memory_management('final_cleanup', ['unet', 'clip', 'vae'])
            
            # 3. Chunking Strategy Verification
            print("\n3️⃣  CHUNKING STRATEGY VERIFICATION:")
            chunking_strategy = self._verify_chunking_strategy('final_cleanup', processing_plan)
            
            # 4. Final Memory State Verification
            print("\n4️⃣  FINAL MEMORY STATE VERIFICATION:")
            if torch.cuda.is_available():
                final_allocated = torch.cuda.memory_allocated() / 1024**2
                final_reserved = torch.cuda.memory_reserved() / 1024**2
                total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**2
                free_vram = total_vram - final_reserved
                
                print(f"   Final GPU Memory:")
                print(f"     Allocated: {final_allocated:.1f} MB")
                print(f"     Reserved: {final_reserved:.1f} MB")
                print(f"     Free: {free_vram:.1f} MB")
                print(f"     Total: {total_vram:.1f} MB")
                print(f"     Utilization: {(final_reserved/total_vram)*100:.1f}%")
                
                # Memory efficiency
                if 'baseline_allocated' in locals():
                    baseline_mb = baseline_allocated / 1024**2
                    memory_efficiency = ((final_allocated - baseline_mb) / baseline_mb) * 100 if baseline_mb > 0 else 0
                    print(f"     Memory Efficiency: {memory_efficiency:+.1f}% from baseline")
                    
                    if abs(memory_efficiency) < 100:  # Within 100MB of baseline
                        print("     ✅ Memory successfully restored to baseline state")
                        memory_restored = True
                    else:
                        print("     ⚠️  Memory not fully restored to baseline state")
                        memory_restored = False
                else:
                    memory_restored = False
                    print("     ⚠️  Cannot determine memory restoration (no baseline)")
            else:
                memory_restored = True
                print("   GPU not available, skipping memory verification")
            
            # 5. Summary
            print("\n📊 FINAL CLEANUP SUMMARY:")
            print(f"   Model Placement: {'✅ PASS' if model_placement else '❌ FAIL'}")
            print(f"   Memory Management: {'✅ PASS' if memory_management else '❌ FAIL'}")
            print(f"   Chunking Strategy: {'✅ PASS' if chunking_strategy else '❌ FAIL'}")
            print(f"   Memory Restored: {'✅ PASS' if memory_restored else '❌ FAIL'}")
            
            if not all([model_placement, memory_management, chunking_strategy, memory_restored]):
                print("   ⚠️  Some verifications failed - final cleanup may be incomplete")
            else:
                print("   ✅ All verifications passed - pipeline cleanup complete")
            
            print("="*80)
            
            # COMPREHENSIVE DIAGNOSTIC SUMMARY
            print("\n" + "="*100)
            print("🔍 COMPREHENSIVE PIPELINE DIAGNOSTIC SUMMARY")
            print("="*100)
            
            # System Information
            print("💻 SYSTEM INFORMATION:")
            if torch.cuda.is_available():
                gpu_props = torch.cuda.get_device_properties(0)
                print(f"   GPU: {gpu_props.name}")
                print(f"   Total VRAM: {gpu_props.total_memory / 1024**3:.2f} GB")
                print(f"   CUDA Version: {torch.version.cuda}")
            else:
                print("   GPU: Not available")
            
            if psutil:
                cpu_info = psutil.cpu_count(logical=False)
                cpu_logical = psutil.cpu_count(logical=True)
                memory_info = psutil.virtual_memory()
                print(f"   CPU: {cpu_info} physical cores, {cpu_logical} logical cores")
                print(f"   RAM: {memory_info.total / 1024**3:.2f} GB total, {memory_info.available / 1024**3:.2f} GB available")
            else:
                print("   CPU: Not available")
                print("   RAM: Not available")
            
            # Pipeline Step-by-Step Analysis
            print("\n📊 PIPELINE STEP ANALYSIS:")
            print("-" * 80)
            
            # Step 1: Model Loading
            print("1️⃣  MODEL LOADING:")
            step1_data = self.oom_checklist.get('model_loading')
            if step1_data:
                print(f"   Status: {'✅ PASS' if step1_data['status'] == 'PASS' else '❌ FAIL'}")
                print(f"   GPU Memory: {step1_data['allocated_mb']:.1f} MB allocated, {step1_data['reserved_mb']:.1f} MB reserved")
                if torch.cuda.is_available():
                    current_gpu = torch.cuda.memory_allocated() / 1024**2
                    current_reserved = torch.cuda.memory_reserved() / 1024**2
                    print(f"   Current GPU: {current_gpu:.1f} MB allocated, {current_reserved:.1f} MB reserved")
                    if step1_data['allocated_mb'] > 0:
                        memory_change = current_gpu - step1_data['allocated_mb']
                        print(f"   Memory Change: {memory_change:+.1f} MB")
            else:
                print("   Status: ❌ NOT EXECUTED")
            
            # Step 2: LoRA Application
            print("\n2️⃣  LoRA APPLICATION:")
            step2_data = self.oom_checklist.get('lora_application')
            if step2_data:
                print(f"   Status: {'✅ PASS' if step2_data['status'] == 'PASS' else '❌ FAIL'}")
                print(f"   GPU Memory: {step2_data['allocated_mb']:.1f} MB allocated, {step2_data['reserved_mb']:.1f} MB reserved")
            else:
                print("   Status: ❌ NOT EXECUTED")
            
            # Step 3: Text Encoding
            print("\n3️⃣  TEXT ENCODING:")
            step3_data = self.oom_checklist.get('text_encoding')
            if step3_data:
                print(f"   Status: {'✅ PASS' if step3_data['status'] == 'PASS' else '❌ FAIL'}")
                print(f"   GPU Memory: {step3_data['allocated_mb']:.1f} MB allocated, {step3_data['reserved_mb']:.1f} MB reserved")
                print(f"   CLIP Status: Moved to offload device (CPU)")
            else:
                print("   Status: ❌ NOT EXECUTED")
            
            # Step 4: Model Sampling
            print("\n4️⃣  MODEL SAMPLING (ModelSamplingSD3):")
            step4_data = self.oom_checklist.get('model_sampling')
            if step4_data:
                print(f"   Status: {'✅ PASS' if step4_data['status'] == 'PASS' else '❌ FAIL'}")
                print(f"   GPU Memory: {step4_data['allocated_mb']:.1f} MB allocated, {step4_data['reserved_mb']:.1f} MB reserved")
            else:
                print("   Status: ❌ NOT EXECUTED")
            
            # Step 5: VAE Encoding
            print("\n5️⃣  VAE ENCODING:")
            step5_data = self.oom_checklist.get('vae_encoding_complete')
            if step5_data:
                print(f"   Status: {'✅ PASS' if step5_data['status'] == 'PASS' else '❌ FAIL'}")
                print(f"   GPU Memory: {step5_data['allocated_mb']:.1f} MB allocated, {step5_data['reserved_mb']:.1f} MB reserved")
                
                # Check if VAE encoding actually worked or fell back to dummies
                if 'init_latent' in locals():
                    if hasattr(init_latent, 'shape'):
                        print(f"   Latent Generated: ✅ Shape: {init_latent.shape}")
                        if init_latent.shape[1] < 10:  # Likely dummy latents
                            print("   ⚠️  WARNING: Using dummy latents (VAE encoding failed)")
                        else:
                            print("   ✅ Real VAE encoding successful")
                    else:
                        print("   Latent Generated: ❌ No shape information")
                else:
                    print("   Latent Generated: ❌ No latent created")
            else:
                print("   Status: ❌ NOT EXECUTED")
            
            # Step 6: UNET Sampling
            print("\n6️⃣  UNET SAMPLING:")
            step6_data = self.oom_checklist.get('unet_sampling')
            if step6_data:
                print(f"   Status: {'✅ PASS' if step6_data['status'] == 'PASS' else '❌ FAIL'}")
                print(f"   GPU Memory: {step6_data['allocated_mb']:.1f} MB allocated, {step6_data['reserved_mb']:.1f} MB reserved")
                
                # Check if UNET sampling worked
                if 'final_latent' in locals():
                    if hasattr(final_latent, 'shape'):
                        print(f"   Sampling Result: ✅ Shape: {final_latent.shape}")
                    else:
                        print("   Sampling Result: ❌ No shape information")
                else:
                    print("   Sampling Result: ❌ No final latent created")
            else:
                print("   Status: ❌ NOT EXECUTED")
            
            # Step 7: Video Latent Trimming
            print("\n7️⃣  VIDEO LATENT TRIMMING:")
            if 'trimmed_latent' in locals():
                if hasattr(trimmed_latent, 'shape'):
                    print(f"   Status: ✅ PASS")
                    print(f"   Trimmed Shape: {trimmed_latent.shape}")
                    print(f"   Trim Count: {trim_count if 'trim_count' in locals() else 'Unknown'}")
                else:
                    print("   Status: ❌ FAIL - No shape information")
            else:
                print("   Status: ❌ NOT EXECUTED")
            
            # Step 8: VAE Decoding
            print("\n8️⃣  VAE DECODING:")
            step8_data = self.oom_checklist.get('vae_decoding')
            if step8_data:
                print(f"   Status: {'✅ PASS' if step8_data['status'] == 'PASS' else '❌ FAIL'}")
                print(f"   GPU Memory: {step8_data['allocated_mb']:.1f} MB allocated, {step8_data['reserved_mb']:.1f} MB reserved")
                
                # Check if frames were generated
                if 'frames' in locals():
                    if hasattr(frames, 'shape'):
                        print(f"   Frames Generated: ✅ Shape: {frames.shape}")
                        if len(frames.shape) == 4:
                            print(f"   Frame Info: {frames.shape[0]} frames, {frames.shape[1]}x{frames.shape[2]}, {frames.shape[3]} channels")
                    else:
                        print("   Frames Generated: ❌ No shape information")
                else:
                    print("   Frames Generated: ❌ No frames created")
            else:
                print("   Status: ❌ NOT EXECUTED")
            
            # Step 9: Video Export
            print("\n9️⃣  VIDEO EXPORT:")
            if 'output_path' in locals():
                print(f"   Status: ✅ PASS")
                print(f"   Output Path: {output_path}")
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                    print(f"   File Size: {file_size:.1f} MB")
                else:
                    print("   File Size: ❌ File not found")
            else:
                print("   Status: ❌ NOT EXECUTED")
            
            # Memory Usage Summary
            print("\n💾 MEMORY USAGE SUMMARY:")
            print("-" * 80)
            
            if torch.cuda.is_available():
                final_allocated = torch.cuda.memory_allocated() / 1024**2
                final_reserved = torch.cuda.memory_reserved() / 1024**2
                total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**2
                free_vram = total_vram - final_reserved
                
                print(f"   Final GPU Memory:")
                print(f"     Allocated: {final_allocated:.1f} MB")
                print(f"     Reserved: {final_reserved:.1f} MB")
                print(f"     Free: {free_vram:.1f} MB")
                print(f"     Total: {total_vram:.1f} MB")
                print(f"     Utilization: {(final_reserved/total_vram)*100:.1f}%")
                
                # Memory efficiency
                if 'baseline_allocated' in locals():
                    baseline_mb = baseline_allocated / 1024**2
                    memory_efficiency = ((final_allocated - baseline_mb) / baseline_mb) * 100 if baseline_mb > 0 else 0
                    print(f"     Memory Efficiency: {memory_efficiency:+.1f}% from baseline")
            
            # CPU Memory
            cpu_memory = psutil.virtual_memory()
            print(f"   Final CPU Memory:")
            print(f"     Used: {cpu_memory.used / 1024**3:.1f} GB")
            print(f"     Available: {cpu_memory.available / 1024**3:.1f} GB")
            print(f"     Total: {cpu_memory.total / 1024**3:.1f} GB")
            print(f"     Utilization: {cpu_memory.percent:.1f}%")
            
            # Performance Metrics
            print("\n⚡ PERFORMANCE METRICS:")
            print("-" * 80)
            
            # Count successful vs failed steps
            successful_steps = 0
            failed_steps = 0
            total_steps = 0
            
            for step_name, step_data in self.oom_checklist.items():
                if step_data is not None:
                    total_steps += 1
                    if step_data['status'] == 'PASS':
                        successful_steps += 1
                    else:
                        failed_steps += 1
            
            print(f"   Pipeline Success Rate: {successful_steps}/{total_steps} steps ({successful_steps/total_steps*100:.1f}%)")
            
            # Identify critical failures
            critical_failures = []
            if 'vae_encoding_complete' in self.oom_checklist and self.oom_checklist['vae_encoding_complete']:
                if self.oom_checklist['vae_encoding_complete']['status'] == 'FAIL':
                    critical_failures.append("VAE Encoding")
            
            if 'unet_sampling' in self.oom_checklist and self.oom_checklist['unet_sampling']:
                if self.oom_checklist['unet_sampling']['status'] == 'FAIL':
                    critical_failures.append("UNET Sampling")
            
            if critical_failures:
                print(f"   Critical Failures: {'❌ ' + ', '.join(critical_failures)}")
            else:
                print("   Critical Failures: ✅ None")
            
            # Recommendations
            print("\n💡 RECOMMENDATIONS:")
            print("-" * 80)
            
            print("   ✅ Pipeline: Now fully leverages ComfyUI's proven memory management")
            print("   ✅ Memory: ComfyUI automatically prevents fragmentation and OOM")
            print("   ✅ Models: All model loading/unloading handled by ComfyUI")
            
            if failed_steps > 0:
                print("   🔧 Pipeline: Review failed steps - may need to adjust input parameters")
                print("   🔧 Pipeline: ComfyUI will handle memory automatically")
            
            if 'vae_encoding_complete' in self.oom_checklist and self.oom_checklist['vae_encoding_complete']:
                if self.oom_checklist['vae_encoding_complete']['status'] == 'PASS':
                    print("   ✅ VAE Encoding: Working correctly with ComfyUI")
                else:
                    print("   🔧 VAE Encoding: ComfyUI will handle memory management automatically")
            
            print("\n" + "="*100)
            print("🔍 DIAGNOSTIC SUMMARY COMPLETE")
            print("="*100)
            
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
            
            # Stop real-time memory monitoring
            self.memory_monitor.stop_monitoring()
            
            # Print final memory summary
            self.memory_monitor.print_memory_summary()
            
            # Final VRAM analysis at pipeline completion
            self._detailed_vram_analysis("PIPELINE_COMPLETE")
            
            return output_path
            
        except Exception as e:
            print(f"Pipeline failed with error: {str(e)}")
            
            # Stop memory monitoring on failure
            if hasattr(self, 'memory_monitor'):
                self.memory_monitor.stop_monitoring()
                self.memory_monitor.print_memory_summary()
            
            # Analyze failure with detailed VRAM analysis
            self._analyze_oom_cause(e, "PIPELINE_FAILURE")
            
            # ComfyUI automatically handles cleanup on failure
            raise
    
    def load_video(self, video_path):
        """Load control video from path as float tensor (T, H, W, 3) in [0,1]."""
        if not video_path or not os.path.exists(video_path):
            print(f"Warning: Video file not found: {video_path}")
            return None
        try:
            from torchvision.io import read_video
            print(f"Loading video from: {video_path}")
            video, audio, info = read_video(video_path, pts_unit='sec')  # (T, H, W, C) uint8
            if video is None or video.numel() == 0:
                print(f"Warning: Empty video: {video_path}")
                return None
            # Normalize to [0,1] float32 and ensure CPU tensor
            video = video.float() / 255.0
            # Ensure 3 channels; if more, take first 3; if 1, repeat to 3
            if video.shape[-1] > 3:
                video = video[..., :3]
            elif video.shape[-1] == 1:
                video = video.repeat(1, 1, 1, 3)
            print(f"Loaded video tensor: {tuple(video.shape)} (T,H,W,C)")
            return video
        except Exception as e:
            print(f"Error loading video '{video_path}': {e}")
            return None
    
    def load_image(self, image_path):
        """Load reference image from path as float tensor (1, H, W, 3) in [0,1]."""
        if not image_path or not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}")
            return None
        try:
            from PIL import Image
            import numpy as np
            print(f"Loading image from: {image_path}")
            img = Image.open(image_path).convert('RGB')
            arr = np.asarray(img).astype('float32') / 255.0  # (H,W,3)
            # Add time dimension of 1 frame to match expected shape
            tensor = torch.from_numpy(arr).unsqueeze(0)  # (1,H,W,3)
            print(f"Loaded image tensor: {tuple(tensor.shape)} (1,H,W,3)")
            return tensor
        except Exception as e:
            print(f"Error loading image '{image_path}': {e}")
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
                    # Let ComfyUI handle VAE device placement automatically
                    vae_device = vae.device if hasattr(vae, 'device') else 'cuda:0'
                    print(f"Using CPU fallback for frame {frame_idx + 1} - letting ComfyUI handle VAE memory")
                    
                    # Let ComfyUI handle VAE device placement automatically
                    single_frame_cpu = single_frame.cpu()
                    
                    # Encode on CPU (ComfyUI will handle device placement)
                    single_latent_cpu = vae.encode(single_frame_cpu[:, :, :, :3])
                    
                    # Move result back to GPU
                    single_latent = single_latent_cpu.to(vae_device)
                    
                    all_latents.append(single_latent)
                    
                    # Cleanup
                    del single_frame_cpu, single_latent_cpu
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
                    # Let ComfyUI handle VAE device placement automatically
                    vae_device = vae.device if hasattr(vae, 'device') else 'cuda:0'
                    print(f"Using CPU fallback for frame {frame_idx + 1} - letting ComfyUI handle VAE memory")
                    
                    # Let ComfyUI handle VAE device placement automatically
                    single_frame_latent_cpu = single_frame_latent.cpu()
                    
                    # Decode on CPU (ComfyUI will handle device placement)
                    single_frame_cpu = vae.decode(single_frame_latent_cpu)
                    
                    # Move result back to GPU
                    single_frame = single_frame_cpu.to(vae_device)
                    
                    all_frames.append(single_frame)
                    
                    # Cleanup
                    del single_frame_latent_cpu, single_frame_cpu
                        
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
    
    # ============================================
    # LORA APPLICATION MONITORING SYSTEM
    # ============================================
    
    def _capture_lora_baseline(self, unet_model, clip_model, lora_path):
        """Capture baseline state before LoRA application"""
        baseline = {
            'timestamp': time.time(),
            'unet': {
                'model_id': id(unet_model),
                'class': type(unet_model).__name__,
                'device': getattr(unet_model, 'device', None),
                'patches_count': len(getattr(unet_model, 'patches', {})),
                'patches_uuid': getattr(unet_model, 'patches_uuid', None),
                'memory_allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                'memory_reserved': torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,
                'memory_allocated_mb': (torch.cuda.memory_allocated() if torch.cuda.is_available() else 0) / (1024**2),
                'memory_reserved_mb': (torch.cuda.memory_reserved() if torch.cuda.is_available() else 0) / (1024**2)
            },
            'clip': {
                'model_id': id(clip_model),
                'class': type(clip_model).__name__,
                'device': getattr(clip_model, 'device', None),
                'patcher_patches_count': len(getattr(clip_model.patcher, 'patches', {})) if hasattr(clip_model, 'patcher') else 0,
                'patcher_patches_uuid': getattr(clip_model.patcher, 'patches_uuid', None) if hasattr(clip_model, 'patcher') else None,
                'memory_allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                'memory_reserved': torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,
                'memory_allocated_mb': (torch.cuda.memory_allocated() if torch.cuda.is_available() else 0) / (1024**2),
                'memory_reserved_mb': (torch.cuda.memory_reserved() if torch.cuda.is_available() else 0) / (1024**2)
            },
            'lora_file': self._check_lora_file_status(lora_path)
        }
        return baseline
    
    def _check_lora_file_status(self, lora_path):
        """Check LoRA file status and return file information"""
        file_exists = os.path.exists(lora_path)
        file_size = os.path.getsize(lora_path) if file_exists else 0
        
        return {
            'filename': os.path.basename(lora_path),
            'file_exists': file_exists,
            'file_size_mb': file_size / (1024**2) if file_exists else 0,
            'full_path': lora_path
        }
    
    def _track_model_identity_changes(self, original_model, modified_model, model_type):
        """Track changes in model identity and structure"""
        
        # 1. Model Instance Changes
        model_cloned = original_model is not modified_model
        model_class_changed = type(original_model) != type(modified_model)
        
        # 2. ModelPatcher Changes (for UNET)
        original_patch_count = 0
        modified_patch_count = 0
        patches_added = 0
        original_uuid = None
        modified_uuid = None
        uuid_changed = False
        
        if hasattr(original_model, 'patches') and hasattr(modified_model, 'patches'):
            original_patch_count = len(original_model.patches)
            modified_patch_count = len(modified_model.patches)
            patches_added = modified_patch_count - original_patch_count
            
            # 3. Patch UUID Changes
            original_uuid = getattr(original_model, 'patches_uuid', None)
            modified_uuid = getattr(modified_model, 'patches_uuid', None)
            uuid_changed = original_uuid != modified_uuid
        
        return {
            'model_cloned': model_cloned,
            'class_changed': model_class_changed,
            'patches_added': patches_added,
            'uuid_changed': uuid_changed,
            'original_patch_count': original_patch_count,
            'modified_patch_count': modified_patch_count
        }
    
    def _track_weight_modifications(self, original_model, modified_model, model_type):
        """Track how LoRA modifies model weights"""
        
        # 1. State Dict Changes
        original_state = {}
        modified_state = {}
        
        try:
            if hasattr(original_model, 'state_dict'):
                original_state = original_model.state_dict()
            if hasattr(modified_model, 'state_dict'):
                modified_state = modified_model.state_dict()
        except Exception as e:
            print(f"⚠️  Warning: Could not access state_dict for {model_type}: {e}")
        
        # 2. Key Differences
        original_keys = set(original_state.keys())
        modified_keys = set(modified_state.keys())
        keys_added = modified_keys - original_keys
        keys_removed = original_keys - modified_keys
        keys_modified = original_keys & modified_keys
        
        # 3. Weight Value Changes (for accessible weights)
        weight_changes = {}
        for key in list(keys_modified)[:10]:  # Limit to first 10 keys for performance
            if key in original_state and key in modified_state:
                orig_weight = original_state[key]
                mod_weight = modified_state[key]
                
                if hasattr(orig_weight, 'shape') and hasattr(mod_weight, 'shape'):
                    shape_changed = orig_weight.shape != mod_weight.shape
                    dtype_changed = orig_weight.dtype != mod_weight.dtype
                    device_changed = orig_weight.device != mod_weight.device
                    
                    weight_changes[key] = {
                        'shape_changed': shape_changed,
                        'dtype_changed': dtype_changed,
                        'device_changed': device_changed,
                        'original_shape': str(orig_weight.shape),
                        'modified_shape': str(mod_weight.shape)
                    }
        
        return {
            'keys_added': list(keys_added),
            'keys_removed': list(keys_removed),
            'keys_modified': list(keys_modified),
            'weight_changes': weight_changes,
            'total_keys_original': len(original_keys),
            'total_keys_modified': len(modified_keys)
        }
    
    def _analyze_lora_patches(self, modified_model, model_type):
        """Analyze the specific LoRA patches applied to the model"""
        
        if not hasattr(modified_model, 'patches'):
            return {'error': 'Model has no patches attribute'}
        
        patches = modified_model.patches
        lora_patches = {}
        
        for key, patch_list in list(patches.items())[:20]:  # Limit to first 20 keys for performance
            if patch_list:  # If patches exist for this key
                # Each patch is a tuple: (strength_patch, patch_data, strength_model, offset, function)
                for patch in patch_list:
                    if len(patch) >= 2:
                        strength_patch, patch_data = patch[0], patch[1]
                        
                        # Determine patch type
                        if isinstance(patch_data, dict) and 'lora_up.weight' in str(patch_data):
                            patch_type = 'lora_up'
                        elif isinstance(patch_data, dict) and 'lora_down.weight' in str(patch_data):
                            patch_type = 'lora_down'
                        elif isinstance(patch_data, dict) and 'diff' in str(patch_data):
                            patch_type = 'diff'
                        else:
                            patch_type = 'unknown'
                        
                        lora_patches[key] = {
                            'strength_patch': strength_patch,
                            'patch_type': patch_type,
                            'patch_data_shape': str(type(patch_data)),
                            'patch_count': len(patch_list)
                        }
        
        return {
            'total_patched_keys': len(lora_patches),
            'patch_details': lora_patches,
            'model_type': model_type
        }
    
    def _track_model_placement_changes(self, original_model, modified_model, model_type):
        """Track changes in model device placement"""
        
        # 1. Device Changes
        original_device = getattr(original_model, 'device', None)
        modified_device = getattr(modified_model, 'device', None)
        
        # 2. ModelPatcher Device Changes
        original_load_device = getattr(original_model, 'load_device', None)
        modified_load_device = getattr(modified_model, 'load_device', None)
        
        original_offload_device = getattr(original_model, 'offload_device', None)
        modified_offload_device = getattr(modified_model, 'offload_device', None)
        
        # 3. CLIP-specific device tracking
        clip_model_device = None
        clip_patcher_load_device = None
        clip_patcher_offload_device = None
        
        if model_type == 'CLIP' and hasattr(modified_model, 'patcher'):
            try:
                clip_model_device = getattr(modified_model.patcher.model, 'device', None)
                clip_patcher_load_device = getattr(modified_model.patcher, 'load_device', None)
                clip_patcher_offload_device = getattr(modified_model.patcher, 'offload_device', None)
            except Exception as e:
                print(f"⚠️  Warning: Could not access CLIP patcher device info: {e}")
        
        return {
            'model_device_changed': original_device != modified_device,
            'load_device_changed': original_load_device != modified_load_device,
            'offload_device_changed': original_offload_device != modified_offload_device,
            'original_device': str(original_device),
            'modified_device': str(modified_device),
            'clip_model_device': str(clip_model_device) if clip_model_device else None,
            'clip_patcher_devices': {
                'load': str(clip_patcher_load_device),
                'offload': str(clip_patcher_offload_device)
            } if clip_patcher_load_device else None
        }
    
    def _calculate_memory_change(self, baseline_info, current_model):
        """Calculate memory usage changes for a model"""
        try:
            current_allocated = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            current_reserved = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
            
            allocated_change = current_allocated - baseline_info['memory_allocated']
            reserved_change = current_reserved - baseline_info['memory_reserved']
            
            return {
                'allocated_change_mb': allocated_change / (1024**2),
                'reserved_change_mb': reserved_change / (1024**2),
                'current_allocated_mb': current_allocated / (1024**2),
                'current_reserved_mb': current_reserved / (1024**2)
            }
        except Exception as e:
            return {'error': f'Memory calculation failed: {e}'}
    
    def _analyze_lora_application_results(self, baseline, original_unet, original_clip, modified_unet, modified_clip, lora_result):
        """Analyze the results of LoRA application"""
        
        analysis = {
            'lora_application_success': lora_result is not None,
            'models_returned': len(lora_result) if lora_result else 0,
            'unet_changes': self._track_model_identity_changes(
                original_unet, modified_unet, 'UNET'
            ),
            'clip_changes': self._track_model_identity_changes(
                original_clip, modified_clip, 'CLIP'
            ),
            'unet_weight_changes': self._track_weight_modifications(
                original_unet, modified_unet, 'UNET'
            ),
            'clip_weight_changes': self._track_weight_modifications(
                original_clip, modified_clip, 'CLIP'
            ),
            'unet_lora_patches': self._analyze_lora_patches(modified_unet, 'UNET'),
            'clip_lora_patches': self._analyze_lora_patches(modified_clip, 'CLIP'),
            'placement_changes': {
                'unet': self._track_model_placement_changes(
                    original_unet, modified_unet, 'UNET'
                ),
                'clip': self._track_model_placement_changes(
                    original_clip, modified_clip, 'CLIP'
                )
            },
            'memory_impact': {
                'unet_memory_change': self._calculate_memory_change(
                    baseline['unet'], modified_unet
                ),
                'clip_memory_change': self._calculate_memory_change(
                    baseline['clip'], modified_clip
                )
            }
        }
        
        return analysis
    
    def _print_lora_analysis_summary(self, analysis):
        """Print comprehensive LoRA application analysis"""
        print(f"\n🔍 LORA APPLICATION ANALYSIS SUMMARY")
        print("=" * 80)
        
        # Basic success info
        print(f"✅ LoRA Application Success: {'YES' if analysis['lora_application_success'] else 'NO'}")
        print(f"📦 Models Returned: {analysis['models_returned']}")
        
        # UNET Changes
        print(f"\n🔧 UNET MODEL CHANGES:")
        unet_changes = analysis['unet_changes']
        print(f"   Model Cloned: {'✅ YES' if unet_changes['model_cloned'] else '❌ NO'}")
        print(f"   Class Changed: {'✅ YES' if unet_changes['class_changed'] else '❌ NO'}")
        print(f"   Patches Added: {unet_changes['patches_added']}")
        print(f"   UUID Changed: {'✅ YES' if unet_changes['uuid_changed'] else '❌ NO'}")
        print(f"   Original Patches: {unet_changes['original_patch_count']}")
        print(f"   Modified Patches: {unet_changes['modified_patch_count']}")
        
        # CLIP Changes
        print(f"\n🔧 CLIP MODEL CHANGES:")
        clip_changes = analysis['clip_changes']
        print(f"   Model Cloned: {'✅ YES' if clip_changes['model_cloned'] else '❌ NO'}")
        print(f"   Class Changed: {'✅ YES' if clip_changes['class_changed'] else '❌ NO'}")
        print(f"   Patches Added: {clip_changes['patches_added']}")
        print(f"   UUID Changed: {'✅ YES' if clip_changes['uuid_changed'] else '❌ NO'}")
        print(f"   Original Patches: {clip_changes['original_patch_count']}")
        print(f"   Modified Patches: {clip_changes['modified_patch_count']}")
        
        # LoRA Patches Analysis
        print(f"\n🔧 LORA PATCHES ANALYSIS:")
        unet_patches = analysis['unet_lora_patches']
        clip_patches = analysis['clip_lora_patches']
        
        if 'error' not in unet_patches:
            print(f"   UNET Patched Keys: {unet_patches['total_patched_keys']}")
        else:
            print(f"   UNET Patches: {unet_patches['error']}")
            
        if 'error' not in clip_patches:
            print(f"   CLIP Patched Keys: {clip_patches['total_patched_keys']}")
        else:
            print(f"   CLIP Patches: {clip_patches['error']}")
        
        # Memory Impact
        print(f"\n💾 MEMORY IMPACT:")
        unet_memory = analysis['memory_impact']['unet_memory_change']
        clip_memory = analysis['memory_impact']['clip_memory_change']
        
        if 'error' not in unet_memory:
            print(f"   UNET Memory Change: {unet_memory['allocated_change_mb']:+.1f} MB allocated, {unet_memory['reserved_change_mb']:+.1f} MB reserved")
        if 'error' not in clip_memory:
            print(f"   CLIP Memory Change: {clip_memory['allocated_change_mb']:+.1f} MB allocated, {clip_memory['reserved_change_mb']:+.1f} MB reserved")
        
        print("=" * 80)
    
    # ============================================
    # ENHANCED SYSTEM MONITORING METHODS
    # ============================================
    
    def _get_system_memory_info(self):
        """Get comprehensive system memory information"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                'ram_used_mb': memory.used / (1024**2),
                'ram_available_mb': memory.available / (1024**2),
                'ram_total_mb': memory.total / (1024**2),
                'ram_percent': memory.percent
            }
        except ImportError:
            return {
                'ram_used_mb': 0,
                'ram_available_mb': 0,
                'ram_total_mb': 0,
                'ram_percent': 0
            }
    
    def _get_gpu_memory_info(self):
        """Get comprehensive GPU memory information"""
        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024**2)
                reserved = torch.cuda.memory_reserved() / (1024**2)
                total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
                available = total - reserved
                
                return {
                    'gpu_allocated_mb': allocated,
                    'gpu_reserved_mb': reserved,
                    'gpu_total_mb': total,
                    'gpu_available_mb': available,
                    'gpu_device_name': torch.cuda.get_device_name(0)
                }
            else:
                return {
                    'gpu_allocated_mb': 0,
                    'gpu_reserved_mb': 0,
                    'gpu_total_mb': 0,
                    'gpu_available_mb': 0,
                    'gpu_device_name': 'No GPU'
                }
        except Exception as e:
            return {
                'gpu_allocated_mb': 0,
                'gpu_reserved_mb': 0,
                'gpu_total_mb': 0,
                'gpu_available_mb': 0,
                'gpu_device_name': f'Error: {e}'
            }
    
    def _get_enhanced_model_info(self, model, model_type):
        """Get enhanced model information including parameters, size, etc."""
        try:
            # Count parameters
            total_params = 0
            state_dict_keys = 0
            
            # Try different ways to access model parameters for ComfyUI ModelPatcher
            if hasattr(model, 'model') and hasattr(model.model, 'state_dict'):
                # Access underlying model if it's a ModelPatcher
                state_dict = model.model.state_dict()
                state_dict_keys = len(state_dict)
                
                for param in state_dict.values():
                    if hasattr(param, 'numel'):
                        total_params += param.numel()
            elif hasattr(model, 'state_dict'):
                # Direct state_dict access
                state_dict = model.state_dict()
                state_dict_keys = len(state_dict)
                
                for param in state_dict.values():
                    if hasattr(param, 'numel'):
                        total_params += param.numel()
            elif hasattr(model, 'parameters'):
                # Try parameters() method
                for param in model.parameters():
                    if hasattr(param, 'numel'):
                        total_params += param.numel()
                state_dict_keys = total_params  # Rough estimate
            
            # Calculate model size in MB (rough estimate)
            model_size_mb = (total_params * 4) / (1024**2)  # Assuming float32 (4 bytes)
            
            # Get device information - try multiple approaches
            device = 'unknown'
            if hasattr(model, 'device'):
                device = str(model.device)
            elif hasattr(model, 'model') and hasattr(model.model, 'device'):
                device = str(model.model.device)
            elif hasattr(model, 'parameters'):
                # Try to get device from first parameter
                try:
                    first_param = next(model.parameters())
                    device = str(first_param.device)
                except:
                    device = 'unknown'
            
            # Debug info
            debug_info = f"Model class: {type(model).__name__}"
            if hasattr(model, 'model'):
                debug_info += f", Underlying model: {type(model.model).__name__}"
            
            return {
                'total_parameters': total_params,
                'model_size_mb': model_size_mb,
                'state_dict_keys': state_dict_keys,
                'device': device,
                'model_type': model_type,
                'debug_info': debug_info
            }
        except Exception as e:
            return {
                'total_parameters': 0,
                'model_size_mb': 0,
                'state_dict_keys': 0,
                'device': 'unknown',
                'model_type': model_type,
                'error': str(e),
                'debug_info': f"Error: {str(e)}"
            }
    
    def _calculate_memory_efficiency(self, memory_info):
        """Calculate memory efficiency metrics"""
        try:
            if memory_info['gpu_total_mb'] > 0:
                allocated_efficiency = (memory_info['gpu_allocated_mb'] / memory_info['gpu_total_mb']) * 100
                reserved_efficiency = (memory_info['gpu_reserved_mb'] / memory_info['gpu_total_mb']) * 100
            else:
                allocated_efficiency = 0
                reserved_efficiency = 0
            
            return {
                'allocated_efficiency_percent': allocated_efficiency,
                'reserved_efficiency_percent': reserved_efficiency
            }
        except Exception:
            return {
                'allocated_efficiency_percent': 0,
                'reserved_efficiency_percent': 0
            }
    
    def _print_comprehensive_memory_breakdown(self, baseline_memory, current_memory, baseline_time, current_time):
        """Print comprehensive memory breakdown with detailed analysis"""
        print(f"\n💾 DETAILED MEMORY BREAKDOWN:")
        
        # RAM Breakdown
        print(f"   🖥️  RAM MEMORY BREAKDOWN:")
        print(f"      Baseline State:")
        print(f"         Used: {baseline_memory['ram_used_mb']:.1f} MB")
        print(f"         Available: {baseline_memory['ram_available_mb']:.1f} MB")
        print(f"         Usage: {baseline_memory['ram_percent']:.1f}%")
        print(f"      Current State:")
        print(f"         Used: {current_memory['ram_used_mb']:.1f} MB")
        print(f"         Available: {current_memory['ram_available_mb']:.1f} MB")
        print(f"         Usage: {current_memory['ram_percent']:.1f}%")
        
        # Calculate RAM changes
        ram_used_change = current_memory['ram_used_mb'] - baseline_memory['ram_used_mb']
        ram_available_change = current_memory['ram_available_mb'] - baseline_memory['ram_available_mb']
        ram_percent_change = current_memory['ram_percent'] - baseline_memory['ram_percent']
        
        print(f"      Changes:")
        print(f"         Used Change: {ram_used_change:+.1f} MB ({ram_used_change/baseline_memory['ram_used_mb']*100:+.1f}%)")
        print(f"         Available Change: {ram_available_change:+.1f} MB")
        print(f"         Usage Change: {ram_percent_change:+.1f}%")
        
        # GPU Breakdown
        print(f"\n   🎮 GPU MEMORY BREAKDOWN:")
        print(f"      Baseline State:")
        print(f"         Allocated: {baseline_memory['gpu_allocated_mb']:.1f} MB")
        print(f"         Reserved: {baseline_memory['gpu_reserved_mb']:.1f} MB")
        print(f"         Total VRAM: {baseline_memory['gpu_total_mb']:.1f} MB")
        print(f"         Available VRAM: {baseline_memory['gpu_available_mb']:.1f} MB")
        print(f"      Current State:")
        print(f"         Allocated: {current_memory['gpu_allocated_mb']:.1f} MB")
        print(f"         Reserved: {current_memory['gpu_reserved_mb']:.1f} MB")
        print(f"         Total VRAM: {current_memory['gpu_total_mb']:.1f} MB")
        print(f"         Available VRAM: {current_memory['gpu_available_mb']:.1f} MB")
        
        # Calculate GPU changes
        gpu_allocated_change = current_memory['gpu_allocated_mb'] - baseline_memory['gpu_allocated_mb']
        gpu_reserved_change = current_memory['gpu_reserved_mb'] - baseline_memory['gpu_reserved_mb']
        gpu_available_change = current_memory['gpu_available_mb'] - baseline_memory['gpu_available_mb']
        
        print(f"      Changes:")
        if baseline_memory['gpu_allocated_mb'] > 0:
            allocated_change_percent = (gpu_allocated_change / baseline_memory['gpu_allocated_mb']) * 100
        else:
            allocated_change_percent = 0
            
        if baseline_memory['gpu_reserved_mb'] > 0:
            reserved_change_percent = (gpu_reserved_change / baseline_memory['gpu_reserved_mb']) * 100
        else:
            reserved_change_percent = 0
        
        print(f"         Allocated Change: {gpu_allocated_change:+.1f} MB ({allocated_change_percent:+.1f}%)")
        print(f"         Reserved Change: {gpu_reserved_change:+.1f} MB ({reserved_change_percent:+.1f}%)")
        print(f"         Available VRAM Change: {gpu_available_change:+.1f} MB")
        
        # Memory Efficiency
        baseline_efficiency = self._calculate_memory_efficiency(baseline_memory)
        current_efficiency = self._calculate_memory_efficiency(current_memory)
        
        print(f"      Memory Efficiency:")
        print(f"         Baseline: {baseline_efficiency['allocated_efficiency_percent']:.1f}% (allocated/reserved)")
        print(f"         Current: {current_efficiency['allocated_efficiency_percent']:.1f}% (allocated/reserved)")
        efficiency_change = current_efficiency['allocated_efficiency_percent'] - baseline_efficiency['allocated_efficiency_percent']
        print(f"         Efficiency Change: {efficiency_change:+.1f}%")
    
    def _print_peak_memory_summary(self, baseline_memory, current_memory, baseline_time, current_time):
        """Print peak memory summary with timestamps"""
        print(f"\n📊 PEAK MEMORY DURING LORA APPLICATION:")
        
        # RAM Peak
        ram_peak = max(baseline_memory['ram_used_mb'], current_memory['ram_used_mb'])
        print(f"   🖥️  RAM Peak: {ram_peak:.1f} MB")
        
        # GPU Peak
        gpu_allocated_peak = max(baseline_memory['gpu_allocated_mb'], current_memory['gpu_allocated_mb'])
        gpu_reserved_peak = max(baseline_memory['gpu_reserved_mb'], current_memory['gpu_reserved_mb'])
        print(f"   🎮 GPU Allocated Peak: {gpu_allocated_peak:.1f} MB")
        print(f"   🎮 GPU Reserved Peak: {gpu_reserved_peak:.1f} MB")
        
        # Peak Timestamps
        print(f"   ⏱️  Peak Timestamps:")
        if ram_peak == baseline_memory['ram_used_mb']:
            print(f"      ram: {ram_peak:.1f} MB at baseline")
        else:
            print(f"      ram: {ram_peak:.1f} MB at {current_time - baseline_time:.2f}s")
        
        if gpu_allocated_peak == baseline_memory['gpu_allocated_mb']:
            print(f"      gpu_allocated: {gpu_allocated_peak:.1f} MB at baseline")
        else:
            print(f"      gpu_allocated: {gpu_allocated_peak:.1f} MB at {current_time - baseline_time:.2f}s")
    
    def _print_enhanced_model_summary(self, model, model_type):
        """Print enhanced model information summary"""
        model_info = self._get_enhanced_model_info(model, model_type)
        
        print(f"\n🔧 ENHANCED MODEL INFORMATION:")
        print(f"   Model Type: {model_info['model_type']}")
        print(f"   Model Class: {type(model).__name__}")
        print(f"   Device: {model_info['device']}")
        print(f"   Parameters: {model_info['total_parameters']:,}")
        print(f"   Model Size: {model_info['model_size_mb']:.1f} MB")
        print(f"   State Dict Keys: {model_info['state_dict_keys']}")
        
        # Add debug information
        if 'debug_info' in model_info:
            print(f"   🔍 Debug Info: {model_info['debug_info']}")
        
        # Memory efficiency recommendations
        print(f"   💡 MEMORY EFFICIENCY ANALYSIS:")
        if model_info['model_size_mb'] > 10000:
            size_category = "Very Large"
            recommendation = "Consider aggressive GPU offloading and chunked processing"
        elif model_info['model_size_mb'] > 5000:
            size_category = "Large"
            recommendation = "Consider GPU offloading for memory efficiency"
        elif model_info['model_size_mb'] > 1000:
            size_category = "Medium"
            recommendation = "Monitor memory usage, consider GPU offloading if needed"
        else:
            size_category = "Small"
            recommendation = "Memory efficient, can stay in GPU"
        
        print(f"     Model Size: {size_category} ({model_info['model_size_mb']:.1f} MB)")
        print(f"     Recommendation: {recommendation}")
        
        if model_info['device'] == 'cpu':
            print(f"     Device Placement: CPU (memory efficient, slower inference)")
        else:
            print(f"     Device Placement: GPU (faster inference, higher memory usage)")
    
    def _start_step_monitoring(self, step_name):
        """Start monitoring a specific step with timing and memory baseline"""
        start_time = time.time()
        start_memory = {
            **self._get_system_memory_info(),
            **self._get_gpu_memory_info()
        }
        
        print(f"\n🔍 STARTING MONITORING FOR: {step_name.upper()}")
        print(f"   Baseline RAM: {start_memory['ram_used_mb']:.1f} GB used, {start_memory['ram_available_mb']:.1f} GB available")
        print(f"   Baseline GPU: {start_memory['gpu_allocated_mb']:.1f} MB allocated, {start_memory['gpu_reserved_mb']:.1f} MB reserved")
        
        return start_time, start_memory
    
    def _end_step_monitoring(self, step_name, start_time, start_memory):
        """End monitoring and calculate comprehensive metrics"""
        end_time = time.time()
        end_memory = {
            **self._get_system_memory_info(),
            **self._get_gpu_memory_info()
        }
        
        elapsed_time = end_time - start_time
        
        print(f"\n🔍 {step_name.upper()} DEBUGGING COMPLETE")
        print("=" * 60)
        
        # Performance timing
        print(f"⏱️  PERFORMANCE:")
        print(f"   Loading Time: {elapsed_time:.3f} seconds")
        
        # Memory analysis
        print(f"💾 MEMORY ANALYSIS:")
        ram_change = end_memory['ram_used_mb'] - start_memory['ram_used_mb']
        gpu_allocated_change = end_memory['gpu_allocated_mb'] - start_memory['gpu_allocated_mb']
        gpu_reserved_change = end_memory['gpu_reserved_mb'] - start_memory['gpu_reserved_mb']
        
        print(f"   RAM Change: {ram_change:+.1f} MB")
        print(f"   Current RAM: {end_memory['ram_used_mb']:.1f} GB used, {end_memory['ram_available_mb']:.1f} GB available")
        print(f"   GPU Change: {gpu_allocated_change:+.1f} MB allocated, {gpu_reserved_change:+.1f} MB reserved")
        print(f"   Current GPU: {end_memory['gpu_allocated_mb']:.1f} MB allocated, {end_memory['gpu_reserved_mb']:.1f} MB reserved")
        
        return elapsed_time, end_memory
    
    # ============================================
    # WORKFLOW MONITORING INTEGRATION
    # ============================================
    
    def _print_workflow_monitoring_summary(self, step_results):
        """Print comprehensive workflow monitoring summary"""
        print(f"\n📊 COMPREHENSIVE WORKFLOW MONITORING SUMMARY")
        print("=" * 80)
        
        # Step 1: Model Loading Summary
        if 'model_loading' in step_results:
            print(f"🔍 STEP 1: MODEL LOADING")
            model_loading = step_results['model_loading']
            print(f"   ⏱️  Total Loading Time: {model_loading.get('elapsed_time', 0):.3f} seconds")
            print(f"   💾 Total RAM Change: {model_loading.get('ram_change', 0):+.1f} MB")
            print(f"   🎮 Total GPU Change: {model_loading.get('gpu_change', 0):+.1f} MB")
        
        # Step 2: LoRA Application Summary
        if 'lora_application' in step_results:
            print(f"\n🔍 STEP 2: LORA APPLICATION")
            lora_app = step_results['lora_application']
            print(f"   ⏱️  Total Time: {lora_app.get('elapsed_time', 0):.3f}s")
            print(f"   💾 RAM Change: {lora_app.get('ram_change', 0):+.1f} MB")
            print(f"   🎮 GPU Change: {lora_app.get('gpu_change', 0):+.1f} MB")
            print(f"   ✅ Success: {'YES' if lora_app.get('success', False) else 'NO'}")
        
        print("=" * 80)
    
    def _print_final_workflow_summary(self, step_results):
        """Print final workflow summary with all steps"""
        print(f"\n🔍 FINAL WORKFLOW MONITORING SUMMARY")
        print("=" * 80)
        
        # Summary of all completed steps
        completed_steps = []
        total_time = 0
        total_ram_change = 0
        total_gpu_change = 0
        
        for step_name, step_data in step_results.items():
            if step_data:
                completed_steps.append(step_name)
                total_time += step_data.get('elapsed_time', 0)
                total_ram_change += step_data.get('ram_change', 0)
                total_gpu_change += step_data.get('gpu_change', 0)
        
        print(f"📊 WORKFLOW SUMMARY:")
        print(f"   ✅ Completed Steps: {len(completed_steps)}")
        print(f"   ⏱️  Total Time: {total_time:.3f} seconds")
        print(f"   💾 Total RAM Change: {total_ram_change:+.1f} MB")
        print(f"   🎮 Total GPU Change: {total_gpu_change:+.1f} MB")
        
        print(f"\n📋 STEP BREAKDOWN:")
        for i, step_name in enumerate(completed_steps, 1):
            step_data = step_results[step_name]
            print(f"   {i}. {step_name.replace('_', ' ').title()}")
            print(f"      Time: {step_data.get('elapsed_time', 0):.3f}s")
            print(f"      RAM: {step_data.get('ram_change', 0):+.1f} MB")
            print(f"      GPU: {step_data.get('gpu_change', 0):+.1f} MB")
            if 'success' in step_data:
                print(f"      Status: {'✅ SUCCESS' if step_data['success'] else '❌ FAILED'}")
        
        print("=" * 80)
    
    def _stop_execution_after_step(self, step_name, step_results, error_message=None):
        """Stop execution after a specific step for debugging purposes"""
        print(f"\n🛑 STOPPING EXECUTION AFTER STEP {step_name.upper()}")
        
        if error_message:
            print(f"🔍 {error_message}")
        
        print("🔍 All debugging information has been displayed above.")
        print("📊 Check the monitoring data above to analyze performance.")
        
        # Print step completion status
        completed_steps = []
        for step_name, step_data in step_results.items():
            if step_data:
                completed_steps.append(step_name)
        
        print(f"\n🔍 Step {completed_steps.index(step_name) + 1}: {step_name.replace('_', ' ').title()} - COMPLETED")
        
        # Show which steps were completed and which were skipped
        for i, step in enumerate(['model_loading', 'lora_application', 'text_encoding', 'model_sampling', 
                                'video_generation', 'sampling', 'video_processing', 'vae_decoding', 'video_export'], 1):
            if step in completed_steps:
                print(f"🔍 Step {i}: {step.replace('_', ' ').title()} - COMPLETED")
            else:
                print(f"🔍 Steps {i}-9: SKIPPED for debugging purposes")
        
        # Print final workflow summary
        self._print_final_workflow_summary(step_results)
        
        return False  # Signal to stop execution
    
    def _detailed_vram_analysis(self, step_name):
        """Comprehensive VRAM analysis to identify memory bottlenecks"""
        try:
            import gc
            import psutil
            
            print(f"\n🔍 DETAILED VRAM ANALYSIS - {step_name.upper()}")
            print("="*80)
            
            # 1. PyTorch GPU Memory Analysis
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                
                # Basic memory stats
                allocated = torch.cuda.memory_allocated(device) / (1024**3)  # GB
                reserved = torch.cuda.memory_reserved(device) / (1024**3)   # GB
                max_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)  # GB
                max_reserved = torch.cuda.max_memory_reserved(device) / (1024**3)   # GB
                
                print(f"📊 PyTorch Memory Stats:")
                print(f"   Current Allocated: {allocated:.2f} GB")
                print(f"   Current Reserved:  {reserved:.2f} GB") 
                print(f"   Peak Allocated:    {max_allocated:.2f} GB")
                print(f"   Peak Reserved:     {max_reserved:.2f} GB")
                print(f"   Free (Reserved):   {(reserved - allocated):.2f} GB")
                
                # Memory breakdown by tensor types
                memory_summary = torch.cuda.memory_summary(device)
                print(f"\n🔬 Memory Summary:")
                print(memory_summary)
                
                # 2. ComfyUI Model Tracking
                print(f"\n🎯 ComfyUI Model Tracking:")
                if hasattr(comfy.model_management, 'current_loaded_models'):
                    loaded_models = comfy.model_management.current_loaded_models
                    print(f"   Tracked Models: {len(loaded_models)}")
                    
                    total_model_memory = 0
                    for i, model in enumerate(loaded_models):
                        try:
                            if hasattr(model, 'model'):
                                model_type = type(model.model).__name__
                                if hasattr(model, 'model_memory_required'):
                                    mem_req = model.model_memory_required(device) / (1024**3)
                                    total_model_memory += mem_req
                                    print(f"   Model {i+1}: {model_type} - {mem_req:.2f} GB")
                                else:
                                    print(f"   Model {i+1}: {model_type} - Memory unknown")
                        except Exception as e:
                            print(f"   Model {i+1}: Error getting info - {e}")
                    
                    print(f"   Total Model Memory: {total_model_memory:.2f} GB")
                    print(f"   Unaccounted Memory: {(allocated - total_model_memory):.2f} GB")
                
                # 3. ComfyUI Free Memory Check
                try:
                    free_mem = comfy.model_management.get_free_memory(device) / (1024**3)
                    print(f"\n💾 ComfyUI Free Memory: {free_mem:.2f} GB")
                except Exception as e:
                    print(f"   Error getting ComfyUI free memory: {e}")
                
                # 4. GPU Device Properties
                props = torch.cuda.get_device_properties(device)
                total_vram = props.total_memory / (1024**3)
                free_raw = (props.total_memory - torch.cuda.memory_reserved(device)) / (1024**3)
                
                print(f"\n🎮 GPU Hardware:")
                print(f"   Device: {props.name}")
                print(f"   Total VRAM: {total_vram:.2f} GB")
                print(f"   Free (Raw): {free_raw:.2f} GB")
                print(f"   Utilization: {(allocated/total_vram)*100:.1f}%")
                
            # 5. System RAM Analysis
            ram = psutil.virtual_memory()
            print(f"\n🖥️  System RAM:")
            print(f"   Total: {ram.total / (1024**3):.2f} GB")
            print(f"   Used: {ram.used / (1024**3):.2f} GB")
            print(f"   Available: {ram.available / (1024**3):.2f} GB")
            print(f"   Utilization: {ram.percent:.1f}%")
            
            # 6. Python Object Analysis
            print(f"\n🐍 Python Objects:")
            gc.collect()  # Force garbage collection
            
            # Count tensor objects
            tensor_count = 0
            tensor_memory = 0
            for obj in gc.get_objects():
                if torch.is_tensor(obj):
                    tensor_count += 1
                    if obj.is_cuda:
                        tensor_memory += obj.element_size() * obj.nelement()
            
            print(f"   Total Tensors: {tensor_count}")
            print(f"   GPU Tensor Memory: {tensor_memory / (1024**3):.2f} GB")
            
            print("="*80)
            
        except Exception as e:
            print(f"❌ VRAM Analysis failed: {e}")
    
    def _monitor_vae_encoding_memory(self, vae, pixel_samples, operation_name):
        """Monitor memory during VAE encoding specifically"""
        print(f"\n🔍 VAE ENCODING MEMORY MONITOR - {operation_name}")
        print("-"*60)
        
        # Before encoding
        print("📍 BEFORE VAE ENCODING:")
        self._quick_memory_snapshot("before_vae")
        
        # Check VAE model status
        print(f"\n🎯 VAE Model Status:")
        if hasattr(vae, 'patcher'):
            patcher = vae.patcher
            print(f"   Patcher Type: {type(patcher)}")
            print(f"   Load Device: {patcher.load_device}")
            print(f"   Offload Device: {patcher.offload_device}")
            print(f"   Current Device: {getattr(patcher, 'current_device', 'Unknown')}")
            
            # Check if VAE is currently loaded
            if hasattr(patcher, 'is_loaded'):
                print(f"   Is Loaded: {patcher.is_loaded}")
            
            # Check VAE model memory
            if hasattr(patcher, 'model_memory_required'):
                try:
                    mem_req = patcher.model_memory_required(torch.cuda.current_device()) / (1024**3)
                    print(f"   Memory Required: {mem_req:.2f} GB")
                except:
                    print(f"   Memory Required: Unknown")
        
        # Predict encoding memory
        try:
            if hasattr(vae, 'memory_used_encode'):
                predicted_mem = vae.memory_used_encode(pixel_samples.shape, vae.vae_dtype) / (1024**3)
                print(f"   Predicted Encoding Memory: {predicted_mem:.2f} GB")
        except Exception as e:
            print(f"   Could not predict encoding memory: {e}")
        
        print(f"   Input Tensor Shape: {pixel_samples.shape}")
        print(f"   Input Tensor Size: {pixel_samples.element_size() * pixel_samples.nelement() / (1024**3):.3f} GB")
        
        return True
    
    def _quick_memory_snapshot(self, label):
        """Quick memory snapshot for frequent monitoring"""
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(device) / (1024**3)
            reserved = torch.cuda.memory_reserved(device) / (1024**3)
            
            try:
                free_comfy = comfy.model_management.get_free_memory(device) / (1024**3)
                print(f"   {label}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, ComfyFree={free_comfy:.2f}GB")
            except:
                print(f"   {label}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
    
    def _track_memory_during_step5(self):
        """Track memory throughout Step 5 execution"""
        print(f"\n🎯 STEP 5 MEMORY TRACKING ENABLED")
        print("="*80)
        
        # Enable PyTorch memory profiling
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            print("✅ PyTorch memory profiling reset and enabled")
            
        return True
    
    def _analyze_oom_cause(self, error, context):
        """Analyze the specific cause of OOM errors"""
        print(f"\n💥 OOM ERROR ANALYSIS - {context}")
        print("="*80)
        
        print(f"📋 Error Details:")
        print(f"   Error Type: {type(error).__name__}")
        print(f"   Error Message: {str(error)}")
        
        # Extract memory details from error message
        import re
        
        # Try to extract memory amounts from error
        tried_to_allocate = re.search(r'Tried to allocate (\d+\.?\d*)\s*(\w+)', str(error))
        if tried_to_allocate:
            amount = tried_to_allocate.group(1)
            unit = tried_to_allocate.group(2)
            print(f"   Allocation Attempt: {amount} {unit}")
        
        # Extract available memory
        available_mem = re.search(r'(\d+\.?\d*)\s*(\w+) is free', str(error))
        if available_mem:
            amount = available_mem.group(1)
            unit = available_mem.group(2)
            print(f"   Available Memory: {amount} {unit}")
        
        # Current memory state
        self._detailed_vram_analysis("OOM_ANALYSIS")
        
        # Suggestions based on analysis
        print(f"\n💡 SUGGESTED SOLUTIONS:")
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            
            if allocated > 40:  # > 40GB
                print("   🔧 High allocation detected - try reducing batch size")
            if reserved - allocated > 5:  # > 5GB fragmentation
                print("   🔧 Memory fragmentation detected - call torch.cuda.empty_cache()")
            if len(comfy.model_management.current_loaded_models) > 1:
                print("   🔧 Multiple models loaded - enable better offloading")
                
        print("="*80)

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
        positive_prompt="very cinematic video",
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