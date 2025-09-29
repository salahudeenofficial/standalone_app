# LORA APPLICATION MONITORING CODE TEMPLATE
# ============================================
# 
# This file contains the complete code structure for all monitoring methods
# currently implemented in workflow_api.py. Use this as a template to replicate
# the monitoring system in your pipeline.py file.
#
# Copy the methods you need and adapt them to your pipeline structure.

import os
import time
import torch
from typing import Dict, Any, Union

class LoRAMonitor:
    """Comprehensive monitoring for LoRA application steps"""
    
    def __init__(self):
        self.monitoring_data = {}
        self.step_start_time = None
    
    def start_monitoring(self, step_name):
        """Start monitoring a specific step"""
        self.step_start_time = time.time()
        print(f"\n🔍 STARTING MONITORING FOR: {step_name.upper()}")
    
    def end_monitoring(self, step_name):
        """End monitoring and calculate elapsed time"""
        if self.step_start_time:
            elapsed_time = time.time() - self.step_start_time
            print(f"⏱️  {step_name.upper()} completed in {elapsed_time:.3f} seconds")
            return elapsed_time
        return 0.0
    
    # ============================================
    # 1. BASELINE CAPTURE METHODS
    # ============================================
    
    def capture_lora_baseline(self, unet_model, clip_model):
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
                'memory_reserved': torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
            },
            'clip': {
                'model_id': id(clip_model),
                'class': type(clip_model).__name__,
                'device': getattr(clip_model, 'device', None),
                'patcher_patches_count': len(getattr(clip_model.patcher, 'patches', {})) if hasattr(clip_model, 'patcher') else 0,
                'patcher_patches_uuid': getattr(clip_model.patcher, 'patches_uuid', None) if hasattr(clip_model, 'patcher') else None,
                'memory_allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                'memory_reserved': torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
            }
        }
        return baseline
    
    def check_lora_file_status(self, lora_filename):
        """Check LoRA file status and return file information"""
        file_exists = os.path.exists(lora_filename)
        file_size = os.path.getsize(lora_filename) if file_exists else 0
        
        return {
            'filename': lora_filename,
            'file_exists': file_exists,
            'file_size_mb': file_size / (1024**2) if file_exists else 0
        }
    
    # ============================================
    # 2. MODEL IDENTITY CHANGES TRACKING
    # ============================================
    
    def track_model_identity_changes(self, original_model, modified_model, model_type):
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
    
    # ============================================
    # 3. WEIGHT MODIFICATIONS TRACKING
    # ============================================
    
    def track_weight_modifications(self, original_model, modified_model, model_type):
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
        for key in keys_modified:
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
    
    # ============================================
    # 4. LoRA PATCHES ANALYSIS
    # ============================================
    
    def analyze_lora_patches(self, modified_model, model_type):
        """Analyze the specific LoRA patches applied to the model"""
        
        if not hasattr(modified_model, 'patches'):
            return {'error': 'Model has no patches attribute'}
        
        patches = modified_model.patches
        lora_patches = {}
        
        for key, patch_list in patches.items():
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
    
    # ============================================
    # 5. MODEL PLACEMENT CHANGES TRACKING
    # ============================================
    
    def track_model_placement_changes(self, original_model, modified_model, model_type):
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
    
    # ============================================
    # 6. MEMORY IMPACT TRACKING
    # ============================================
    
    def calculate_memory_change(self, baseline_info, current_model):
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
    
    # ============================================
    # 7. COMPREHENSIVE ANALYSIS
    # ============================================
    
    def analyze_lora_application_results(self, baseline, modified_unet, modified_clip, lora_result):
        """Analyze the results of LoRA application"""
        
        analysis = {
            'lora_application_success': lora_result is not None,
            'models_returned': len(lora_result) if lora_result else 0,
            'unet_changes': self.track_model_identity_changes(
                baseline['unet'], modified_unet, 'UNET'
            ),
            'clip_changes': self.track_model_identity_changes(
                baseline['clip'], modified_clip, 'CLIP'
            ),
            'unet_weight_changes': self.track_weight_modifications(
                baseline['unet'], modified_clip, 'UNET'
            ),
            'clip_weight_changes': self.track_weight_modifications(
                baseline['clip'], modified_clip, 'CLIP'
            ),
            'unet_lora_patches': self.analyze_lora_patches(modified_unet, 'UNET'),
            'clip_lora_patches': self.analyze_lora_patches(modified_clip, 'CLIP'),
            'placement_changes': {
                'unet': self.track_model_placement_changes(
                    baseline['unet'], modified_unet, 'UNET'
                ),
                'clip': self.track_model_placement_changes(
                    baseline['clip'], modified_clip, 'CLIP'
                )
            },
            'memory_impact': {
                'unet_memory_change': self.calculate_memory_change(
                    baseline['unet'], modified_unet
                ),
                'clip_memory_change': self.calculate_memory_change(
                    baseline['clip'], modified_clip
                )
            }
        }
        
        return analysis
    
    # ============================================
    # 8. OUTPUT AND REPORTING METHODS
    # ============================================
    
    def print_lora_analysis_summary(self, analysis):
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
    
    def print_comprehensive_summary(self, lora_analysis=None):
        """Print comprehensive summary including both model loading and LoRA application"""
        print(f"\n📊 COMPREHENSIVE WORKFLOW MONITORING SUMMARY")
        print("=" * 80)
        
        # Step 1: Model Loading Summary (implement as needed)
        print(f"🔍 STEP 1: MODEL LOADING")
        # Add your model loading summary here
        
        # Step 2: LoRA Application Summary
        if lora_analysis:
            print(f"\n🔍 STEP 2: LORA APPLICATION")
            self.print_lora_analysis_summary(lora_analysis)
        else:
            print(f"\n🔍 STEP 2: LORA APPLICATION - No analysis available")
        
        print("=" * 80)

# ============================================
# USAGE EXAMPLE
# ============================================

def example_usage():
    """Example of how to use the LoRA monitoring system"""
    
    # Initialize monitor
    monitor = LoRAMonitor()
    
    # Start monitoring
    monitor.start_monitoring("lora_application")
    
    # Capture baseline (before LoRA)
    unet_model = None  # Your UNET model here
    clip_model = None  # Your CLIP model here
    
    baseline = monitor.capture_lora_baseline(unet_model, clip_model)
    
    # Check LoRA file
    lora_status = monitor.check_lora_file_status("lora.safetensors")
    
    # Apply LoRA (your LoRA application code here)
    # modified_unet, modified_clip = apply_lora(unet_model, clip_model, lora_file)
    
    # Analyze results
    # analysis = monitor.analyze_lora_application_results(baseline, modified_unet, modified_clip, result)
    
    # Print summary
    # monitor.print_lora_analysis_summary(analysis)
    
    # End monitoring
    monitor.end_monitoring("lora_application")

if __name__ == "__main__":
    example_usage() 