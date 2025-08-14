#!/usr/bin/env python3
"""
Memory Manager for Intermediate Tensor Cleanup
Handles explicit tensor dereferencing and memory cleanup to reduce peak VRAM usage
"""

import torch
import gc
import logging
from typing import Any, List, Optional, Union
import weakref

class MemoryManager:
    """Manages intermediate tensor cleanup and memory optimization"""
    
    def __init__(self, auto_cleanup_threshold_mb: float = 1000.0):
        self.logger = logging.getLogger(__name__)
        self.tracked_tensors = {}  # Track tensors for cleanup
        self.cleanup_points = []   # Track cleanup points in pipeline
        self.auto_cleanup_threshold_mb = auto_cleanup_threshold_mb  # Auto-cleanup when total tracked size exceeds this
        self.cleanup_callbacks = []  # Callbacks to run during cleanup
        
    def track_tensor(self, tensor, name: str, cleanup_priority: int = 1) -> None:
        """Track a tensor or other object for potential cleanup"""
        if tensor is None:
            return
        
        # Handle different types of objects
        if isinstance(tensor, torch.Tensor):
            # It's a PyTorch tensor
            tensor_id = id(tensor)
            self.tracked_tensors[tensor_id] = {
                'tensor': weakref.ref(tensor),
                'name': name,
                'type': 'tensor',
                'shape': tensor.shape,
                'dtype': tensor.dtype,
                'device': tensor.device,
                'cleanup_priority': cleanup_priority,
                'size_mb': self._get_tensor_size_mb(tensor)
            }
            
            self.logger.debug(f"Tracking tensor {name}: {tensor.shape} on {tensor.device} ({self.tracked_tensors[tensor_id]['size_mb']:.2f} MB)")
        elif isinstance(tensor, list):
            # It's a list - track each item individually if they're tensors
            self.logger.debug(f"Tracking list {name} with {len(tensor)} items")
            for i, item in enumerate(tensor):
                if isinstance(item, torch.Tensor):
                    item_name = f"{name}[{i}]"
                    self.track_tensor(item, item_name, cleanup_priority)
        elif isinstance(tensor, (tuple, set)):
            # Handle other iterable types
            self.logger.debug(f"Tracking {type(tensor).__name__} {name} with {len(tensor)} items")
            for i, item in enumerate(tensor):
                if isinstance(item, torch.Tensor):
                    item_name = f"{name}[{i}]"
                    self.track_tensor(item, item_name, cleanup_priority)
        else:
            # It's some other type - try to create a weak reference
            try:
                tensor_id = id(tensor)
                self.tracked_tensors[tensor_id] = {
                    'tensor': weakref.ref(tensor),
                    'name': name,
                    'type': type(tensor).__name__,
                    'shape': getattr(tensor, 'shape', 'unknown'),
                    'dtype': getattr(tensor, 'dtype', 'unknown'),
                    'device': getattr(tensor, 'device', 'unknown'),
                    'cleanup_priority': cleanup_priority,
                    'size_mb': self._get_tensor_size_mb(tensor)
                }
                
                self.logger.debug(f"Tracking object {name} of type {type(tensor).__name__}")
            except Exception as e:
                self.logger.warning(f"Could not track object {name} of type {type(tensor).__name__}: {e}")
                return
        
        # Check if we need to auto-cleanup
        self._check_auto_cleanup()
    
    def add_cleanup_callback(self, callback_func) -> None:
        """Add a callback function to run during cleanup"""
        self.cleanup_callbacks.append(callback_func)
    
    def cleanup_tensor(self, tensor, name: str = "unknown") -> None:
        """Explicitly cleanup a specific tensor or object"""
        if tensor is None:
            return
            
        tensor_id = id(tensor)
        
        # Log cleanup
        size_mb = self._get_tensor_size_mb(tensor)
        if isinstance(tensor, torch.Tensor):
            self.logger.info(f"Cleaning up tensor {name}: {tensor.shape} ({size_mb:.2f} MB)")
        else:
            self.logger.info(f"Cleaning up object {name} of type {type(tensor).__name__} ({size_mb:.2f} MB)")
        
        # Handle different types of objects
        if isinstance(tensor, torch.Tensor):
            # Move to CPU first if on GPU to free VRAM immediately
            if tensor.device.type == 'cuda':
                try:
                    tensor.cpu()
                except:
                    pass
        elif isinstance(tensor, (list, tuple, set)):
            # For iterables, cleanup each tensor item
            for item in tensor:
                if isinstance(item, torch.Tensor):
                    self.cleanup_tensor(item, f"{name}_item")
        
        # Delete the object
        del tensor
        
        # Remove from tracking
        if tensor_id in self.tracked_tensors:
            del self.tracked_tensors[tensor_id]
        
        # Force garbage collection for immediate cleanup
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Run cleanup callbacks
        self._run_cleanup_callbacks()
    
    def cleanup_tensors_by_priority(self, max_priority: int = 1) -> None:
        """Clean up tensors up to a certain priority level"""
        tensors_to_cleanup = []
        tensor_ids_to_remove = []
        
        # First pass: collect tensors to cleanup and IDs to remove
        for tensor_id, info in self.tracked_tensors.items():
            if info['cleanup_priority'] <= max_priority:
                # Get the actual tensor if it still exists
                tensor = info['tensor']()
                if tensor is not None:
                    tensors_to_cleanup.append((tensor, info['name']))
                # Mark for removal
                tensor_ids_to_remove.append(tensor_id)
        
        # Second pass: remove from tracking
        for tensor_id in tensor_ids_to_remove:
            del self.tracked_tensors[tensor_id]
        
        # Clean up the tensors
        for tensor, name in tensors_to_cleanup:
            self.cleanup_tensor(tensor, name)
        
        if tensors_to_cleanup:
            self.logger.info(f"Cleaned up {len(tensors_to_cleanup)} tensors with priority <= {max_priority}")
    
    def cleanup_all_tracked_tensors(self) -> None:
        """Clean up all tracked tensors"""
        self.logger.info("Cleaning up all tracked tensors")
        self.cleanup_tensors_by_priority(max_priority=999)
    
    def cleanup_intermediate_results(self, step_name: str, keep_tensors: List[str] = None) -> None:
        """Clean up intermediate results after a pipeline step"""
        if keep_tensors is None:
            keep_tensors = []
        
        self.logger.info(f"Cleaning up intermediate results after {step_name}")
        
        # Clean up all tracked tensors except those we want to keep
        tensors_to_cleanup = []
        tensor_ids_to_remove = []
        
        # First pass: collect tensors to cleanup and IDs to remove
        for tensor_id, info in self.tracked_tensors.items():
            if info['name'] not in keep_tensors:
                tensor = info['tensor']()
                if tensor is not None:
                    tensors_to_cleanup.append((tensor, info['name']))
                # Mark for removal
                tensor_ids_to_remove.append(tensor_id)
        
        # Second pass: remove from tracking
        for tensor_id in tensor_ids_to_remove:
            del self.tracked_tensors[tensor_id]
        
        # Clean up the tensors
        for tensor, name in tensors_to_cleanup:
            self.cleanup_tensor(tensor, name)
        
        # Force memory cleanup
        self._force_memory_cleanup()
        
        if tensors_to_cleanup:
            self.logger.info(f"Cleaned up {len(tensors_to_cleanup)} intermediate tensors after {step_name}")
    
    def _force_memory_cleanup(self) -> None:
        """Force aggressive memory cleanup"""
        # Python garbage collection
        gc.collect()
        
        # CUDA cache cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        # Empty Python cache
        import sys
        if hasattr(sys, 'getallocatedblocks'):
            # This is a CPython-specific feature
            pass
    
    def _check_auto_cleanup(self) -> None:
        """Check if we need to trigger auto-cleanup based on memory threshold"""
        stats = self.get_memory_stats()
        if stats['total_tracked_size_mb'] > self.auto_cleanup_threshold_mb:
            self.logger.info(f"Auto-cleanup triggered: {stats['total_tracked_size_mb']:.2f} MB > {self.auto_cleanup_threshold_mb:.2f} MB")
            self._auto_cleanup()
    
    def _auto_cleanup(self) -> None:
        """Automatically cleanup low-priority tensors to stay under threshold"""
        # Sort tensors by priority (highest priority = lowest number)
        tensors_by_priority = []
        for tensor_id, info in self.tracked_tensors.items():
            tensor = info['tensor']()
            if tensor is not None:
                tensors_by_priority.append((tensor_id, info['priority'], info['size_mb'], info['name']))
        
        # Sort by priority (ascending) and size (descending)
        tensors_by_priority.sort(key=lambda x: (x[1], -x[2]))
        
        # Clean up tensors until we're under threshold
        cleaned_size = 0
        tensor_ids_to_remove = []
        
        for tensor_id, priority, size_mb, name in tensors_by_priority:
            if priority > 1:  # Don't auto-cleanup high-priority tensors
                stats = self.get_memory_stats()
                if stats['total_tracked_size_mb'] - cleaned_size <= self.auto_cleanup_threshold_mb:
                    break
                
                # Clean up this tensor
                tensor = self.tracked_tensors[tensor_id]['tensor']()
                if tensor is not None:
                    # Use a safer cleanup method that doesn't modify tracked_tensors during iteration
                    self._safe_cleanup_tensor(tensor, name, tensor_id)
                    cleaned_size += size_mb
                    tensor_ids_to_remove.append(tensor_id)
        
        # Remove cleaned tensors from tracking
        for tensor_id in tensor_ids_to_remove:
            if tensor_id in self.tracked_tensors:
                del self.tracked_tensors[tensor_id]
        
        if cleaned_size > 0:
            self.logger.info(f"Auto-cleanup completed: freed {cleaned_size:.2f} MB")
    
    def _safe_cleanup_tensor(self, tensor, name: str, tensor_id: int) -> None:
        """Safely cleanup a tensor without modifying tracked_tensors during iteration"""
        # Log cleanup
        size_mb = self._get_tensor_size_mb(tensor)
        if isinstance(tensor, torch.Tensor):
            self.logger.info(f"Cleaning up tensor {name}: {tensor.shape} ({size_mb:.2f} MB)")
        else:
            self.logger.info(f"Cleaning up object {name} of type {type(tensor).__name__} ({size_mb:.2f} MB)")
        
        # Handle different types of objects
        if isinstance(tensor, torch.Tensor):
            # Move to CPU first if on GPU to free VRAM immediately
            if tensor.device.type == 'cuda':
                try:
                    tensor.cpu()
                except:
                    pass
        elif isinstance(tensor, (list, tuple, set)):
            # For iterables, cleanup each tensor item
            for item in tensor:
                if isinstance(item, torch.Tensor):
                    self._safe_cleanup_tensor(item, f"{name}_item", id(item))
        
        # Delete the object
        del tensor
        
        # Force garbage collection for immediate cleanup
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _run_cleanup_callbacks(self) -> None:
        """Run all registered cleanup callbacks"""
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                self.logger.warning(f"Cleanup callback failed: {e}")
    
    def _get_tensor_size_mb(self, tensor) -> float:
        """Calculate tensor size in MB"""
        if tensor is None:
            return 0.0
        
        try:
            if isinstance(tensor, torch.Tensor):
                # Get element size in bytes
                element_size = tensor.element_size()
                # Calculate total size
                total_elements = tensor.numel()
                total_size_bytes = total_elements * element_size
                return total_size_bytes / (1024 * 1024)  # Convert to MB
            elif isinstance(tensor, list):
                # For lists, sum the size of all tensor items
                total_size = 0.0
                for item in tensor:
                    if isinstance(item, torch.Tensor):
                        total_size += self._get_tensor_size_mb(item)
                return total_size
            elif isinstance(tensor, (tuple, set)):
                # For other iterables, sum the size of all tensor items
                total_size = 0.0
                for item in tensor:
                    if isinstance(item, torch.Tensor):
                        total_size += self._get_tensor_size_mb(item)
                return total_size
            else:
                # For other objects, try to estimate size
                try:
                    # Try to get size using sys.getsizeof
                    import sys
                    size_bytes = sys.getsizeof(tensor)
                    return size_bytes / (1024 * 1024)  # Convert to MB
                except:
                    return 0.0
        except:
            return 0.0
    
    def get_memory_stats(self) -> dict:
        """Get current memory statistics"""
        stats = {
            'tracked_tensors': len(self.tracked_tensors),
            'total_tracked_size_mb': 0.0,
            'tensor_details': []
        }
        
        for tensor_id, info in self.tracked_tensors.items():
            tensor = info['tensor']()
            if tensor is not None:
                stats['total_tracked_size_mb'] += info['size_mb']
                
                # Handle different types of objects
                if info.get('type') == 'tensor':
                    shape = info.get('shape', 'unknown')
                    device = str(info.get('device', 'unknown'))
                else:
                    shape = info.get('shape', 'unknown')
                    device = str(info.get('device', 'unknown'))
                
                stats['tensor_details'].append({
                    'name': info['name'],
                    'type': info.get('type', 'unknown'),
                    'shape': shape,
                    'device': device,
                    'size_mb': info['size_mb'],
                    'priority': info['cleanup_priority']
                })
        
        return stats
    
    def print_memory_stats(self) -> None:
        """Print current memory statistics"""
        stats = self.get_memory_stats()
        self.logger.info("=== Memory Manager Stats ===")
        self.logger.info(f"Tracked Tensors: {stats['tracked_tensors']}")
        self.logger.info(f"Total Tracked Size: {stats['total_tracked_size_mb']:.2f} MB")
        
        if stats['tensor_details']:
            self.logger.info("Object Details:")
            for tensor_info in stats['tensor_details']:
                self.logger.info(f"  {tensor_info['name']}: {tensor_info['type']} {tensor_info['shape']} on {tensor_info['device']} ({tensor_info['size_mb']:.2f} MB, priority {tensor_info['priority']})")
        
        self.logger.info("==========================")
    
    def add_cleanup_point(self, step_name: str, description: str) -> None:
        """Add a cleanup point for monitoring"""
        timestamp = None
        try:
            if torch.cuda.is_available():
                timestamp = torch.cuda.Event()
                timestamp.record()
        except Exception as e:
            self.logger.warning(f"Failed to create CUDA event for {step_name}: {e}")
        
        self.cleanup_points.append({
            'step': step_name,
            'description': description,
            'timestamp': timestamp
        })
    
    def mark_cleanup_point_complete(self, step_name: str) -> None:
        """Mark a cleanup point as complete and log timing"""
        for point in self.cleanup_points:
            if point['step'] == step_name and point['timestamp'] is not None:
                try:
                    if torch.cuda.is_available():
                        point['timestamp'].synchronize()
                        elapsed_ms = point['timestamp'].elapsed_time(point['timestamp'])
                        self.logger.info(f"Cleanup point {step_name} completed in {elapsed_ms:.2f} ms")
                except Exception as e:
                    self.logger.warning(f"Failed to get timing for cleanup point {step_name}: {e}")
                break 