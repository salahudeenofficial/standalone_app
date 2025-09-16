#!/usr/bin/env python3
"""
Multithreaded Memory Tracking System
Monitors GPU memory usage in real-time during model loading and operations
"""

import threading
import time
import torch
import psutil
import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import queue
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class MemorySnapshot:
    """Memory snapshot at a specific point in time"""
    timestamp: float
    gpu_allocated_mb: float
    gpu_reserved_mb: float
    gpu_free_mb: float
    gpu_total_mb: float
    cpu_memory_mb: float
    cpu_percent: float
    operation: str
    step: str
    details: Dict

class MemoryTracker:
    """Multithreaded GPU memory tracker"""
    
    def __init__(self, interval: float = 0.1, log_file: Optional[str] = None):
        """
        Initialize memory tracker
        
        Args:
            interval: Sampling interval in seconds
            log_file: Optional file to log memory data
        """
        self.interval = interval
        self.log_file = log_file
        self.is_tracking = False
        self.tracking_thread = None
        self.memory_snapshots: List[MemorySnapshot] = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        
        # GPU info
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        else:
            self.gpu_total_mb = 0
        
        # CPU info
        self.cpu_total_mb = psutil.virtual_memory().total / (1024**2)
        
        print(f"🔍 Memory Tracker initialized:")
        print(f"   GPU Available: {self.gpu_available}")
        print(f"   GPU Total: {self.gpu_total_mb:.1f} MB")
        print(f"   CPU Total: {self.cpu_total_mb:.1f} MB")
        print(f"   Sampling Interval: {interval}s")
        if log_file:
            print(f"   Log File: {log_file}")
    
    def _get_memory_snapshot(self, operation: str = "", step: str = "", details: Dict = None) -> MemorySnapshot:
        """Get current memory snapshot"""
        timestamp = time.time()
        
        # GPU memory
        if self.gpu_available:
            gpu_allocated = torch.cuda.memory_allocated() / (1024**2)
            gpu_reserved = torch.cuda.memory_reserved() / (1024**2)
            gpu_free = self.gpu_total_mb - gpu_reserved
        else:
            gpu_allocated = gpu_reserved = gpu_free = 0
        
        # CPU memory
        cpu_memory = psutil.virtual_memory()
        cpu_memory_mb = cpu_memory.used / (1024**2)
        cpu_percent = cpu_memory.percent
        
        return MemorySnapshot(
            timestamp=timestamp,
            gpu_allocated_mb=gpu_allocated,
            gpu_reserved_mb=gpu_reserved,
            gpu_free_mb=gpu_free,
            gpu_total_mb=self.gpu_total_mb,
            cpu_memory_mb=cpu_memory_mb,
            cpu_percent=cpu_percent,
            operation=operation,
            step=step,
            details=details or {}
        )
    
    def _tracking_loop(self):
        """Main tracking loop running in separate thread"""
        print(f"🔍 Memory tracking started (interval: {self.interval}s)")
        
        while not self.stop_event.is_set():
            try:
                snapshot = self._get_memory_snapshot("continuous_tracking", "background")
                
                with self.lock:
                    self.memory_snapshots.append(snapshot)
                
                # Log to file if specified
                if self.log_file:
                    self._log_snapshot(snapshot)
                
                # Sleep until next sample
                self.stop_event.wait(self.interval)
                
            except Exception as e:
                print(f"❌ Error in memory tracking loop: {e}")
                break
        
        print(f"🔍 Memory tracking stopped")
    
    def _log_snapshot(self, snapshot: MemorySnapshot):
        """Log memory snapshot to file"""
        try:
            log_entry = {
                'timestamp': snapshot.timestamp,
                'datetime': datetime.fromtimestamp(snapshot.timestamp).isoformat(),
                'gpu_allocated_mb': snapshot.gpu_allocated_mb,
                'gpu_reserved_mb': snapshot.gpu_reserved_mb,
                'gpu_free_mb': snapshot.gpu_free_mb,
                'gpu_total_mb': snapshot.gpu_total_mb,
                'cpu_memory_mb': snapshot.cpu_memory_mb,
                'cpu_percent': snapshot.cpu_percent,
                'operation': snapshot.operation,
                'step': snapshot.step,
                'details': snapshot.details
            }
            
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            print(f"❌ Error logging memory snapshot: {e}")
    
    def start_tracking(self):
        """Start memory tracking in background thread"""
        if self.is_tracking:
            print("⚠️  Memory tracking already running")
            return
        
        self.is_tracking = True
        self.stop_event.clear()
        self.tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.tracking_thread.start()
        
        print(f"✅ Memory tracking started")
    
    def stop_tracking(self):
        """Stop memory tracking"""
        if not self.is_tracking:
            print("⚠️  Memory tracking not running")
            return
        
        self.stop_event.set()
        if self.tracking_thread:
            self.tracking_thread.join(timeout=2.0)
        
        self.is_tracking = False
        print(f"✅ Memory tracking stopped")
    
    def log_memory_event(self, operation: str, step: str = "", details: Dict = None):
        """Log a specific memory event"""
        snapshot = self._get_memory_snapshot(operation, step, details)
        
        with self.lock:
            self.memory_snapshots.append(snapshot)
        
        if self.log_file:
            self._log_snapshot(snapshot)
        
        # Print real-time info
        print(f"🔍 MEMORY EVENT: {operation}")
        print(f"   Step: {step}")
        print(f"   GPU: {snapshot.gpu_allocated_mb:.1f}MB allocated, {snapshot.gpu_reserved_mb:.1f}MB reserved, {snapshot.gpu_free_mb:.1f}MB free")
        print(f"   CPU: {snapshot.cpu_memory_mb:.1f}MB used ({snapshot.cpu_percent:.1f}%)")
        if details:
            print(f"   Details: {details}")
    
    def get_memory_summary(self) -> Dict:
        """Get summary of memory usage"""
        if not self.memory_snapshots:
            return {"error": "No memory data available"}
        
        with self.lock:
            snapshots = self.memory_snapshots.copy()
        
        if not snapshots:
            return {"error": "No memory data available"}
        
        # Calculate statistics
        gpu_allocated_values = [s.gpu_allocated_mb for s in snapshots]
        gpu_reserved_values = [s.gpu_reserved_mb for s in snapshots]
        cpu_memory_values = [s.cpu_memory_mb for s in snapshots]
        
        return {
            "total_samples": len(snapshots),
            "duration_seconds": snapshots[-1].timestamp - snapshots[0].timestamp,
            "gpu_allocated": {
                "min": min(gpu_allocated_values),
                "max": max(gpu_allocated_values),
                "avg": sum(gpu_allocated_values) / len(gpu_allocated_values),
                "current": gpu_allocated_values[-1]
            },
            "gpu_reserved": {
                "min": min(gpu_reserved_values),
                "max": max(gpu_reserved_values),
                "avg": sum(gpu_reserved_values) / len(gpu_reserved_values),
                "current": gpu_reserved_values[-1]
            },
            "cpu_memory": {
                "min": min(cpu_memory_values),
                "max": max(cpu_memory_values),
                "avg": sum(cpu_memory_values) / len(cpu_memory_values),
                "current": cpu_memory_values[-1]
            },
            "gpu_total_mb": self.gpu_total_mb,
            "cpu_total_mb": self.cpu_total_mb
        }
    
    def print_memory_summary(self):
        """Print detailed memory usage summary"""
        summary = self.get_memory_summary()
        
        if "error" in summary:
            print(f"❌ {summary['error']}")
            return
        
        print("\n" + "="*80)
        print("📊 MEMORY TRACKING SUMMARY")
        print("="*80)
        print(f"📈 Total Samples: {summary['total_samples']}")
        print(f"⏱️  Duration: {summary['duration_seconds']:.2f} seconds")
        print(f"📊 Sampling Rate: {summary['total_samples'] / summary['duration_seconds']:.1f} samples/sec")
        
        print(f"\n🎮 GPU MEMORY:")
        print(f"   Total Available: {summary['gpu_total_mb']:.1f} MB")
        print(f"   Allocated: {summary['gpu_allocated']['current']:.1f} MB (min: {summary['gpu_allocated']['min']:.1f}, max: {summary['gpu_allocated']['max']:.1f}, avg: {summary['gpu_allocated']['avg']:.1f})")
        print(f"   Reserved: {summary['gpu_reserved']['current']:.1f} MB (min: {summary['gpu_reserved']['min']:.1f}, max: {summary['gpu_reserved']['max']:.1f}, avg: {summary['gpu_reserved']['avg']:.1f})")
        print(f"   Free: {summary['gpu_total_mb'] - summary['gpu_reserved']['current']:.1f} MB")
        
        print(f"\n💻 CPU MEMORY:")
        print(f"   Total Available: {summary['cpu_total_mb']:.1f} MB")
        print(f"   Used: {summary['cpu_memory']['current']:.1f} MB (min: {summary['cpu_memory']['min']:.1f}, max: {summary['cpu_memory']['max']:.1f}, avg: {summary['cpu_memory']['avg']:.1f})")
        
        # Calculate memory efficiency
        gpu_utilization = (summary['gpu_reserved']['max'] / summary['gpu_total_mb']) * 100
        cpu_utilization = (summary['cpu_memory']['max'] / summary['cpu_total_mb']) * 100
        
        print(f"\n📈 UTILIZATION:")
        print(f"   GPU Peak Utilization: {gpu_utilization:.1f}%")
        print(f"   CPU Peak Utilization: {cpu_utilization:.1f}%")
        
        print("="*80)
    
    def clear_snapshots(self):
        """Clear all memory snapshots"""
        with self.lock:
            self.memory_snapshots.clear()
        print("🧹 Memory snapshots cleared")

def track_memory_during_operation(tracker: MemoryTracker, operation_name: str, step_name: str = "", details: Dict = None):
    """Decorator to track memory during a specific operation"""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            # Log start of operation
            tracker.log_memory_event(f"{operation_name}_start", step_name, details)
            
            try:
                # Execute the function
                result = func(*args, **kwargs)
                
                # Log end of operation
                tracker.log_memory_event(f"{operation_name}_end", step_name, details)
                
                return result
                
            except Exception as e:
                # Log error
                tracker.log_memory_event(f"{operation_name}_error", step_name, {"error": str(e)})
                raise
        
        return wrapper
    return decorator

def create_memory_tracker(interval: float = 0.1, log_file: Optional[str] = None) -> MemoryTracker:
    """Create and configure a memory tracker"""
    return MemoryTracker(interval=interval, log_file=log_file)

# Example usage
if __name__ == "__main__":
    # Create memory tracker
    tracker = create_memory_tracker(interval=0.1, log_file="memory_tracking.log")
    
    # Start tracking
    tracker.start_tracking()
    
    # Simulate some operations
    print("🔍 Simulating memory operations...")
    
    # Simulate GPU memory allocation
    if torch.cuda.is_available():
        tracker.log_memory_event("test_operation", "test_step", {"description": "Testing memory tracking"})
        
        # Allocate some memory
        test_tensor = torch.randn(1000, 1000).cuda()
        tracker.log_memory_event("gpu_allocation", "test_step", {"tensor_shape": test_tensor.shape})
        
        # Free memory
        del test_tensor
        torch.cuda.empty_cache()
        tracker.log_memory_event("gpu_deallocation", "test_step", {"description": "Freed test tensor"})
    
    # Stop tracking
    tracker.stop_tracking()
    
    # Print summary
    tracker.print_memory_summary()
    
    print("✅ Memory tracking test completed!")
