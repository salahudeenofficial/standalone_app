#!/usr/bin/env python3
"""
Test script for the real-time memory monitoring system
"""

import time
import threading
import queue
from collections import deque
import torch
import gc

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
            'gpu_utilization': 0,
            'tensor_count': 0
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
                
                # Tensor count (quick estimate)
                snapshot['tensor_count'] = len([obj for obj in gc.get_objects() if torch.is_tensor(obj) and obj.is_cuda])
                
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
        alert += f"\n   Tensors: {snapshot['tensor_count']}"
        
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
        except queue.Empty:
            pass
    
    def change_label(self, new_label):
        """Change monitoring label (e.g., for different pipeline steps)"""
        self.event_queue.put({
            'type': 'label_change',
            'label': new_label
        })
    
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

def test_memory_monitor():
    """Test the memory monitoring system"""
    print("🧪 Testing Real-Time Memory Monitor")
    print("="*50)
    
    # Create monitor
    monitor = RealTimeMemoryMonitor(sample_interval=0.2, max_samples=50)
    
    # Start monitoring
    monitor.start_monitoring("TEST_START")
    
    # Simulate some GPU operations
    print("\n🔍 Simulating GPU operations...")
    
    # Create some tensors to consume memory
    tensors = []
    for i in range(5):
        # Create a 1GB tensor
        tensor = torch.randn(1024, 1024, 256, device='cuda' if torch.cuda.is_available() else 'cpu')
        tensors.append(tensor)
        print(f"   Created tensor {i+1}: {tensor.shape} on {tensor.device}")
        
        # Change label to show progression
        monitor.change_label(f"TENSOR_{i+1}")
        time.sleep(0.5)
    
    # Wait a bit more
    time.sleep(1.0)
    
    # Clean up tensors
    print("\n🧹 Cleaning up tensors...")
    for i, tensor in enumerate(tensors):
        del tensor
        print(f"   Deleted tensor {i+1}")
        time.sleep(0.3)
    
    # Wait for final monitoring
    time.sleep(1.0)
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    # Print summary
    monitor.print_memory_summary()
    
    print("\n✅ Memory monitoring test completed!")

if __name__ == "__main__":
    test_memory_monitor() 