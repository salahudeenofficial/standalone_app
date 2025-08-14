#!/usr/bin/env python3
"""
Chunked Processor for Video Frame Processing
Processes video frames in optimal batch sizes to reduce peak VRAM usage
"""

import torch
import math
import logging
from typing import List, Tuple, Optional, Callable, Any
from pathlib import Path

class ChunkedProcessor:
    """Processes video frames in chunks to optimize memory usage"""
    
    def __init__(self, memory_manager=None, model_manager=None):
        self.memory_manager = memory_manager
        self.model_manager = model_manager
        self.logger = logging.getLogger(__name__)
        
        # Default chunk sizes (can be adjusted based on VRAM)
        self.default_chunk_sizes = {
            'vae_encode': 4,    # Process 4 frames at a time for VAE encoding (reduced from 8)
            'vae_decode': 2,    # Process 2 frames at a time for VAE decoding (reduced from 4)
            'unet_process': 8   # Process 8 frames at a time for UNET (reduced from 16)
        }
        
        # Memory thresholds for dynamic chunk sizing
        self.memory_thresholds = {
            'low_vram': 4.0,    # GB - use smaller chunks
            'medium_vram': 8.0,  # GB - use medium chunks
            'high_vram': 16.0    # GB - use larger chunks
        }
        
        # More conservative chunk sizes for VRAM-constrained environments
        self.conservative_chunk_sizes = {
            'vae_encode': 2,     # Very conservative VAE encoding
            'vae_decode': 1,     # Very conservative VAE decoding
            'unet_process': 4    # Conservative UNET processing
        }
        
        # Advanced chunking strategies
        self.chunking_strategies = {
            'conservative': 0.5,    # Use 50% of calculated chunk size
            'balanced': 1.0,        # Use 100% of calculated chunk size
            'aggressive': 1.5       # Use 150% of calculated chunk size
        }
        
        self.current_strategy = 'balanced'
    
    def force_conservative_chunking(self) -> None:
        """Force very conservative chunking for memory-constrained environments"""
        self.logger.warning("Forcing conservative chunking due to memory constraints")
        self.current_strategy = 'conservative'
        # Override default chunk sizes with very conservative ones
        self.default_chunk_sizes = self.conservative_chunk_sizes.copy()
    
    def get_optimal_chunk_size(self, operation: str, frame_count: int, 
                              width: int, height: int, channels: int = 3) -> int:
        """Calculate optimal chunk size based on available VRAM and frame dimensions"""
        
        if not torch.cuda.is_available():
            return frame_count  # No GPU, process all at once
        
        # Get available VRAM
        try:
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            allocated_vram_gb = torch.cuda.memory_allocated(0) / 1024**3
            free_vram_gb = total_vram_gb - allocated_vram_gb
        except Exception as e:
            self.logger.warning(f"Failed to get VRAM info: {e}")
            total_vram_gb = 8.0  # Default fallback
            allocated_vram_gb = 0.0
            free_vram_gb = 8.0
        
        self.logger.info(f"VRAM Status - Total: {total_vram_gb:.2f} GB, "
                        f"Allocated: {allocated_vram_gb:.2f} GB, "
                        f"Free: {free_vram_gb:.2f} GB")
        
        # Estimate memory per frame for this operation
        frame_size_mb = self._estimate_frame_memory_size(width, height, channels, operation)
        
        # Calculate how many frames we can process based on available VRAM
        # Leave some buffer for other operations
        safety_buffer = 0.5  # Use only 50% of free VRAM (more conservative)
        usable_vram_mb = free_vram_gb * 1024 * safety_buffer
        
        # Calculate optimal chunk size
        optimal_chunk_size = max(1, int(usable_vram_mb / frame_size_mb))
        
        # Apply strategy multiplier
        strategy_multiplier = self.chunking_strategies[self.current_strategy]
        optimal_chunk_size = int(optimal_chunk_size * strategy_multiplier)
        
        # Apply operation-specific limits
        max_chunk_size = self._get_max_chunk_size_for_operation(operation, total_vram_gb)
        optimal_chunk_size = min(optimal_chunk_size, max_chunk_size, frame_count)
        
        # Ensure chunk size is reasonable
        if optimal_chunk_size < 1:
            optimal_chunk_size = 1
        elif optimal_chunk_size > frame_count:
            optimal_chunk_size = frame_count
        
        self.logger.info(f"Optimal chunk size for {operation}: {optimal_chunk_size} frames "
                        f"(frame size: {frame_size_mb:.2f} MB, "
                        f"usable VRAM: {usable_vram_mb:.2f} MB, "
                        f"strategy: {self.current_strategy})")
        
        return optimal_chunk_size
    
    def set_chunking_strategy(self, strategy: str) -> None:
        """Set the chunking strategy (conservative, balanced, aggressive)"""
        if strategy not in self.chunking_strategies:
            raise ValueError(f"Invalid strategy: {strategy}. Must be one of {list(self.chunking_strategies.keys())}")
        
        self.current_strategy = strategy
        self.logger.info(f"Chunking strategy set to: {strategy}")
    
    def get_vram_status(self) -> dict:
        """Get current VRAM status for monitoring"""
        if not torch.cuda.is_available():
            return {'available': False}
        
        try:
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            allocated_vram_gb = torch.cuda.memory_allocated(0) / 1024**3
            reserved_vram_gb = torch.cuda.memory_reserved(0) / 1024**3
            free_vram_gb = total_vram_gb - allocated_vram_gb
            
            return {
                'available': True,
                'total_gb': total_vram_gb,
                'allocated_gb': allocated_vram_gb,
                'reserved_gb': reserved_vram_gb,
                'free_gb': free_vram_gb,
                'utilization_percent': (allocated_vram_gb / total_vram_gb) * 100
            }
        except Exception as e:
            self.logger.warning(f"Failed to get VRAM status: {e}")
            return {
                'available': False,
                'error': str(e)
            }
    
    def should_adjust_strategy(self) -> bool:
        """Determine if we should adjust the chunking strategy based on VRAM pressure"""
        status = self.get_vram_status()
        if not status['available']:
            return False
        
        # If utilization is very high, consider switching to conservative
        if status['utilization_percent'] > 90:
            if self.current_strategy != 'conservative':
                self.logger.warning("High VRAM utilization detected, switching to conservative strategy")
                self.current_strategy = 'conservative'
                return True
        
        # If utilization is very low, consider switching to aggressive
        elif status['utilization_percent'] < 30:
            if self.current_strategy != 'aggressive':
                self.logger.info("Low VRAM utilization detected, switching to aggressive strategy")
                self.current_strategy = 'aggressive'
                return True
        
        return False
    
    def _check_memory_pressure(self) -> None:
        """Check if we need to force conservative chunking due to memory pressure"""
        try:
            if torch.cuda.is_available():
                allocated_vram_gb = torch.cuda.memory_allocated(0) / 1024**3
                total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                utilization = (allocated_vram_gb / total_vram_gb) * 100
                
                # If VRAM utilization is very high, force conservative chunking
                if utilization > 85:
                    self.force_conservative_chunking()
                # If we're getting close to OOM, force very conservative chunking
                elif utilization > 95:
                    self.logger.error("Critical VRAM usage detected! Forcing ultra-conservative chunking")
                    self.current_strategy = 'conservative'
                    # Use even smaller chunks
                    self.default_chunk_sizes = {
                        'vae_encode': 1,
                        'vae_decode': 1,
                        'unet_process': 2
                    }
        except Exception as e:
            self.logger.warning(f"Failed to check memory pressure: {e}")
    
    def _estimate_frame_memory_size(self, width: int, height: int, channels: int, operation: str) -> float:
        """Estimate memory usage per frame for a given operation"""
        
        # Base frame size in bytes (assuming float32)
        base_frame_size = width * height * channels * 4  # 4 bytes per float32
        
        # Operation-specific multipliers
        multipliers = {
            'vae_encode': 2.0,    # VAE encoding uses more memory
            'vae_decode': 1.5,    # VAE decoding
            'unet_process': 3.0,  # UNET processing uses most memory
            'latent_space': 0.125  # Latent space is 8x smaller
        }
        
        multiplier = multipliers.get(operation, 1.0)
        frame_size_mb = (base_frame_size * multiplier) / (1024 * 1024)
        
        return frame_size_mb
    
    def _get_max_chunk_size_for_operation(self, operation: str, total_vram_gb: float) -> int:
        """Get maximum chunk size based on VRAM capacity and operation type"""
        
        if total_vram_gb < self.memory_thresholds['low_vram']:
            # Low VRAM - use very small chunks
            max_sizes = self.conservative_chunk_sizes
        elif total_vram_gb < self.memory_thresholds['medium_vram']:
            # Medium VRAM - use moderate chunks
            max_sizes = {
                'vae_encode': 6,
                'vae_decode': 3,
                'unet_process': 12
            }
        else:
            # High VRAM - use larger chunks
            max_sizes = {
                'vae_encode': 12,
                'vae_decode': 6,
                'unet_process': 24
            }
        
        return max_sizes.get(operation, 4)
    
    def process_in_chunks(self, frames: torch.Tensor, operation: str, 
                         process_func: Callable, chunk_size: Optional[int] = None,
                         **kwargs) -> torch.Tensor:
        """Process frames in chunks using the provided function"""
        
        if frames is None or frames.numel() == 0:
            return frames
        
        frame_count = frames.shape[0]
        
        # Determine chunk size if not provided
        if chunk_size is None:
            chunk_size = self.get_optimal_chunk_size(
                operation, frame_count, frames.shape[-2], frames.shape[-1], frames.shape[-1]
            )
        
        self.logger.info(f"Processing {frame_count} frames in chunks of {chunk_size} "
                        f"for operation: {operation}")
        
        # If chunk size is >= frame count, process all at once
        if chunk_size >= frame_count:
            return process_func(frames, **kwargs)
        
        # Process in chunks
        results = []
        for i in range(0, frame_count, chunk_size):
            end_idx = min(i + chunk_size, frame_count)
            chunk = frames[i:end_idx]
            
            self.logger.debug(f"Processing chunk {i//chunk_size + 1}/{(frame_count + chunk_size - 1)//chunk_size}: "
                            f"frames {i} to {end_idx-1}")
            
            # Check memory pressure before processing each chunk
            self._check_memory_pressure()
            
            # Process this chunk
            try:
                chunk_result = process_func(chunk, **kwargs)
                results.append(chunk_result)
            except torch.cuda.OutOfMemoryError:
                self.logger.error("OOM during chunk processing! Reducing chunk size and retrying...")
                # Force ultra-conservative chunking
                self.current_strategy = 'conservative'
                self.default_chunk_sizes = {
                    'vae_encode': 1,
                    'vae_decode': 1,
                    'unet_process': 1
                }
                # Retry with smaller chunk
                smaller_chunk = chunk[:chunk_size//2] if chunk_size > 1 else chunk
                chunk_result = process_func(smaller_chunk, **kwargs)
                results.append(chunk_result)
            
            # Clean up chunk if memory manager is available
            if self.memory_manager:
                self.memory_manager.cleanup_tensor(chunk, f"chunk_{i}_{end_idx}")
            
            # Force memory cleanup between chunks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Combine results
        if len(results) == 1:
            return results[0]
        else:
            return torch.cat(results, dim=0)
    
    def vae_encode_chunked(self, vae, frames: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode frames using VAE in chunks"""
        
        def encode_chunk(chunk):
            return vae.encode(chunk)
        
        return self.process_in_chunks(
            frames, 'vae_encode', encode_chunk, **kwargs
        )
    
    def vae_decode_chunked(self, vae, latents: torch.Tensor, **kwargs) -> torch.Tensor:
        """Decode latents using VAE in chunks"""
        
        def decode_chunk(chunk):
            return vae.decode(chunk)
        
        return self.process_in_chunks(
            latents, 'vae_decode', decode_chunk, **kwargs
        )
    
    def unet_process_chunked(self, unet, latents: torch.Tensor, **kwargs) -> torch.Tensor:
        """Process latents using UNET in chunks"""
        
        def process_chunk(chunk):
            # This would be the UNET processing logic
            # For now, just return the chunk as-is
            return chunk
        
        return self.process_in_chunks(
            latents, 'unet_process', process_chunk, **kwargs
        )
    
    def get_processing_plan(self, frame_count: int, width: int, height: int, 
                           operations: List[str]) -> dict:
        """Generate a processing plan with optimal chunk sizes for each operation"""
        
        plan = {}
        
        for operation in operations:
            chunk_size = self.get_optimal_chunk_size(operation, frame_count, width, height)
            num_chunks = math.ceil(frame_count / chunk_size)
            
            plan[operation] = {
                'chunk_size': chunk_size,
                'num_chunks': num_chunks,
                'estimated_memory_mb': chunk_size * self._estimate_frame_memory_size(width, height, 3, operation),
                'processing_order': []
            }
            
            # Generate processing order for chunks
            for i in range(num_chunks):
                start_frame = i * chunk_size
                end_frame = min((i + 1) * chunk_size, frame_count)
                plan[operation]['processing_order'].append({
                    'chunk_id': i,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'frame_count': end_frame - start_frame
                })
        
        return plan
    
    def print_processing_plan(self, plan: dict) -> None:
        """Print the processing plan in a readable format"""
        
        self.logger.info("=== Chunked Processing Plan ===")
        
        for operation, details in plan.items():
            self.logger.info(f"\n{operation.upper()}:")
            self.logger.info(f"  Chunk Size: {details['chunk_size']} frames")
            self.logger.info(f"  Total Chunks: {details['num_chunks']}")
            self.logger.info(f"  Estimated Memory per Chunk: {details['estimated_memory_mb']:.2f} MB")
            
            self.logger.info("  Chunk Details:")
            for chunk_info in details['processing_order']:
                self.logger.info(f"    Chunk {chunk_info['chunk_id']}: "
                               f"Frames {chunk_info['start_frame']}-{chunk_info['end_frame']-1} "
                               f"({chunk_info['frame_count']} frames)")
        
        self.logger.info("===============================")
    
    def estimate_total_processing_time(self, plan: dict, estimated_time_per_chunk: float = 1.0) -> float:
        """Estimate total processing time based on the plan"""
        
        total_time = 0.0
        
        for operation, details in plan.items():
            operation_time = details['num_chunks'] * estimated_time_per_chunk
            total_time += operation_time
            
            self.logger.info(f"Estimated time for {operation}: {operation_time:.2f}s "
                           f"({details['num_chunks']} chunks × {estimated_time_per_chunk:.2f}s)")
        
        self.logger.info(f"Total estimated processing time: {total_time:.2f}s")
        return total_time 