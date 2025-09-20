"""
Video export component for the standalone pipeline
Converts decoded frames to MP4 video format
"""

import torch
import numpy as np
import cv2
from pathlib import Path

class VideoExporter:
    """Export video frames to MP4 format"""
    
    def __init__(self, fps=24):
        """Initialize with desired FPS"""
        self.fps = fps
        
    def export_video(self, frames, output_path):
        """Export frames to MP4 video"""
        if frames is None or len(frames) == 0:
            raise ValueError("No frames to export")
            
        # CRITICAL FIX: Handle different frame formats
        print(f"🔍 VideoExporter: Input frames shape: {frames.shape}")
        
        # Handle tensor reshaping BEFORE converting to numpy
        if isinstance(frames, torch.Tensor):
            # Handle different tensor formats
            if len(frames.shape) == 5:  # (batch, channels, frames, height, width)
                print(f"🔧 Reshaping from (batch, channels, frames, height, width) to (frames, height, width, channels)")
                batch_size, channels, num_frames, height, width = frames.shape
                # Reshape: (batch, channels, frames, height, width) -> (frames, height, width, channels)
                frames = frames.squeeze(0)  # Remove batch dimension: (channels, frames, height, width)
                frames = frames.permute(1, 2, 3, 0)  # (frames, height, width, channels)
                print(f"✅ Reshaped frames: {frames.shape}")
            elif len(frames.shape) == 4:  # Already in correct format
                if frames.shape[1] == 3:  # (batch, channels, height, width) - single frame
                    print(f"🔧 Single frame detected, reshaping to (1, height, width, channels)")
                    batch_size, channels, height, width = frames.shape
                    frames = frames.squeeze(0)  # Remove batch: (channels, height, width)
                    frames = frames.permute(1, 2, 0)  # (height, width, channels)
                    frames = frames.unsqueeze(0)  # Add frame dimension: (1, height, width, channels)
                    print(f"✅ Single frame reshaped: {frames.shape}")
                else:  # Assume (frames, height, width, channels)
                    print(f"✅ Frames already in correct format: {frames.shape}")
            else:
                raise ValueError(f"Unsupported tensor shape: {frames.shape}, expected 4D or 5D tensor")
            
            # Convert to numpy AFTER reshaping
            frames = frames.cpu().numpy()
        else:
            # Already numpy array - handle reshaping with numpy operations
            if len(frames.shape) == 5:  # (batch, channels, frames, height, width)
                print(f"🔧 Reshaping numpy array from (batch, channels, frames, height, width) to (frames, height, width, channels)")
                batch_size, channels, num_frames, height, width = frames.shape
                # Reshape: (batch, channels, frames, height, width) -> (frames, height, width, channels)
                frames = frames.squeeze(0)  # Remove batch dimension: (channels, frames, height, width)
                frames = np.transpose(frames, (1, 2, 3, 0))  # (frames, height, width, channels)
                print(f"✅ Reshaped frames: {frames.shape}")
            elif len(frames.shape) == 4:  # Already in correct format
                if frames.shape[1] == 3:  # (batch, channels, height, width) - single frame
                    print(f"🔧 Single frame detected, reshaping to (1, height, width, channels)")
                    batch_size, channels, height, width = frames.shape
                    frames = frames.squeeze(0)  # Remove batch: (channels, height, width)
                    frames = np.transpose(frames, (1, 2, 0))  # (height, width, channels)
                    frames = np.expand_dims(frames, 0)  # Add frame dimension: (1, height, width, channels)
                    print(f"✅ Single frame reshaped: {frames.shape}")
                else:  # Assume (frames, height, width, channels)
                    print(f"✅ Frames already in correct format: {frames.shape}")
            else:
                raise ValueError(f"Unsupported array shape: {frames.shape}, expected 4D or 5D array")
        
        # Ensure frames are in correct format (frames, height, width, channels)
        if len(frames.shape) != 4:
            raise ValueError(f"Expected 4D frames (frames, height, width, channels), got {frames.shape}")
            
        # Ensure frames are in correct format (H, W, C) and range [0, 255]
        if frames.dtype != np.uint8:
            frames = (frames * 255).clip(0, 255).astype(np.uint8)
            
        # Get video dimensions from first frame
        height, width = frames[0].shape[:2]
        print(f"🔍 Video dimensions: {width}x{height}, {len(frames)} frames")
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
        
        try:
            # Write frames
            print(f"🔍 Writing {len(frames)} frames to video...")
            for i, frame in enumerate(frames):
                if i < 5 or i % 10 == 0:  # Log first 5 frames and every 10th frame
                    print(f"🔍 Frame {i+1}: shape={frame.shape}, dtype={frame.dtype}, range=[{frame.min():.1f}, {frame.max():.1f}]")
                
                # Validate frame format
                if len(frame.shape) != 3:
                    raise ValueError(f"Frame {i} has wrong shape: {frame.shape}, expected (height, width, channels)")
                if frame.shape[2] not in [1, 3]:
                    raise ValueError(f"Frame {i} has wrong channel count: {frame.shape[2]}, expected 1 or 3")
                
                # Convert RGB to BGR if needed
                if frame.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                
                # Write frame
                success = out.write(frame_bgr)
                if not success:
                    print(f"⚠️  Warning: Failed to write frame {i+1}")
                
        except Exception as e:
            print(f"❌ Error during video export: {e}")
            raise
        finally:
            out.release()
            
        print(f"✅ Video exported successfully to: {output_path}")
        return output_path 