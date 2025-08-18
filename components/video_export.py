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
            
        # Convert frames to numpy arrays
        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()
            
        # CRITICAL FIX: Handle different frame formats
        print(f"🔍 VideoExporter: Input frames shape: {frames.shape}")
        
        # Remove batch dimension if present
        if len(frames.shape) == 5:  # (batch, frames, height, width, channels)
            print(f"🔧 Removing batch dimension: {frames.shape[0]} -> {frames.shape[1]} frames")
            frames = frames.squeeze(0)  # Remove batch dimension
            print(f"✅ Fixed frames shape: {frames.shape}")
        
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