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
            
        # Ensure frames are in correct format (H, W, C) and range [0, 255]
        if frames.dtype != np.uint8:
            frames = (frames * 255).clip(0, 255).astype(np.uint8)
            
        # Get video dimensions
        height, width = frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
        
        try:
            # Write frames
            for frame in frames:
                # Convert RGB to BGR if needed
                if frame.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                out.write(frame_bgr)
                
        finally:
            out.release()
            
        print(f"Video exported to: {output_path}")
        return output_path 