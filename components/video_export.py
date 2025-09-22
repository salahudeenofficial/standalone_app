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
        print(f"🔍 Frame data before conversion: dtype={frames.dtype}, range=[{frames.min():.1f}, {frames.max():.1f}]")
        if frames.dtype != np.uint8:
            # Check if data is already in [0, 1] range or [0, 255] range
            if frames.max() <= 1.0:
                print(f"🔧 Converting from [0, 1] range to [0, 255]")
                frames = (frames * 255).clip(0, 255).astype(np.uint8)
            else:
                print(f"🔧 Converting from [0, {frames.max():.1f}] range to [0, 255]")
                frames = frames.clip(0, 255).astype(np.uint8)
        print(f"🔍 Frame data after conversion: dtype={frames.dtype}, range=[{frames.min():.1f}, {frames.max():.1f}]")
            
        # Get video dimensions from first frame
        height, width = frames[0].shape[:2]
        print(f"🔍 Video dimensions: {width}x{height}, {len(frames)} frames")
        
        # Create video writer with better codec support
        # Try multiple codecs in order of preference
        codecs_to_try = [
            ('mp4v', 'mp4v'),
            ('XVID', 'XVID'), 
            ('MJPG', 'MJPG'),
            ('H264', 'H264'),
            ('avc1', 'avc1')
        ]
        
        out = None
        successful_codec = None
        
        for codec_name, fourcc_str in codecs_to_try:
            print(f"🔧 Trying codec: {codec_name}")
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
            
            if out.isOpened():
                print(f"✅ Successfully initialized video writer with {codec_name} codec")
                successful_codec = codec_name
                break
            else:
                print(f"❌ Failed to initialize with {codec_name} codec")
                out.release()
                out = None
        
        if out is None or not out.isOpened():
            raise RuntimeError(f"Failed to initialize video writer with any codec. Tried: {[c[0] for c in codecs_to_try]}")
        
        try:
            # Write frames
            print(f"🔍 Writing {len(frames)} frames to video...")
            frames_written = 0
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
                
                # Additional validation and debugging
                if i < 3:  # Debug first few frames
                    print(f"   🔍 Frame {i+1} debug:")
                    print(f"      Original frame shape: {frame.shape}, dtype: {frame.dtype}")
                    print(f"      BGR frame shape: {frame_bgr.shape}, dtype: {frame_bgr.dtype}")
                    print(f"      BGR frame range: [{frame_bgr.min()}, {frame_bgr.max()}]")
                    print(f"      Video writer dimensions: {out.get(cv2.CAP_PROP_FRAME_WIDTH)}x{out.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
                    print(f"      Video writer FPS: {out.get(cv2.CAP_PROP_FPS)}")
                
                # Try to enhance frame if it's too dark
                if frame_bgr.max() < 50:  # Very dark frame
                    print(f"   🔧 Frame {i+1} is very dark (max={frame_bgr.max()}), enhancing...")
                    # Enhance contrast
                    frame_bgr = cv2.convertScaleAbs(frame_bgr, alpha=2.0, beta=50)
                    print(f"   🔧 Enhanced frame range: [{frame_bgr.min()}, {frame_bgr.max()}]")
                
                # Write frame
                success = out.write(frame_bgr)
                if success:
                    frames_written += 1
                    if i < 3:  # Log success for first few frames
                        print(f"   ✅ Frame {i+1} written successfully")
                else:
                    print(f"⚠️  Warning: Failed to write frame {i+1}")
                    # Additional debugging for failed frames
                    print(f"   Frame shape: {frame_bgr.shape}, dtype: {frame_bgr.dtype}")
                    print(f"   Frame range: [{frame_bgr.min()}, {frame_bgr.max()}]")
                    print(f"   Video writer status: isOpened={out.isOpened()}")
                    
                    # Try writing a test frame
                    if i == 0:  # Only for first frame
                        print(f"   🔧 Trying to write a test white frame...")
                        test_frame = np.full((height, width, 3), 255, dtype=np.uint8)
                        test_success = out.write(test_frame)
                        print(f"   🔧 Test frame result: {test_success}")
                
        except Exception as e:
            print(f"❌ Error during video export: {e}")
            raise
        finally:
            out.release()
            
        # Verify video was created and has content
        if frames_written == 0:
            print(f"❌ No frames were written to video")
            print(f"🔧 Attempting fallback: saving frames as individual images...")
            
            # Fallback: save frames as individual images
            import os
            frames_dir = output_path.replace('.mp4', '_frames')
            os.makedirs(frames_dir, exist_ok=True)
            
            for i, frame in enumerate(frames):
                frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
                # Convert RGB to BGR for OpenCV
                if frame.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                cv2.imwrite(frame_path, frame_bgr)
            
            print(f"✅ Fallback successful: saved {len(frames)} frames to {frames_dir}")
            return frames_dir
        
        # Check if file exists and has reasonable size
        if not os.path.exists(output_path):
            raise RuntimeError(f"Video file was not created: {output_path}")
        
        file_size = os.path.getsize(output_path)
        if file_size < 1024:  # Less than 1KB is suspicious
            print(f"⚠️  Warning: Video file is very small ({file_size} bytes), may be corrupted")
        
        print(f"✅ Video exported successfully to: {output_path}")
        print(f"   📊 Frames written: {frames_written}/{len(frames)}")
        print(f"   📊 File size: {file_size / (1024*1024):.2f} MB")
        return output_path 