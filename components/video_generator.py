"""
Video generation component for the standalone pipeline
Adapted from ComfyUI WanVaceToVideo but simplified for direct use
"""

import torch
import sys
from pathlib import Path

# Add the comfy modules to path
sys.path.insert(0, str(Path(__file__).parent.parent / "comfy"))
sys.path.insert(0, str(Path(__file__).parent.parent / "comfy_extras"))

import comfy.model_management
import comfy.utils
import comfy.latent_formats

class WanVaceToVideo:
    """Generate initial video latents from reference image and control video"""
    
    def encode(self, positive, negative, vae, width, height, length, batch_size, 
               strength, control_video=None, control_masks=None, reference_image=None, 
               chunked_processor=None, chunk_size=None, force_downscale=False):
        """Encode reference image and control video to initial latents"""
        
        # Handle frame downscaling if needed
        if force_downscale:
            target_width = 256
            target_height = 448
            print(f"Downscaling frames from {width}x{height} to {target_width}x{target_height} to reduce memory usage")
            width = target_width
            height = target_height
        
        latent_length = ((length - 1) // 4) + 1
        
        # Process control video
        if control_video is not None:
            # Debug: Show tensor shapes
            print(f"5a. Control video shape before processing: {control_video.shape}")
            
            # Downscale if needed
            if force_downscale:
                print(f"5a. Downscaling control video to {target_width}x{target_height}")
                control_video = comfy.utils.common_upscale(
                    control_video[:length].movedim(-1, 1), target_width, target_height, "bilinear", "center"
                ).movedim(1, -1)
            else:
                print(f"5a. Upscaling control video to {width}x{height}")
                control_video = comfy.utils.common_upscale(
                    control_video[:length].movedim(-1, 1), width, height, "bilinear", "center"
                ).movedim(1, -1)
            
            print(f"5a. Control video shape after upscaling: {control_video.shape}")
            
            if control_video.shape[0] < length:
                control_video = torch.nn.functional.pad(
                    control_video, (0, 0, 0, 0, 0, 0, 0, length - control_video.shape[0]), value=0.5
                )
        else:
            control_video = torch.ones((length, height, width, 3)) * 0.5
            
        # Process reference image
        if reference_image is not None:
            # Debug: Show tensor shapes
            print(f"5a. Reference image shape before processing: {reference_image.shape}")
            
            # Downscale if needed
            if force_downscale:
                print(f"5a. Downscaling reference image to {target_width}x{target_height}")
                reference_image = comfy.utils.common_upscale(
                    reference_image[:1].movedim(-1, 1), target_width, target_height, "bilinear", "center"
                ).movedim(1, -1)
            else:
                print(f"5a. Upscaling reference image to {width}x{height}")
                reference_image = comfy.utils.common_upscale(
                    reference_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
                ).movedim(1, -1)
            
            print(f"5a. Reference image shape after upscaling: {reference_image.shape}")
            
            # Note: Reference image will be encoded by the main pipeline later
            # We just prepare the pixel format here, don't encode to latent yet
            # reference_image = vae.encode(reference_image[:, :, :, :3])  # REMOVED - duplicate encoding
            # reference_image = torch.cat([  # REMOVED - duplicate processing
            #     reference_image, 
            #     comfy.latent_formats.Wan21().process_out(torch.zeros_like(reference_image))
            # ], dim=1)
            
        # Process masks
        if control_masks is None:
            mask = torch.ones((length, height, width, 1))
        else:
            mask = control_masks
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            
            # Downscale if needed
            if force_downscale:
                mask = comfy.utils.common_upscale(mask[:length], target_width, target_height, "bilinear", "center").movedim(1, -1)
            else:
                mask = comfy.utils.common_upscale(mask[:length], width, height, "bilinear", "center").movedim(1, -1)
            
            if mask.shape[0] < length:
                mask = torch.nn.functional.pad(
                    mask, (0, 0, 0, 0, 0, 0, 0, length - mask.shape[0]), value=1.0
                )
                
        # Process control video latents
        control_video = control_video - 0.5
        inactive = (control_video * (1 - mask)) + 0.5
        reactive = (control_video * mask) + 0.5
        
        # Use chunked VAE encoding if available
        if chunked_processor is not None and chunk_size is not None and length > chunk_size:
            print(f"Using chunked VAE encoding: processing {length} frames in chunks of {chunk_size}")
            
            # Encode inactive frames in chunks
            inactive_latents = []
            print(f"5a. Starting VAE encoding of {length} inactive frames in chunks of {chunk_size}")
            
            for i in range(0, length, chunk_size):
                end_idx = min(i + chunk_size, length)
                chunk = inactive[i:end_idx, :, :, :3]
                print(f"5a. Processing inactive chunk {i//chunk_size + 1}: frames {i}-{end_idx-1}, shape: {chunk.shape}")
                
                try:
                    chunk_latent = vae.encode(chunk)
                    print(f"5a. Chunk {i//chunk_size + 1} encoded successfully: {chunk_latent.shape}")
                    print(f"5a. Chunk {i//chunk_size + 1} latent dimensions: {chunk_latent.dim()}")
                    inactive_latents.append(chunk_latent)
                except torch.cuda.OutOfMemoryError:
                    print(f"OOM encoding chunk {i//chunk_size + 1}! Processing frame by frame...")
                    # Process frame by frame if chunk fails
                    for j in range(i, end_idx):
                        single_frame = inactive[j:j+1, :, :, :3]
                        print(f"5a. Processing single frame {j}, shape: {single_frame.shape}")
                        try:
                            single_latent = vae.encode(single_frame)
                            print(f"5a. Frame {j} encoded successfully: {single_latent.shape}")
                            # Ensure consistent shape for single frame latents
                            if single_latent.dim() == 4:  # Should be [1, 4, H//8, W//8]
                                inactive_latents.append(single_latent)
                            else:
                                print(f"Warning: Unexpected latent shape for frame {j}: {single_latent.shape}")
                                # Create dummy latent with correct shape
                                dummy_latent = torch.zeros((1, 4, height // 8, width // 8), 
                                                        device=single_latent.device, dtype=single_latent.dtype)
                                inactive_latents.append(dummy_latent)
                        except Exception as frame_error:
                            print(f"Error encoding frame {j}: {frame_error}")
                            # Create dummy latent for failed frame
                            dummy_latent = torch.zeros((1, 4, height // 8, width // 8), 
                                                    device=comfy.model_management.intermediate_device())
                            inactive_latents.append(dummy_latent)
                        
                        del single_frame
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                # Clean up chunk to free memory
                del chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            print(f"5a. Completed inactive frame encoding. Total latents: {len(inactive_latents)}")
            
            # Verify all latents have consistent shapes before concatenating
            if inactive_latents:
                # Handle both 4D and 5D latent formats
                first_latent = inactive_latents[0]
                if first_latent.dim() == 5:
                    # 5D format: [batch, frames, channels, height, width]
                    expected_shape = first_latent.shape[2:]  # Skip batch and frames dimensions
                    print(f"Expected 5D latent shape: {expected_shape}")
                    
                    # Validate and fix any mismatched latents
                    for i, latent in enumerate(inactive_latents):
                        if latent.shape[2:] != expected_shape:
                            print(f"Fixing 5D latent {i} shape: {latent.shape} -> {expected_shape}")
                            # Force correct shape for WAN VAE compatibility
                            # WAN VAE produces 10 channels but UNET expects 4
                            if latent.shape[2] == 10:  # WAN VAE 10-channel output
                                print(f"   WAN VAE detected: converting 10 channels to 4 channels")
                                # Take first 4 channels from WAN VAE output
                                corrected_latent = latent[:, :, :4, :, :]
                                inactive_latents[i] = corrected_latent
                                print(f"   Corrected latent shape: {corrected_latent.shape}")
                            else:
                                # Create dummy latent with correct shape
                                inactive_latent = torch.zeros((latent.shape[0], latent.shape[1], 4, height // 8, width // 8), 
                                                            device=latent.device, dtype=latent.dtype)
                                inactive_latents[i] = inactive_latent
                else:
                    # 4D format: [batch, channels, height, width]
                    expected_shape = first_latent.shape[1:]  # Skip batch dimension
                    print(f"Expected 4D latent shape: {expected_shape}")
                    
                    # Validate and fix any mismatched latents
                    for i, latent in enumerate(inactive_latents):
                        if latent.shape[1:] != expected_shape:
                            print(f"Fixing 4D latent {i} shape: {latent.shape} -> {expected_shape}")
                            # Create dummy latent with correct shape
                            inactive_latents[i] = torch.zeros((latent.shape[0], 4, height // 8, width // 8), 
                                                            device=latent.device, dtype=latent.dtype)
            
            # Concatenate along the appropriate dimension
            if inactive_latents and inactive_latents[0].dim() == 5:
                # 5D format: concatenate along frame dimension (dim=1)
                inactive = torch.cat(inactive_latents, dim=1)
                print(f"5a. Concatenated 5D inactive latents: {inactive.shape}")
            else:
                # 4D format: concatenate along batch dimension (dim=0)
                inactive = torch.cat(inactive_latents, dim=0)
                print(f"5a. Concatenated 4D inactive latents: {inactive.shape}")
            del inactive_latents
            
            # Encode reactive frames in chunks
            reactive_latents = []
            print(f"5a. Starting VAE encoding of {length} reactive frames in chunks of {chunk_size}")
            
            for i in range(0, length, chunk_size):
                end_idx = min(i + chunk_size, length)
                chunk = reactive[i:end_idx, :, :, :3]
                print(f"5a. Processing reactive chunk {i//chunk_size + 1}: frames {i}-{end_idx-1}, shape: {chunk.shape}")
                
                try:
                    chunk_latent = vae.encode(chunk)
                    print(f"5a. Reactive chunk {i//chunk_size + 1} encoded successfully: {chunk_latent.shape}")
                    print(f"5a. Reactive chunk {i//chunk_size + 1} latent dimensions: {chunk_latent.dim()}")
                    reactive_latents.append(chunk_latent)
                except torch.cuda.OutOfMemoryError:
                    print(f"OOM encoding reactive chunk {i//chunk_size + 1}! Processing frame by frame...")
                    # Process frame by frame if chunk fails
                    for j in range(i, end_idx):
                        single_frame = reactive[j:j+1, :, :, :3]
                        print(f"5a. Processing reactive single frame {j}, shape: {single_frame.shape}")
                        try:
                            single_latent = vae.encode(single_frame)
                            print(f"5a. Reactive frame {j} encoded successfully: {single_latent.shape}")
                            # Ensure consistent shape for single frame latents
                            if single_latent.dim() == 4:  # Should be [1, 4, H//8, W//8]
                                reactive_latents.append(single_latent)
                            else:
                                print(f"Warning: Unexpected latent shape for reactive frame {j}: {single_latent.shape}")
                                # Create dummy latent with correct shape
                                dummy_latent = torch.zeros((1, 4, height // 8, width // 8), 
                                                        device=single_latent.device, dtype=single_latent.dtype)
                                reactive_latents.append(dummy_latent)
                        except Exception as frame_error:
                            print(f"Error encoding reactive frame {j}: {frame_error}")
                            # Create dummy latent for failed frame
                            dummy_latent = torch.zeros((1, 4, height // 8, width // 8), 
                                                    device=comfy.model_management.intermediate_device())
                            reactive_latents.append(dummy_latent)
                        
                        del single_frame
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                # Clean up chunk to free memory
                del chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            print(f"5a. Completed reactive frame encoding. Total latents: {len(reactive_latents)}")
            
            # Verify all reactive latents have consistent shapes before concatenating
            if reactive_latents:
                # Handle both 4D and 5D latent formats
                first_latent = reactive_latents[0]
                if first_latent.dim() == 5:
                    # 5D format: [batch, frames, channels, height, width]
                    expected_shape = first_latent.shape[2:]  # Skip batch and frames dimensions
                    print(f"Expected 5D reactive latent shape: {expected_shape}")
                    
                    # Validate and fix any mismatched latents
                    for i, latent in enumerate(reactive_latents):
                        if latent.shape[2:] != expected_shape:
                            print(f"Fixing 5D reactive latent {i} shape: {latent.shape} -> {expected_shape}")
                            # Force correct shape for WAN VAE compatibility
                            # WAN VAE produces 10 channels but UNET expects 4
                            if latent.shape[2] == 10:  # WAN VAE 10-channel output
                                print(f"   WAN VAE detected: converting 10 channels to 4 channels")
                                # Take first 4 channels from WAN VAE output
                                corrected_latent = latent[:, :, :4, :, :]
                                reactive_latents[i] = corrected_latent
                                print(f"   Corrected latent shape: {corrected_latent.shape}")
                            else:
                                # Create dummy latent with correct shape
                                reactive_latent = torch.zeros((latent.shape[0], latent.shape[1], 4, height // 8, width // 8), 
                                                            device=latent.device, dtype=latent.dtype)
                                reactive_latents[i] = reactive_latent
                else:
                    # 4D format: [batch, channels, height, width]
                    expected_shape = first_latent.shape[1:]  # Skip batch dimension
                    print(f"Expected 4D reactive latent shape: {expected_shape}")
                    
                    # Validate and fix any mismatched latents
                    for i, latent in enumerate(reactive_latents):
                        if latent.shape[1:] != expected_shape:
                            print(f"Fixing 4D reactive latent {i} shape: {latent.shape} -> {expected_shape}")
                            # Create dummy latent with correct shape
                            reactive_latents[i] = torch.zeros((latent.shape[0], 4, height // 8, width // 8), 
                                                            device=latent.device, dtype=latent.dtype)
            
            # Concatenate along the appropriate dimension
            if reactive_latents and reactive_latents[0].dim() == 5:
                # 5D format: concatenate along frame dimension (dim=1)
                reactive = torch.cat(reactive_latents, dim=1)
                print(f"5a. Concatenated 5D reactive latents: {reactive.shape}")
            else:
                # 4D format: concatenate along batch dimension (dim=0)
                reactive = torch.cat(reactive_latents, dim=0)
                print(f"5a. Concatenated 4D reactive latents: {reactive.shape}")
            del reactive_latents
            
        else:
            # Use regular encoding
            inactive = vae.encode(inactive[:, :, :, :3])
            reactive = vae.encode(reactive[:, :, :, :3])
        
        # Ensure both inactive and reactive have the same tensor format before concatenation
        print(f"5a. Normalizing tensor dimensions for concatenation...")
        print(f"5a. Inactive tensor: {inactive.shape} (dim={inactive.dim()})")
        print(f"5a. Reactive tensor: {reactive.shape} (dim={reactive.dim()})")
        
        # Normalize to 5D format if possible, otherwise to 4D format
        if inactive.dim() == 5 and reactive.dim() == 4:
            # Convert reactive from 4D to 5D: [frames, channels, height, width] -> [1, frames, channels, height, width]
            reactive = reactive.unsqueeze(0)
            print(f"5a. Converted reactive to 5D: {reactive.shape}")
        elif inactive.dim() == 4 and reactive.dim() == 5:
            # Convert inactive from 4D to 5D: [frames, channels, height, width] -> [1, frames, channels, height, width]
            inactive = inactive.unsqueeze(0)
            print(f"5a. Converted inactive to 5D: {inactive.shape}")
        elif inactive.dim() == 4 and reactive.dim() == 4:
            # Both are 4D, keep as is
            print(f"5a. Both tensors are 4D, keeping format")
        elif inactive.dim() == 5 and reactive.dim() == 5:
            # Both are 5D, keep as is
            print(f"5a. Both tensors are 5D, keeping format")
        
        # Now concatenate along the appropriate dimension
        if inactive.dim() == 5 and reactive.dim() == 5:
            # 5D format: concatenate along frame dimension (dim=1)
            control_video_latent = torch.cat((inactive, reactive), dim=1)
            print(f"5a. Concatenated 5D tensors along frame dimension: {control_video_latent.shape}")
        elif inactive.dim() == 4 and reactive.dim() == 4:
            # 4D format: concatenate along frame dimension (dim=0)
            control_video_latent = torch.cat((inactive, reactive), dim=0)
            print(f"5a. Concatenated 4D tensors along frame dimension: {control_video_latent.shape}")
        else:
            # This shouldn't happen, but handle it gracefully
            print(f"5a. Warning: Unexpected tensor dimensions, attempting concatenation...")
            try:
                control_video_latent = torch.cat((inactive, reactive), dim=1)
                print(f"5a. Concatenation successful: {control_video_latent.shape}")
            except Exception as e:
                print(f"5a. Error during concatenation: {e}")
                # Create a dummy control video latent as fallback
                control_video_latent = torch.zeros((1, length, 4, height // 8, width // 8), 
                                                device=comfy.model_management.intermediate_device())
                print(f"5a. Created dummy control video latent: {control_video_latent.shape}")
        
        if reference_image is not None:
            # Encode reference image through VAE to get consistent latent dimensions
            print(f"5a. Encoding reference image through VAE for consistent latent dimensions...")
            print(f"5a. Reference image shape before VAE encoding: {reference_image.shape}")
            
            try:
                # Encode reference image to get latent representation
                reference_latent = vae.encode(reference_image[:, :, :, :3])
                print(f"5a. Reference image encoded successfully: {reference_latent.shape}")
                
                # Ensure reference latent has the same format as control video latent
                if reference_latent.dim() == 4 and control_video_latent.dim() == 5:
                    # Convert 4D reference latent to 5D format to match control video
                    # Add frame dimension: [batch, channels, height, width] -> [batch, 1, channels, height, width]
                    reference_latent = reference_latent.unsqueeze(1)
                    print(f"5a. Converted reference latent to 5D: {reference_latent.shape}")
                
                # Now concatenate along the appropriate dimension
                if control_video_latent.dim() == 5:
                    # 5D format: concatenate along frame dimension (dim=1)
                    control_video_latent = torch.cat((reference_latent, control_video_latent), dim=1)
                    print(f"5a. Concatenated reference and control video latents (5D): {control_video_latent.shape}")
                else:
                    # 4D format: concatenate along batch dimension (dim=0)
                    control_video_latent = torch.cat((reference_latent, control_video_latent), dim=0)
                    print(f"5a. Concatenated reference and control video latents (4D): {control_video_latent.shape}")
                    
            except Exception as e:
                print(f"5a. Warning: Failed to encode reference image: {e}")
                print(f"5a. Skipping reference image concatenation...")
                # Continue without reference image if encoding fails
            
        # Process mask for latent space
        vae_stride = 8
        height_mask = height // vae_stride
        width_mask = width // vae_stride
        mask = mask.view(length, height_mask, vae_stride, width_mask, vae_stride)
        mask = mask.permute(2, 4, 0, 1, 3)
        mask = mask.reshape(vae_stride * vae_stride, length, height_mask, width_mask)
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0), size=(latent_length, height_mask, width_mask), mode='nearest-exact'
        ).squeeze(0)
        
        trim_latent = 0
        if reference_image is not None:
            # Since we're now using VAE-encoded reference images, we need to adjust the mask logic
            # The reference image is now a latent with shape [batch, 1, channels, height, width] for 5D format
            # or [batch, channels, height, width] for 4D format
            
            if control_video_latent.dim() == 5:
                # 5D format: reference is at index 0 in frame dimension
                reference_frames = 1
                print(f"5a. 5D format: reference image contributes 1 frame to mask")
            else:
                # 4D format: reference is at index 0 in batch dimension
                reference_frames = 1
                print(f"5a. 4D format: reference image contributes 1 frame to mask")
            
            # Create mask padding for reference image
            mask_pad = torch.zeros_like(mask[:, :reference_frames, :, :])
            mask = torch.cat((mask_pad, mask), dim=1)
            latent_length += reference_frames
            trim_latent = reference_frames
            print(f"5a. Updated mask and latent length: mask={mask.shape}, latent_length={latent_length}, trim_latent={trim_latent}")
        
        mask = mask.unsqueeze(0)
        
        # Set conditioning values
        positive = self._set_conditioning_values(
            positive, {"vace_frames": [control_video_latent], "vace_mask": [mask], "vace_strength": [strength]}
        )
        negative = self._set_conditioning_values(
            negative, {"vace_frames": [control_video_latent], "vace_mask": [mask], "vace_strength": [strength]}
        )
        
        # Create initial latent - MUST be 4 channels for UNET compatibility
        # WAN VAE produces 10 channels but UNET expects 4 channels
        print(f"5a. Creating final latent with correct shape for UNET compatibility")
        print(f"5a. UNET expects: 4 channels, WAN VAE produces: 10 channels")
        print(f"5a. Using: 4 channels (UNET requirement)")
        
        latent = torch.zeros(
            [batch_size, 4, latent_length, height // 8, width // 8], 
            device=comfy.model_management.intermediate_device()
        )
        out_latent = {"samples": latent}
        
        # Final validation: ensure latent has correct shape for UNET
        final_shape = out_latent["samples"].shape
        print(f"5a. Final latent shape validation:")
        print(f"   Shape: {final_shape}")
        print(f"   Channels: {final_shape[1]} (expected: 4)")
        
        if final_shape[1] != 4:
            print(f"   ❌ ERROR: Wrong channel count! UNET expects 4 channels")
            print(f"   🔧 Forcing correction to 4 channels...")
            # Force correct shape
            corrected_latent = torch.zeros(
                [final_shape[0], 4, final_shape[2], final_shape[3], final_shape[4]], 
                device=out_latent["samples"].device,
                dtype=out_latent["samples"].dtype
            )
            out_latent["samples"] = corrected_latent
            print(f"   ✅ Corrected latent shape: {out_latent['samples'].shape}")
        else:
            print(f"   ✅ SUCCESS: Latent has correct 4 channels for UNET")
        
        return (positive, negative, out_latent, trim_latent)
    
    def _set_conditioning_values(self, conditioning, values, append=True):
        """Helper to set conditioning values - uses ComfyUI's node_helpers exactly"""
        # Import and use ComfyUI's conditioning_set_values function
        from comfy.node_helpers import conditioning_set_values
        return conditioning_set_values(conditioning, values, append) 