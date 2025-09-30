"""
Standalone VAE Implementation
A complete, self-contained implementation of ComfyUI's VAE class
with all dependencies included and using standalone ModelPatcher.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
import json
from typing import Dict, Any, Optional, Tuple, Callable
from standalone_model_patcher import ModelPatcher, create_model_patcher
from wan_vae_components import WanVAE

# ComfyUI-compatible Conv2d with weight/bias casting
class ComfyUICompatibleConv2d(nn.Conv2d):
    """Conv2d layer that mimics ComfyUI's ops.Conv2d weight/bias casting behavior"""
    
    def forward(self, input):
        # Apply weight and bias casting like ComfyUI's ops.Conv2d
        weight = self.weight.to(input.dtype).to(input.device)
        bias = self.bias.to(input.dtype).to(input.device) if self.bias is not None else None
        return self._conv_forward(input, weight, bias)

# ComfyUI-compatible Downsample with weight/bias casting
class ComfyUICompatibleDownsample(nn.Module):
    """Downsample layer that mimics ComfyUI's Downsample behavior"""
    
    def __init__(self, in_channels, with_conv=True, stride=2):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # Use ComfyUI-compatible Conv2d with padding=0 and manual padding
            self.conv = ComfyUICompatibleConv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=0)
    
    def forward(self, x):
        if self.with_conv:
            if x.ndim == 4:
                pad = (0, 1, 0, 1)
                mode = "constant"
                x = torch.nn.functional.pad(x, pad, mode=mode, value=0)
            elif x.ndim == 5:
                pad = (1, 1, 1, 1, 2, 0)
                mode = "replicate"
                x = torch.nn.functional.pad(x, pad, mode=mode)
            x = self.conv(x)
        return x

# ComfyUI-compatible Upsample with weight/bias casting
class ComfyUICompatibleUpsample(nn.Module):
    """Upsample layer that mimics ComfyUI's Upsample behavior"""
    
    def __init__(self, in_channels, with_conv=True, scale_factor=2.0):
        super().__init__()
        self.with_conv = with_conv
        self.scale_factor = scale_factor
        
        if self.with_conv:
            self.conv = ComfyUICompatibleConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
    
    def interpolate_up(self, x, scale_factor):
        """ComfyUI's interpolate_up function"""
        try:
            return torch.nn.functional.interpolate(x, scale_factor=scale_factor, mode="nearest")
        except:  # operation not implemented for bf16
            orig_shape = list(x.shape)
            out_shape = orig_shape[:2]
            for i in range(len(orig_shape) - 2):
                out_shape.append(round(orig_shape[i + 2] * scale_factor[i]))
            out = torch.empty(out_shape, dtype=x.dtype, layout=x.layout, device=x.device)
            split = 8
            l = out.shape[1] // split
            for i in range(0, out.shape[1], l):
                out[:,i:i+l] = torch.nn.functional.interpolate(x[:,i:i+l].to(torch.float32), scale_factor=scale_factor, mode="nearest").to(x.dtype)
            return out
    
    def forward(self, x):
        scale_factor = self.scale_factor
        if isinstance(scale_factor, (int, float)):
            scale_factor = (scale_factor,) * (x.ndim - 2)

        if x.ndim == 5 and scale_factor[0] > 1.0:
            t = x.shape[2]
            if t > 1:
                a, b = x.split((1, t - 1), dim=2)
                del x
                b = self.interpolate_up(b, scale_factor)
            else:
                a = x

            a = self.interpolate_up(a.squeeze(2), scale_factor=scale_factor[1:]).unsqueeze(2)
            if t > 1:
                x = torch.cat((a, b), dim=2)
            else:
                x = a
        else:
            x = self.interpolate_up(x, scale_factor)
        
        if self.with_conv:
            x = self.conv(x)
        return x

# ComfyUI-compatible GroupNorm with weight/bias casting
class ComfyUICompatibleGroupNorm(nn.GroupNorm):
    """GroupNorm layer that mimics ComfyUI's ops.GroupNorm weight/bias casting behavior"""
    
    def forward(self, input):
        # Apply weight and bias casting like ComfyUI's ops.GroupNorm
        weight = self.weight.to(input.dtype).to(input.device) if self.weight is not None else None
        bias = self.bias.to(input.dtype).to(input.device) if self.bias is not None else None
        return F.group_norm(input, self.num_groups, weight, bias, self.eps)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def dtype_size(dtype):
    """Get size of dtype in bytes"""
    if dtype == torch.float32:
        return 4
    elif dtype == torch.float16:
        return 2
    elif dtype == torch.bfloat16:
        return 2
    elif dtype == torch.float8_e4m3fn or dtype == torch.float8_e5m2:
        return 1
    else:
        return 4  # Default to float32 size


def get_torch_device():
    """Get the best available torch device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def vae_device():
    """Get VAE device"""
    return get_torch_device()


def vae_offload_device():
    """Get VAE offload device"""
    return torch.device("cpu")


def intermediate_device():
    """Get intermediate device"""
    return get_torch_device()


def vae_dtype(device, working_dtypes):
    """Get optimal VAE dtype"""
    if device.type == "cpu":
        return torch.float32
    else:
        # Prefer float16 for GPU
        if torch.float16 in working_dtypes:
            return torch.float16
        elif torch.bfloat16 in working_dtypes:
            return torch.bfloat16
        else:
            return torch.float32


# ============================================================================
# BASIC VAE COMPONENTS
# ============================================================================

class DiagonalGaussianRegularizer(nn.Module):
    """Diagonal Gaussian Regularizer for VAE"""
    def __init__(self):
        super().__init__()
    
    def forward(self, sample):
        mean, logvar = torch.chunk(sample, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(std)
        return z, mean, logvar


class ResnetBlock(nn.Module):
    """ResNet block for VAE"""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.norm1 = ComfyUICompatibleGroupNorm(32, in_channels)
        self.conv1 = ComfyUICompatibleConv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = ComfyUICompatibleGroupNorm(32, out_channels)
        self.conv2 = ComfyUICompatibleConv2d(out_channels, out_channels, 3, padding=1)
        self.dropout = torch.nn.Dropout(dropout, inplace=True)
        
        if in_channels != out_channels:
            self.nin_shortcut = ComfyUICompatibleConv2d(in_channels, out_channels, 1)
        else:
            self.nin_shortcut = nn.Identity()
    
    def forward(self, x):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = F.silu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.nin_shortcut(x)


class AttnBlock(nn.Module):
    """Attention block for VAE"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = ComfyUICompatibleGroupNorm(32, channels)
        self.q = ComfyUICompatibleConv2d(channels, channels, 1)
        self.k = ComfyUICompatibleConv2d(channels, channels, 1)
        self.v = ComfyUICompatibleConv2d(channels, channels, 1)
        self.proj_out = ComfyUICompatibleConv2d(channels, channels, 1)
    
    def forward(self, x):
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        
        # Compute attention
        b, c, h, w = q.shape
        q = q.view(b, c, h * w).transpose(1, 2)
        k = k.view(b, c, h * w)
        v = v.view(b, c, h * w).transpose(1, 2)
        
        attn = torch.bmm(q, k) * (c ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        h = torch.bmm(attn, v)
        h = h.transpose(1, 2).view(b, c, h, w)
        h = self.proj_out(h)
        
        return x + h


class Encoder(nn.Module):
    """VAE Encoder"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ch = config['ch']
        self.ch_mult = config['ch_mult']
        self.num_res_blocks = config['num_res_blocks']
        self.attn_resolutions = config.get('attn_resolutions', [])
        self.dropout = config.get('dropout', 0.0)
        self.in_channels = config['in_channels']
        self.resolution = config['resolution']
        
        self.conv_in = ComfyUICompatibleConv2d(self.in_channels, self.ch, 3, padding=1)
        
        self.down = nn.ModuleList()
        ch = self.ch
        ds = 1
        resolution = self.resolution
        
        for i_level, mult in enumerate(self.ch_mult):
            for i_block in range(self.num_res_blocks):
                self.down.append(ResnetBlock(ch, mult * self.ch, dropout=self.dropout))
                ch = mult * self.ch
                if resolution in self.attn_resolutions:
                    self.down.append(AttnBlock(ch))
            
            if i_level != len(self.ch_mult) - 1:
                self.down.append(ComfyUICompatibleDownsample(ch, with_conv=True, stride=2))
                ds *= 2
                resolution //= 2
        
        self.mid_block_1 = ResnetBlock(ch, ch, dropout=self.dropout)
        self.mid_attn_1 = AttnBlock(ch)
        self.mid_block_2 = ResnetBlock(ch, ch, dropout=self.dropout)
        
        self.norm_out = ComfyUICompatibleGroupNorm(32, ch)
        self.conv_out = ComfyUICompatibleConv2d(ch, config['z_channels'] * 2, 3, padding=1)
    
    def forward(self, x):
        h = self.conv_in(x)
        
        for layer in self.down:
            h = layer(h)
        
        h = self.mid_block_1(h)
        h = self.mid_attn_1(h)
        h = self.mid_block_2(h)
        
        h = F.silu(self.norm_out(h))
        h = self.conv_out(h)
        
        return h


class Decoder(nn.Module):
    """VAE Decoder"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ch = config['ch']
        self.ch_mult = config['ch_mult']
        self.num_res_blocks = config['num_res_blocks']
        self.attn_resolutions = config.get('attn_resolutions', [])
        self.dropout = config.get('dropout', 0.0)
        self.out_ch = config['out_ch']
        self.z_channels = config['z_channels']
        self.resolution = config['resolution']
        
        self.conv_in = ComfyUICompatibleConv2d(self.z_channels, self.ch, 3, padding=1)
        
        self.up = nn.ModuleList()
        ch = self.ch
        ds = 1
        resolution = self.resolution // (2 ** (len(self.ch_mult) - 1))
        
        for i_level, mult in enumerate(reversed(self.ch_mult)):
            for i_block in range(self.num_res_blocks):
                self.up.append(ResnetBlock(ch, mult * self.ch, dropout=self.dropout))
                ch = mult * self.ch
                if resolution in self.attn_resolutions:
                    self.up.append(AttnBlock(ch))
            
            if i_level != len(self.ch_mult) - 1:
                self.up.append(ComfyUICompatibleUpsample(ch, with_conv=True, scale_factor=2.0))
                ds *= 2
                resolution *= 2
        
        self.mid_block_1 = ResnetBlock(ch, ch, dropout=self.dropout)
        self.mid_attn_1 = AttnBlock(ch)
        self.mid_block_2 = ResnetBlock(ch, ch, dropout=self.dropout)
        
        self.norm_out = ComfyUICompatibleGroupNorm(32, ch)
        self.conv_out = ComfyUICompatibleConv2d(ch, self.out_ch, 3, padding=1)
    
    def forward(self, z):
        h = self.conv_in(z)
        
        for layer in self.up:
            h = layer(h)
        
        h = self.mid_block_1(h)
        h = self.mid_attn_1(h)
        h = self.mid_block_2(h)
        
        h = F.silu(self.norm_out(h))
        h = self.conv_out(h)
        
        return h


class AutoencoderKL(nn.Module):
    """AutoencoderKL implementation"""
    def __init__(self, ddconfig, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder = Encoder(ddconfig)
        self.decoder = Decoder(ddconfig)
        self.quant_conv = nn.Conv2d(ddconfig['z_channels'] * 2, embed_dim * 2, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig['z_channels'], 1)
        self.regularizer = DiagonalGaussianRegularizer()
    
    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        z, mean, logvar = self.regularizer(moments)
        # For ComfyUI WAN VAE compatibility: return only mean (mu) like ComfyUI
        return mean
    
    def decode(self, z):
        z = self.post_quant_conv(z)
        return self.decoder(z)
    
    def forward(self, x):
        z, mean, logvar = self.encode(x)
        dec = self.decode(z)
        return dec, mean, logvar


class AutoencodingEngine(nn.Module):
    """AutoencodingEngine implementation"""
    def __init__(self, regularizer_config, encoder_config, decoder_config):
        super().__init__()
        self.encoder = Encoder(encoder_config['params'])
        self.decoder = Decoder(decoder_config['params'])
        self.regularizer = DiagonalGaussianRegularizer()
    
    def encode(self, x):
        h = self.encoder(x)
        z, mean, logvar = self.regularizer(h)
        return z, mean, logvar
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z, mean, logvar = self.encode(x)
        dec = self.decode(z)
        return dec, mean, logvar


# ============================================================================
# WAN VAE IMPLEMENTATION
# ============================================================================

class WanVAE_Simplified(nn.Module):  # Our simplified version
    """WAN VAE implementation"""
    def __init__(self, dim=96, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2, 
                 attn_scales=[], temperal_downsample=[False, True, True], dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.dropout = dropout
        
        # Build encoder
        self.encoder = self._build_encoder()
        
        # Build decoder
        self.decoder = self._build_decoder()
        
        # Regularizer
        self.regularizer = DiagonalGaussianRegularizer()
    
    def _build_encoder(self):
        """Build encoder network"""
        layers = []
        
        # Input convolution
        layers.append(nn.Conv2d(3, self.dim, 3, padding=1))
        
        # Downsampling blocks
        ch = self.dim
        for i, mult in enumerate(self.dim_mult):
            for j in range(self.num_res_blocks):
                layers.append(ResnetBlock(ch, mult * self.dim, dropout=self.dropout))
                ch = mult * self.dim
                
                # Add attention if specified
                if i in self.attn_scales:
                    layers.append(AttnBlock(ch))
            
            # Downsample (except last level)
            if i < len(self.dim_mult) - 1:
                layers.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
        
        # Middle blocks
        layers.append(ResnetBlock(ch, ch, dropout=self.dropout))
        layers.append(AttnBlock(ch))
        layers.append(ResnetBlock(ch, ch, dropout=self.dropout))
        
        # Output
        layers.append(nn.GroupNorm(32, ch))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(ch, self.z_dim * 2, 3, padding=1))
        
        return nn.Sequential(*layers)
    
    def _build_decoder(self):
        """Build decoder network"""
        layers = []
        
        # Input convolution
        layers.append(nn.Conv2d(self.z_dim, self.dim * self.dim_mult[-1], 3, padding=1))
        
        # Middle blocks
        ch = self.dim * self.dim_mult[-1]
        layers.append(ResnetBlock(ch, ch, dropout=self.dropout))
        layers.append(AttnBlock(ch))
        layers.append(ResnetBlock(ch, ch, dropout=self.dropout))
        
        # Upsampling blocks
        for i, mult in enumerate(reversed(self.dim_mult)):
            for j in range(self.num_res_blocks):
                layers.append(ResnetBlock(ch, mult * self.dim, dropout=self.dropout))
                ch = mult * self.dim
                
                # Add attention if specified
                if (len(self.dim_mult) - 1 - i) in self.attn_scales:
                    layers.append(AttnBlock(ch))
            
            # Upsample (except last level)
            if i < len(self.dim_mult) - 1:
                layers.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
        
        # Output
        layers.append(nn.GroupNorm(32, ch))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(ch, 3, 3, padding=1))
        
        return nn.Sequential(*layers)
    
    def encode(self, x, dtype=None):
        x = x.to(dtype) if dtype is not None else x
        h = self.encoder(x)
        z, mean, logvar = self.regularizer(h)
        return z, mean, logvar
    
    def decode(self, z, dtype=None):
        z = z.to(dtype) if dtype is not None else z
        return self.decoder(z)
    
    def forward(self, x):
        z, mean, logvar = self.encode(x)
        dec = self.decode(z)
        return dec, mean, logvar


# ============================================================================
# SPECIALIZED VAE MODELS
# ============================================================================

class TAESD(nn.Module):
    """Tiny Autoencoder for Stable Diffusion"""
    def __init__(self, latent_channels=4):
        super().__init__()
        self.latent_channels = latent_channels
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, latent_channels, 3, padding=1),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


class StageA(nn.Module):
    """Stage A VAE for Stable Cascade"""
    def __init__(self):
        super().__init__()
        # Simplified implementation
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 16, 3, padding=1),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(16, 256, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 3, 3, padding=1),
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


class StageC_coder(nn.Module):
    """Stage C coder for Stable Cascade"""
    def __init__(self):
        super().__init__()
        # Simplified implementation
        self.encoder = nn.Sequential(
            nn.Conv2d(16, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 16, 3, padding=1),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(16, 128, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 16, 3, padding=1),
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


class AudioOobleckVAE(nn.Module):
    """Audio Oobleck VAE"""
    def __init__(self):
        super().__init__()
        # Simplified audio VAE implementation
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv1d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 2, 3, padding=1),
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


# ============================================================================
# MAIN VAE CLASS
# ============================================================================

class VAE:
    """
    Standalone VAE implementation that supports multiple VAE types
    without depending on ComfyUI modules.
    """
    
    def __init__(self, sd=None, device=None, config=None, dtype=None, metadata=None):
        if sd is None:
            sd = {}
        
        # Initialize default properties
        self.memory_used_encode = lambda shape, dtype: (1767 * shape[2] * shape[3]) * dtype_size(dtype)
        self.memory_used_decode = lambda shape, dtype: (2178 * shape[2] * shape[3] * 64) * dtype_size(dtype)
        self.downscale_ratio = 8
        self.upscale_ratio = 8
        self.latent_channels = 4
        self.latent_dim = 2
        self.output_channels = 3
        self.process_input = lambda image: image * 2.0 - 1.0
        self.process_output = lambda image: torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
        self.working_dtypes = [torch.bfloat16, torch.float32]
        self.disable_offload = False
        
        self.downscale_index_formula = None
        self.upscale_index_formula = None
        self.extra_1d_channel = None
        
        # Initialize latent format for proper scaling (ComfyUI style)
        # self.latent_format = None
        # self._init_latent_format()
        
        # Detect VAE type and initialize
        if config is None:
            self._detect_and_init_vae(sd, metadata)
        else:
            self.first_stage_model = AutoencoderKL(**(config['params']))
        
        if self.first_stage_model is not None:
            self.first_stage_model = self.first_stage_model.eval()
            
            # Load state dict
            m, u = self.first_stage_model.load_state_dict(sd, strict=False)
            if len(m) > 0:
                logging.warning("Missing VAE keys {}".format(m))
            if len(u) > 0:
                logging.debug("Leftover VAE keys {}".format(u))
        
        # Set up device and dtype
        if device is None:
            device = vae_device()
        self.device = device
        offload_device = vae_offload_device()
        
        if dtype is None:
            dtype = vae_dtype(self.device, self.working_dtypes)
        self.vae_dtype = dtype
        
        if self.first_stage_model is not None:
            self.first_stage_model.to(self.vae_dtype)
        
        self.output_device = intermediate_device()
        
        # Create ModelPatcher
        if self.first_stage_model is not None:
            self.patcher = create_model_patcher(
                self.first_stage_model, 
                load_device=self.device, 
                offload_device=offload_device
            )
            logging.info("VAE load device: {}, offload device: {}, dtype: {}".format(
                self.device, offload_device, self.vae_dtype))
        else:
            self.patcher = None
    
    def _detect_and_init_vae(self, sd, metadata):
        # Debug prints removed for production
        """Detect VAE type and initialize appropriate model"""
        
        # Check for diffusers format
        if 'decoder.up_blocks.0.resnets.0.norm1.weight' in sd.keys():
            sd = self._convert_diffusers_vae(sd)
        
        # Detect VAE type based on keys
        if "decoder.mid.block_1.mix_factor" in sd:
            # Video VAE
            encoder_config = {
                'double_z': True, 'z_channels': 4, 'resolution': 256, 
                'in_channels': 3, 'out_ch': 3, 'ch': 128, 
                'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 
                'attn_resolutions': [], 'dropout': 0.0
            }
            decoder_config = encoder_config.copy()
            decoder_config["video_kernel_size"] = [3, 1, 1]
            decoder_config["alpha"] = 0.0
            self.first_stage_model = AutoencodingEngine(
                regularizer_config={'target': "DiagonalGaussianRegularizer"},
                encoder_config={'target': "Encoder", 'params': encoder_config},
                decoder_config={'target': "VideoDecoder", 'params': decoder_config}
            )
            
        elif "taesd_decoder.1.weight" in sd:
            # TAESD
            self.latent_channels = sd["taesd_decoder.1.weight"].shape[1]
            self.first_stage_model = TAESD(latent_channels=self.latent_channels)
            
        elif "vquantizer.codebook.weight" in sd:
            # VQGan (Stage A)
            self.first_stage_model = StageA()
            self.downscale_ratio = 4
            self.upscale_ratio = 4
            self.process_input = lambda image: image * 2.0 - 1.0
            self.process_output = lambda image: torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
            
        elif "backbone.1.0.block.0.1.num_batches_tracked" in sd:
            # EffNet encoder
            self.first_stage_model = StageC_coder()
            self.downscale_ratio = 32
            self.latent_channels = 16
            new_sd = {}
            for k in sd:
                new_sd["encoder.{}".format(k)] = sd[k]
            sd = new_sd
            
        elif "blocks.11.num_batches_tracked" in sd:
            # Previewer decoder
            self.first_stage_model = StageC_coder()
            self.latent_channels = 16
            new_sd = {}
            for k in sd:
                new_sd["previewer.{}".format(k)] = sd[k]
            sd = new_sd
            
        elif "encoder.backbone.1.0.block.0.1.num_batches_tracked" in sd:
            # Combined effnet and previewer
            self.first_stage_model = StageC_coder()
            self.downscale_ratio = 32
            self.latent_channels = 16
            
        elif "decoder.middle.0.residual.0.gamma" in sd:
            # WAN VAE detection
            # WAN VAE detected - setting correct process_input/output
            self.process_input = lambda image: image * 2.0 - 1.0
            self.process_output = lambda image: torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
            if "decoder.upsamples.0.upsamples.0.residual.2.weight" in sd:  # Wan 2.2 VAE
                self.upscale_ratio = (lambda a: max(0, a * 4 - 3), 16, 16)
                self.upscale_index_formula = (4, 16, 16)
                self.downscale_ratio = (lambda a: max(0, math.floor((a + 3) / 4)), 16, 16)
                self.downscale_index_formula = (4, 16, 16)
                self.latent_dim = 3  # Real WAN VAE uses 3D encoder
                self.latent_channels = 48
                ddconfig = {"dim": 160, "z_dim": self.latent_channels, "dim_mult": [1, 2, 4, 4], "num_res_blocks": 2, "attn_scales": [], "temperal_downsample": [False, True, True], "dropout": 0.0}
                from wan_vae_components.vae import WanVAE
                self.first_stage_model = WanVAE(**ddconfig)
                self.working_dtypes = [torch.bfloat16, torch.float16, torch.float32]
                self.memory_used_encode = lambda shape, dtype: 3300 * shape[3] * shape[4] * dtype_size(dtype)
                self.memory_used_decode = lambda shape, dtype: 8000 * shape[3] * shape[4] * (16 * 16) * dtype_size(dtype)
            else:  # Wan 2.1 VAE
                self.upscale_ratio = (lambda a: max(0, a * 4 - 3), 8, 8)
                self.upscale_index_formula = (4, 8, 8)
                self.downscale_ratio = (lambda a: max(0, math.floor((a + 3) / 4)), 8, 8)
                self.downscale_index_formula = (4, 8, 8)
                self.latent_dim = 3  # Real WAN VAE uses 3D encoder
                self.latent_channels = 16
                ddconfig = {"dim": 96, "z_dim": self.latent_channels, "dim_mult": [1, 2, 4, 4], "num_res_blocks": 2, "attn_scales": [], "temperal_downsample": [False, True, True], "dropout": 0.0}
                from wan_vae_components.vae import WanVAE
                self.first_stage_model = WanVAE(**ddconfig)
                self.working_dtypes = [torch.bfloat16, torch.float16, torch.float32]
                self.memory_used_encode = lambda shape, dtype: 6000 * shape[3] * shape[4] * dtype_size(dtype)
                self.memory_used_decode = lambda shape, dtype: 7000 * shape[3] * shape[4] * (8 * 8) * dtype_size(dtype)
                
        elif "decoder.conv_in.weight" in sd:
            # Standard SD VAE
            ddconfig = {
                'double_z': True, 'z_channels': 4, 'resolution': 256, 
                'in_channels': 3, 'out_ch': 3, 'ch': 128, 
                'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 
                'attn_resolutions': [], 'dropout': 0.0
            }
            
            # Check for x4 upscaler
            if 'encoder.down.2.downsample.conv.weight' not in sd and 'decoder.up.3.upsample.conv.weight' not in sd:
                ddconfig['ch_mult'] = [1, 2, 4]
                self.downscale_ratio = 4
                self.upscale_ratio = 4
            
            self.latent_channels = ddconfig['z_channels'] = sd["decoder.conv_in.weight"].shape[1]
            
            if 'post_quant_conv.weight' in sd:
                self.first_stage_model = AutoencoderKL(ddconfig=ddconfig, embed_dim=sd['post_quant_conv.weight'].shape[1])
            else:
                self.first_stage_model = AutoencodingEngine(
                    regularizer_config={'target': "DiagonalGaussianRegularizer"},
                    encoder_config={'target': "Encoder", 'params': ddconfig},
                    decoder_config={'target': "Decoder", 'params': ddconfig}
                )
                
        elif "decoder.layers.1.layers.0.beta" in sd:
            # Audio VAE
            # Audio VAE detected
            self.first_stage_model = AudioOobleckVAE()
            self.memory_used_encode = lambda shape, dtype: (1000 * shape[2]) * dtype_size(dtype)
            self.memory_used_decode = lambda shape, dtype: (1000 * shape[2] * 2048) * dtype_size(dtype)
            self.latent_channels = 64
            self.output_channels = 2
            self.upscale_ratio = 2048
            self.downscale_ratio = 2048
            self.latent_dim = 1
            self.process_output = lambda audio: audio
            self.process_input = lambda audio: audio
            self.working_dtypes = [torch.float16, torch.bfloat16, torch.float32]
            self.disable_offload = True
            
            
        else:
            logging.warning("WARNING: No VAE weights detected, VAE not initialized.")
            self.first_stage_model = None
    
    def _convert_diffusers_vae(self, sd):
        """Convert diffusers VAE format to standard format"""
        # Simplified conversion - in practice this would be more complex
        new_sd = {}
        for k, v in sd.items():
            # Basic key mapping
            if k.startswith("encoder."):
                new_sd[k[8:]] = v  # Remove "encoder." prefix
            elif k.startswith("decoder."):
                new_sd[k[8:]] = v  # Remove "decoder." prefix
            else:
                new_sd[k] = v
        return new_sd
    
    def _init_latent_format(self):
        """Initialize latent format for proper scaling (ComfyUI style)"""
        try:
            # Import Wan21 latent format from ComfyUI
            import sys
            import os
            comfy_path = os.path.join(os.path.dirname(__file__), '..', 'comfy')
            if comfy_path not in sys.path:
                sys.path.insert(0, comfy_path)
            
            from latent_formats import Wan21
            self.latent_format = Wan21()
            print(f"✅ Initialized Wan21 latent format for proper scaling")
        except ImportError:
            print(f"⚠️  Could not import Wan21 latent format, using fallback scaling")
            self.latent_format = None
    
    def throw_exception_if_invalid(self):
        """Check if VAE is valid"""
        if self.first_stage_model is None:
            raise RuntimeError("ERROR: VAE is invalid: None\n\nIf the VAE is from a checkpoint loader node your checkpoint does not contain a valid VAE.")
    
    def spacial_compression_encode(self):
        """Get spatial compression ratio for encoding"""
        try:
            return self.downscale_ratio[-1]
        except:
            return self.downscale_ratio
    
    def spacial_compression_decode(self):
        """Get spatial compression ratio for decoding"""
        try:
            return self.upscale_ratio[-1]
        except:
            return self.upscale_ratio
    
    def temporal_compression_decode(self):
        """Get temporal compression ratio for decoding"""
        try:
            return round(self.upscale_ratio[0](8192) / 8192)
        except:
            return None
    
    def vae_encode_crop_pixels(self, pixels):
        """Crop pixels to be divisible by downscale ratio"""
        downscale_ratio = self.spacial_compression_encode()
        
        # For 5D input [B, C, T, H, W], we want spatial dims [T, H, W]
        # For 4D input [B, H, W, C], we want spatial dims [H, W]
        if pixels.ndim == 5:  # [B, C, T, H, W]
            dims = pixels.shape[2:]  # [T, H, W] - spatial dimensions only
            start_dim = 2  # Start from dimension 2 (T)
        else:  # [B, H, W, C] or similar
            dims = pixels.shape[1:-1]  # [H, W] - spatial dimensions only
            start_dim = 1  # Start from dimension 1 (H)
        
        for d in range(len(dims)):
            x = (dims[d] // downscale_ratio) * downscale_ratio
            x_offset = (dims[d] % downscale_ratio) // 2
            if x != dims[d] and x > 0:  # Only crop if result is positive
                pixels = pixels.narrow(start_dim + d, x_offset, x)
            elif x == 0:  # Handle case where x becomes 0
                # For very small dimensions, don't crop to avoid empty tensors
                print(f"⚠️  Warning: Dimension {d} would become 0 after cropping, keeping original size")
        return pixels
    
    def encode(self, pixel_samples):
        """Encode input to latent space with proper downscaling logic"""
        self.throw_exception_if_invalid()
        
        # Crop pixels to be divisible by downscale ratio
        pixel_samples = self.vae_encode_crop_pixels(pixel_samples)
        
        # Move channel dimension to correct position (ComfyUI style)
        # Only apply movedim for 4D tensors [B, H, W, C] -> [B, C, H, W]
        if pixel_samples.ndim == 4:
            pixel_samples = pixel_samples.movedim(-1, 1)
        # For 5D tensors [B, C, T, H, W], no dimension change needed
        if self.latent_dim == 3 and pixel_samples.ndim < 5:
            pixel_samples = pixel_samples.movedim(1, 0).unsqueeze(0)
        
        try:
            # Calculate memory usage
            memory_used = self.memory_used_encode(pixel_samples.shape, self.vae_dtype)
            
            # Load models to GPU (ComfyUI style)
            from wan_vae_components.model_management import load_models_gpu, get_free_memory
            load_models_gpu([self.patcher], memory_required=memory_used, force_full_load=self.disable_offload)
            free_memory = get_free_memory(self.device)
            batch_number = int(free_memory / max(1, memory_used))
            batch_number = max(1, batch_number)
            
            samples = None
            for x in range(0, pixel_samples.shape[0], batch_number):
                pixels_in = self.process_input(pixel_samples[x:x + batch_number]).to(self.vae_dtype).to(self.device)
                out = self.first_stage_model.encode(pixels_in).to(self.output_device).float()
                if samples is None:
                    samples = torch.empty((pixel_samples.shape[0],) + tuple(out.shape[1:]), device=self.output_device)
                samples[x:x + batch_number] = out
                
        except Exception as e:
            # Check if it's an OOM error and try tiled encoding
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                logging.warning("Warning: Ran out of memory when regular VAE encoding, retrying with tiled VAE encoding.")
                # TODO: Implement tiled encoding fallback
                raise e
            else:
                logging.warning(f"Warning: VAE encoding failed: {e}")
                raise e
        
        # Video reshape logic removed - ComfyUI doesn't do this in encode method
        
        return samples
    
    def decode(self, z):
        """Decode latent to output space"""
        if self.first_stage_model is None:
            raise RuntimeError("VAE not initialized")
        
        # Decode (ComfyUI style - no dtype parameter)
        if hasattr(self.first_stage_model, 'decode'):
            x = self.first_stage_model.decode(z)
        else:
            # Fallback for models without decode method
            x = self.first_stage_model.decoder(z)
        
        # Process output
        x = self.process_output(x)
        
        return x
    
    def forward(self, x):
        """Forward pass through VAE"""
        if self.first_stage_model is None:
            raise RuntimeError("VAE not initialized")
        
        if hasattr(self.first_stage_model, 'forward'):
            return self.first_stage_model(x)
        else:
            # Fallback
            z, mean, logvar = self.encode(x)
            x_recon = self.decode(z)
            return x_recon, mean, logvar
    
    def load_state_dict(self, state_dict, strict=False):
        """Load state dict into VAE"""
        if self.first_stage_model is None:
            raise RuntimeError("VAE not initialized")
        
        return self.first_stage_model.load_state_dict(state_dict, strict=strict)
    
    def to(self, device):
        """Move VAE to device"""
        if self.first_stage_model is not None:
            self.first_stage_model.to(device)
        self.device = device
        return self
    
    def eval(self):
        """Set VAE to evaluation mode"""
        if self.first_stage_model is not None:
            self.first_stage_model.eval()
        return self
    
    def train(self):
        """Set VAE to training mode"""
        if self.first_stage_model is not None:
            self.first_stage_model.train()
        return self


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_vae(state_dict=None, device=None, dtype=None, metadata=None):
    """Convenience function to create VAE"""
    return VAE(sd=state_dict, device=device, dtype=dtype, metadata=metadata)


def load_vae_from_checkpoint(checkpoint_path, device=None, dtype=None):
    """Load VAE from checkpoint file"""
    import torch
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract VAE state dict
    vae_sd = {}
    for k, v in checkpoint.items():
        if k.startswith('vae.'):
            vae_sd[k[4:]] = v  # Remove 'vae.' prefix
    
    return VAE(sd=vae_sd, device=device, dtype=dtype)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    print("Standalone VAE Implementation")
    print("This provides VAE functionality without ComfyUI dependencies")
    
    # Create a simple VAE
    vae = create_vae()
    print(f"VAE created with {vae.latent_channels} latent channels")
    print(f"Downscale ratio: {vae.downscale_ratio}, Upscale ratio: {vae.upscale_ratio}")
    
    # Test with dummy data
    if vae.first_stage_model is not None:
        dummy_input = torch.randn(1, 3, 64, 64)
        print(f"Input shape: {dummy_input.shape}")
        
        with torch.no_grad():
            output = vae.forward(dummy_input)
            if isinstance(output, tuple):
                print(f"Output shape: {output[0].shape}")
            else:
                print(f"Output shape: {output.shape}")
    
    print("Standalone VAE implementation complete!")
