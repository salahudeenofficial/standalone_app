"""
3D WAN VAE Components
Implements CausalConv3d and other 3D components for proper video processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalConv3d(nn.Module):
    """Causal 3D Convolution for video processing"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        
        # Create 3D convolution
        self.conv3d = nn.Conv3d(in_channels, out_channels, self.kernel_size, 
                               stride=self.stride, padding=self.padding, bias=bias)
        
        # For causal convolution, we need to handle temporal dimension specially
        self.temporal_padding = (self.kernel_size[0] - 1, 0, 0) if len(self.kernel_size) == 3 else (self.kernel_size[0] - 1, 0)
    
    def forward(self, x, feat_cache=None):
        # Apply causal padding to temporal dimension
        if len(x.shape) == 5:  # [B, C, T, H, W]
            x = F.pad(x, self.temporal_padding, mode='constant', value=0)
        
        return self.conv3d(x)


class ResidualBlock3d(nn.Module):
    """3D Residual Block for video processing"""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = CausalConv3d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = CausalConv3d(out_channels, out_channels, 3, padding=1)
        self.dropout = nn.Dropout(dropout)
        
        if in_channels != out_channels:
            self.nin_shortcut = CausalConv3d(in_channels, out_channels, 1)
        else:
            self.nin_shortcut = nn.Identity()
    
    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = F.relu(self.norm1(x))
        h = self.conv1(h, feat_cache)
        h = F.relu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h, feat_cache)
        
        return h + self.nin_shortcut(x)


class AttentionBlock3d(nn.Module):
    """3D Attention Block for video processing"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(32, channels)
        self.q = CausalConv3d(channels, channels, 1)
        self.k = CausalConv3d(channels, channels, 1)
        self.v = CausalConv3d(channels, channels, 1)
        self.proj_out = CausalConv3d(channels, channels, 1)
    
    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.norm(x)
        q = self.q(h, feat_cache)
        k = self.k(h, feat_cache)
        v = self.v(h, feat_cache)
        
        # Compute attention
        b, c, t, h, w = q.shape
        q = q.view(b, c, t * h * w).transpose(1, 2)
        k = k.view(b, c, t * h * w)
        v = v.view(b, c, t * h * w).transpose(1, 2)
        
        attn = torch.bmm(q, k) * (c ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        h = torch.bmm(attn, v)
        h = h.transpose(1, 2).view(b, c, t, h, w)
        h = self.proj_out(h, feat_cache)
        
        return x + h


class Resample3d(nn.Module):
    """3D Resampling for downsampling/upsampling"""
    def __init__(self, channels, mode='downsample3d'):
        super().__init__()
        self.channels = channels
        self.mode = mode
        
        if mode == 'downsample3d':
            self.conv = CausalConv3d(channels, channels, 3, stride=(2, 2, 2), padding=1)
        elif mode == 'upsample3d':
            self.conv = nn.ConvTranspose3d(channels, channels, 3, stride=(2, 2, 2), padding=1, output_padding=1)
        elif mode == 'downsample2d':
            self.conv = CausalConv3d(channels, channels, 3, stride=(1, 2, 2), padding=1)
        elif mode == 'upsample2d':
            self.conv = nn.ConvTranspose3d(channels, channels, 3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1))
    
    def forward(self, x, feat_cache=None, feat_idx=[0]):
        return self.conv(x)


class RMSNorm(nn.Module):
    """Root Mean Square Normalization"""
    def __init__(self, dim, images=False):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.images = images
    
    def forward(self, x):
        if self.images:
            # For images: normalize over spatial dimensions
            norm = x.norm(dim=(2, 3), keepdim=True) / math.sqrt(x.shape[2] * x.shape[3])
        else:
            # For features: normalize over channel dimension
            norm = x.norm(dim=1, keepdim=True) / math.sqrt(x.shape[1])
        
        return x / (norm + 1e-8) * self.scale
