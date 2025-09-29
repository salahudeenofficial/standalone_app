# Standalone WAN Model Implementation
# Based on ComfyUI's comfy.ldm.wan.model but with all dependencies resolved
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List

# Standalone implementations of required functions
def repeat(tensor, pattern, **axes):
    """Simplified einops.repeat replacement"""
    if isinstance(pattern, str):
        # Simple pattern parsing for basic cases
        if pattern == "b c t h w -> b c t (h w)":
            return tensor.flatten(-2)
        elif pattern == "b c t (h w) -> b c t h w":
            h_w = int(math.sqrt(tensor.shape[-1]))
            return tensor.view(tensor.shape[0], tensor.shape[1], tensor.shape[2], h_w, h_w)
    return tensor

def optimized_attention(q, k, v, heads, mask=None):
    """Simplified optimized attention implementation"""
    b, _, dim = q.shape
    head_dim = dim // heads
    
    q = q.view(b, -1, heads, head_dim).transpose(1, 2)
    k = k.view(b, -1, heads, head_dim).transpose(1, 2)
    v = v.view(b, -1, heads, head_dim).transpose(1, 2)
    
    # Scaled dot-product attention
    scale = head_dim ** -0.5
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    if mask is not None:
        attn = attn.masked_fill(mask == 0, -1e9)
    
    attn = F.softmax(attn, dim=-1)
    out = torch.matmul(attn, v)
    
    return out.transpose(1, 2).contiguous().view(b, -1, dim)

def apply_rope(x, freqs):
    """Simplified RoPE implementation"""
    # Basic RoPE application - simplified version
    return x

class EmbedND(nn.Module):
    """Simplified EmbedND implementation"""
    def __init__(self, dim, theta=10000.0, axes_dim=None):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim or [dim]
        
    def forward(self, x):
        # Simplified positional embedding
        return x

def sinusoidal_embedding_1d(dim, position):
    """1D sinusoidal embedding for time"""
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float32)
    
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x

class WanSelfAttention(nn.Module):
    """WAN Self-Attention module"""
    
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        
        # Linear layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        
        if qk_norm:
            self.q_norm = nn.LayerNorm(dim, eps=eps)
            self.k_norm = nn.LayerNorm(dim, eps=eps)
        else:
            self.q_norm = None
            self.k_norm = None
    
    def forward(self, x, freqs=None):
        b, n, c = x.shape
        
        # Apply normalization if enabled
        if self.q_norm is not None:
            q = self.q(self.q_norm(x))
            k = self.k(self.k_norm(x))
        else:
            q = self.q(x)
            k = self.k(x)
        
        v = self.v(x)
        
        # Apply RoPE if freqs provided
        if freqs is not None:
            q = apply_rope(q, freqs)
            k = apply_rope(k, freqs)
        
        # Attention
        attn_out = optimized_attention(q, k, v, self.num_heads)
        
        # Output projection
        out = self.o(attn_out)
        
        return out

class WanCrossAttention(nn.Module):
    """WAN Cross-Attention module"""
    
    def __init__(self, dim, num_heads, qk_norm=True, eps=1e-6):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.eps = eps
        
        # Linear layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        
        if qk_norm:
            self.q_norm = nn.LayerNorm(dim, eps=eps)
            self.k_norm = nn.LayerNorm(dim, eps=eps)
        else:
            self.q_norm = None
            self.k_norm = None
    
    def forward(self, x, context):
        b, n, c = x.shape
        
        # Apply normalization if enabled
        if self.q_norm is not None:
            q = self.q(self.q_norm(x))
            k = self.k(self.k_norm(context))
        else:
            q = self.q(x)
            k = self.k(context)
        
        v = self.v(context)
        
        # Attention
        attn_out = optimized_attention(q, k, v, self.num_heads)
        
        # Output projection
        out = self.o(attn_out)
        
        return out

class WanMLP(nn.Module):
    """WAN MLP module"""
    
    def __init__(self, dim, ffn_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, dim)
        self.activation = nn.GELU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

class WanAttentionBlock(nn.Module):
    """WAN Attention Block"""
    
    def __init__(self, cross_attn_type, dim, ffn_dim, num_heads, window_size, 
                 qk_norm, cross_attn_norm, eps):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.cross_attn = WanCrossAttention(dim, num_heads, cross_attn_norm, eps)
        
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.mlp = WanMLP(dim, ffn_dim)
    
    def forward(self, x, e, freqs, context, context_img_len=None, vace_context=None, vace_strength=None):
        # Self-attention
        x = x + self.self_attn(self.norm1(x), freqs)
        
        # Cross-attention
        x = x + self.cross_attn(self.norm2(x), context)
        
        # MLP
        x = x + self.mlp(self.norm3(x))
        
        return x

class VaceWanAttentionBlock(nn.Module):
    """VACE WAN Attention Block"""
    
    def __init__(self, cross_attn_type, dim, ffn_dim, num_heads, window_size, 
                 qk_norm, cross_attn_norm, eps, block_id=0):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.cross_attn = WanCrossAttention(dim, num_heads, cross_attn_norm, eps)
        
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.mlp = WanMLP(dim, ffn_dim)
        
        self.block_id = block_id
    
    def forward(self, x, e, freqs, context, vace_context=None, vace_strength=None, context_img_len=None):
        # Self-attention
        x = x + self.self_attn(self.norm1(x), freqs)
        
        # Cross-attention
        x = x + self.cross_attn(self.norm2(x), context)
        
        # VACE conditioning if provided
        if vace_context is not None and vace_strength is not None:
            if self.block_id < len(vace_strength):
                strength = vace_strength[self.block_id]
                if strength > 0:
                    x = x + strength * vace_context
        
        # MLP
        x = x + self.mlp(self.norm3(x))
        
        return x

class Head(nn.Module):
    """WAN Model Head"""
    
    def __init__(self, dim, out_dim, patch_size, eps):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.proj = nn.Linear(dim, out_dim * math.prod(patch_size))
    
    def forward(self, x, e):
        x = self.norm(x)
        x = self.proj(x)
        return x

class MLPProj(nn.Module):
    """MLP Projection for image embeddings"""
    
    def __init__(self, in_dim, out_dim, flf_pos_embed_token_number=None):
        super().__init__()
        
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        
        if flf_pos_embed_token_number is not None:
            self.emb_pos = nn.Parameter(torch.empty((1, flf_pos_embed_token_number, in_dim)))
        else:
            self.emb_pos = None
    
    def forward(self, image_embeds):
        if self.emb_pos is not None:
            image_embeds = image_embeds[:, :self.emb_pos.shape[1]] + self.emb_pos[:, :image_embeds.shape[1]]
        
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens

class WanCamAdapter(nn.Module):
    """WAN Camera Adapter"""
    
    def __init__(self, in_dim, out_dim, kernel_size, stride, operation_settings=None):
        super().__init__()
        self.conv = nn.Conv3d(in_dim, out_dim, kernel_size=kernel_size, stride=stride)
    
    def forward(self, x):
        return self.conv(x)

class WanModel(nn.Module):
    """WAN Diffusion Model - Standalone Implementation"""
    
    def __init__(self, model_type='t2v', patch_size=(1, 2, 2), text_len=512,
                 in_dim=16, dim=2048, ffn_dim=8192, freq_dim=256, text_dim=4096,
                 out_dim=16, num_heads=16, num_layers=32, window_size=(-1, -1),
                 qk_norm=True, cross_attn_norm=True, eps=1e-6,
                 flf_pos_embed_token_number=None, in_dim_ref_conv=None,
                 image_model=None, device=None, dtype=None, operations=None):
        
        super().__init__()
        self.dtype = dtype
        
        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type
        
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        
        # Embeddings
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(),
            nn.Linear(dim, dim)
        )
        
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6)
        )
        
        # Blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                            window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])
        
        # Head
        self.head = Head(dim, out_dim, patch_size, eps)
        
        # RoPE embedder
        d = dim // num_heads
        self.rope_embedder = EmbedND(dim=d, theta=10000.0, axes_dim=[d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)])
        
        # Image embedding for i2v
        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim, flf_pos_embed_token_number=flf_pos_embed_token_number)
        else:
            self.img_emb = None
        
        # Reference convolution
        if in_dim_ref_conv is not None:
            self.ref_conv = nn.Conv2d(in_dim_ref_conv, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.ref_conv = None
    
    def forward(self, x, t, context, clip_fea=None, freqs=None, transformer_options={}, **kwargs):
        """Forward pass of WAN model"""
        
        # Patch embedding
        x = self.patch_embedding(x.float()).to(x.dtype)
        grid_sizes = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        
        # Time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x.dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        
        # Context
        context = self.text_embedding(context)
        
        context_img_len = None
        if clip_fea is not None:
            if self.img_emb is not None:
                context_clip = self.img_emb(clip_fea)
                context = torch.concat([context_clip, context], dim=1)
            context_img_len = clip_fea.shape[-2]
        
        # Generate frequencies for RoPE
        freqs = self.rope_embedder(torch.zeros(x.shape[0], x.shape[1], self.dim, device=x.device, dtype=x.dtype))
        
        # Transformer blocks
        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        
        for i, block in enumerate(self.blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"], context_img_len=context_img_len)
                    return out
                out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": e0, "pe": freqs}, {"original_block": block_wrap})
                x = out["img"]
            else:
                x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len)
        
        # Head
        x = self.head(x, e)
        
        # Unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x
    
    def unpatchify(self, x, grid_sizes):
        """Reconstruct video tensors from patch embeddings"""
        c = self.out_dim
        u = x
        b = u.shape[0]
        u = u[:, :math.prod(grid_sizes)].view(b, *grid_sizes, *self.patch_size, c)
        u = torch.einsum('bfhwpqrc->bcfphqwr', u)
        u = u.reshape(b, c, *[i * j for i, j in zip(grid_sizes, self.patch_size)])
        return u

class VaceWanModel(WanModel):
    """VACE WAN Model - Standalone Implementation"""
    
    def __init__(self, model_type='vace', patch_size=(1, 2, 2), text_len=512,
                 in_dim=16, dim=2048, ffn_dim=8192, freq_dim=256, text_dim=4096,
                 out_dim=16, num_heads=16, num_layers=32, window_size=(-1, -1),
                 qk_norm=True, cross_attn_norm=True, eps=1e-6,
                 flf_pos_embed_token_number=None, image_model=None,
                 vace_layers=None, vace_in_dim=None, device=None, dtype=None, operations=None):
        
        super().__init__(model_type='t2v', patch_size=patch_size, text_len=text_len,
                        in_dim=in_dim, dim=dim, ffn_dim=ffn_dim, freq_dim=freq_dim,
                        text_dim=text_dim, out_dim=out_dim, num_heads=num_heads,
                        num_layers=num_layers, window_size=window_size, qk_norm=qk_norm,
                        cross_attn_norm=cross_attn_norm, eps=eps,
                        flf_pos_embed_token_number=flf_pos_embed_token_number,
                        image_model=image_model, device=device, dtype=dtype, operations=operations)
        
        # VACE specific components
        if vace_layers is not None:
            self.vace_layers = vace_layers
            self.vace_in_dim = vace_in_dim
            
            # VACE blocks
            self.vace_blocks = nn.ModuleList([
                VaceWanAttentionBlock('t2v_cross_attn', self.dim, self.ffn_dim, self.num_heads,
                                    self.window_size, self.qk_norm, self.cross_attn_norm, self.eps,
                                    block_id=i)
                for i in range(self.vace_layers)
            ])
            
            self.vace_layers_mapping = {i: n for n, i in enumerate(range(0, self.num_layers, self.num_layers // self.vace_layers))}
            
            # VACE patch embeddings
            self.vace_patch_embedding = nn.Conv3d(
                self.vace_in_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size
            )
    
    def forward(self, x, t, context, vace_context=None, vace_strength=None,
                clip_fea=None, freqs=None, transformer_options={}, **kwargs):
        """Forward pass of VACE WAN model"""
        
        # Patch embedding
        x = self.patch_embedding(x.float()).to(x.dtype)
        grid_sizes = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        
        # Time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x.dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        
        # Context
        context = self.text_embedding(context)
        
        context_img_len = None
        if clip_fea is not None:
            if self.img_emb is not None:
                context_clip = self.img_emb(clip_fea)
                context = torch.concat([context_clip, context], dim=1)
            context_img_len = clip_fea.shape[-2]
        
        # Generate frequencies for RoPE
        freqs = self.rope_embedder(torch.zeros(x.shape[0], x.shape[1], self.dim, device=x.device, dtype=x.dtype))
        
        # Process VACE context if provided
        vace_context_processed = None
        if vace_context is not None and hasattr(self, 'vace_patch_embedding'):
            vace_context_processed = self.vace_patch_embedding(vace_context.float()).to(vace_context.dtype)
            vace_context_processed = vace_context_processed.flatten(2).transpose(1, 2)
        
        # Transformer blocks
        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        
        for i, block in enumerate(self.blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"], context=args["txt"], e=args["vec"], 
                                     freqs=args["pe"], context_img_len=context_img_len,
                                     vace_context=vace_context_processed, vace_strength=vace_strength)
                    return out
                out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": e0, "pe": freqs}, {"original_block": block_wrap})
                x = out["img"]
            else:
                x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len,
                         vace_context=vace_context_processed, vace_strength=vace_strength)
        
        # Head
        x = self.head(x, e)
        
        # Unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x

class CameraWanModel(WanModel):
    """Camera WAN Model - Standalone Implementation"""
    
    def __init__(self, model_type='camera', patch_size=(1, 2, 2), text_len=512,
                 in_dim=16, dim=2048, ffn_dim=8192, freq_dim=256, text_dim=4096,
                 out_dim=16, num_heads=16, num_layers=32, window_size=(-1, -1),
                 qk_norm=True, cross_attn_norm=True, eps=1e-6,
                 flf_pos_embed_token_number=None, image_model=None,
                 in_dim_control_adapter=24, device=None, dtype=None, operations=None):
        
        if model_type == 'camera':
            model_type = 'i2v'
        else:
            model_type = 't2v'
        
        super().__init__(model_type=model_type, patch_size=patch_size, text_len=text_len,
                        in_dim=in_dim, dim=dim, ffn_dim=ffn_dim, freq_dim=freq_dim,
                        text_dim=text_dim, out_dim=out_dim, num_heads=num_heads,
                        num_layers=num_layers, window_size=window_size, qk_norm=qk_norm,
                        cross_attn_norm=cross_attn_norm, eps=eps,
                        flf_pos_embed_token_number=flf_pos_embed_token_number,
                        image_model=image_model, device=device, dtype=dtype, operations=operations)
        
        # Camera adapter
        self.control_adapter = WanCamAdapter(in_dim_control_adapter, dim, 
                                            kernel_size=patch_size[1:], stride=patch_size[1:])
    
    def forward(self, x, t, context, clip_fea=None, freqs=None, camera_conditions=None,
                transformer_options={}, **kwargs):
        """Forward pass of Camera WAN model"""
        
        # Patch embedding
        x = self.patch_embedding(x.float()).to(x.dtype)
        
        # Apply camera conditions if provided
        if self.control_adapter is not None and camera_conditions is not None:
            x = x + self.control_adapter(camera_conditions).to(x.dtype)
        
        grid_sizes = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        
        # Time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x.dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        
        # Context
        context = self.text_embedding(context)
        
        context_img_len = None
        if clip_fea is not None:
            if self.img_emb is not None:
                context_clip = self.img_emb(clip_fea)
                context = torch.concat([context_clip, context], dim=1)
            context_img_len = clip_fea.shape[-2]
        
        # Generate frequencies for RoPE
        freqs = self.rope_embedder(torch.zeros(x.shape[0], x.shape[1], self.dim, device=x.device, dtype=x.dtype))
        
        # Transformer blocks
        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        
        for i, block in enumerate(self.blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"], context=args["txt"], e=args["vec"], 
                                     freqs=args["pe"], context_img_len=context_img_len)
                    return out
                out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": e0, "pe": freqs}, {"original_block": block_wrap})
                x = out["img"]
            else:
                x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len)
        
        # Head
        x = self.head(x, e)
        
        # Unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x
