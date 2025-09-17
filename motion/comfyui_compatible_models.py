"""
ComfyUI-Compatible WAN 2.1 Vace Model Classes
Exact replication of ComfyUI's model implementations
"""

import torch
import torch.nn as nn
import math
import logging
from typing import Optional, Dict, Any, List, Tuple

# Import ComfyUI's operations
try:
    import comfy.ops
    COMFYUI_OPS_AVAILABLE = True
except ImportError:
    COMFYUI_OPS_AVAILABLE = False
    logging.warning("ComfyUI operations not available, using standard PyTorch operations")

def sinusoidal_embedding_1d(dim, position):
    """ComfyUI's exact sinusoidal embedding implementation"""
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float32)
    
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x

def repeat_e(e, x):
    """ComfyUI's exact repeat_e implementation"""
    repeats = 1
    if e.shape[1] > 1:
        repeats = x.shape[1] // e.shape[1]
    if repeats == 1:
        return e
    return torch.repeat_interleave(e, repeats, dim=1)

class WanSelfAttention(nn.Module):
    """ComfyUI's exact WanSelfAttention implementation"""
    
    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6, 
                 operation_settings={}):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # Use ComfyUI operations if available, otherwise standard PyTorch
        if COMFYUI_OPS_AVAILABLE and operation_settings.get("operations"):
            ops = operation_settings.get("operations")
            device = operation_settings.get("device")
            dtype = operation_settings.get("dtype")
            
            self.q = ops.Linear(dim, dim, device=device, dtype=dtype)
            self.k = ops.Linear(dim, dim, device=device, dtype=dtype)
            self.v = ops.Linear(dim, dim, device=device, dtype=dtype)
            self.o = ops.Linear(dim, dim, device=device, dtype=dtype)
            self.norm_q = ops.RMSNorm(dim, eps=eps, elementwise_affine=True, device=device, dtype=dtype) if qk_norm else nn.Identity()
            self.norm_k = ops.RMSNorm(dim, eps=eps, elementwise_affine=True, device=device, dtype=dtype) if qk_norm else nn.Identity()
        else:
            # Fallback to standard PyTorch
            self.q = nn.Linear(dim, dim)
            self.k = nn.Linear(dim, dim)
            self.v = nn.Linear(dim, dim)
            self.o = nn.Linear(dim, dim)
            self.norm_q = nn.LayerNorm(dim, eps=eps, elementwise_affine=True) if qk_norm else nn.Identity()
            self.norm_k = nn.LayerNorm(dim, eps=eps, elementwise_affine=True) if qk_norm else nn.Identity()

    def forward(self, x, freqs):
        """ComfyUI's exact forward implementation"""
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n * d)
            return q, k, v

        q, k, v = qkv_fn(x)
        
        # Apply RoPE if freqs provided
        if freqs is not None:
            try:
                from comfy.ldm.flux.math import apply_rope
                q, k = apply_rope(q, k, freqs)
            except ImportError:
                logging.warning("ComfyUI's apply_rope not available, skipping RoPE")
        
        # Use ComfyUI's optimized attention if available
        try:
            from comfy.ldm.modules.attention import optimized_attention
            x = optimized_attention(
                q.view(b, s, n * d),
                k.view(b, s, n * d),
                v,
                heads=self.num_heads,
            )
        except ImportError:
            # Fallback to standard attention
            q = q.view(b, s, n * d)
            k = k.view(b, s, n * d)
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v, num_heads=self.num_heads)

        x = self.o(x)
        return x

class WanT2VCrossAttention(WanSelfAttention):
    """ComfyUI's exact WanT2VCrossAttention implementation"""
    
    def forward(self, x, context, **kwargs):
        """ComfyUI's exact cross attention forward"""
        # compute query, key, value
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(context))
        v = self.v(context)

        # compute attention
        try:
            from comfy.ldm.modules.attention import optimized_attention
            x = optimized_attention(q, k, v, heads=self.num_heads)
        except ImportError:
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v, num_heads=self.num_heads)

        x = self.o(x)
        return x

class WanAttentionBlock(nn.Module):
    """ComfyUI's exact WanAttentionBlock implementation"""
    
    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6, 
                 operation_settings={}):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # Use ComfyUI operations if available
        if COMFYUI_OPS_AVAILABLE and operation_settings.get("operations"):
            ops = operation_settings.get("operations")
            device = operation_settings.get("device")
            dtype = operation_settings.get("dtype")
            
            self.norm1 = ops.LayerNorm(dim, eps, elementwise_affine=False, device=device, dtype=dtype)
            self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps, operation_settings=operation_settings)
            self.norm3 = ops.LayerNorm(dim, eps, elementwise_affine=True, device=device, dtype=dtype) if cross_attn_norm else nn.Identity()
            self.cross_attn = WanT2VCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps, operation_settings=operation_settings)
            self.norm2 = ops.LayerNorm(dim, eps, elementwise_affine=False, device=device, dtype=dtype)
            self.ffn = nn.Sequential(
                ops.Linear(dim, ffn_dim, device=device, dtype=dtype), 
                nn.GELU(approximate='tanh'),
                ops.Linear(ffn_dim, dim, device=device, dtype=dtype)
            )
            self.modulation = nn.Parameter(torch.empty(1, 6, dim, device=device, dtype=dtype))
        else:
            # Fallback to standard PyTorch
            self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
            self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps, operation_settings=operation_settings)
            self.norm3 = nn.LayerNorm(dim, eps=eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
            self.cross_attn = WanT2VCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps, operation_settings=operation_settings)
            self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
            self.ffn = nn.Sequential(
                nn.Linear(dim, ffn_dim), 
                nn.GELU(approximate='tanh'),
                nn.Linear(ffn_dim, dim)
            )
            self.modulation = nn.Parameter(torch.empty(1, 6, dim))

    def forward(self, x, e, freqs, context, context_img_len=257):
        """ComfyUI's exact forward implementation"""
        # Handle modulation
        if e.ndim < 4:
            e = (self.modulation + e).chunk(6, dim=1)
        else:
            e = (self.modulation.unsqueeze(0) + e).unbind(2)

        # self-attention
        y = self.self_attn(
            self.norm1(x) * (1 + repeat_e(e[1], x)) + repeat_e(e[0], x),
            freqs)

        x = x + y * repeat_e(e[2], x)

        # cross-attention & ffn
        x = x + self.cross_attn(self.norm3(x), context, context_img_len=context_img_len)
        y = self.ffn(self.norm2(x) * (1 + repeat_e(e[4], x)) + repeat_e(e[3], x))
        x = x + y * repeat_e(e[5], x)
        return x

class VaceWanAttentionBlock(WanAttentionBlock):
    """ComfyUI's exact VaceWanAttentionBlock implementation"""
    
    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 block_id=0,
                 operation_settings={}
    ):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps, operation_settings=operation_settings)
        self.block_id = block_id
        
        if COMFYUI_OPS_AVAILABLE and operation_settings.get("operations"):
            ops = operation_settings.get("operations")
            device = operation_settings.get("device")
            dtype = operation_settings.get("dtype")
            
            if block_id == 0:
                self.before_proj = ops.Linear(self.dim, self.dim, device=device, dtype=dtype)
            self.after_proj = ops.Linear(self.dim, self.dim, device=device, dtype=dtype)
        else:
            if block_id == 0:
                self.before_proj = nn.Linear(self.dim, self.dim)
            self.after_proj = nn.Linear(self.dim, self.dim)

    def forward(self, c, x, **kwargs):
        """ComfyUI's exact Vace forward implementation"""
        if self.block_id == 0:
            c = self.before_proj(c) + x
        c = super().forward(c, **kwargs)
        c_skip = self.after_proj(c)
        return c_skip, c

class Head(nn.Module):
    """ComfyUI's exact Head implementation"""
    
    def __init__(self, dim, out_dim, patch_size, eps=1e-6, operation_settings={}):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        
        if COMFYUI_OPS_AVAILABLE and operation_settings.get("operations"):
            ops = operation_settings.get("operations")
            device = operation_settings.get("device")
            dtype = operation_settings.get("dtype")
            
            self.norm = ops.LayerNorm(dim, eps, elementwise_affine=False, device=device, dtype=dtype)
            self.head = ops.Linear(dim, out_dim, device=device, dtype=dtype)
            self.modulation = nn.Parameter(torch.empty(1, 2, dim, device=device, dtype=dtype))
        else:
            self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
            self.head = nn.Linear(dim, out_dim)
            self.modulation = nn.Parameter(torch.empty(1, 2, dim))

    def forward(self, x, e):
        """ComfyUI's exact head forward implementation"""
        if e.ndim < 3:
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        else:
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).unbind(2)

        x = (self.head(self.norm(x) * (1 + repeat_e(e[1], x)) + repeat_e(e[0], x)))
        return x

class ComfyUIWanModel(torch.nn.Module):
    """ComfyUI's exact WanModel implementation"""
    
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 flf_pos_embed_token_number=None,
                 in_dim_ref_conv=None,
                 image_model=None,
                 device=None,
                 dtype=None,
                 operations=None,
                 ):
        super().__init__()
        self.dtype = dtype
        operation_settings = {"operations": operations, "device": device, "dtype": dtype}

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

        # embeddings
        if COMFYUI_OPS_AVAILABLE and operations:
            self.patch_embedding = operations.Conv3d(
                in_dim, dim, kernel_size=patch_size, stride=patch_size, device=device, dtype=torch.float32)
            self.text_embedding = nn.Sequential(
                operations.Linear(text_dim, dim, device=device, dtype=dtype), 
                nn.GELU(approximate='tanh'),
                operations.Linear(dim, dim, device=device, dtype=dtype)
            )
            self.time_embed = nn.Sequential(
                operations.Linear(freq_dim, dim, device=device, dtype=dtype), 
                nn.SiLU(), 
                operations.Linear(dim, dim, device=device, dtype=dtype)
            )
            self.time_projection = nn.Sequential(
                nn.SiLU(), 
                operations.Linear(dim, dim * 6, device=device, dtype=dtype)
            )
        else:
            self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
            self.text_embedding = nn.Sequential(
                nn.Linear(text_dim, dim), 
                nn.GELU(approximate='tanh'),
                nn.Linear(dim, dim)
            )
            self.time_embed = nn.Sequential(
                nn.Linear(freq_dim, dim), 
                nn.SiLU(), 
                nn.Linear(dim, dim)
            )
            self.time_projection = nn.Sequential(
                nn.SiLU(), 
                nn.Linear(dim, dim * 6)
            )

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps, operation_settings=operation_settings)
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps, operation_settings=operation_settings)

        # RoPE embedder
        try:
            from comfy.ldm.flux.layers import EmbedND
            d = dim // num_heads
            self.rope_embedder = EmbedND(dim=d, theta=10000.0, axes_dim=[d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)])
        except ImportError:
            logging.warning("ComfyUI's EmbedND not available, using identity")
            self.rope_embedder = nn.Identity()

        # Image embedding for i2v
        if model_type == 'i2v':
            try:
                from comfy.ldm.flux.layers import MLPProj
                self.img_emb = MLPProj(1280, dim, flf_pos_embed_token_number=flf_pos_embed_token_number, operation_settings=operation_settings)
            except ImportError:
                logging.warning("ComfyUI's MLPProj not available, using identity")
                self.img_emb = nn.Identity()
        else:
            self.img_emb = None

        # Reference convolution
        if in_dim_ref_conv is not None:
            if COMFYUI_OPS_AVAILABLE and operations:
                self.ref_conv = operations.Conv2d(in_dim_ref_conv, dim, kernel_size=patch_size[1:], stride=patch_size[1:], device=device, dtype=dtype)
            else:
                self.ref_conv = nn.Conv2d(in_dim_ref_conv, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.ref_conv = None

    def forward(self, x, timestep, context, clip_fea=None, time_dim_concat=None, transformer_options={}, **kwargs):
        """ComfyUI's exact forward implementation"""
        bs, c, t, h, w = x.shape
        
        # Pad to patch size
        try:
            import comfy.ldm.common_dit
            x = comfy.ldm.common_dit.pad_to_patch_size(x, self.patch_size)
        except ImportError:
            logging.warning("ComfyUI's pad_to_patch_size not available, using identity")
        
        patch_size = self.patch_size
        t_len = ((t + (patch_size[0] // 2)) // patch_size[0])
        h_len = ((h + (patch_size[1] // 2)) // patch_size[1])
        w_len = ((w + (patch_size[2] // 2)) // patch_size[2])

        if time_dim_concat is not None:
            try:
                import comfy.ldm.common_dit
                time_dim_concat = comfy.ldm.common_dit.pad_to_patch_size(time_dim_concat, self.patch_size)
                x = torch.cat([x, time_dim_concat], dim=2)
                t_len = ((x.shape[2] + (patch_size[0] // 2)) // patch_size[0])
            except ImportError:
                logging.warning("ComfyUI's pad_to_patch_size not available for time_dim_concat")

        if self.ref_conv is not None and "reference_latent" in kwargs:
            t_len += 1

        # Create image IDs for RoPE
        img_ids = torch.zeros((t_len, h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[:, :, :, 0] = img_ids[:, :, :, 0] + torch.linspace(0, t_len - 1, steps=t_len, device=x.device, dtype=x.dtype).reshape(-1, 1, 1)
        img_ids[:, :, :, 1] = img_ids[:, :, :, 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype).reshape(1, -1, 1)
        img_ids[:, :, :, 2] = img_ids[:, :, :, 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype).reshape(1, 1, -1)
        
        try:
            from einops import repeat
            img_ids = repeat(img_ids, "t h w c -> b (t h w) c", b=bs)
        except ImportError:
            img_ids = img_ids.view(1, -1, 3).repeat(bs, 1, 1)

        freqs = self.rope_embedder(img_ids).movedim(1, 2)
        return self.forward_orig(x, timestep, context, clip_fea=clip_fea, freqs=freqs, transformer_options=transformer_options, **kwargs)[:, :, :t, :h, :w]

    def forward_orig(self, x, t, context, clip_fea=None, freqs=None, transformer_options={}, **kwargs):
        """ComfyUI's exact forward_orig implementation"""
        # embeddings
        x = self.patch_embedding(x.float()).to(x.dtype)
        grid_sizes = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)

        # time embeddings
        e = self.time_embed(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(dtype=x[0].dtype))
        e = e.reshape(t.shape[0], -1, e.shape[-1])
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))

        full_ref = None
        if self.ref_conv is not None:
            full_ref = kwargs.get("reference_latent", None)
            if full_ref is not None:
                full_ref = self.ref_conv(full_ref).flatten(2).transpose(1, 2)
                x = torch.concat((full_ref, x), dim=1)

        # context
        context = self.text_embedding(context)

        context_img_len = None
        if clip_fea is not None:
            if self.img_emb is not None:
                context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
                context = torch.concat([context_clip, context], dim=1)
            context_img_len = clip_fea.shape[-2]

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

        # head
        x = self.head(x, e)

        if full_ref is not None:
            x = x[:, full_ref.shape[1]:]

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x

    def unpatchify(self, x, grid_sizes):
        """ComfyUI's exact unpatchify implementation"""
        c = self.out_dim
        u = x
        b = u.shape[0]
        u = u[:, :math.prod(grid_sizes)].view(b, *grid_sizes, *self.patch_size, c)
        u = torch.einsum('bfhwpqrc->bcfphqwr', u)
        u = u.reshape(b, c, *[i * j for i, j in zip(grid_sizes, self.patch_size)])
        return u

class ComfyUIVaceWanModel(ComfyUIWanModel):
    """ComfyUI's exact VaceWanModel implementation"""
    
    def __init__(self,
                 model_type='vace',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 flf_pos_embed_token_number=None,
                 image_model=None,
                 vace_layers=None,
                 vace_in_dim=None,
                 device=None,
                 dtype=None,
                 operations=None,
                 ):

        super().__init__(model_type='t2v', patch_size=patch_size, text_len=text_len, in_dim=in_dim, dim=dim, ffn_dim=ffn_dim, freq_dim=freq_dim, text_dim=text_dim, out_dim=out_dim, num_heads=num_heads, num_layers=num_layers, window_size=window_size, qk_norm=qk_norm, cross_attn_norm=cross_attn_norm, eps=eps, flf_pos_embed_token_number=flf_pos_embed_token_number, image_model=image_model, device=device, dtype=dtype, operations=operations)
        operation_settings = {"operations": operations, "device": device, "dtype": dtype}

        # Vace
        if vace_layers is not None:
            self.vace_layers = vace_layers
            self.vace_in_dim = vace_in_dim
            # vace blocks
            self.vace_blocks = nn.ModuleList([
                VaceWanAttentionBlock('t2v_cross_attn', self.dim, self.ffn_dim, self.num_heads, self.window_size, self.qk_norm, self.cross_attn_norm, self.eps, block_id=i, operation_settings=operation_settings)
                for i in range(self.vace_layers)
            ])

            self.vace_layers_mapping = {i: n for n, i in enumerate(range(0, self.num_layers, self.num_layers // self.vace_layers))}
            # vace patch embeddings
            if COMFYUI_OPS_AVAILABLE and operations:
                self.vace_patch_embedding = operations.Conv3d(
                    self.vace_in_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size, device=device, dtype=torch.float32
                )
            else:
                self.vace_patch_embedding = nn.Conv3d(
                    self.vace_in_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size
                )

    def forward_orig(self, x, t, context, vace_context, vace_strength, clip_fea=None, freqs=None, transformer_options={}, **kwargs):
        """ComfyUI's exact Vace forward_orig implementation"""
        # embeddings
        x = self.patch_embedding(x.float()).to(x.dtype)
        grid_sizes = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)

        # time embeddings
        e = self.time_embed(
            sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x[0].dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # context
        context = self.text_embedding(context)

        context_img_len = None
        if clip_fea is not None:
            if self.img_emb is not None:
                context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
                context = torch.concat([context_clip, context], dim=1)
            context_img_len = clip_fea.shape[-2]

        orig_shape = list(vace_context.shape)
        vace_context = vace_context.movedim(0, 1).reshape([-1] + orig_shape[2:])
        c = self.vace_patch_embedding(vace_context.float()).to(vace_context.dtype)
        c = c.flatten(2).transpose(1, 2)
        c = list(c.split(orig_shape[0], dim=0))

        # arguments
        x_orig = x

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

            ii = self.vace_layers_mapping.get(i, None)
            if ii is not None:
                for iii in range(len(c)):
                    c_skip, c[iii] = self.vace_blocks[ii](c[iii], x=x_orig, e=e0, freqs=freqs, context=context, context_img_len=context_img_len)
                    x += c_skip * vace_strength[iii]
                del c_skip
        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x
