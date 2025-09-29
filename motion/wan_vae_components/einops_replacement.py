"""
Manual einops rearrange implementation - expanded to match ComfyUI usage patterns
"""

def rearrange(tensor, pattern, **kwargs):
    """Expanded rearrange implementation for VAE patterns (borrowed from ComfyUI usage)."""
    
    # Basic video patterns used in VAE
    if pattern == 'b c t h w -> (b t) c h w':
        b, c, t, h, w = tensor.shape
        return tensor.view(b*t, c, h, w)
    elif pattern == '(b t) c h w -> b c t h w':
        b_t, c, h, w = tensor.shape
        t = kwargs.get('t', 4)  # Default to 4 if not provided
        b = b_t // t
        return tensor.view(b, t, c, h, w).transpose(1, 2)
    elif pattern == '(b t) c h w-> b c t h w':  # Note: no space before ->
        b_t, c, h, w = tensor.shape
        t = kwargs.get('t', 4)  # Default to 4 if not provided
        b = b_t // t
        return tensor.view(b, t, c, h, w).transpose(1, 2)
    
    # Additional patterns commonly used in VAE attention
    elif pattern == 'b c h w -> (b h w) c':
        b, c, h, w = tensor.shape
        return tensor.view(b, c, h*w).transpose(1, 2).contiguous().view(b*h*w, c)
    elif pattern == '(b h w) c -> b c h w':
        b_h_w, c = tensor.shape
        h = kwargs.get('h', 8)  # Default height
        w = kwargs.get('w', 8)  # Default width
        b = b_h_w // (h * w)
        return tensor.view(b, h, w, c).transpose(1, 3).transpose(2, 3).contiguous()
    
    # Channel dimension patterns
    elif pattern == 'b c h w -> b h w c':
        return tensor.permute(0, 2, 3, 1)
    elif pattern == 'b h w c -> b c h w':
        return tensor.permute(0, 3, 1, 2)
    
    # Flatten patterns
    elif pattern == 'b c h w -> b (c h w)':
        return tensor.view(tensor.shape[0], -1)
    elif pattern == 'b (c h w) -> b c h w':
        b, c_h_w = tensor.shape
        c = kwargs.get('c', 3)  # Default channels
        h = kwargs.get('h', 8)  # Default height
        w = kwargs.get('w', 8)  # Default width
        return tensor.view(b, c, h, w)
    
    # Chunk patterns
    elif pattern == 'b (n c) h w -> b n c h w':
        b, n_c, h, w = tensor.shape
        n = kwargs.get('n', 2)  # Default number of chunks
        c = n_c // n
        return tensor.view(b, n, c, h, w)
    elif pattern == 'b n c h w -> b (n c) h w':
        b, n, c, h, w = tensor.shape
        return tensor.view(b, n*c, h, w)
    
    else:
        raise NotImplementedError(f'Pattern {pattern} not implemented in einops replacement')
