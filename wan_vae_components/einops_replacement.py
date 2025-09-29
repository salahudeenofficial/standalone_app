"""
Manual einops rearrange implementation
"""

def rearrange(tensor, pattern, **kwargs):
    """Simple rearrange implementation for basic patterns"""
    # Handle keyword arguments like t=t
    for key, value in kwargs.items():
        globals()[key] = value
    
    if pattern == "(b t) c h w-> b c t h w" or pattern == "(b t) c h w -> b c t h w":
        b_t, c, h, w = tensor.shape
        # Use provided t if available, otherwise default to 4
        t = kwargs.get('t', b_t // 4 if b_t % 4 == 0 else 4)
        b = b_t // t
        return tensor.view(b, t, c, h, w).transpose(1, 2)
    elif pattern == "b c t h w-> (b t) c h w" or pattern == "b c t h w -> (b t) c h w":
        b, c, t, h, w = tensor.shape
        return tensor.transpose(1, 2).contiguous().view(b*t, c, h, w)
    else:
        raise NotImplementedError(f"Pattern {pattern} not implemented")
