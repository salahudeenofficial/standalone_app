"""
Manual einops rearrange implementation
"""

def rearrange(tensor, pattern):
    """Simple rearrange implementation for basic patterns"""
    if pattern == "(b t) c h w-> b c t h w":
        b_t, c, h, w = tensor.shape
        b = b_t // 4  # Assuming t=4 for WAN VAE
        t = 4
        return tensor.view(b, t, c, h, w).transpose(1, 2)
    elif pattern == "b c t h w-> (b t) c h w":
        b, c, t, h, w = tensor.shape
        return tensor.transpose(1, 2).contiguous().view(b*t, c, h, w)
    else:
        raise NotImplementedError(f"Pattern {pattern} not implemented")
