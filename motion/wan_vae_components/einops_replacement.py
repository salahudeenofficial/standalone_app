"""
Manual einops rearrange implementation
"""

def rearrange(tensor, pattern, **kwargs):
    """Simple rearrange implementation for basic patterns"""
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
    else:
        raise NotImplementedError(f'Pattern {pattern} not implemented')
