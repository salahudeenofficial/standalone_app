"""
ComfyUI-compatible Utility Module for Motion Pipeline
Standalone implementation of ComfyUI's ldm/util.py functions
No external dependencies - pure PyTorch/numpy only
"""

import importlib
import logging
from inspect import isfunction
from typing import Optional, Union, Any, Dict

import torch
from torch import optim
import numpy as np

# Handle PIL dependencies gracefully
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL not available - text image functions will be disabled")


def exists(x):
    """Check if value exists (not None)"""
    return x is not None


def default(val, d):
    """Return val if it exists, otherwise return d (resolved if callable)"""
    if exists(val):
        return val
    return d() if isfunction(d) else d


def log_txt_as_img(wh, xc, size=10):
    """
    Convert text captions to image tensors
    
    Args:
        wh: Tuple of (width, height) for output image
        xc: List of caption strings
        size: Font size for text
        
    Returns:
        Tensor of shape [B, 3, H, W] with text rendered as images
    """
    if not PIL_AVAILABLE:
        logging.warning("PIL not available, returning zeros for text images")
        return torch.zeros(len(xc), 3, wh[1], wh[0])
    
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        
        try:
            # Try to use default font, fallback to basic if unavailable
            try:
                font = ImageFont.truetype('DejaVuSans.ttf', size=size)
            except (OSError, IOError):
                try:
                    font = ImageFont.truetype('/System/Library/Fonts/Arial.ttf', size=size)
                except (OSError, IOError):
                    font = ImageFont.load_default()
                    
        except:
            font = ImageFont.load_default()
        
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start:start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            logging.warning("Can't encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    
    txts = np.stack(txts)
    txts = torch.tensor(txts, dtype=torch.float32)
    return txts


def ismap(x):
    """Check if tensor is a map (height map, depth map, etc.) - more than 3 channels"""
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x):
    """Check if tensor is an image (3 or 1 channels)"""
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    Reference: https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    
    Args:
        tensor: Input tensor
        
    Returns:
        Mean across all dimensions except batch dimension
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def count_params(model, verbose=False):
    """
    Count parameters in a model
    
    Args:
        model: PyTorch model
        verbose: Whether to log parameter count
        
    Returns:
        Total number of parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        logging.info(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    """
    Instantiate object from configuration dictionary
    
    Args:
        config: Dictionary with 'target' key and parameters
        
    Returns:
        Instantiated object
    """
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    """
    Get class from string representation
    
    Args:
        string: String like "module.module.ClassName"
        reload: Whether to reload module
        
    Returns:
        Class ready for instantiation
    """
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


class AdamWwithEMAandWings(optim.Optimizer):
    """
    AdamW optimizer with Exponential Moving Average (EMA) of parameters.
    
    Credit: https://gist.github.com/crowsonkb/65f7265353f403714fce3b2595e0b298
    
    This optimizer maintains EMA versions of parameters for improved training stability.
    """
    
    def __init__(self, params, lr=1.e-3, betas=(0.9, 0.999), eps=1.e-8,
                 weight_decay=1.e-2, amsgrad=False, ema_decay=0.9999,
                 ema_power=1., param_names=()):
        """AdamW that saves EMA versions of the parameters."""
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= ema_decay <= 1.0:
            raise ValueError("Invalid ema_decay value: {}".format(ema_decay))
            
        defaults = dict(lr=lr, betas=betas, eps=eps,
                       weight_decay=weight_decay, amsgrad=amsgrad, ema_decay=ema_decay,
                       ema_power=ema_power, param_names=param_names)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            ema_params_with_grad = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group['amsgrad']
            beta1,beta2 = group['betas']
            ema_decay = group['ema_decay']
            ema_power = group['ema_power']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of parameter values
                    state['param_exp_avg'] = p.detach().float().clone()

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                ema_params_with_grad.append(state['param_exp_avg'])

                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                # update the steps for each param group update
                state['step'] += 1
                # record the step after step update (as singleton tensor)
                state_steps.append(torch.tensor([state['step']], dtype=torch.int32, device=p.device))

            optim._functional.adamw(params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    state_steps,
                    amsgrad=amsgrad,
                    beta1=beta1,
                    beta2=beta2,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    eps=group['eps'],
                    maximize=False)

            cur_ema_decay = min(ema_decay, 1 - state['step'] ** -ema_power)
            for param, ema_param in zip(params_with_grad, ema_params_with_grad):
                ema_param.mul_(cur_ema_decay).add_(param.float(), alpha=1 - cur_ema_decay)

        return loss


# Motion pipeline specific utilities
def safe_device_cast(tensor, device, dtype=None):
    """Safely cast tensor to device and dtype"""
    if device is None:
        device = tensor.device
    if dtype is None:
        dtype = tensor.dtype
    return tensor.to(device=device, dtype=dtype)


def get_device_info(device=None):
    """Get detailed device information"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    info = {
        'device': str(device),
        'type': device.type,
    }
    
    if device.type == 'cuda':
        info.update({
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count(),
            'cuda_current_device': torch.cuda.current_device(),
            'memory_allocated': torch.cuda.memory_allocated(device) / 1024**3,  # GB
            'max_memory_allocated': torch.cuda.max_memory_allocated(device) / 1024**3,  # GB
            'memory_reserved': torch.cuda.memory_reserved(device) / 1024**3,  # GB
        })
    elif device.type == 'mps':
        info['mps_available'] = torch.backends.mps.is_available()
    
    return info


def tensor_info(tensor, name="Tensor"):
    """Get comprehensive tensor information"""
    info = {
        'name': name,
        'shape': tuple(tensor.shape),
        'dtype': str(tensor.dtype),
        'device': str(tensor.device),
        'requires_grad': tensor.requires_grad,
        'memory': tensor.element_size() * tensor.nelement() / 1024**2,  # MB
    }
    
    try:
        info.update({
            'min': tensor.min().item(),
            'max': tensor.max().item(),
            'mean': tensor.mean().item(),
            'std': tensor.std().item(),
            'has_nan': torch.isnan(tensor).any().item(),
            'has_inf': torch.isinf(tensor).any().item(),
        })
    except:
        pass  # Skip stats if tensor computation fails
    
    return info


# Export main functions and classes
__all__ = [
    'exists',
    'default', 
    'log_txt_as_img',
    'ismap',
    'isimage',
    'mean_flat',
    'count_params',
    'instantiate_from_config',
    'get_obj_from_str',
    'AdamWwithEMAandWings',
    'safe_device_cast',
    'get_device_info',
    'tensor_info',
    'PIL_AVAILABLE'
]

# Log module initialization
logging.info(f"Motion pipeline utilities loaded (PIL available: {PIL_AVAILABLE})")
