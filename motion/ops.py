"""
Motion Pipeline Operations Module
Port of ComfyUI's ops.py for standalone usage
"""

import torch
import logging
import contextlib

# Motion Pipeline Internal Dependencies (ComfyUI compatible)
from motion.model_management_standalone import *
from motion.cli_args import args, PerformanceFeature
from motion.rmsnorm import rms_norm, RMSNorm

def calc_mantissa(abs_x, exponent, normal_mask, MANTISSA_BITS, EXPONENT_BIAS, generator=None):
    mantissa_scaled = torch.where(
        normal_mask,
        (abs_x / (2.0 ** (exponent - EXPONENT_BIAS)) - 1.0) * (2**MANTISSA_BITS),
        (abs_x / (2.0 ** (-EXPONENT_BIAS + 1 - MANTISSA_BITS)))
    )
    mantissa_scaled += torch.rand(mantissa_scaled.size(), dtype=mantissa_scaled.dtype, layout=mantissa_scaled.layout, device=mantissa_scaled.device, generator=generator)
    return mantissa_scaled.floor() / (2**MANTISSA_BITS)

def manual_stochastic_round_to_float8(x, dtype, generator=None):
    """Manual stochastic rounding for FP8 formats"""
    if dtype == torch.float8_e4m3fn:
        EXPONENT_BITS, MANTISSA_BITS, EXPONENT_BIAS = 4, 3, 7
    elif dtype == torch.float8_e5m2:
        EXPONENT_BITS, MANTISSA_BITS, EXPONENT_BIAS = 5, 2, 15
    else:
        raise ValueError("Unsupported dtype")

    x = x.half()
    sign = torch.sign(x)
    abs_x = x.abs()
    sign = torch.where(abs_x == 0, 0, sign)

    # Combine exponent calculation and clamping
    exponent = torch.clamp(
        torch.floor(torch.log2(abs_x)) + EXPONENT_BIAS,
        0, 2**EXPONENT_BITS - 1
    )

    # Combine mantissa calculation and rounding  
    normal_mask = ~(exponent == 0)
    abs_x[:] = calc_mantissa(abs_x, exponent, normal_mask, MANTISSA_BITS, EXPONENT_BIAS, generator=generator)

    sign *= torch.where(
        normal_mask,
        (2.0 ** (exponent - EXPONENT_BIAS)) * (1.0 + abs_x),
        (2.0 ** (-EXPONENT_BIAS + 1)) * abs_x
    )

    # Clamp to supported range
    inf_value = torch.finfo(torch.float8_e4m3fn).max if dtype == torch.float8_e4m3fn else torch.finfo(torch.float8_e5m2).max
    torch.clamp(sign, min=-inf_value, max=inf_value, out=sign)
    return sign

def stochastic_rounding(value, dtype, seed=0):
    """Proper stochastic rounding implementation for FP8 quantization"""
    if dtype == torch.float32:
        return value.to(dtype=torch.float32)
    if dtype == torch.float16:
        return value.to(dtype=torch.float16)
    if dtype == torch.bfloat16:
        return value.to(dtype=torch.bfloat16)
    if dtype == torch.float8_e4m3fn or dtype == torch.float8_e5m2:
        generator = torch.Generator(device=value.device)
        generator.manual_seed(seed)
        output = torch.empty_like(value, dtype=dtype)
        num_slices = max(1, (value.numel() / (4096 * 4096)))
        slice_size = max(1, round(value.shape[0] / num_slices))
        for i in range(0, value.shape[0], slice_size):
            output[i:i+slice_size].copy_(manual_stochastic_round_to_float8(value[i:i+slice_size], dtype, generator=generator))
        return output

    return value.to(dtype=dtype)




def scaled_dot_product_attention(q, k, v, *args, **kwargs):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, *args, **kwargs)


try:
    if torch.cuda.is_available():
        from torch.nn.attention import SDPBackend, sdpa_kernel
        import inspect
        if "set_priority" in inspect.signature(sdpa_kernel).parameters:
            SDPA_BACKEND_PRIORITY = [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]

            SDPA_BACKEND_PRIORITY.insert(0, SDPBackend.CUDNN_ATTENTION)

            def scaled_dot_product_attention(q, k, v, *args, **kwargs):
                with sdpa_kernel(SDPA_BACKEND_PRIORITY, set_priority=True):
                    return torch.nn.functional.scaled_dot_product_attention(q, k, v, *args, **kwargs)
        else:
            logging.warning("Torch version too old to set sdpa backend priority.")
except (ModuleNotFoundError, TypeError):
    logging.warning("Could not set sdpa backend priority.")

# cast_to is imported from model_management_standalone

def cast_to_input(weight, input, non_blocking=False, copy=True):
    return cast_to(weight, input.dtype, input.device, non_blocking=non_blocking, copy=copy)

def cast_bias_weight(s, input=None, dtype=None, device=None, bias_dtype=None):
    if input is not None:
        if dtype is None:
            dtype = input.dtype
        if bias_dtype is None:
            bias_dtype = dtype
        if device is None:
            device = input.device

    offload_stream = get_offload_stream(device)
    if offload_stream is not None:
        wf_context = offload_stream
    else:
        wf_context = contextlib.nullcontext()

    bias = None
    non_blocking = device_supports_non_blocking(device)
    if s.bias is not None:
        has_function = len(s.bias_function) > 0
        bias = cast_to(s.bias, bias_dtype, device, non_blocking=non_blocking, copy=has_function, stream=offload_stream)

        if has_function:
            with wf_context:
                for f in s.bias_function:
                    bias = f(bias)

    has_function = len(s.weight_function) > 0
    weight = cast_to(s.weight, dtype, device, non_blocking=non_blocking, copy=has_function, stream=offload_stream)
    if has_function:
        with wf_context:
            for f in s.weight_function:
                weight = f(weight)

    sync_stream(device, offload_stream)
    return weight, bias

class CastWeightBiasOp:
    comfy_cast_weights = False
    weight_function = []
    bias_function = []

class disable_weight_init:
    class Linear(torch.nn.Linear, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.linear(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv1d(torch.nn.Conv1d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv2d(torch.nn.Conv2d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv3d(torch.nn.Conv3d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class GroupNorm(torch.nn.GroupNorm, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.group_norm(input, self.num_groups, weight, bias, self.eps)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class LayerNorm(torch.nn.LayerNorm, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            if self.weight is not None:
                weight, bias = cast_bias_weight(self, input)
            else:
                weight = None
                bias = None
            return torch.nn.functional.layer_norm(input, self.normalized_shape, weight, bias, self.eps)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class RMSNorm(RMSNorm, CastWeightBiasOp):
        def reset_parameters(self):
            self.bias = None
            return None

        def forward_comfy_cast_weights(self, input):
            if self.weight is not None:
                weight, bias = cast_bias_weight(self, input)
            else:
                weight = None
            return rms_norm(input, weight, self.eps)  # TODO: switch to commented out line when old torch is deprecated
            # return torch.nn.functional.rms_norm(input, self.normalized_shape, weight, self.eps)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class ConvTranspose2d(torch.nn.ConvTranspose2d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input, output_size=None):
            num_spatial_dims = 2
            output_padding = self._output_padding(
                input, output_size, self.stride, self.padding, self.kernel_size,
                num_spatial_dims, self.dilation)

            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.conv_transpose2d(
                input, weight, bias, self.stride, self.padding,
                output_padding, self.groups, self.dilation)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class ConvTranspose1d(torch.nn.ConvTranspose1d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input, output_size=None):
            num_spatial_dims = 1
            output_padding = self._output_padding(
                input, output_size, self.stride, self.padding, self.kernel_size,
                num_spatial_dims, self.dilation)

            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.conv_transpose1d(
                input, weight, bias, self.stride, self.padding,
                output_padding, self.groups, self.dilation)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Embedding(torch.nn.Embedding, CastWeightBiasOp):
        def reset_parameters(self):
            self.bias = None
            return None

        def forward_comfy_cast_weights(self, input, out_dtype=None):
            output_dtype = out_dtype
            if self.weight.dtype == torch.float16 or self.weight.dtype == torch.bfloat16:
                out_dtype = None
            weight, bias = cast_bias_weight(self, device=input.device, dtype=out_dtype)
            return torch.nn.functional.embedding(input, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse).to(dtype=output_dtype)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                if "out_dtype" in kwargs:
                    kwargs.pop("out_dtype")
                return super().forward(*args, **kwargs)

    @classmethod
    def conv_nd(s, dims, *args, **kwargs):
        if dims == 2:
            return s.Conv2d(*args, **kwargs)
        elif dims == 3:
            return s.Conv3d(*args, **kwargs)
        else:
            raise ValueError(f"unsupported dimensions: {dims}")


class manual_cast(disable_weight_init):
    class Linear(disable_weight_init.Linear):
        comfy_cast_weights = True

    class Conv1d(disable_weight_init.Conv1d):
        comfy_cast_weights = True

    class Conv2d(disable_weight_init.Conv2d):
        comfy_cast_weights = True

    class Conv3d(disable_weight_init.Conv3d):
        comfy_cast_weights = True

    class GroupNorm(disable_weight_init.GroupNorm):
        comfy_cast_weights = True

    class LayerNorm(disable_weight_init.LayerNorm):
        comfy_cast_weights = True

    class ConvTranspose2d(disable_weight_init.ConvTranspose2d):
        comfy_cast_weights = True

    class ConvTranspose1d(disable_weight_init.ConvTranspose1d):
        comfy_cast_weights = True

    class RMSNorm(disable_weight_init.RMSNorm):
        comfy_cast_weights = True

    class Embedding(disable_weight_init.Embedding):
        comfy_cast_weights = True


def fp8_linear(self, input):
    dtype = self.weight.dtype
    if dtype not in [torch.float8_e4m3fn]:
        return None

    tensor_2d = False
    if len(input.shape) == 2:
        tensor_2d = True
        input = input.unsqueeze(1)

    input_shape = input.shape
    input_dtype = input.dtype
    if len(input.shape) == 3:
        w, bias = cast_bias_weight(self, input, dtype=dtype, bias_dtype=input_dtype)
        w = w.t()

        scale_weight = self.scale_weight
        scale_input = self.scale_input
        if scale_weight is None:
            scale_weight = torch.ones((), device=input.device, dtype=torch.float32)
        else:
            scale_weight = scale_weight.to(input.device)

        if scale_input is None:
            scale_input = torch.ones((), device=input.device, dtype=torch.float32)
            input = torch.clamp(input, min=-448, max=448, out=input)
            input = input.reshape(-1, input_shape[2]).to(dtype).contiguous()
        else:
            scale_input = scale_input.to(input.device)
            input = (input * (1.0 / scale_input).to(input_dtype)).reshape(-1, input_shape[2]).to(dtype).contiguous()

        if bias is not None:
            o = torch._scaled_mm(input, w, out_dtype=input_dtype, bias=bias, scale_a=scale_input, scale_b=scale_weight)
        else:
            o = torch._scaled_mm(input, w, out_dtype=input_dtype, scale_a=scale_input, scale_b=scale_weight)

        if isinstance(o, tuple):
            o = o[0]

        if tensor_2d:
            return o.reshape(input_shape[0], -1)

        return o.reshape((-1, input_shape[1], self.weight.shape[0]))

    return None

class fp8_ops(manual_cast):
    class Linear(manual_cast.Linear):
        def reset_parameters(self):
            self.scale_weight = None
            self.scale_input = None
            return None

        def forward_comfy_cast_weights(self, input):
            try:
                out = fp8_linear(self, input)
                if out is not None:
                    return out
            except Exception as e:
                logging.info("Exception during fp8 op: {}".format(e))

            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.linear(input, weight, bias)

def scaled_fp8_ops(fp8_matrix_mult=False, scale_input=False, override_dtype=None):
    logging.info("Using scaled fp8: fp8 matrix mult: {}, scale input: {}".format(fp8_matrix_mult, scale_input))
    class scaled_fp8_op(manual_cast):
        class Linear(manual_cast.Linear):
            def __init__(self, *args, **kwargs):
                if override_dtype is not None:
                    kwargs['dtype'] = override_dtype
                super().__init__(*args, **kwargs)

            def reset_parameters(self):
                if not hasattr(self, 'scale_weight'):
                    self.scale_weight = torch.nn.parameter.Parameter(data=torch.ones((), device=self.weight.device, dtype=torch.float32), requires_grad=False)

                if not scale_input:
                    self.scale_input = None

                if not hasattr(self, 'scale_input'):
                    self.scale_input = torch.nn.parameter.Parameter(data=torch.ones((), device=self.weight.device, dtype=torch.float32), requires_grad=False)
                return None

            def forward_comfy_cast_weights(self, input):
                if fp8_matrix_mult:
                    out = fp8_linear(self, input)
                    if out is not None:
                        return out

                weight, bias = cast_bias_weight(self, input)

                if weight.numel() < input.numel(): #TODO: optimize
                    return torch.nn.functional.linear(input, weight * self.scale_weight.to(device=weight.device, dtype=weight.dtype), bias)
                else:
                    return torch.nn.functional.linear(input * self.scale_weight.to(device=weight.device, dtype=weight.dtype), weight, bias)

            def convert_weight(self, weight, inplace=False, **kwargs):
                if inplace:
                    weight *= self.scale_weight.to(device=weight.device, dtype=weight.dtype)
                    return weight
                else:
                    return weight * self.scale_weight.to(device=weight.device, dtype=weight.dtype)

            def set_weight(self, weight, inplace_update=False, seed=None, **kwargs):
                weight = stochastic_rounding(weight / self.scale_weight.to(device=weight.device, dtype=weight.dtype), self.weight.dtype, seed=seed)
                if inplace_update:
                    self.weight.data.copy_(weight)
                else:
                    self.weight = torch.nn.Parameter(weight, requires_grad=False)

    return scaled_fp8_op

CUBLAS_IS_AVAILABLE = False
try:
    from cublas_ops import CublasLinear
    CUBLAS_IS_AVAILABLE = True
except ImportError:
    pass

if CUBLAS_IS_AVAILABLE:
    class cublas_ops(disable_weight_init):
        class Linear(CublasLinear, disable_weight_init.Linear):
            def reset_parameters(self):
                return None

            def forward_comfy_cast_weights(self, input):
                return super().forward(input)

            def forward(self, *args, **kwargs):
                return super().forward(*args, **kwargs)

def pick_operations(weight_dtype, compute_dtype, load_device=None, disable_fast_fp8=False, fp8_optimizations=False, scaled_fp8=None):
    fp8_compute = supports_fp8_compute(load_device)
    if scaled_fp8 is not None:
        return scaled_fp8_ops(fp8_matrix_mult=fp8_compute and fp8_optimizations, scale_input=fp8_optimizations, override_dtype=scaled_fp8)

    if (
        fp8_compute and
        (fp8_optimizations or PerformanceFeature.Fp8MatrixMultiplication in args.fast) and
        not disable_fast_fp8
    ):
        return fp8_ops

    if (
        PerformanceFeature.CublasOps in args.fast and
        CUBLAS_IS_AVAILABLE and
        weight_dtype == torch.float16 and
        (compute_dtype == torch.float16 or compute_dtype is None)
    ):
        logging.info("Using cublas ops")
        return cublas_ops

    if compute_dtype is None or weight_dtype == compute_dtype:
        return disable_weight_init

    return manual_cast
