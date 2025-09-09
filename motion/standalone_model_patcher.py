"""
Standalone ModelPatcher Implementation
A complete, self-contained implementation of ComfyUI's ModelPatcher
with all dependencies included in a single file.

This provides memory management, model patching, device management,
and weight modification capabilities for PyTorch models.
"""

import torch
import uuid
import collections
import copy
import logging
import math
import weakref
import gc
from typing import Callable, Optional, Dict, List, Any, Union
from enum import Enum


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def string_to_seed(data):
    """Generate deterministic seed from string data"""
    crc = 0xFFFFFFFF
    for byte in data:
        if isinstance(byte, str):
            byte = ord(byte)
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xEDB88320
            else:
                crc >>= 1
    return crc ^ 0xFFFFFFFF


def get_attr(obj, attr: str):
    """Retrieves a nested attribute from an object using dot notation"""
    attrs = attr.split(".")
    for name in attrs:
        obj = getattr(obj, name)
    return obj


def set_attr(obj, attr: str, value):
    """Sets a nested attribute on an object using dot notation"""
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    setattr(obj, attrs[-1], value)


def copy_to_param(obj, attr, value):
    """Inplace update tensor instead of replacing it"""
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    prev = getattr(obj, attrs[-1])
    prev.data.copy_(value)


def set_attr_param(obj, attr, value):
    """Set parameter attribute"""
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    setattr(obj, attrs[-1], value)


def cast_to_device(tensor, device, dtype, copy=False):
    """Cast tensor to device and dtype"""
    if copy:
        return tensor.to(device=device, dtype=dtype, copy=True)
    else:
        return tensor.to(device=device, dtype=dtype)


def module_size(module):
    """Calculate module size in bytes"""
    module_mem = 0
    for param in module.parameters():
        module_mem += param.nelement() * param.element_size()
    for buffer in module.buffers():
        module_mem += buffer.nelement() * buffer.element_size()
    return module_mem


# ============================================================================
# STOCHASTIC ROUNDING
# ============================================================================

def stochastic_rounding(value, dtype, seed=0):
    """Stochastic rounding for different dtypes"""
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


def manual_stochastic_round_to_float8(x, dtype, generator=None):
    """Manual stochastic rounding for float8 types"""
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

    exponent = torch.clamp(
        torch.floor(torch.log2(abs_x)) + EXPONENT_BIAS,
        0, 2**EXPONENT_BITS - 1
    )

    normal_mask = ~(exponent == 0)

    abs_x[:] = calc_mantissa(abs_x, exponent, normal_mask, MANTISSA_BITS, EXPONENT_BIAS, generator=generator)

    sign *= torch.where(
        normal_mask,
        (2.0 ** (exponent - EXPONENT_BIAS)) * (1.0 + abs_x),
        (2.0 ** (-EXPONENT_BIAS + 1)) * abs_x
    )

    torch.clamp(sign, min=float('-inf'), max=float('inf'), out=sign)
    return sign


def calc_mantissa(abs_x, exponent, normal_mask, MANTISSA_BITS, EXPONENT_BIAS, generator=None):
    """Calculate mantissa for stochastic rounding"""
    mantissa_scaled = torch.where(
        normal_mask,
        (abs_x / (2.0 ** (exponent - EXPONENT_BIAS)) - 1.0) * (2**MANTISSA_BITS),
        (abs_x / (2.0 ** (-EXPONENT_BIAS + 1 - MANTISSA_BITS)))
    )
    return mantissa_scaled.floor() / (2**MANTISSA_BITS)


# ============================================================================
# WEIGHT CALCULATION (LoRA)
# ============================================================================

def calculate_weight(patches, weight, key, intermediate_dtype=torch.float32, original_weights=None):
    """Calculate weight with patches applied (LoRA, etc.)"""
    for p in patches:
        strength = p[0]
        v = p[1]
        strength_model = p[2]
        offset = p[3]
        function = p[4]
        if function is None:
            function = lambda a: a

        old_weight = None
        if offset is not None:
            old_weight = weight
            weight = weight.narrow(offset[0], offset[1], offset[2])

        if strength_model != 1.0:
            weight *= strength_model

        if isinstance(v, list):
            v = (calculate_weight(v[1:], v[0][1](cast_to_device(v[0][0], weight.device, intermediate_dtype, copy=True), inplace=True), key, intermediate_dtype=intermediate_dtype), )

        if len(v) == 1:
            patch_type = "diff"
        elif len(v) == 2:
            patch_type = v[0]
            v = v[1]

        if patch_type == "diff":
            diff: torch.Tensor = v[0]
            do_pad_weight = len(v) > 1 and v[1].get('pad_weight', False)
            if do_pad_weight and diff.shape != weight.shape:
                logging.info("Pad weight {} from {} to shape: {}".format(key, weight.shape, diff.shape))
                weight = pad_tensor_to_shape(weight, diff.shape)

            if strength != 0.0:
                if diff.shape != weight.shape:
                    logging.warning("WARNING SHAPE MISMATCH {} WEIGHT NOT MERGED {} != {}".format(key, diff.shape, weight.shape))
                else:
                    weight += function(strength * cast_to_device(diff, weight.device, weight.dtype))
        elif patch_type == "set":
            weight.copy_(v[0])
        elif patch_type == "model_as_lora":
            target_weight: torch.Tensor = v[0]
            if original_weights and key in original_weights:
                diff_weight = cast_to_device(target_weight, weight.device, intermediate_dtype) - \
                              cast_to_device(original_weights[key][0][0], weight.device, intermediate_dtype)
                weight += function(strength * cast_to_device(diff_weight, weight.device, weight.dtype))
        else:
            logging.warning("patch type not recognized {} {}".format(patch_type, key))

        if old_weight is not None:
            weight = old_weight

    return weight


def pad_tensor_to_shape(tensor: torch.Tensor, new_shape: List[int]) -> torch.Tensor:
    """Pad tensor to new shape"""
    if tensor.shape == new_shape:
        return tensor

    padded_tensor = torch.zeros(new_shape, dtype=tensor.dtype, device=tensor.device)
    
    orig_slices = []
    new_slices = []
    
    for i in range(len(tensor.shape)):
        orig_slices.append(slice(0, tensor.shape[i]))
        new_slices.append(slice(0, min(tensor.shape[i], new_shape[i])))
    
    padded_tensor[new_slices] = tensor[orig_slices]
    return padded_tensor


# ============================================================================
# MEMORY MANAGEMENT
# ============================================================================

class MemoryCounter:
    """Memory counter for tracking available memory"""
    def __init__(self, initial: int, minimum=0):
        self.value = initial
        self.minimum = minimum

    def use(self, weight: torch.Tensor):
        """Use memory for a weight tensor"""
        weight_size = weight.nelement() * weight.element_size()
        if self.is_useable(weight_size):
            self.decrement(weight_size)
            return True
        return False

    def is_useable(self, used: int):
        """Check if memory is available"""
        return self.value - used > self.minimum

    def decrement(self, used: int):
        """Decrement available memory"""
        self.value -= used


# ============================================================================
# LOW VRAM PATCHES
# ============================================================================

class LowVramPatch:
    """Low VRAM patch for memory-efficient weight modifications"""
    def __init__(self, key, patches):
        self.key = key
        self.patches = patches

    def __call__(self, weight):
        """Apply patches to weight"""
        intermediate_dtype = weight.dtype
        if intermediate_dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            intermediate_dtype = torch.float32
            return stochastic_rounding(
                calculate_weight(self.patches[self.key], weight.to(intermediate_dtype), self.key, intermediate_dtype=intermediate_dtype), 
                weight.dtype, 
                seed=string_to_seed(self.key)
            )
        return calculate_weight(self.patches[self.key], weight, self.key, intermediate_dtype=intermediate_dtype)


def move_weight_functions(m, device):
    """Move weight functions to device"""
    if device is None:
        return 0

    memory = 0
    if hasattr(m, "weight_function"):
        for f in m.weight_function:
            if hasattr(f, "move_to"):
                memory += f.move_to(device=device)

    if hasattr(m, "bias_function"):
        for f in m.bias_function:
            if hasattr(f, "move_to"):
                memory += f.move_to(device=device)
    return memory


def wipe_lowvram_weight(m):
    """Clean up low VRAM weights"""
    if hasattr(m, "prev_comfy_cast_weights"):
        m.comfy_cast_weights = m.prev_comfy_cast_weights
        del m.prev_comfy_cast_weights

    if hasattr(m, "weight_function"):
        m.weight_function = []

    if hasattr(m, "bias_function"):
        m.bias_function = []


# ============================================================================
# WEIGHT ACCESS FUNCTIONS
# ============================================================================

def get_key_weight(model, key):
    """Get weight tensor and associated functions for a model key"""
    set_func = None
    convert_func = None
    op_keys = key.rsplit('.', 1)
    if len(op_keys) < 2:
        weight = get_attr(model, key)
    else:
        op = get_attr(model, op_keys[0])
        try:
            set_func = getattr(op, "set_{}".format(op_keys[1]))
        except AttributeError:
            pass

        try:
            convert_func = getattr(op, "convert_{}".format(op_keys[1]))
        except AttributeError:
            pass

        weight = getattr(op, op_keys[1])
        if convert_func is not None:
            weight = get_attr(model, key)

    return weight, set_func, convert_func


# ============================================================================
# CONTEXT MANAGERS
# ============================================================================

class AutoPatcherEjector:
    """Context manager for safe model injection handling"""
    def __init__(self, model: 'ModelPatcher', skip_and_inject_on_exit_only=False):
        self.model = model
        self.was_injected = False
        self.prev_skip_injection = False
        self.skip_and_inject_on_exit_only = skip_and_inject_on_exit_only

    def __enter__(self):
        self.was_injected = False
        self.prev_skip_injection = self.model.skip_injection
        if self.skip_and_inject_on_exit_only:
            self.model.skip_injection = True
        if self.model.is_injected:
            self.model.eject_model()
            self.was_injected = True

    def __exit__(self, *args):
        if self.skip_and_inject_on_exit_only:
            self.model.skip_injection = self.prev_skip_injection
            self.model.inject_model()
        if self.was_injected and not self.model.skip_injection:
            self.model.inject_model()
        self.model.skip_injection = self.prev_skip_injection


# ============================================================================
# MAIN MODELPATCHER CLASS
# ============================================================================

class ModelPatcher:
    """
    Standalone ModelPatcher implementation
    
    Provides memory management, model patching, device management,
    and weight modification capabilities for PyTorch models.
    """
    
    def __init__(self, model, load_device, offload_device, size=0, weight_inplace_update=False):
        # Core model and device management
        self.model = model
        self.load_device = load_device
        self.offload_device = offload_device
        self.weight_inplace_update = weight_inplace_update
        
        # Memory tracking
        self.size = size
        self.model_size()
        
        # Patching system
        self.patches = {}
        self.backup = {}
        self.object_patches = {}
        self.object_patches_backup = {}
        self.weight_wrapper_patches = {}
        
        # Model options and configuration
        self.model_options = {"transformer_options": {}}
        self.patches_uuid = uuid.uuid4()
        self.parent = None
        self.force_cast_weights = False
        
        # Advanced features (simplified)
        self.attachments = {}
        self.additional_models = {}
        self.callbacks = {}
        self.wrappers = {}
        self.injections = {}
        
        # Injection system
        self.is_injected = False
        self.skip_injection = False
        
        # Initialize model attributes
        if not hasattr(self.model, 'device'):
            logging.debug("Model doesn't have a device attribute.")
            self.model.device = offload_device
        elif self.model.device is None:
            self.model.device = offload_device

        if not hasattr(self.model, 'model_loaded_weight_memory'):
            self.model.model_loaded_weight_memory = 0

        if not hasattr(self.model, 'lowvram_patch_counter'):
            self.model.lowvram_patch_counter = 0

        if not hasattr(self.model, 'model_lowvram'):
            self.model.model_lowvram = False

        if not hasattr(self.model, 'current_weight_patches_uuid'):
            self.model.current_weight_patches_uuid = None

    def model_size(self):
        """Calculate total model size in bytes"""
        if self.size > 0:
            return self.size
        self.size = module_size(self.model)
        return self.size

    def loaded_size(self):
        """Get currently loaded memory usage"""
        return self.model.model_loaded_weight_memory

    def memory_required(self, input_shape):
        """Calculate memory needed for inference"""
        if hasattr(self.model, 'memory_required'):
            return self.model.memory_required(input_shape=input_shape)
        return self.model_size()

    def clone(self):
        """Create deep copy of ModelPatcher"""
        n = self.__class__(self.model, self.load_device, self.offload_device, self.size, weight_inplace_update=self.weight_inplace_update)
        n.patches = {}
        for k in self.patches:
            n.patches[k] = self.patches[k][:]
        n.patches_uuid = self.patches_uuid

        n.object_patches = self.object_patches.copy()
        n.weight_wrapper_patches = self.weight_wrapper_patches.copy()
        n.model_options = copy.deepcopy(self.model_options)
        n.backup = self.backup
        n.object_patches_backup = self.object_patches_backup
        n.parent = self
        n.force_cast_weights = self.force_cast_weights

        # attachments
        n.attachments = {}
        for k in self.attachments:
            if hasattr(self.attachments[k], "on_model_patcher_clone"):
                n.attachments[k] = self.attachments[k].on_model_patcher_clone()
            else:
                n.attachments[k] = self.attachments[k]
        
        # additional models
        for k, c in self.additional_models.items():
            n.additional_models[k] = [x.clone() for x in c]
        
        # callbacks
        for k, c in self.callbacks.items():
            n.callbacks[k] = {}
            for k1, c1 in c.items():
                n.callbacks[k][k1] = c1.copy()
        
        # wrappers
        for k, w in self.wrappers.items():
            n.wrappers[k] = {}
            for k1, w1 in w.items():
                n.wrappers[k][k1] = w1.copy()
        
        # injection
        n.is_injected = self.is_injected
        n.skip_injection = self.skip_injection
        for k, i in self.injections.items():
            n.injections[k] = i.copy()

        return n

    def is_clone(self, other):
        """Check if other is clone of this model"""
        if hasattr(other, 'model') and self.model is other.model:
            return True
        return False

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        """Add weight patches (LoRA, etc.)"""
        with self.use_ejected():
            p = set()
            model_sd = self.model.state_dict()
            for k in patches:
                offset = None
                function = None
                if isinstance(k, str):
                    key = k
                else:
                    offset = k[1]
                    key = k[0]
                    if len(k) > 2:
                        function = k[2]

                if key in model_sd:
                    p.add(k)
                    current_patches = self.patches.get(key, [])
                    current_patches.append((strength_patch, patches[k], strength_model, offset, function))
                    self.patches[key] = current_patches

            self.patches_uuid = uuid.uuid4()
            return list(p)

    def get_key_patches(self, filter_prefix=None):
        """Get all patches for model keys"""
        model_sd = self.model_state_dict()
        p = {}
        for k in model_sd:
            if filter_prefix is not None:
                if not k.startswith(filter_prefix):
                    continue
            bk = self.backup.get(k, None)
            weight, set_func, convert_func = get_key_weight(self.model, k)
            if bk is not None:
                weight = bk.weight
            if convert_func is None:
                convert_func = lambda a, **kwargs: a

            if k in self.patches:
                p[k] = [(weight, convert_func)] + self.patches[k]
            else:
                p[k] = [(weight, convert_func)]
        return p

    def model_state_dict(self, filter_prefix=None):
        """Get model state dict with patches applied"""
        with self.use_ejected():
            sd = self.model.state_dict()
            keys = list(sd.keys())
            if filter_prefix is not None:
                for k in keys:
                    if not k.startswith(filter_prefix):
                        sd.pop(k)
            return sd

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False):
        """Apply patches to specific weight"""
        if key not in self.patches:
            return

        weight, set_func, convert_func = get_key_weight(self.model, key)
        inplace_update = self.weight_inplace_update or inplace_update

        if key not in self.backup:
            self.backup[key] = collections.namedtuple('Dimension', ['weight', 'inplace_update'])(
                weight.to(device=self.offload_device, copy=inplace_update), inplace_update
            )

        if device_to is not None:
            temp_weight = cast_to_device(weight, device_to, torch.float32, copy=True)
        else:
            temp_weight = weight.to(torch.float32, copy=True)
        if convert_func is not None:
            temp_weight = convert_func(temp_weight, inplace=True)

        out_weight = calculate_weight(self.patches[key], temp_weight, key)
        if set_func is None:
            out_weight = stochastic_rounding(out_weight, weight.dtype, seed=string_to_seed(key))
            if inplace_update:
                copy_to_param(self.model, key, out_weight)
            else:
                set_attr_param(self.model, key, out_weight)
        else:
            set_func(out_weight, inplace_update=inplace_update, seed=string_to_seed(key))

    def _load_list(self):
        """Get list of modules to load"""
        loading = []
        for n, m in self.model.named_modules():
            params = []
            skip = False
            for name, param in m.named_parameters(recurse=False):
                params.append(name)
            for name, param in m.named_parameters(recurse=True):
                if name not in params:
                    skip = True
                    break
            if not skip and (hasattr(m, "comfy_cast_weights") or len(params) > 0):
                loading.append((module_size(m), n, m, params))
        return loading

    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        """Load model to device with memory optimization"""
        with self.use_ejected():
            self.unpatch_hooks()
            mem_counter = 0
            patch_counter = 0
            lowvram_counter = 0
            loading = self._load_list()

            load_completely = []
            loading.sort(reverse=True)
            for x in loading:
                n = x[1]
                m = x[2]
                params = x[3]
                module_mem = x[0]

                lowvram_weight = False

                weight_key = "{}.weight".format(n)
                bias_key = "{}.bias".format(n)

                if not full_load and hasattr(m, "comfy_cast_weights"):
                    if mem_counter + module_mem >= lowvram_model_memory:
                        lowvram_weight = True
                        lowvram_counter += 1
                        if hasattr(m, "prev_comfy_cast_weights"):
                            continue

                cast_weight = self.force_cast_weights
                if lowvram_weight:
                    if hasattr(m, "comfy_cast_weights"):
                        m.weight_function = []
                        m.bias_function = []

                    if weight_key in self.patches:
                        if force_patch_weights:
                            self.patch_weight_to_device(weight_key)
                        else:
                            m.weight_function = [LowVramPatch(weight_key, self.patches)]
                            patch_counter += 1
                    if bias_key in self.patches:
                        if force_patch_weights:
                            self.patch_weight_to_device(bias_key)
                        else:
                            m.bias_function = [LowVramPatch(bias_key, self.patches)]
                            patch_counter += 1

                    cast_weight = True
                else:
                    if hasattr(m, "comfy_cast_weights"):
                        wipe_lowvram_weight(m)

                    if full_load or mem_counter + module_mem < lowvram_model_memory:
                        mem_counter += module_mem
                        load_completely.append((module_mem, n, m, params))

                if cast_weight and hasattr(m, "comfy_cast_weights"):
                    m.prev_comfy_cast_weights = m.comfy_cast_weights
                    m.comfy_cast_weights = True

                if weight_key in self.weight_wrapper_patches:
                    m.weight_function.extend(self.weight_wrapper_patches[weight_key])

                if bias_key in self.weight_wrapper_patches:
                    m.bias_function.extend(self.weight_wrapper_patches[bias_key])

                mem_counter += move_weight_functions(m, device_to)

            load_completely.sort(reverse=True)
            for x in load_completely:
                n = x[1]
                m = x[2]
                params = x[3]
                if hasattr(m, "comfy_patched_weights"):
                    if m.comfy_patched_weights == True:
                        continue

                for param in params:
                    self.patch_weight_to_device("{}.{}".format(n, param), device_to=device_to)

                logging.debug("lowvram: loaded module regularly {} {}".format(n, m))
                m.comfy_patched_weights = True

            for x in load_completely:
                x[2].to(device_to)

            if lowvram_counter > 0:
                logging.info("loaded partially {} {} {}".format(lowvram_model_memory / (1024 * 1024), mem_counter / (1024 * 1024), patch_counter))
                self.model.model_lowvram = True
            else:
                logging.info("loaded completely {} {} {}".format(lowvram_model_memory / (1024 * 1024), mem_counter / (1024 * 1024), full_load))
                self.model.model_lowvram = False
                if full_load:
                    self.model.to(device_to)
                    mem_counter = self.model_size()

            self.model.lowvram_patch_counter += patch_counter
            self.model.device = device_to
            self.model.model_loaded_weight_memory = mem_counter
            self.model.current_weight_patches_uuid = self.patches_uuid

    def patch_model(self, device_to=None, lowvram_model_memory=0, load_weights=True, force_patch_weights=False):
        """Apply all patches and load model"""
        with self.use_ejected():
            for k in self.object_patches:
                old = set_attr(self.model, k, self.object_patches[k])
                if k not in self.object_patches_backup:
                    self.object_patches_backup[k] = old

            if lowvram_model_memory == 0:
                full_load = True
            else:
                full_load = False

            if load_weights:
                self.load(device_to, lowvram_model_memory=lowvram_model_memory, force_patch_weights=force_patch_weights, full_load=full_load)
        self.inject_model()
        return self.model

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        """Remove all patches and restore original weights"""
        self.eject_model()
        if unpatch_weights:
            self.unpatch_hooks()
            if self.model.model_lowvram:
                for m in self.model.modules():
                    move_weight_functions(m, device_to)
                    wipe_lowvram_weight(m)

                self.model.model_lowvram = False
                self.model.lowvram_patch_counter = 0

            keys = list(self.backup.keys())

            for k in keys:
                bk = self.backup[k]
                if bk.inplace_update:
                    copy_to_param(self.model, k, bk.weight)
                else:
                    set_attr_param(self.model, k, bk.weight)

            self.model.current_weight_patches_uuid = None
            self.backup.clear()

            if device_to is not None:
                self.model.to(device_to)
                self.model.device = device_to
            self.model.model_loaded_weight_memory = 0

            for m in self.model.modules():
                if hasattr(m, "comfy_patched_weights"):
                    del m.comfy_patched_weights

        keys = list(self.object_patches_backup.keys())
        for k in keys:
            set_attr(self.model, k, self.object_patches_backup[k])

        self.object_patches_backup.clear()

    def partially_unload(self, device_to, memory_to_free=0):
        """Unload parts of model to free memory"""
        with self.use_ejected():
            hooks_unpatched = False
            memory_freed = 0
            patch_counter = 0
            unload_list = self._load_list()
            unload_list.sort()
            for unload in unload_list:
                if memory_to_free < memory_freed:
                    break
                module_mem = unload[0]
                n = unload[1]
                m = unload[2]
                params = unload[3]

                lowvram_possible = hasattr(m, "comfy_cast_weights")
                if hasattr(m, "comfy_patched_weights") and m.comfy_patched_weights == True:
                    move_weight = True
                    for param in params:
                        key = "{}.{}".format(n, param)
                        bk = self.backup.get(key, None)
                        if bk is not None:
                            if not lowvram_possible:
                                move_weight = False
                                break

                            if not hooks_unpatched:
                                self.unpatch_hooks()
                                hooks_unpatched = True

                            if bk.inplace_update:
                                copy_to_param(self.model, key, bk.weight)
                            else:
                                set_attr_param(self.model, key, bk.weight)
                            self.backup.pop(key)

                    weight_key = "{}.weight".format(n)
                    bias_key = "{}.bias".format(n)
                    if move_weight:
                        cast_weight = self.force_cast_weights
                        m.to(device_to)
                        module_mem += move_weight_functions(m, device_to)
                        if lowvram_possible:
                            if weight_key in self.patches:
                                m.weight_function.append(LowVramPatch(weight_key, self.patches))
                                patch_counter += 1
                            if bias_key in self.patches:
                                m.bias_function.append(LowVramPatch(bias_key, self.patches))
                                patch_counter += 1
                            cast_weight = True

                        if cast_weight:
                            m.prev_comfy_cast_weights = m.comfy_cast_weights
                            m.comfy_cast_weights = True
                        m.comfy_patched_weights = False
                        memory_freed += module_mem
                        logging.debug("freed {}".format(n))

            self.model.model_lowvram = True
            self.model.lowvram_patch_counter += patch_counter
            self.model.model_loaded_weight_memory -= memory_freed
            return memory_freed

    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        """Load model partially to save memory"""
        with self.use_ejected(skip_and_inject_on_exit_only=True):
            unpatch_weights = self.model.current_weight_patches_uuid is not None and (self.model.current_weight_patches_uuid != self.patches_uuid or force_patch_weights)
            used = self.model.model_loaded_weight_memory
            self.unpatch_model(self.offload_device, unpatch_weights=unpatch_weights)
            if unpatch_weights:
                extra_memory += (used - self.model.model_loaded_weight_memory)

            self.patch_model(load_weights=False)
            full_load = False
            if self.model.model_lowvram == False and self.model.model_loaded_weight_memory > 0:
                return 0
            if self.model.model_loaded_weight_memory + extra_memory > self.model_size():
                full_load = True
            current_used = self.model.model_loaded_weight_memory
            try:
                self.load(device_to, lowvram_model_memory=current_used + extra_memory, force_patch_weights=force_patch_weights, full_load=full_load)
            except Exception as e:
                self.detach()
                raise e

            return self.model.model_loaded_weight_memory - current_used

    def detach(self, unpatch_all=True):
        """Completely detach and clean up model"""
        self.eject_model()
        self.model_patches_to(self.offload_device)
        if unpatch_all:
            self.unpatch_model(self.offload_device, unpatch_weights=unpatch_all)
        return self.model

    def current_loaded_device(self):
        """Get current device"""
        return self.model.device

    def model_patches_to(self, device):
        """Move patches to device"""
        to = self.model_options["transformer_options"]
        if "patches" in to:
            patches = to["patches"]
            for name in patches:
                patch_list = patches[name]
                for i in range(len(patch_list)):
                    if hasattr(patch_list[i], "to"):
                        patch_list[i] = patch_list[i].to(device)
        if "patches_replace" in to:
            patches = to["patches_replace"]
            for name in patches:
                patch_list = patches[name]
                for k in patch_list:
                    if hasattr(patch_list[k], "to"):
                        patch_list[k] = patch_list[k].to(device)
        if "model_function_wrapper" in self.model_options:
            wrap_func = self.model_options["model_function_wrapper"]
            if hasattr(wrap_func, "to"):
                self.model_options["model_function_wrapper"] = wrap_func.to(device)

    def model_dtype(self):
        """Get model dtype"""
        if hasattr(self.model, "get_dtype"):
            return self.model.get_dtype()

    def add_object_patch(self, name, obj):
        """Replace model object"""
        self.object_patches[name] = obj

    def get_model_object(self, name: str) -> torch.nn.Module:
        """Get model object (patched or original)"""
        if name in self.object_patches:
            return self.object_patches[name]
        else:
            if name in self.object_patches_backup:
                return self.object_patches_backup[name]
            else:
                return get_attr(self.model, name)

    def set_model_compute_dtype(self, dtype):
        """Set computation dtype"""
        self.add_object_patch("manual_cast_dtype", dtype)
        if dtype is not None:
            self.force_cast_weights = True
        self.patches_uuid = uuid.uuid4()

    def add_weight_wrapper(self, name, function):
        """Add weight wrapper function"""
        self.weight_wrapper_patches[name] = self.weight_wrapper_patches.get(name, []) + [function]
        self.patches_uuid = uuid.uuid4()

    def set_model_patch(self, patch, name):
        """Add transformer patch"""
        to = self.model_options["transformer_options"]
        if "patches" not in to:
            to["patches"] = {}
        to["patches"][name] = to["patches"].get(name, []) + [patch]

    def set_model_attn1_patch(self, patch):
        """Patch self-attention layers"""
        self.set_model_patch(patch, "attn1_patch")

    def set_model_attn2_patch(self, patch):
        """Patch cross-attention layers"""
        self.set_model_patch(patch, "attn2_patch")

    def set_model_input_block_patch(self, patch):
        """Patch input blocks"""
        self.set_model_patch(patch, "input_block_patch")

    def set_model_output_block_patch(self, patch):
        """Patch output blocks"""
        self.set_model_patch(patch, "output_block_patch")

    def set_model_sampler_cfg_function(self, sampler_cfg_function, disable_cfg1_optimization=False):
        """Set CFG sampling function"""
        if len(sampler_cfg_function.__code__.co_varnames) == 3:
            self.model_options["sampler_cfg_function"] = lambda args: sampler_cfg_function(args["cond"], args["uncond"], args["cond_scale"])
        else:
            self.model_options["sampler_cfg_function"] = sampler_cfg_function
        if disable_cfg1_optimization:
            self.model_options["disable_cfg1_optimization"] = True

    def set_attachments(self, key: str, attachment):
        """Attach object to model"""
        self.attachments[key] = attachment

    def get_attachment(self, key: str):
        """Get attached object"""
        return self.attachments.get(key, None)

    def set_additional_models(self, key: str, models: List['ModelPatcher']):
        """Add additional models (ControlNet, etc.)"""
        self.additional_models[key] = models

    def get_additional_models(self):
        """Get all additional models"""
        all_models = []
        for models in self.additional_models.values():
            all_models.extend(models)
        return all_models

    def add_callback(self, call_type: str, callback: Callable):
        """Add event callback"""
        if call_type not in self.callbacks:
            self.callbacks[call_type] = {}
        if None not in self.callbacks[call_type]:
            self.callbacks[call_type][None] = []
        self.callbacks[call_type][None].append(callback)

    def get_all_callbacks(self, call_type: str):
        """Get all callbacks of type"""
        c_list = []
        if call_type in self.callbacks:
            for c in self.callbacks[call_type].values():
                c_list.extend(c)
        return c_list

    def use_ejected(self, skip_and_inject_on_exit_only=False):
        """Context manager for safe injection handling"""
        return AutoPatcherEjector(self, skip_and_inject_on_exit_only)

    def inject_model(self):
        """Inject model modifications"""
        if self.is_injected or self.skip_injection:
            return
        for injections in self.injections.values():
            for inj in injections:
                inj.inject(self)
                self.is_injected = True
        if self.is_injected:
            for callback in self.get_all_callbacks("ON_INJECT_MODEL"):
                callback(self)

    def eject_model(self):
        """Remove model injections"""
        if not self.is_injected:
            return
        for injections in self.injections.values():
            for inj in injections:
                inj.eject(self)
        self.is_injected = False
        for callback in self.get_all_callbacks("ON_EJECT_MODEL"):
            callback(self)

    def unpatch_hooks(self, whitelist_keys_set=None):
        """Remove hook patches (simplified implementation)"""
        # Simplified hook system - can be extended as needed
        pass

    def cleanup(self):
        """Clean up all resources"""
        self.unpatch_hooks()
        if hasattr(self.model, "current_patcher"):
            self.model.current_patcher = None
        for callback in self.get_all_callbacks("ON_CLEANUP"):
            callback(self)

    def __del__(self):
        """Destructor"""
        self.detach(unpatch_all=False)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_model_patcher(model, load_device=None, offload_device=None):
    """Convenience function to create ModelPatcher"""
    if load_device is None:
        load_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if offload_device is None:
        offload_device = torch.device("cpu")
    
    return ModelPatcher(model, load_device, offload_device)


def load_model_to_device(model_patcher, device=None, lowvram_model_memory=0):
    """Convenience function to load model to device"""
    if device is None:
        device = model_patcher.load_device
    
    model_patcher.patch_model(device_to=device, lowvram_model_memory=lowvram_model_memory)
    return model_patcher


def unload_model_from_device(model_patcher, device=None):
    """Convenience function to unload model from device"""
    if device is None:
        device = model_patcher.offload_device
    
    model_patcher.unpatch_model(device_to=device)
    return model_patcher


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    import torch.nn as nn
    
    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)
            self.device = torch.device("cpu")
        
        def forward(self, x):
            return self.linear(x)
    
    # Create model
    model = SimpleModel()
    
    # Create ModelPatcher
    patcher = create_model_patcher(
        model, 
        load_device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        offload_device=torch.device("cpu")
    )
    
    print(f"Model size: {patcher.model_size() / (1024*1024):.2f} MB")
    
    # Load model to device
    load_model_to_device(patcher)
    print(f"Model loaded to: {patcher.current_loaded_device()}")
    
    # Add some patches (example)
    patches = {
        "linear.weight": torch.randn(5, 10) * 0.1,
        "linear.bias": torch.randn(5) * 0.1
    }
    patcher.add_patches(patches, strength_patch=0.5)
    
    # Unload model
    unload_model_from_device(patcher)
    print("Model unloaded")
    
    print("Standalone ModelPatcher implementation complete!")
