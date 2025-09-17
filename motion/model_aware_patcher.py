#!/usr/bin/env python3
"""
Model-aware patching system for ComfyUI-style partial loading
Based on ComfyUI's approach of handling different model architectures intelligently
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Any, Optional, Type
import inspect

# Import our ComfyUI-style operations
from comfyui_ops import ComfyUILinear, ComfyUIConv2d, ComfyUIConv3d, ComfyUIGroupNorm, ComfyUILayerNorm

class ModelAwarePatcher:
    """
    Model-aware patcher that handles different architectures intelligently
    """
    
    def __init__(self):
        self.model_handlers = {}
        self.register_default_handlers()
    
    def register_model_handler(self, model_class: Type, handler_func):
        """Register a custom handler for a specific model type"""
        self.model_handlers[model_class] = handler_func
    
    def register_default_handlers(self):
        """Register default handlers for common model types"""
        # Register VaceWanModel handler
        try:
            from standalone_sd import VaceWanModel
            self.register_model_handler(VaceWanModel, self._patch_vace_wan_model)
        except ImportError:
            logging.warning("VaceWanModel not available for patching")
        
        # Register generic UNet handler
        try:
            from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel
            self.register_model_handler(UNetModel, self._patch_generic_unet)
        except ImportError:
            logging.warning("UNetModel not available for patching")
    
    def patch_model(self, model: nn.Module) -> nn.Module:
        """
        Patch a model using model-aware approach
        """
        model_type = type(model)
        model_name = model_type.__name__
        
        logging.info(f"🔧 Model-aware patching for {model_name}...")
        
        # Check if we have a specific handler for this model type
        if model_type in self.model_handlers:
            handler = self.model_handlers[model_type]
            logging.info(f"  ✅ Using specific handler for {model_name}")
            return handler(model)
        
        # Check inheritance hierarchy for handlers
        for handler_type, handler_func in self.model_handlers.items():
            if isinstance(model, handler_type):
                logging.info(f"  ✅ Using inherited handler {handler_type.__name__} for {model_name}")
                return handler_func(model)
        
        # Fall back to generic patching
        logging.info(f"  🔄 Using generic patching for {model_name}")
        return self._patch_generic_model(model)
    
    def _patch_vace_wan_model(self, model) -> nn.Module:
        """
        Specific handler for VaceWanModel architecture
        """
        logging.info("  🎯 Patching VaceWanModel with architecture-aware approach...")
        
        patched_count = 0
        
        # VaceWanModel has a specific structure with vace_blocks
        if hasattr(model, 'vace_blocks'):
            logging.info("  📊 Found vace_blocks - patching attention and MLP layers...")
            
            # Patch vace_blocks
            for i, block in enumerate(model.vace_blocks):
                block_patched = self._patch_vace_block(block, f"vace_blocks.{i}")
                if block_patched:
                    patched_count += 1
        
        # Patch other layers
        patched_count += self._patch_remaining_layers(model, "vace_wan_model")
        
        logging.info(f"  ✅ VaceWanModel patching complete: {patched_count} components patched")
        return model
    
    def _patch_vace_block(self, block, block_name: str) -> bool:
        """
        Patch a single VaceWanModel block
        """
        patched = False
        
        # Patch self-attention layers
        if hasattr(block, 'self_attn'):
            self._patch_attention_layer(block.self_attn, f"{block_name}.self_attn")
            patched = True
        
        # Patch cross-attention layers
        if hasattr(block, 'cross_attn'):
            self._patch_attention_layer(block.cross_attn, f"{block_name}.cross_attn")
            patched = True
        
        # Patch MLP layers
        if hasattr(block, 'mlp'):
            self._patch_mlp_layer(block.mlp, f"{block_name}.mlp")
            patched = True
        
        # Patch normalization layers
        for norm_name in ['norm1', 'norm2', 'norm3']:
            if hasattr(block, norm_name):
                self._patch_norm_layer(getattr(block, norm_name), f"{block_name}.{norm_name}")
                patched = True
        
        return patched
    
    def _patch_attention_layer(self, attn_layer, layer_name: str):
        """
        Patch attention layer components (q, k, v, o, norm layers)
        """
        # Patch q, k, v, o projections
        for proj_name in ['q', 'k', 'v', 'o']:
            if hasattr(attn_layer, proj_name):
                proj_layer = getattr(attn_layer, proj_name)
                if isinstance(proj_layer, nn.Linear):
                    new_proj = ComfyUILinear(
                        proj_layer.in_features,
                        proj_layer.out_features,
                        proj_layer.bias is not None
                    )
                    new_proj.weight.data = proj_layer.weight.data.clone()
                    if proj_layer.bias is not None:
                        new_proj.bias.data = proj_layer.bias.data.clone()
                    setattr(attn_layer, proj_name, new_proj)
                    logging.debug(f"    ✅ Patched {layer_name}.{proj_name}")
        
        # Patch normalization layers
        for norm_name in ['norm_q', 'norm_k']:
            if hasattr(attn_layer, norm_name):
                norm_layer = getattr(attn_layer, norm_name)
                if isinstance(norm_layer, nn.LayerNorm):
                    new_norm = ComfyUILayerNorm(
                        norm_layer.normalized_shape,
                        norm_layer.eps,
                        norm_layer.elementwise_affine
                    )
                    if norm_layer.weight is not None:
                        new_norm.weight.data = norm_layer.weight.data.clone()
                    if norm_layer.bias is not None:
                        new_norm.bias.data = norm_layer.bias.data.clone()
                    setattr(attn_layer, norm_name, new_norm)
                    logging.debug(f"    ✅ Patched {layer_name}.{norm_name}")
    
    def _patch_mlp_layer(self, mlp_layer, layer_name: str):
        """
        Patch MLP layer components (fc1, fc2)
        """
        for fc_name in ['fc1', 'fc2']:
            if hasattr(mlp_layer, fc_name):
                fc_layer = getattr(mlp_layer, fc_name)
                if isinstance(fc_layer, nn.Linear):
                    new_fc = ComfyUILinear(
                        fc_layer.in_features,
                        fc_layer.out_features,
                        fc_layer.bias is not None
                    )
                    new_fc.weight.data = fc_layer.weight.data.clone()
                    if fc_layer.bias is not None:
                        new_fc.bias.data = fc_layer.bias.data.clone()
                    setattr(mlp_layer, fc_name, new_fc)
                    logging.debug(f"    ✅ Patched {layer_name}.{fc_name}")
    
    def _patch_norm_layer(self, norm_layer, layer_name: str):
        """
        Patch normalization layers
        """
        if isinstance(norm_layer, nn.LayerNorm):
            new_norm = ComfyUILayerNorm(
                norm_layer.normalized_shape,
                norm_layer.eps,
                norm_layer.elementwise_affine
            )
            if norm_layer.weight is not None:
                new_norm.weight.data = norm_layer.weight.data.clone()
            if norm_layer.bias is not None:
                new_norm.bias.data = norm_layer.bias.data.clone()
            # Replace the layer
            parent = self._get_parent_module(norm_layer)
            if parent is not None:
                attr_name = self._get_attribute_name(parent, norm_layer)
                if attr_name:
                    setattr(parent, attr_name, new_norm)
                    logging.debug(f"    ✅ Patched {layer_name}")
        elif isinstance(norm_layer, nn.GroupNorm):
            new_norm = ComfyUIGroupNorm(
                norm_layer.num_groups,
                norm_layer.num_channels,
                norm_layer.eps,
                norm_layer.affine
            )
            if norm_layer.weight is not None:
                new_norm.weight.data = norm_layer.weight.data.clone()
            if norm_layer.bias is not None:
                new_norm.bias.data = norm_layer.bias.data.clone()
            # Replace the layer
            parent = self._get_parent_module(norm_layer)
            if parent is not None:
                attr_name = self._get_attribute_name(parent, norm_layer)
                if attr_name:
                    setattr(parent, attr_name, new_norm)
                    logging.debug(f"    ✅ Patched {layer_name}")
    
    def _patch_generic_unet(self, model) -> nn.Module:
        """
        Generic handler for UNet models
        """
        logging.info("  🎯 Patching UNet with generic approach...")
        return self._patch_generic_model(model)
    
    def _patch_generic_model(self, model) -> nn.Module:
        """
        Generic patching for any model
        """
        patched_count = 0
        
        # Get all modules to patch
        modules_to_patch = []
        for name, module in model.named_modules():
            modules_to_patch.append((name, module))
        
        for name, module in modules_to_patch:
            if isinstance(module, nn.Linear):
                new_module = ComfyUILinear(
                    module.in_features,
                    module.out_features,
                    module.bias is not None
                )
                new_module.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    new_module.bias.data = module.bias.data.clone()
                
                if self._replace_module(model, name, new_module):
                    patched_count += 1
                    logging.debug(f"  ✅ Patched Linear: {name}")
            
            elif isinstance(module, nn.Conv2d):
                new_module = ComfyUIConv2d(
                    module.in_channels,
                    module.out_channels,
                    module.kernel_size,
                    module.stride,
                    module.padding,
                    module.dilation,
                    module.groups,
                    module.bias is not None,
                    module.padding_mode
                )
                new_module.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    new_module.bias.data = module.bias.data.clone()
                
                if self._replace_module(model, name, new_module):
                    patched_count += 1
                    logging.debug(f"  ✅ Patched Conv2d: {name}")
            
            elif isinstance(module, nn.Conv3d):
                new_module = ComfyUIConv3d(
                    module.in_channels,
                    module.out_channels,
                    module.kernel_size,
                    module.stride,
                    module.padding,
                    module.dilation,
                    module.groups,
                    module.bias is not None,
                    module.padding_mode
                )
                new_module.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    new_module.bias.data = module.bias.data.clone()
                
                if self._replace_module(model, name, new_module):
                    patched_count += 1
                    logging.debug(f"  ✅ Patched Conv3d: {name}")
            
            elif isinstance(module, nn.GroupNorm):
                new_module = ComfyUIGroupNorm(
                    module.num_groups,
                    module.num_channels,
                    module.eps,
                    module.affine
                )
                if module.weight is not None:
                    new_module.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    new_module.bias.data = module.bias.data.clone()
                
                if self._replace_module(model, name, new_module):
                    patched_count += 1
                    logging.debug(f"  ✅ Patched GroupNorm: {name}")
            
            elif isinstance(module, nn.LayerNorm):
                new_module = ComfyUILayerNorm(
                    module.normalized_shape,
                    module.eps,
                    module.elementwise_affine
                )
                if module.weight is not None:
                    new_module.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    new_module.bias.data = module.bias.data.clone()
                
                if self._replace_module(model, name, new_module):
                    patched_count += 1
                    logging.debug(f"  ✅ Patched LayerNorm: {name}")
        
        logging.info(f"  ✅ Generic patching complete: {patched_count} layers patched")
        return model
    
    def _patch_remaining_layers(self, model, model_name: str) -> int:
        """
        Patch remaining layers that weren't handled by specific handlers
        """
        patched_count = 0
        
        # Patch other common layers
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d, nn.GroupNorm, nn.LayerNorm)):
                # Skip if already patched by specific handler
                if hasattr(module, 'weight_function'):
                    continue
                
                # Apply generic patching
                if isinstance(module, nn.Linear):
                    new_module = ComfyUILinear(
                        module.in_features,
                        module.out_features,
                        module.bias is not None
                    )
                    new_module.weight.data = module.weight.data.clone()
                    if module.bias is not None:
                        new_module.bias.data = module.bias.data.clone()
                    
                    if self._replace_module(model, name, new_module):
                        patched_count += 1
                        logging.debug(f"  ✅ Patched remaining Linear: {name}")
        
        return patched_count
    
    def _replace_module(self, model, module_name: str, new_module) -> bool:
        """
        Replace a module in the model hierarchy
        """
        try:
            parent_name = '.'.join(module_name.split('.')[:-1])
            if parent_name:
                # Find parent module
                parent_module = None
                for name, module in model.named_modules():
                    if name == parent_name:
                        parent_module = module
                        break
                
                if parent_module is not None:
                    attr_name = module_name.split('.')[-1]
                    setattr(parent_module, attr_name, new_module)
                    return True
                else:
                    logging.warning(f"    ⚠️  Could not find parent module '{parent_name}' for '{module_name}'")
                    return False
            else:
                # Root module
                logging.warning(f"    ⚠️  Attempting to patch root module '{module_name}' - skipping")
                return False
        except Exception as e:
            logging.warning(f"    ⚠️  Error patching '{module_name}': {e}")
            return False
    
    def _get_parent_module(self, module):
        """
        Get the parent module of a given module (if possible)
        """
        # This is a simplified approach - in practice, we'd need to track the model hierarchy
        return None
    
    def _get_attribute_name(self, parent, module):
        """
        Get the attribute name of a module within its parent
        """
        # This is a simplified approach - in practice, we'd need to track the model hierarchy
        return None

# Global instance
model_aware_patcher = ModelAwarePatcher()

def patch_model_with_comfyui_ops(model):
    """
    Patch a model using model-aware approach
    """
    return model_aware_patcher.patch_model(model)
