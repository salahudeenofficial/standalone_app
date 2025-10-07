import torch
import logging 
from enum import Enum
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from motion.model_patcher import ModelPatcher
import motion.model_management_standalone as model_management
import motion.patcher_extension 
import motion.conds 
import motion.ops
import motion.latent_format 
import motion.model_sampling
import math
import motion.text_encoders.sd3_clip
import motion.text_encoders.wan
import motion.model_base
import motion.supported_model_base

class WAN21_T2V(motion.supported_model_base.BASE):
    unet_config = {
        "image_model": "wan2.1",
        "model_type": "t2v",
    }

    sampling_settings = {
        "shift": 8.0,
    }

    unet_extra_config = {}
    latent_format = motion.latent_format.Wan21

    memory_usage_factor = 1.0

    supported_inference_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def __init__(self, unet_config):
        super().__init__(unet_config)
        self.memory_usage_factor = self.unet_config.get("dim", 2000) / 2000

    def get_model(self, state_dict, prefix="", device=None):
        out = motion.model_base.WAN21(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        pref = self.text_encoder_key_prefix[0]
        t5_detect = motion.text_encoders.sd3_clip.t5_xxl_detect(state_dict, "{}umt5xxl.transformer.".format(pref))
        return motion.supported_model_base.ClipTarget(motion.text_encoders.wan.WanT5Tokenizer, motion.text_encoders.wan.te(**t5_detect))

class WAN21_Vace(WAN21_T2V):
    unet_config = {
        "image_model": "wan2.1",
        "model_type": "vace",
    }

    def __init__(self, unet_config):
        super().__init__(unet_config)
        self.memory_usage_factor = 1.2 * self.memory_usage_factor

    def get_model(self, state_dict, prefix="", device=None):
        out = motion.model_base.WAN21_Vace(self, image_to_video=False, device=device)
        return out

# List of all supported models for model detection
models = [WAN21_T2V, WAN21_Vace]
