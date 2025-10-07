from __future__ import annotations
from typing import TYPE_CHECKING,Callable
import enum
import math
import torch
import numpy as np
import itertools
import logging

if TYPE_CHECKING:
    from comfy.model_patcher import ModelPatcher, PatcherInjection
    from comfy.model_base import BaseModel
    from comfy.sd import CLIP
import motion.lora
import motion.model_management_standalone
import motion.patcher_extension

class EnumHookMode(enum.Enum):
    '''
    Priority of hook memory optimization vs. speed, mostly related to WeightHooks.

    MinVram: No caching will occur for any operations related to hooks.
    MaxSpeed: Excess VRAM (and RAM, once VRAM is sufficiently depleted) will be used to cache hook weights when switching hook groups.
    '''
    MinVram = "minvram"
    MaxSpeed = "maxspeed"

class EnumHookType(enum.Enum):
    '''
    Hook types, each of which has different expected behavior.
    '''
    Weight = "weight"
    ObjectPatch = "object_patch"
    AdditionalModels = "add_models"
    TransformerOptions = "transformer_options"
    Injections = "add_injections"

class EnumHookScope(enum.Enum):
    '''
    Hook scope types.
    '''
    HookedOnly = "hooked_only"
    Global = "global"

class _HookRef:
    pass

class Hook:
    '''
    Base hook class.
    '''
    def __init__(self, hook_type: EnumHookType, hook_scope: EnumHookScope):
        self.hook_type = hook_type
        self.hook_scope = hook_scope

class WeightHook(Hook):
    '''
    Hook responsible for tracking weights to be applied to some model/clip.

    Note, value of hook_scope is ignored and is treated as HookedOnly.
    '''
    def __init__(self, strength_model=1.0, strength_clip=1.0):
        super().__init__(hook_type=EnumHookType.Weight, hook_scope=EnumHookScope.HookedOnly)
        self.weights: dict = None
        self.weights_clip: dict = None
        self.need_weight_init = True
        self._strength_model = strength_model
        self._strength_clip = strength_clip
        self.hook_scope = EnumHookScope.HookedOnly # this value does not matter for WeightHooks, just for docs

    @property
    def strength_model(self):
        return self._strength_model * self.strength

    @property
    def strength_clip(self):
        return self._strength_clip * self.strength

    def add_hook_patches(self, model: ModelPatcher, model_options: dict, target_dict: dict[str], registered: HookGroup):
        if not self.should_register(model, model_options, target_dict, registered):
            return False
        weights = None

        target = target_dict.get('target', None)
        if target == EnumWeightTarget.Clip:
            strength = self._strength_clip
        else:
            strength = self._strength_model

        if self.need_weight_init:
            key_map = {}
            if target == EnumWeightTarget.Clip:
                key_map = comfy.lora.model_lora_keys_clip(model.model, key_map)
            else:
                key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
            weights = comfy.lora.load_lora(self.weights, key_map, log_missing=False)
        else:
            if target == EnumWeightTarget.Clip:
                weights = self.weights_clip
            else:
                weights = self.weights
        model.add_hook_patches(hook=self, patches=weights, strength_patch=strength)
        registered.add(self)
        return True
        # TODO: add logs about any keys that were not applied

    def clone(self):
        c: WeightHook = super().clone()
        c.weights = self.weights
        c.weights_clip = self.weights_clip
        c.need_weight_init = self.need_weight_init
        c._strength_model = self._strength_model
        c._strength_clip = self._strength_clip
        return c



class HookGroup:
    '''
    Stores groups of hooks, and allows them to be queried by type.

    To prevent breaking their functionality, never modify the underlying self.hooks or self._hook_dict vars directly;
    always use the provided functions on HookGroup.
    '''
    def __init__(self):
        self.hooks: list[Hook] = []
        self._hook_dict: dict[EnumHookType, list[Hook]] = {}

    def __len__(self):
        return len(self.hooks)

    def add(self, hook: Hook):
        if hook not in self.hooks:
            self.hooks.append(hook)
            self._hook_dict.setdefault(hook.hook_type, []).append(hook)

    def remove(self, hook: Hook):
        if hook in self.hooks:
            self.hooks.remove(hook)
            self._hook_dict[hook.hook_type].remove(hook)

    def get_type(self, hook_type: EnumHookType):
        return self._hook_dict.get(hook_type, [])

    def contains(self, hook: Hook):
        return hook in self.hooks

    def is_subset_of(self, other: HookGroup):
        self_hooks = set(self.hooks)
        other_hooks = set(other.hooks)
        return self_hooks.issubset(other_hooks)

    def new_with_common_hooks(self, other: HookGroup):
        c = HookGroup()
        for hook in self.hooks:
            if other.contains(hook):
                c.add(hook.clone())
        return c

    def clone(self):
        c = HookGroup()
        for hook in self.hooks:
            c.add(hook.clone())
        return c

    def clone_and_combine(self, other: HookGroup):
        c = self.clone()
        if other is not None:
            for hook in other.hooks:
                c.add(hook.clone())
        return c

    def set_keyframes_on_hooks(self, hook_kf: HookKeyframeGroup):
        if hook_kf is None:
            hook_kf = HookKeyframeGroup()
        else:
            hook_kf = hook_kf.clone()
        for hook in self.hooks:
            hook.hook_keyframe = hook_kf

    def get_hooks_for_clip_schedule(self):
        scheduled_hooks: dict[WeightHook, list[tuple[tuple[float,float], HookKeyframe]]] = {}
        # only care about WeightHooks, for now
        for hook in self.get_type(EnumHookType.Weight):
            hook: WeightHook
            hook_schedule = []
            # if no hook keyframes, assign default value
            if len(hook.hook_keyframe.keyframes) == 0:
                hook_schedule.append(((0.0, 1.0), None))
                scheduled_hooks[hook] = hook_schedule
                continue
            # find ranges of values
            prev_keyframe = hook.hook_keyframe.keyframes[0]
            for keyframe in hook.hook_keyframe.keyframes:
                if keyframe.start_percent > prev_keyframe.start_percent and not math.isclose(keyframe.strength, prev_keyframe.strength):
                    hook_schedule.append(((prev_keyframe.start_percent, keyframe.start_percent), prev_keyframe))
                    prev_keyframe = keyframe
                elif keyframe.start_percent == prev_keyframe.start_percent:
                    prev_keyframe = keyframe
            # create final range, assuming last start_percent was not 1.0
            if not math.isclose(prev_keyframe.start_percent, 1.0):
                hook_schedule.append(((prev_keyframe.start_percent, 1.0), prev_keyframe))
            scheduled_hooks[hook] = hook_schedule
        # hooks should not have their schedules in a list of tuples
        all_ranges: list[tuple[float, float]] = []
        for range_kfs in scheduled_hooks.values():
            for t_range, keyframe in range_kfs:
                all_ranges.append(t_range)
        # turn list of ranges into boundaries
        boundaries_set = set(itertools.chain.from_iterable(all_ranges))
        boundaries_set.add(0.0)
        boundaries = sorted(boundaries_set)
        real_ranges = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
        # with real ranges defined, give appropriate hooks w/ keyframes for each range
        scheduled_keyframes: list[tuple[tuple[float,float], list[tuple[WeightHook, HookKeyframe]]]] = []
        for t_range in real_ranges:
            hooks_schedule = []
            for hook, val in scheduled_hooks.items():
                keyframe = None
                # check if is a keyframe that works for the current t_range
                for stored_range, stored_kf in val:
                    # if stored start is less than current end, then fits - give it assigned keyframe
                    if stored_range[0] < t_range[1] and stored_range[1] > t_range[0]:
                        keyframe = stored_kf
                        break
                hooks_schedule.append((hook, keyframe))
            scheduled_keyframes.append((t_range, hooks_schedule))
        return scheduled_keyframes

    def reset(self):
        for hook in self.hooks:
            hook.reset()

    @staticmethod
    def combine_all_hooks(hooks_list: list[HookGroup], require_count=0) -> HookGroup:
        actual: list[HookGroup] = []
        for group in hooks_list:
            if group is not None:
                actual.append(group)
        if len(actual) < require_count:
            raise Exception(f"Need at least {require_count} hooks to combine, but only had {len(actual)}.")
        # if no hooks, then return None
        if len(actual) == 0:
            return None
        # if only 1 hook, just return itself without cloning
        elif len(actual) == 1:
            return actual[0]
        final_hook: HookGroup = None
        for hook in actual:
            if final_hook is None:
                final_hook = hook.clone()
            else:
                final_hook = final_hook.clone_and_combine(hook)
        return final_hook


def create_transformer_options_from_hooks(model: ModelPatcher, hooks: HookGroup,  transformer_options: dict[str]=None):
    # if no hooks or is not a ModelPatcher for sampling, return empty dict
    if hooks is None or model.is_clip:
        return {}
    if transformer_options is None:
        transformer_options = {}
    for hook in hooks.get_type(EnumHookType.TransformerOptions):
        hook: TransformerOptionsHook
        hook.on_apply_hooks(model, transformer_options)
    return transformer_options

