# Fix the model_lora_keys_unet function to handle diffusion_model prefix mismatch

def model_lora_keys_unet(model, key_map: Dict[str, str] = None) -> Dict[str, str]:
    """Generate LoRA key mapping for UNet model - WAN focused"""
    if key_map is None:
        key_map = {}
    
    sd = model.state_dict()
    sdk = sd.keys()

    # Generic mapping for all weight parameters
    for k in sdk:
        if k.endswith(".weight"):
            # Standard LoRA format
            key_lora = k[:-len(".weight")].replace(".", "_")
            key_map[f"lora_unet_{key_lora}"] = k
            key_map[k[:-len(".weight")]] = k  # Generic format

            # CRITICAL FIX: Handle diffusion_model prefix mismatch
            # LoRA keys have "diffusion_model." prefix, but model keys don't
            # Create mappings for both cases
            diffusion_model_key = f"diffusion_model.{k}"
            key_map[diffusion_model_key[:-len(".weight")]] = k  # Map diffusion_model.blocks.0.cross_attn.k -> blocks.0.cross_attn.k.weight
            
            # Also create the reverse mapping for LoRA keys
            wan_key = k[:-len(".weight")].replace(".", "_")
            key_map[f"diffusion_model.{k[:-len('.weight')]}"] = k  # Map diffusion_model.blocks.0.cross_attn.k -> blocks.0.cross_attn.k.weight
            
        else:
            key_map[k] = k  # Generic format for non-weight parameters
            # Also map with diffusion_model prefix
            key_map[f"diffusion_model.{k}"] = k

    return key_map

print("Fixed model_lora_keys_unet function created!")
print("Key changes:")
print("1. Added mapping for diffusion_model.{model_key} -> model_key")
print("2. This handles the case where LoRA keys have diffusion_model prefix but model keys don't")
