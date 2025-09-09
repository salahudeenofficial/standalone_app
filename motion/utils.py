import torch
import safetensors.torch




def load_torch_file(file_path,device=None):
    if device is None:
        device = torch.device("cpu")
    if file_path.lower().endswith(".safetensors"):
        try:
            with safetensors.safe_open(file_path,framework="pt",device=device.type) as f:
                sd = {}
                for k in f.keys():
                    tensor = f.get_tensor(k)
                    tensor = tensor.to(device)
                    sd[k] = tensor
        except Exception as e:
            raise ValueError(f"Error loading safetensors file: {e}")
    else:
        raise ValueError(f"Unsupported file extension: {file_path}")
    
    return sd

def calculate_parameters(sd, prefix=""):
    params = 0
    for k in sd.keys():
        if k.startswith(prefix):
            w = sd[k]
            params += w.nelement()
    return params