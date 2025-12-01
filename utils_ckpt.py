import torch
import os

POSSIBLE_KEYS = ("model_state", "model_state_dict", "state_dict", "model", "state")

def inspect_checkpoint(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data = torch.load(path, map_location="cpu")
    info = {"exists": True, "type": type(data).__name__}
    if isinstance(data, dict):
        info["keys"] = list(data.keys())
    else:
        info["keys"] = None
    return info

def _unwrap_state_dict(obj):
    if isinstance(obj, dict) and obj:
        sample_vals = list(obj.values())[:5]
        if all(isinstance(v, torch.Tensor) or hasattr(v, "dtype") for v in sample_vals):
            return obj
        for k in POSSIBLE_KEYS:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return None

def _clean_state_dict(state_dict):
    new = {}
    for k,v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("model."):
            nk = nk[len("model."):]
        new[nk] = v
    return new

def load_state_dict_from_checkpoint(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data = torch.load(path, map_location="cpu")
    state = _unwrap_state_dict(data)
    if state is None and isinstance(data, dict):
        candidates = [(k,v) for k,v in data.items() if isinstance(v, dict)]
        if candidates:
            key, cand = max(candidates, key=lambda kv: len(kv[1]))
            state = cand
    if state is None and not isinstance(data, dict):
        raise RuntimeError("Checkpoint contains a model object, not a state_dict.")
    if state is None:
        raise RuntimeError("Could not find a state_dict inside checkpoint.")
    return _clean_state_dict(state)