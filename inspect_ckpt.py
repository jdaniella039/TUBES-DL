import torch, os, pprint

def load_checkpoint_to_model(ckpt_path, model, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    # jika file menyimpan model object langsung
    if not isinstance(ckpt, dict):
        return ckpt  # caller harus menangani object model langsung
    # kemungkinan key yang dipakai
    for key in ("model_state", "model_state_dict", "state_dict", "model"):
        if key in ckpt:
            state = ckpt[key]
            break
    else:
        # jika dict itu sendiri adalah state_dict (kecuali ada metadata lain)
        state = ckpt
    # jika state berisi nested dict with parameter names
    if isinstance(state, dict):
        model.load_state_dict(state)
        return model
    raise RuntimeError("Could not find usable state_dict in checkpoint")

ckpt = r"swin_t_output\best_swin_t.pth"
print("exists:", os.path.exists(ckpt), "size(MB):", os.path.getsize(ckpt)/1024/1024)
data = torch.load(ckpt, map_location="cpu")
print("type:", type(data))
if isinstance(data, dict):
    print("keys:", list(data.keys()))
    pprint.pprint({k: type(v) for k,v in data.items()})
else:
    print("Loaded object type (not dict):", type(data))