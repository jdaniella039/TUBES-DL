# fix_checkpoint.py
import torch
import os
from collections import OrderedDict

# Ubah path ini ke file .pth kamu
orig_path = r"C:\DATA JOY\Tugas\SEM 7\DL\TUBES DL\swin_t_output\best_swin_t.pth"
out_path = orig_path.replace(".pth", "_fixed.pth")

print("Loading:", orig_path)
ckpt = torch.load(orig_path, map_location="cpu")
print("Type:", type(ckpt))

# helper: apakah ini tampak seperti state_dict (mapping nama->tensor)
def looks_like_state_dict(x):
    if isinstance(x, (dict, OrderedDict)):
        # check some typical tensor values
        for k,v in list(x.items())[:5]:
            # values are tensors or lists/tuples of tensors
            if hasattr(v, "shape") or hasattr(v, "dtype"):
                return True
        # fallback: keys look like 'backbone.conv1.weight'
        return any('.' in str(k) for k in x.keys())
    return False

state = None
if isinstance(ckpt, dict):
    print("Keys in checkpoint:", list(ckpt.keys()))
    # search common keys
    for candidate in ("model_state_dict", "state_dict", "model", "module", "net"):
        if candidate in ckpt and looks_like_state_dict(ckpt[candidate]):
            state = ckpt[candidate]
            print("Found state_dict under key:", candidate)
            break
    # sometimes it's nested under 'checkpoint' or 'net' etc.
    if state is None:
        for k,v in ckpt.items():
            if looks_like_state_dict(v):
                state = v
                print("Found state_dict under nested key:", k)
                break

else:
    # not a dict -> probably raw state_dict (OrderedDict)
    if looks_like_state_dict(ckpt):
        state = ckpt
        print("Checkpoint itself looks like a raw state_dict (not a dict wrapper).")

if state is None:
    raise RuntimeError("Tidak dapat menemukan state_dict di dalam checkpoint. "
                       "Cek output di atas. Jika file korup, coba generate ulang checkpoint.")

# build a new dict that includes multiple common keys so loader pasti menemukan satu
new_ckpt = {
    "model": state,
    "state_dict": state,
    "model_state_dict": state
}

# optionally preserve other metadata (epoch, optimizer) if present
if isinstance(ckpt, dict):
    for k in ("epoch","optimizer","scaler"):
        if k in ckpt:
            new_ckpt[k] = ckpt[k]

torch.save(new_ckpt, out_path)
print("Saved fixed checkpoint to:", out_path)
