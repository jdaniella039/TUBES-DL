import torch

checkpoint_path = "path_to_best_swin_t_fixed.pth"  # Gantilah dengan path file kamu
out_path = "path_to_corrected_model.pth"  # Tempat untuk menyimpan model yang sudah diperbaiki

ckpt = torch.load(checkpoint_path)
if isinstance(ckpt, dict) and "model_state_dict" not in ckpt:
    # Menambahkan key model_state_dict jika belum ada
    ckpt["model_state_dict"] = ckpt.get("state_dict", None) or ckpt.get("model", None)

torch.save(ckpt, out_path)
print(f"Checkpoint yang sudah diperbaiki disimpan di {out_path}")
