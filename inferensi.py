# inferensi.py
"""
Inferensi & evaluasi untuk Swin-Tiny.
Output:
 - inference_results.csv  (filename, true_label, pred_label, pred_prob)
 - inference_metrics.csv  (per-class precision/recall/f1/support)
Usage: edit CONFIG section below, lalu:
    python inferensi.py
"""
import os
import torch
import timm
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, classification_report
import traceback

# ---------------- CONFIG (ubah sesuai lingkungan kamu) ----------------
# Pastikan path ini menunjuk ke checkpoint yang benar; default menunjuk ke file yg biasanya tersimpan
CHECKPOINT_PATH = r"C:\DATA JOY\Tugas\SEM 7\DL\TUBES DL\swin_t_output\best_swin_t.pth"

# Path ke folder test (ImageFolder structure): TEST_DIR/<class_name>/*.jpg
# Sesuaikan: sebelumnya kamu pakai 'val' bukan 'valid'
TEST_DIR = r"C:\DATA JOY\Tugas\SEM 7\DL\TUBES DL\clean_train_temp\val"

# Output names
OUT_PRED_CSV = "inference_results.csv"
OUT_METRICS_CSV = "inference_metrics.csv"

# Settings
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0   # safer for Windows; ubah >0 di Linux
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------------------------------------------------------------

def build_swin_t(num_classes, device):
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=num_classes)
    model.to(device)
    model.eval()
    return model

def load_checkpoint_to_model(model, ckpt_path, device):
    """Robust loader: handle
       - checkpoint saved as dict with key 'model_state' or 'state_dict' or 'model_state_dict'
       - checkpoint saved as plain state_dict
       - raise informative errors
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # load checkpoint (simple)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # if ckpt is dict, try common keys
    state = None
    if isinstance(ckpt, dict):
        for k in ('model_state','model_state_dict','state_dict','model','state'):
            if k in ckpt:
                state = ckpt[k]
                break
        # fallback: treat top-level dict as state_dict if values look like tensors
        if state is None:
            sample_vals = list(ckpt.values())[:5]
            if sample_vals and all(hasattr(v, 'shape') or torch.is_tensor(v) for v in sample_vals):
                state = ckpt
    else:
        # if it's not dict but has state_dict attribute (unlikely) attempt direct use
        if hasattr(ckpt, '__dict__'):
            # cannot reliably extract => raise
            raise RuntimeError("Checkpoint appears to be a saved model object, not a state_dict. Save model.state_dict() instead or use a loader that loads the full model object.")
        raise RuntimeError("Unsupported checkpoint format")

    if state is None or not isinstance(state, dict):
        raise RuntimeError("Could not find a model state_dict in checkpoint.")

    # clean common prefixes
    clean_state = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("model."):
            nk = nk[len("model."):]
        clean_state[nk] = v

    # load into model (use strict=False to be tolerant)
    load_msg = model.load_state_dict(clean_state, strict=False)
    return load_msg

def main():
    try:
        print("Device:", DEVICE)
        # check test dir
        if not os.path.isdir(TEST_DIR):
            raise SystemExit(f"Test directory not found: {TEST_DIR}")

        # transforms & dataloader
        transform = T.Compose([
            T.Resize(int(IMG_SIZE * 1.14)),
            T.CenterCrop(IMG_SIZE),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        dataset = ImageFolder(TEST_DIR, transform=transform)
        class_names = dataset.classes
        print(f"Found {len(dataset)} images across {len(class_names)} classes.")
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        # build & load model
        model = build_swin_t(num_classes=len(class_names), device=DEVICE)
        print("Model built. Loading checkpoint...")
        msg = load_checkpoint_to_model(model, CHECKPOINT_PATH, DEVICE)
        print("Load message:", msg)

        # inference
        model.eval()
        filenames, true_labels, pred_labels, pred_probs = [], [], [], []
        idx_global = 0
        with torch.no_grad():
            for imgs, targets in tqdm(loader, desc="Inference"):
                imgs = imgs.to(DEVICE)
                logits = model(imgs)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                batch_sz = imgs.size(0)
                for i in range(batch_sz):
                    path, _ = dataset.samples[idx_global]
                    filenames.append(os.path.basename(path))
                    true_labels.append(class_names[targets[i].item()])
                    pred_labels.append(class_names[preds[i]])
                    pred_probs.append(float(probs[i, preds[i]]))
                    idx_global += 1

        print("Processed", len(filenames), "images. Saving results...")

        # save predictions
        df = pd.DataFrame({
            "filename": filenames,
            "true_label": true_labels,
            "pred_label": pred_labels,
            "pred_prob": pred_probs
        })
        df.to_csv(OUT_PRED_CSV, index=False)
        print("Saved predictions to", OUT_PRED_CSV)

        # compute metrics
        y_true = [class_names.index(x) for x in true_labels]
        y_pred = [class_names.index(x) for x in pred_labels]

        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=range(len(class_names)), zero_division=0)
        metrics_df = pd.DataFrame({
            "class": class_names,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support
        })
        metrics_df.to_csv(OUT_METRICS_CSV, index=False)
        print("Saved metrics to", OUT_METRICS_CSV)

        # print summary
        print("\nClassification report:\n")
        print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()

if __name__ == "__main__":
    main()
