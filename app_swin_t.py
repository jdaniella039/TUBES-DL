# app_swin_t.py
import os
import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import timm
import numpy as np

# ---------------- CONFIG (ubah kalau perlu) ----------------
DEFAULT_PROJECT_ROOT = os.getcwd()  # biasanya: "C:/DATA JOY/Tugas/SEM 7/DL/TUBES DL"
DEFAULT_CHECKPOINT = "https://drive.google.com/uc?export=download&id=1TMmw78DykVNPR0_dEh4069dJ5uXycUHJ"
DEFAULT_CLASSES_DIR = os.path.join(DEFAULT_PROJECT_ROOT, "clean_train_temp", "train")
IMG_SIZE = 224

# ----------------- util -----------------
@st.cache_data
def get_class_names(classes_dir):
    if not os.path.isdir(classes_dir):
        return []
    items = [d for d in sorted(os.listdir(classes_dir)) if os.path.isdir(os.path.join(classes_dir, d))]
    return items

def build_swin_t(num_classes, device):
    # create architecture (swin tiny)
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=num_classes)
    model.to(device)
    model.eval()
    return model

def load_checkpoint(model, ckpt_path, device):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    # ckpt might be:
    # 1) direct state_dict
    # 2) dict with 'model_state_dict' or 'state_dict'
    if isinstance(ckpt, dict):
        # common keys
        for k in ('model_state_dict', 'state_dict', 'model'):
            if k in ckpt:
                state = ckpt[k]
                break
        else:
            # maybe already a state_dict (but packed under other keys) - try heuristics:
            state = None
            # fallback: if first tensor-like, assume dict is state_dict
            # check if values are tensors
            sample_val = next(iter(ckpt.values()))
            if hasattr(sample_val, 'shape') or torch.is_tensor(sample_val):
                state = ckpt
    else:
        state = ckpt

    if state is None:
        raise RuntimeError("Could not find a model state_dict in checkpoint.")
    # try loading, allow mismatch (strict=False) so final head can be different
    msg = model.load_state_dict(state, strict=False)
    return msg

# Preprocess: same as training - use imagenet mean/std
def preprocess_pil(img: Image.Image):
    transform = T.Compose([
        T.Resize(int(IMG_SIZE*1.14)),   # slightly larger then center crop
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return transform(img).unsqueeze(0)  # 1 x C x H x W

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Swin-T Inference", layout="wide")
st.title("Swin-T Inference (Streamlit)")

# sidebar config
st.sidebar.header("Config")
proj_root = st.sidebar.text_input("Project root (folder)", value=DEFAULT_PROJECT_ROOT)
ckpt_path = st.sidebar.text_input("Checkpoint path (.pth)", value=DEFAULT_CHECKPOINT)
classes_dir = st.sidebar.text_input("Train classes dir (used to build labels)", value=DEFAULT_CLASSES_DIR)
use_gpu = st.sidebar.checkbox("Use GPU (if available)", value=False)
top_k = st.sidebar.slider("Top K results", min_value=1, max_value=10, value=3)

device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
st.sidebar.write(f"Device: {device}")

# load class names
class_names = get_class_names(classes_dir)
if len(class_names) == 0:
    st.sidebar.error(f"No class subfolders found in: {classes_dir}")
else:
    st.sidebar.success(f"Found {len(class_names)} classes")

# load model button to avoid heavy load on refresh
if st.sidebar.button("Load model"):
    if len(class_names) == 0:
        st.sidebar.error("Cannot build model - no classes.")
    else:
        with st.spinner("Building model..."):
            try:
                model = build_swin_t(num_classes=len(class_names), device=device)
                msg = load_checkpoint(model, ckpt_path, device)
                st.sidebar.success("Model loaded. (load msg: {})".format(msg))
                # persist in session
                st.session_state['model'] = model
                st.session_state['class_names'] = class_names
            except Exception as e:
                st.sidebar.error(f"Error loading model: {e}")

# if previously loaded, show info
if 'model' in st.session_state:
    st.info("Model ready — upload image(s) below for inference.")

uploaded = st.file_uploader("Upload image(s) (jpg/png). You can upload multiple.", type=['jpg','jpeg','png','bmp'], accept_multiple_files=True)
cols = st.columns( min(4, max(1, (len(uploaded) or 1))) )

if uploaded and 'model' in st.session_state:
    model = st.session_state['model']
    names = st.session_state['class_names']
    for i, up in enumerate(uploaded):
        try:
            img = Image.open(up).convert("RGB")
        except Exception as e:
            st.error(f"Cannot open {up.name}: {e}")
            continue
        # preprocess
        x = preprocess_pil(img).to(device)
        with torch.no_grad():
            logits = model(x)  # shape (1, num_classes)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            top_idx = np.argsort(probs)[::-1][:top_k]
            top_probs = probs[top_idx]
            top_labels = [names[idx] if idx < len(names) else f"cls_{idx}" for idx in top_idx]

        col = cols[i % len(cols)]
        col.image(img, use_column_width='always', caption=up.name)
        # results
        lines = [f"**{top_labels[0]}** — {top_probs[0]*100:.2f}%"]
        for lab, p in zip(top_labels[1:], top_probs[1:]):
            lines.append(f"{lab} — {p*100:.2f}%")
        col.markdown("\n\n".join(lines))
        # bar chart
        chart_data = {lab: float(p) for lab,p in zip(top_labels, top_probs)}
        try:
            import pandas as pd
            df = pd.DataFrame.from_dict(chart_data, orient='index', columns=['prob'])
            col.bar_chart(df)
        except Exception:
            # fallback
            col.write(chart_data)

elif uploaded and 'model' not in st.session_state:
    st.warning("Model belum diload. Tekan 'Load model' di kiri (sidebar) terlebih dahulu.")

st.write("---")
st.write("Tips:")
st.write("- Jika checkpoint tidak terdeteksi, pastikan path `.pth` benar (contoh default: `swin_t_output/best_swin_t.pth`).")
st.write("- Nama kelas diambil dari subfolder di `clean_train_temp/train`. Kalau kamu punya file `labels.txt`, bisa ubah `get_class_names()` untuk load dari file.")
st.write("- Dependencies: `torch`, `torchvision`, `timm`, `streamlit`, `Pillow`.")
