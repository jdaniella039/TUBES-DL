import os
import traceback
import torch
import timm
import streamlit as st

PROJECT_ROOT = os.getcwd()
DEFAULT_CKPT = os.path.join(PROJECT_ROOT, "swin_t_output", "best_swin_t.pth")
TRAIN_DIR_DEF = os.path.join(PROJECT_ROOT, "clean_train_temp", "train")

ckpt_path = st.sidebar.text_input("Checkpoint path (.pth)", value=DEFAULT_CKPT)
train_classes_dir = st.sidebar.text_input("Train classes dir (used to build labels)", value=TRAIN_DIR_DEF)

def _clean_state(sd):
    return {k.replace("module.", "").replace("model.", ""): v for k, v in sd.items()}

if st.sidebar.button("Inspect checkpoint"):
    try:
        if not os.path.exists(ckpt_path):
            st.sidebar.error(f"Checkpoint not found: {os.path.abspath(ckpt_path)}")
        else:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            st.sidebar.write("type:", type(ckpt).__name__)
            if isinstance(ckpt, dict):
                st.sidebar.write("keys:", list(ckpt.keys()))
            else:
                st.sidebar.write("Checkpoint is not a dict (saved model object).")
    except Exception:
        st.sidebar.error("Inspect failed; see traceback below")
        st.sidebar.text(traceback.format_exc())

if st.sidebar.button("Load model"):
    try:
        if not os.path.exists(ckpt_path):
            st.sidebar.error("Checkpoint not found. Upload or provide correct path.")
        else:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            # cari state_dict di beberapa key umum (termasuk model_state)
            state = None
            for key in ("model_state", "model_state_dict", "state_dict", "state", "model"):
                if isinstance(ckpt, dict) and key in ckpt:
                    state = ckpt[key]; break
            if state is None and isinstance(ckpt, dict):
                state = ckpt  # fallback jika checkpoint itu sendiri adalah state_dict
            if not isinstance(state, dict):
                st.sidebar.error("Could not find a state_dict in checkpoint. Use Inspect first.")
            else:
                state = _clean_state(state)
                # infer num_classes dari folder train jika tersedia
                if os.path.isdir(train_classes_dir):
                    num_classes = len([d for d in os.listdir(train_classes_dir) if os.path.isdir(os.path.join(train_classes_dir, d))])
                else:
                    st.sidebar.error("Train classes dir not found; set it correctly.")
                    raise SystemExit
                model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=num_classes)
                model.load_state_dict(state, strict=False)
                model.eval()
                st.session_state["model"] = model
                st.success("Model loaded successfully.")
    except Exception:
        st.sidebar.error("Error loading model — see traceback below")
        st.sidebar.text(traceback.format_exc())