import os
import streamlit as st
import torch
import timm

PROJECT_ROOT = os.getcwd()  # makes paths repo-relative (works on Streamlit Cloud)
DEFAULT_CKPT = os.path.join(PROJECT_ROOT, "swin_t_output", "best_swin_t.pth")

# sidebar inputs
ckpt_path = st.sidebar.text_input("Checkpoint path (.pth)", value=DEFAULT_CKPT)
train_classes_dir = st.sidebar.text_input("Train classes dir (used to build labels)", value=os.path.join(PROJECT_ROOT, "clean_train_temp", "train"))

# helpful check
if not os.path.exists(ckpt_path):
    st.sidebar.warning(f"Checkpoint not found at: {os.path.abspath(ckpt_path)}. If running on Streamlit Cloud, either commit the file (or use Git LFS) or provide a downloadable URL and use the 'Download checkpoint' helper below.")
    if st.sidebar.button("Download checkpoint from URL"):
        url = st.sidebar.text_input("Checkpoint URL", value="")
        if url:
            try:
                import requests
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                r = requests.get(url, stream=True)
                with open(ckpt_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.sidebar.success("Downloaded checkpoint.")
            except Exception as e:
                st.sidebar.error(f"Download failed: {e}")

# Load model button (robust load for ckpt['model_state'])
if st.sidebar.button("Load model"):
    try:
        if not os.path.exists(ckpt_path):
            st.sidebar.error("Checkpoint not found. Inspect path or download first.")
        else:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state = None
            for key in ("model_state", "model_state_dict", "state_dict", "state", "model"):
                if isinstance(ckpt, dict) and key in ckpt:
                    state = ckpt[key]; break
            if state is None and isinstance(ckpt, dict):
                state = ckpt
            if not isinstance(state, dict):
                st.sidebar.error("Checkpoint does not contain a state_dict — run inspect_ckpt.py locally to see keys.")
            else:
                # clean keys and build model
                state = {k.replace("module.","").replace("model.",""): v for k,v in state.items()}
                num_classes = len([d for d in os.listdir(train_classes_dir) if os.path.isdir(os.path.join(train_classes_dir, d))]) if os.path.exists(train_classes_dir) else None
                if num_classes is None:
                    st.sidebar.error("Could not infer num_classes from train_classes_dir.")
                else:
                    model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=num_classes)
                    model.load_state_dict(state, strict=False)
                    model.eval()
                    st.session_state["model"] = model
                    st.success("Model loaded successfully.")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        st.sidebar.text(str(e))