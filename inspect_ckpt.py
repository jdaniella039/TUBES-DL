import torch, timm
import streamlit as st

# pastikan num_classes sudah tersedia (mis. dari folder train classes)
# num_classes = len(classes)

# --- Load model (paste di sidebar handler) ---
ckpt_path = st.text_input("Checkpoint path (.pth)", value="swin_t_output/best_swin_t.pth")

if st.button("Inspect checkpoint"):
    try:
        info = torch.load(ckpt_path, map_location="cpu")
        st.json({"type": type(info).__name__, "keys": list(info.keys()) if isinstance(info, dict) else None})
    except Exception as e:
        st.error(f"Inspect failed: {e}")

if st.button("Load model"):
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = None
        for key in ("model_state","model_state_dict","state_dict","model"):
            if isinstance(ckpt, dict) and key in ckpt:
                state = ckpt[key]; break
        if state is None and isinstance(ckpt, dict):
            state = ckpt
        if state is None:
            st.error("Could not find a state_dict in checkpoint.")
        else:
            # cleanup keys like "module." or "model."
            state = {k.replace("module.","").replace("model.",""): v for k,v in state.items()}
            model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=num_classes)
            model.load_state_dict(state, strict=False)
            model.eval()
            st.session_state["model"] = model
            st.success("Model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model: {e}")