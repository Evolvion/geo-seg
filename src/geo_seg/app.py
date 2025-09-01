import argparse
import numpy as np
from PIL import Image
import torch
import streamlit as st

# Configure Streamlit page early
st.set_page_config(page_title="Geo-Seg Demo", page_icon="🗺️", layout="centered")

try:
    # When run as a script via Streamlit, use absolute import
    from geo_seg.models.build import create_model  # type: ignore
except Exception:  # fallback when run as a module
    from .models.build import create_model

# Pillow resampling constants
try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR  # Pillow >= 9
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:  # pragma: no cover - older Pillow
    RESAMPLE_BILINEAR = Image.BILINEAR  # type: ignore[attr-defined]
    RESAMPLE_NEAREST = Image.NEAREST  # type: ignore[attr-defined]


def _resize_to_multiple(img: Image.Image, m: int = 4) -> tuple[Image.Image, tuple[int, int]]:
    w, h = img.size
    w2, h2 = w - (w % m), h - (h % m)
    if (w2, h2) != (w, h):
        img = img.resize((w2, h2), RESAMPLE_BILINEAR)
    return img, (w, h)


def load_image_to_tensor(img: Image.Image) -> tuple[torch.Tensor, tuple[int, int]]:
    img = img.convert("RGB")
    img_resized, orig_size = _resize_to_multiple(img, m=4)
    arr = np.asarray(img_resized, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return t, orig_size


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None)
    args = parser.parse_args(argv)

    st.title("Geo-Seg Demo")
    st.write("Load an image and predict a mask.")

    # Sidebar controls
    with st.sidebar:
        st.subheader("Display Settings")
        overlay_alpha = st.slider("Overlay opacity", 0.0, 1.0, 0.5, 0.05)

    model = create_model("unet", 1)
    if args.ckpt:
        try:
            state = torch.load(args.ckpt, map_location="cpu")["model"]
            model.load_state_dict(state)
        except Exception as e:
            st.warning(f"Failed to load checkpoint: {e}")
    model.eval()

    uploaded = st.file_uploader("Upload PNG image", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Input", use_container_width=True)
        if st.button("Predict"):
            x, orig_size = load_image_to_tensor(img)
            with torch.no_grad():
                logits = model(x)
                probs = torch.sigmoid(logits[:, 0])
                mask = (probs >= 0.5).float().squeeze(0).numpy()

            # Resize mask back to original size for display
            mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(
                orig_size, RESAMPLE_NEAREST
            )

            # Red overlay for predicted class
            base_rgba = img.convert("RGBA")
            mask_arr = (np.array(mask_img) > 127).astype(np.uint8)
            alpha_val = int(max(0.0, min(1.0, overlay_alpha)) * 255)
            overlay = np.zeros((mask_arr.shape[0], mask_arr.shape[1], 4), dtype=np.uint8)
            overlay[..., 0] = 255  # red
            overlay[..., 3] = mask_arr * alpha_val
            overlay_img = Image.fromarray(overlay, mode="RGBA")
            composite = Image.alpha_composite(base_rgba, overlay_img)

            c1, c2 = st.columns(2)
            with c1:
                st.image(composite, caption="Input + Predicted Mask (red)", use_container_width=True)
                st.caption("Legend: red overlay = predicted foreground (class 1)")
            with c2:
                st.image(mask_img, caption="Binary Mask", use_container_width=True)


if __name__ == "__main__":
    main()
