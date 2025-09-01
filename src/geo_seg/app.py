import argparse
import io
import time
import numpy as np
from PIL import Image
import torch
import streamlit as st

# Configure Streamlit page early
st.set_page_config(page_title="Geo-Seg Demo", page_icon="🗺️", layout="centered")

# Configure Streamlit page early
st.set_page_config(page_title="Geo-Seg Demo", page_icon="🗺️", layout="centered")

try:
    # When run as a script via Streamlit, use absolute imports
    from geo_seg.models.build import create_model  # type: ignore
    from geo_seg.config import default_cfg  # type: ignore
    from geo_seg.utils.preprocess import (
        resize_and_pad_to_square,
        to_model_tensor,
        project_mask_back_to_original,
    )  # type: ignore
except Exception:  # fallback when run as a module
    from .models.build import create_model
    from .config import default_cfg
    from .utils.preprocess import resize_and_pad_to_square, to_model_tensor, project_mask_back_to_original

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
    with st.expander("Notes"):
        st.markdown(
            "- Day‑1 demo runs with random weights unless a checkpoint is provided.\n"
            "- Input is resized and padded to the training size, then predictions are mapped back.\n"
            "- Threshold controls coverage; saved mask is binary PNG."
        )

    # Sidebar controls
    with st.sidebar:
        st.subheader("Display Settings")
        overlay_alpha = st.slider("Overlay opacity", 0.0, 1.0, 0.5, 0.05)

    model = create_model("unet", 1)
    loaded_ckpt = False
    if args.ckpt:
        try:
            state = torch.load(args.ckpt, map_location="cpu")["model"]
            model.load_state_dict(state)
            loaded_ckpt = True
        except Exception as e:
            st.warning(f"Failed to load checkpoint: {e}")
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    # Status banner
    if loaded_ckpt:
        st.success("Checkpoint loaded", icon="✅")
    else:
        st.warning("Random weights (Day‑1)", icon="⚠️")

    # Sidebar controls
    with st.sidebar:
        st.subheader("Settings")
        max_side = 1024
        st.caption(f"Max image side: {max_side}px")
        thresh = st.slider("Threshold", 0.1, 0.9, 0.5, 0.05)
        show_binary_only = st.checkbox("Show binary mask only", value=False)
        morph = st.checkbox("Denoise display (morphology)", value=False)
        morph_radius = st.slider("Morph radius", 1, 5, 2, 1, disabled=not morph)

    # Sample tile option
    c_left, c_right = st.columns([1, 1])
    sample_clicked = c_right.button("Use sample tile")

    uploaded = c_left.file_uploader("Upload PNG/JPG image", type=["png", "jpg", "jpeg"])

    if sample_clicked and uploaded is None:
        # Generate a sample RGB image (checkerboard noise) sized to config image_size
        sz = int(default_cfg.data.image_size)
        arr = (np.random.rand(sz, sz, 3) * 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        img_name = "sample.png"
    elif uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        img_name = getattr(uploaded, "name", "upload.png")
    else:
        img = None
        img_name = None

    if img is not None:
        # Guardrails: max side
        max_side = 1024
        if max(img.size) > max_side:
            st.error(f"Image too large: {img.size}. Please use max side ≤ {max_side}.")
            return

        # Preprocess: resize+pad to training size
        padded, info = resize_and_pad_to_square(img, int(default_cfg.data.image_size))
        x = to_model_tensor(padded).to(device)

        # Show size info
        st.caption(
            f"Original: {info.orig_size[0]}×{info.orig_size[1]} → Padded canvas: {info.canvas_size[0]}×{info.canvas_size[1]} (resized region: {info.resized_size[0]}×{info.resized_size[1]})"
        )

        # Inference
        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = model(x)
            probs = torch.sigmoid(logits[:, 0]).squeeze(0).cpu().numpy()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        st.caption(f"Inference: {dt_ms:.1f} ms on CPU")

        # Cache probs in session for live thresholding
        st.session_state["probs_canvas"] = probs
        # Threshold
        probs = st.session_state.get("probs_canvas", None)
        if probs is None:
            st.error("No probabilities available.")
            return
        mask_canvas = (probs >= float(thresh)).astype(np.uint8)
        coverage = float(mask_canvas.mean() * 100.0)
        st.caption(f"Mask coverage: {coverage:.1f}%")

        # Map mask back to original resolution
        mask_img = project_mask_back_to_original(mask_canvas, info)

        # Optional morphology (display only)
        if morph:
            try:
                import numpy as _np
                from skimage.morphology import disk, binary_opening, binary_closing

                m = (_np.array(mask_img) > 127)
                se = disk(int(morph_radius))
                m = binary_opening(m, se)
                m = binary_closing(m, se)
                mask_img = Image.fromarray((_np.uint8(m) * 255), mode="L")
            except Exception as e:
                st.info(f"Morphology unavailable: {e}")

        # Display
        if show_binary_only:
            st.image(mask_img, caption="Binary Mask", use_container_width=True)
        else:
            base_rgba = img.convert("RGBA")
            mask_arr = (np.array(mask_img) > 127).astype(np.uint8)
            alpha_val = int(0.5 * 255)
            overlay = np.zeros((mask_arr.shape[0], mask_arr.shape[1], 4), dtype=np.uint8)
            overlay[..., 0] = 255  # red
            overlay[..., 3] = mask_arr * alpha_val
            overlay_img = Image.fromarray(overlay, mode="RGBA")
            composite = Image.alpha_composite(base_rgba, overlay_img)
            st.image(composite, caption="Input + Predicted Mask (red)", use_container_width=True)
            st.caption("Legend: red overlay = predicted foreground (class 1)")

        # Download prediction
        if img_name:
            stem = img_name.rsplit(".", 1)[0]
            out_name = f"{stem}_mask.png"
            buf = io.BytesIO()
            mask_img.save(buf, format="PNG")
            st.download_button(
                label="Download mask PNG",
                data=buf.getvalue(),
                file_name=out_name,
                mime="image/png",
            )


if __name__ == "__main__":
    main()
