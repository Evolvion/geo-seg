import argparse
import numpy as np
from PIL import Image
import torch
import streamlit as st

try:
    # When run as a script via Streamlit, use absolute import
    from geo_seg.models.build import create_model  # type: ignore
except Exception:  # fallback when run as a module
    from .models.build import create_model


def load_image_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return t


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None)
    args = parser.parse_args(argv)

    st.title("Geo-Seg Demo")
    st.write("Load an image and predict a mask.")

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
        img = Image.open(uploaded)
        st.image(img, caption="Input", use_column_width=True)
        if st.button("Predict"):
            x = load_image_to_tensor(img)
            with torch.no_grad():
                logits = model(x)
                probs = torch.sigmoid(logits[:, 0])
                mask = (probs >= 0.5).float().squeeze(0).numpy()
            st.image(mask, caption="Mask", use_column_width=True, clamp=True)


if __name__ == "__main__":
    main()
