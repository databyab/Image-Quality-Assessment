from pathlib import Path
import sys
import tempfile
from io import BytesIO


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from PIL import Image, UnidentifiedImageError

from app.inference import PhotoQualityPredictor

MODEL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "baseline"
    / "checkpoints"
    / "best_model.pt"
)


LABELS = {
    "blur": "Blur",
    "noise": "Noise",
    "underexposure": "Underexposure",
    "overexposure": "Overexposure",
    "compression": "Compression",
}

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000


@st.cache_resource
def load_model():
    return PhotoQualityPredictor(str(MODEL_PATH))


st.set_page_config(
    page_title="PhotoIQ",
    page_icon="📷",
    layout="wide",
)


st.title("📷 PhotoIQ")
st.subheader("Photo quality analysis")

st.write(
    "Upload a photograph and PhotoIQ will analyze it "
    "for common technical quality defects."
)


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
)


if uploaded_file is not None:
    raw_data = uploaded_file.getvalue()

    if len(raw_data) > MAX_UPLOAD_BYTES:
        st.error("Image is too large. The maximum upload size is 10 MB.")
        st.stop()

    try:
        with Image.open(BytesIO(raw_data)) as decoded_image:
            width, height = decoded_image.size

            if width * height > MAX_IMAGE_PIXELS:
                st.error("Image dimensions are too large.")
                st.stop()

            decoded_image.load()
            image = decoded_image.convert("RGB")
    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        OSError,
    ):
        st.error("The uploaded file is not a valid image.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption=uploaded_file.name,
            use_container_width=True,
        )

    with col2:
        st.write("### Analysis")

        predictor = load_model()

        suffix = ".png" if uploaded_file.type == "image/png" else ".jpg"
        image_format = "PNG" if suffix == ".png" else "JPEG"
        temporary_file = tempfile.NamedTemporaryFile(
            prefix="photoiq_",
            suffix=suffix,
            delete=False,
        )
        temp_path = Path(temporary_file.name)

        try:
            temporary_file.close()
            image.save(temp_path, format=image_format)
            results = predictor.predict(str(temp_path))
        finally:
            temp_path.unlink(missing_ok=True)

        for defect, result in results.items():
            probability = result["probability"]

            st.write(LABELS[defect])

            st.progress(
                min(max(probability, 0.0), 1.0),
                text=f"{probability:.1%}",
            )

            if result["detected"]:
                st.warning("Detected")
            else:
                st.success("Not detected")
