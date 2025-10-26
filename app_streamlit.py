import streamlit as st
from ultralytics import YOLO
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="YOLO Detector", layout="wide")
st.title("YOLO Object Detection")
st.caption("Upload an image and the app will run detection automatically.")


@st.cache_resource
def load_model(weights_path: str):
    return YOLO(weights_path)


WEIGHTS = "train27/weights/best.pt"
model = load_model(WEIGHTS)


uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded:
    # Left: original, Right: result
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(uploaded, use_container_width=True)

    with col2:
        st.subheader("Detection")
        image = Image.open(uploaded).convert("RGB")
        with st.spinner("Running YOLO…"):
            results = model(image, verbose=False)
            plotted = results[0].plot()[:, :, ::-1]  # BGR -> RGB
        st.image(plotted, use_container_width=True)

        # Download annotated image
        buf = BytesIO()
        Image.fromarray(plotted).save(buf, format="PNG")
        st.download_button(
            label="Download annotated image",
            data=buf.getvalue(),
            file_name="prediction.png",
            mime="image/png"
        )
else:
    st.info("Upload a JPG or PNG to get started.")