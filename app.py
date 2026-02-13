import streamlit as st
import pandas as pd
from PIL import Image
import os

# Download models on first run
from download_models import download_models

# Ensure models are downloaded
if not os.path.exists("models/EnsembleNiT_cnn.h5"):
    with st.spinner("Downloading models for first-time setup... This may take a few minutes."):
        download_models()

from hdf_inference import load_hdf_models, predict_hdf

# PAGE CONFIG

st.set_page_config(
    page_title="HDF-Net Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumor Classification")
st.caption("Hybrid Deep Feature-Compressed Ensemble Network (HDF-Net)")
st.markdown("---")

# LOAD MODELS (CACHE SAFELY)

@st.cache_resource(show_spinner=False)
def get_models():
    return load_hdf_models()

models = get_models()

# FILE UPLOADER

uploaded_file = st.file_uploader(
    "Upload MRI Scan (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

# INFERENCE

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded MRI", width=300)

    with st.spinner("Running HDF-Net inference..."):
        label, confidence, probs = predict_hdf(image, models)

    with col2:
        st.subheader("Prediction")
        st.success(label)
        st.metric("Confidence", f"{confidence:.2f}%")

    st.markdown("---")

    # PROBABILITY DISTRIBUTION

    st.subheader("Class Probabilities")

    prob_df = pd.DataFrame(
        {
            "Class": list(probs.keys()),
            "Probability (%)": list(probs.values())
        }
    )

    prob_df = prob_df.set_index("Class")

    st.bar_chart(prob_df)

    st.info(
        "Prediction generated using HDF-Net "
        "(VGG16 + CAE + XGBoost + Random Forest with weighted fusion)"
    )
