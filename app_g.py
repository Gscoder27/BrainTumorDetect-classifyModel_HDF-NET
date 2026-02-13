import streamlit as st
from models.hdf_inference2 import HDFPredictor

st.title("HDF-NET: Brain Tumor Detection")
st.write("Upload an MRI scan to detect tumor type.")

# Initialize the predictor once (cache this resource!)
@st.cache_resource
def get_predictor():
    return HDFPredictor()

predictor = get_predictor()

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded MRI", use_column_width=True)
    
    if st.button("Analyze"):
        with st.spinner("Processing with HDF-NET..."):
            try:
                # Get prediction
                label, confidence = predictor.predict(uploaded_file)
                
                # Display Result
                st.success(f"Prediction: **{label}**")
                st.info(f"Confidence: **{confidence * 100:.2f}%**")
                
                # Logic check for Glioma specifically
                if label == "No Tumor" and confidence < 0.60:
                     st.warning("⚠️ The model predicts 'No Tumor' but confidence is low. This might be a subtle Glioma. Please consult a radiologist.")
                     
            except Exception as e:
                st.error(f"Error during prediction: {e}")