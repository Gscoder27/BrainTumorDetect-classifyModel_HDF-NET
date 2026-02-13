import numpy as np
import tensorflow as tf
import joblib
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input

# Version checks for debugging
import sklearn
import xgboost

# print("TensorFlow version:", tf.__version_

IMG_SIZE = 224

# MUST match dataset folder order used during training
# Update this order to match test_gen.class_indices from Colab
CLASS_NAMES = ["Meningioma", "glioma", "No Tumor", "Pituitary"]
# CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# For display purposes, we'll create nice names
DISPLAY_NAMES = {
    "glioma": "Glioma",
    "Meningioma": "Meningioma",
    "No Tumor": "No Tumor",
    "Pituitary": "Pituitary"
}

# LOAD MODELS (IDENTICAL TO COLAB)

def load_hdf_models():
    cnn = tf.keras.models.load_model(
        "models/EnsembleNiT_cnn.h5", compile=False
    )

    cae = tf.keras.models.load_model(
        "models/EnsembleNiT_cae.h5", compile=False
    )

    scaler = joblib.load("models/EnsembleNiT_scaler.pkl")
    xgb = joblib.load("models/EnsembleNiT_xgb.pkl")
    rf = joblib.load("models/EnsembleNiT_rf.pkl")

    # EXACT layers used during training
    feature_model = tf.keras.Model(
        inputs=cnn.input,
        outputs=cnn.get_layer("bottleneck").output  # 256-D
    )

    encoder = tf.keras.Model(
        inputs=cae.input,
        outputs=cae.get_layer("latent").output      # 32-D
    )

    return cnn, feature_model, encoder, scaler, xgb, rf

# IMAGE PREPROCESSING (VGG16 STANDARD)

def preprocess_image(img: Image.Image):
    from tensorflow.keras.preprocessing.image import img_to_array
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr

# FINAL HDF-NET INFERENCE (MATCHES COLAB)

def predict_hdf(image: Image.Image, models):
    cnn, feature_model, encoder, scaler, xgb, rf = models

    img_tensor = preprocess_image(image)

    # 1. CNN probabilities
    cnn_probs = cnn.predict(img_tensor, verbose=0)

    # 2. Extract 256-D CNN features
    cnn_features = feature_model.predict(img_tensor, verbose=0)

    # 3. Scale features (same scaler used during training)
    cnn_features_scaled = scaler.transform(cnn_features)

    # 4. CAE compression → 32-D latent vector
    latent = encoder.predict(cnn_features_scaled, verbose=0)

    # 5. ML probabilities
    xgb_probs = xgb.predict_proba(latent)
    rf_probs = rf.predict_proba(latent)

    # 6. Weighted fusion (EXACT COLAB LOGIC)
    final_probs = (
        0.4 * cnn_probs + 0.4 * xgb_probs + 0.2 * rf_probs
    )

    pred_idx = int(np.argmax(final_probs))
    confidence = float(final_probs[0][pred_idx] * 100)

    # Get the class name
    predicted_class = CLASS_NAMES[pred_idx]
    display_name = DISPLAY_NAMES[predicted_class]

    probs = {
        CLASS_NAMES[i]: float(final_probs[0][i] * 100)
        for i in range(len(CLASS_NAMES))
    }

    # Debug logging for troubleshooting
    # print("cnn_probs:", cnn_probs)
    # print("xgb_probs:", xgb_probs)
    # print("rf_probs:", rf_probs)
    # print("final_probs:", final_probs)
    # print("pred_idx:", pred_idx)
    # print("CLASS_NAMES:", CLASS_NAMES)

    return CLASS_NAMES[pred_idx], confidence, probs
