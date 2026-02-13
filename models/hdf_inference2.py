import numpy as np
import cv2
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.losses import mse
from PIL import Image

# ---------------------------------------------------------
# GLOBAL CONFIG
# ---------------------------------------------------------
# Define your class labels exactly as they appeared in Colab training
# (Verify this order: usually alphabetical!)

CLASSES = {
    0: 'Glioma',
    1: 'Meningioma',
    2: 'No Tumor',
    3: 'Pituitary'
}

class HDFPredictor:
    def __init__(self):
        # 1. Load Models
        cnn_full = load_model('models/EnsembleNiT_cnn.h5') 
        self.autoencoder = load_model('models/EnsembleNiT_cae.h5', custom_objects={'mse': mse})  
        self.xgb_model = joblib.load('models/EnsembleNiT_xgb.pkl') 
        self.rf_model = joblib.load('models/EnsembleNiT_rf.pkl')
        
        # 2. Extract feature extractor (bottleneck layer = 256-D)
        self.cnn_full = cnn_full
        self.feature_model = tf.keras.Model(
            inputs=cnn_full.input,
            outputs=cnn_full.get_layer("bottleneck").output
        )
        
        # 3. Extract encoder for latent representation (32-D)
        self.encoder = tf.keras.Model(
            inputs=self.autoencoder.input,
            outputs=self.autoencoder.get_layer("latent").output
        )  
        
        # 4. LOAD THE SAVED SCALER (Critical Fix!)
        try:
            self.scaler = joblib.load('models/EnsembleNiT_scaler.pkl')
            print("SUCCESS: Loaded scaler.pkl")
        except:
            print("ERROR: scaler.pkl not found! Predictions will be wrong.")
            # Fallback (dangerous, but prevents crash)
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler() 

    def preprocess(self, image_file):
        """
        Prepares the image exactly like VGG16 expects (BGR, Zero-centered).
        """
        # Load as RGB (standard for Streamlit/PIL)
        img = Image.open(image_file).convert('RGB')
        img = np.array(img)
        
        # Resize to VGG standard (224, 224)
        img = cv2.resize(img, (224, 224))
        
        # CONVERT RGB -> BGR (Critical for VGG weights!)
        img = img[:, :, ::-1] 
        
        # Preprocess input (Zero-centering based on ImageNet mean)
        # This replaces manual division by 255 if you used VGG preprocess_input in Colab
        img = preprocess_input(img)
        
        # Add batch dimension: (1, 224, 224, 3)
        return np.expand_dims(img, axis=0)

    def predict(self, image_file):
        # 1. Preprocess Image
        processed_img = self.preprocess(image_file)
        
        # 2. Get CNN probabilities (from full network)
        cnn_probs = self.cnn_full.predict(processed_img, verbose=0)
        
        # 3. Extract 256-D bottleneck features
        deep_features = self.feature_model.predict(processed_img, verbose=0)
        
        # 4. NORMALIZE FEATURES using scaler
        scaled_features = self.scaler.transform(deep_features)
        
        # 5. Get latent representation (32-D) from encoder
        latent_features = self.encoder.predict(scaled_features, verbose=0) 
        
        # 6. Ensemble Prediction
        xgb_probs = self.xgb_model.predict_proba(latent_features)
        rf_probs = self.rf_model.predict_proba(latent_features)
        
        # 7. Weighted Fusion (0.4 CNN + 0.4 XGB + 0.2 RF) - EXACT COLAB WEIGHTS
        final_prob = (0.4 * cnn_probs) + (0.4 * xgb_probs) + (0.2 * rf_probs)
        prediction_idx = np.argmax(final_prob)
        confidence = np.max(final_prob)
        
        return CLASSES[prediction_idx], confidence