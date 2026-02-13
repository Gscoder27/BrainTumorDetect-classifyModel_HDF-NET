import os
import gdown
import streamlit as st

def download_models():
    """Download model files from Google Drive if they don't exist locally"""
    
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Model file IDs from Google Drive links
    model_files = {
        "EnsembleNiT_cnn.h5": "1k66GuldLIi89kduImiUE0hWPagw5xAgs",
        "EnsembleNiT_cae.h5": "1Ter2kLJs7W_GQQCoRg27FL0PkIj_NoAR",
        "EnsembleNiT_xgb.pkl": "1jB6kiIiNT7HBPXKJikuRiADltog1B_lf",
        "EnsembleNiT_rf.pkl": "1aIsFQkfQd0OZnPt-Ql8b8DBe7x8eiIkx",
        "EnsembleNiT_scaler.pkl": "1SsLUG0NG_6q_-J3QLy9JvGxusH9OWcK8"
    }
    
    for filename, file_id in model_files.items():
        filepath = os.path.join(models_dir, filename)
        
        if not os.path.exists(filepath):
            st.info(f"Downloading {filename}...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, filepath, quiet=False)
            st.success(f"✓ {filename} downloaded")
        else:
            st.success(f"✓ {filename} already exists")

if __name__ == "__main__":
    download_models()
