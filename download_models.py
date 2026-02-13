import os
import requests
import streamlit as st

def download_file_from_google_drive(file_id, destination):
    """Download a file from Google Drive using direct download URL"""
    
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

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
            try:
                st.info(f"Downloading {filename}... This may take a few minutes.")
                download_file_from_google_drive(file_id, filepath)
                st.success(f"✓ {filename} downloaded successfully!")
            except Exception as e:
                st.error(f"❌ Failed to download {filename}: {str(e)}")
                st.warning("Please ensure the Google Drive files are shared with 'Anyone with the link' permission.")
                raise
        else:
            st.success(f"✓ {filename} already exists")

if __name__ == "__main__":
    download_models()
