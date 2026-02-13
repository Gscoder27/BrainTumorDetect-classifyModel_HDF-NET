import os
import requests
import streamlit as st

def download_file_from_github_release(url, destination):
    """Download a file from GitHub Releases"""
    
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise error for bad status codes
    
    total_size = 0
    CHUNK_SIZE = 32768
    
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                total_size += len(chunk)
    
    return total_size

def download_models():
    """Download model files from GitHub Releases if they don't exist locally"""
    
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # GitHub Release base URL
    GITHUB_RELEASE_URL = "https://github.com/Gscoder27/BrainTumorDetect-classifyModel_HDF-NET/releases/download/Saved_Model"
    
    # Model files to download
    model_files = [
        "EnsembleNiT_cnn.h5",
        "EnsembleNiT_cae.h5",
        "EnsembleNiT_xgb.pkl",
        "EnsembleNiT_rf.pkl",
        "EnsembleNiT_scaler.pkl"
    ]
    
    for filename in model_files:
        filepath = os.path.join(models_dir, filename)
        
        if not os.path.exists(filepath):
            try:
                url = f"{GITHUB_RELEASE_URL}/{filename}"
                st.info(f"⏳ Downloading {filename} from GitHub Releases...")
                file_size = download_file_from_github_release(url, filepath)
                st.success(f"✓ {filename} downloaded successfully! ({file_size / 1024 / 1024:.1f} MB)")
            except Exception as e:
                st.error(f"❌ Failed to download {filename}: {str(e)}")
                st.warning(
                    f"**Troubleshooting:**\n"
                    f"1. Verify the file exists at: {url}\n"
                    f"2. Check that the GitHub Release v1.0.0 is published\n"
                    f"3. Ensure the release is public"
                )
                raise
        else:
            file_size = os.path.getsize(filepath)
            st.success(f"✓ {filename} already exists ({file_size / 1024 / 1024:.1f} MB)")

if __name__ == "__main__":
    download_models()
