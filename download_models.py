import os
import requests
import streamlit as st
from urllib.parse import parse_qs, urlparse

def download_file_from_google_drive(file_id, destination):
    """Download a file from Google Drive, handling large file confirmations"""
    
    def get_confirm_token(response):
        """Extract confirmation token from response"""
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None
    
    def save_response_content(response, destination):
        """Save response content to file in chunks"""
        CHUNK_SIZE = 32768
        total_size = 0
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    total_size += len(chunk)
        return total_size

    # Try direct download first
    URL = "https://drive.google.com/uc?export=download"
    
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    
    # Check if we got a confirmation page (for large files)
    token = get_confirm_token(response)
    
    if token:
        # Large file - need to confirm download
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    else:
        # Check if response is HTML (virus scan warning page)
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' in content_type:
            # Parse the HTML to find the actual download link
            html_content = response.text
            
            # Look for the download URL in the HTML
            if 'download' in html_content:
                # Try to find UUID in the response
                import re
                uuid_match = re.search(r'download.*?uuid=([^&"]+)', html_content)
                if uuid_match:
                    uuid = uuid_match.group(1)
                    # Construct new download URL with UUID
                    params = {'id': file_id, 'export': 'download', 'confirm': 't', 'uuid': uuid}
                    response = session.get(URL, params=params, stream=True)
    
    # Verify we're getting binary content, not HTML
    content_type = response.headers.get('Content-Type', '')
    if 'text/html' in content_type:
        raise Exception(
            f"Failed to download file. Google Drive returned HTML instead of the file. "
            f"Please ensure the file is shared with 'Anyone with the link' permission."
        )
    
    # Save the file
    file_size = save_response_content(response, destination)
    
    # Verify file was downloaded (should be > 1MB for model files)
    if file_size < 1000000:
        raise Exception(f"Downloaded file is too small ({file_size} bytes). Download may have failed.")
    
    return file_size

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
                st.info(f"⏳ Downloading {filename}... This may take a few minutes.")
                file_size = download_file_from_google_drive(file_id, filepath)
                st.success(f"✓ {filename} downloaded successfully! ({file_size / 1024 / 1024:.1f} MB)")
            except Exception as e:
                st.error(f"❌ Failed to download {filename}: {str(e)}")
                st.warning(
                    "**Troubleshooting:**\n"
                    "1. Ensure Google Drive files are shared with 'Anyone with the link'\n"
                    "2. Try downloading manually and uploading to GitHub Releases\n"
                    "3. Check the deployment logs for more details"
                )
                raise
        else:
            file_size = os.path.getsize(filepath)
            st.success(f"✓ {filename} already exists ({file_size / 1024 / 1024:.1f} MB)")

if __name__ == "__main__":
    download_models()
