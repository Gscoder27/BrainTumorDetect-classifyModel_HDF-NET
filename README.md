# BrainTumorDetectionModel_HDF-NET

A proposed machine learning model to detect Brain Tumor and classify into 4 different categories: **Glioma**, **Meningioma**, **Pituitary**, and **No Tumor**.

## Architecture

![HDF-NET Architecture](HDF-NET_Arch.png)

## Overview

This project implements the HDF-NET (Hierarchical Deep Fusion Network) model for brain tumor detection and classification using MRI scans.

## Features

- Brain tumor detection
- Multi-class classification (Glioma, Meningioma, Pituitary, No Tumor)
- Deep learning-based approach
- Streamlit web interface for easy inference

## Model Downloads

The trained model files are hosted on **GitHub Releases** for easy access and reliable downloads.

### Pre-trained Models

| Model | Description | Download Link |
|-------|-------------|---------------|
| **EnsembleNiT_cnn.h5** | CNN ensemble model | [Download](https://github.com/Gscoder27/BrainTumorDetect-classifyModel_HDF-NET/releases/download/v1.0.0/EnsembleNiT_cnn.h5) |
| **EnsembleNiT_cae.h5** | CAE ensemble model | [Download](https://github.com/Gscoder27/BrainTumorDetect-classifyModel_HDF-NET/releases/download/v1.0.0/EnsembleNiT_cae.h5) |
| **EnsembleNiT_xgb.pkl** | XGBoost/Gradient Boosting model | [Download](https://github.com/Gscoder27/BrainTumorDetect-classifyModel_HDF-NET/releases/download/v1.0.0/EnsembleNiT_xgb.pkl) |
| **EnsembleNiT_rf.pkl** | Random Forest model | [Download](https://github.com/Gscoder27/BrainTumorDetect-classifyModel_HDF-NET/releases/download/v1.0.0/EnsembleNiT_rf.pkl) |
| **EnsembleNiT_scaler.pkl** | Feature scaler | [Download](https://github.com/Gscoder27/BrainTumorDetect-classifyModel_HDF-NET/releases/download/v1.0.0/EnsembleNiT_scaler.pkl) |

**Note:** When deploying to Streamlit Cloud, models are automatically downloaded from GitHub Releases on first run.

### How to Use Downloaded Models

1. Download the model files from the links above
2. Create a `models/` directory in the project root if it doesn't exist
3. Place the downloaded `.h5` and `.pkl` files in the `models/` directory
4. Run the inference scripts (`hdf_inference.py` or `app.py`)


### Training Your Own Models

Alternatively, you can train the models yourself using the provided Jupyter notebooks:
- `HDF_NET.ipynb` - Train the HDF-NET model
- `Ensemble(NiT).ipynb` - Train the ensemble models

## 🚀 Deployment

### Deploy to Streamlit Cloud

This app is ready to deploy on Streamlit Cloud with automatic model downloading:

1. **Fork or use this repository**
   - Repository: `https://github.com/Gscoder27/BrainTumorDetect-classifyModel_HDF-NET`

2. **Go to Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

3. **Deploy the app**
   - Click "New app"
   - Repository: `Gscoder27/BrainTumorDetect-classifyModel_HDF-NET`
   - Branch: `main`
   - Main file path: `app.py`
   - Click "Deploy"

4. **Wait for deployment**
   - The app will automatically download the model files from Google Drive on first run
   - This may take 3-5 minutes for the initial setup

### Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The models will be automatically downloaded on first run.
