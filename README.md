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

The trained model files are not included in this repository due to GitHub's file size limitations. Download the pre-trained models from the links below:

### Pre-trained Models

| Model | Description | Download Link |
|-------|-------------|---------------|
| **EnsembleNiT_cnn.h5** | CNN ensemble model | [Download](https://drive.google.com/file/d/1k66GuldLIi89kduImiUE0hWPagw5xAgs/view?usp=drive_link) |
| **EnsembleNiT_cae.h5** | CAE ensemble model | [Download](https://drive.google.com/file/d/1Ter2kLJs7W_GQQCoRg27FL0PkIj_NoAR/view?usp=drive_link) |
| **EnsembleNiT_xgb.pkl** | XGBoost/Gradient Boosting model | [Download](https://drive.google.com/file/d/1jB6kiIiNT7HBPXKJikuRiADltog1B_lf/view?usp=drive_link) |
| **EnsembleNiT_rf.pkl** | Random Forest model | [Download](https://drive.google.com/file/d/1aIsFQkfQd0OZnPt-Ql8b8DBe7x8eiIkx/view?usp=drive_link) |
| **EnsembleNiT_scaler.pkl** | Feature scaler | [Download](https://drive.google.com/file/d/1SsLUG0NG_6q_-J3QLy9JvGxusH9OWcK8/view?usp=drive_link) |

### How to Use Downloaded Models

1. Download the model files from the links above
2. Create a `models/` directory in the project root if it doesn't exist
3. Place the downloaded `.h5` and `.pkl` files in the `models/` directory
4. Run the inference scripts (`hdf_inference.py` or `app.py`)

### Training Your Own Models

Alternatively, you can train the models yourself using the provided Jupyter notebooks:
- `HDF_NET.ipynb` - Train the HDF-NET model
- `Ensemble(NiT).ipynb` - Train the ensemble models
