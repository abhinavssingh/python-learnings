# Enhancing Dental X-rays Using Autoencoders

## Overview

This project implements a Deep Learning based Denoising Autoencoder using TensorFlow to enhance dental panoramic X-ray images.

The model learns to reconstruct clean dental X-rays from noisy versions, improving image quality and supporting downstream diagnostic workflows.

## Key Features

- Dental X-ray denoising
- Convolutional Autoencoder architecture
- Synthetic noise generation
- Image reconstruction
- Model evaluation using MAE and MSE
- Training visualization
- Reconstruction comparison grids
- Automated HTML report generation

## Dataset

Input dataset:

```text
datasets/Dental-Panaromic-Autoencoder.npz
```

The dataset contains:

- Clean dental X-ray images
- Training set
- Test set
- Grayscale image processing

## Processing Pipeline

```text
Clean Dental X-rays
          ↓
Noise Injection
          ↓
Noisy Images
          ↓
Denoising Autoencoder
          ↓
Reconstructed Images
          ↓
Evaluation & Reporting
```

## Model Architecture

The solution uses a CNN-based Denoising Autoencoder:

- Encoder Network
- Latent Representation
- Decoder Network
- Image Reconstruction Output

## Training Configuration

- Epochs: 50
- Batch Size: 16
- Optimizer: Adam
- Learning Rate: 0.001
- Loss Function: Mean Squared Error (MSE)
- Metric: Mean Absolute Error (MAE)
- Early Stopping Enabled
- Learning Rate Reduction Enabled

## Noise Configuration

```text
Noise Factor: 0.2
Random Seed: 42
```

## Evaluation Metrics

- Test Loss (MSE)
- Mean Absolute Error (MAE)
- Reconstruction Quality Assessment

## Generated Visualizations

- Original Dental X-ray Images
- Noisy X-ray Images
- Reconstructed Images
- Training Loss Curve
- Validation Loss Curve
- MAE Curve

## Generated Report

```text
reports/dental_autoencoder_report.html
```

## Run

```bash
python dental_autoencoder.py
```

## Project Structure

```text
project/
├── dental_autoencoder.py
├── datasets/
│   └── Dental-Panaromic-Autoencoder.npz
├── reports/
│   └── dental_autoencoder_report.html
└── README.md
```

## Use Cases

- Medical image enhancement
- Dental imaging research
- Noise reduction in X-rays
- Computer-aided diagnosis preprocessing
- Healthcare AI experimentation
