# Face Mask Detection Using Transfer Learning

## Overview

This project implements a Face Mask Detection solution using Transfer Learning with TensorFlow. The application benchmarks multiple state-of-the-art pre-trained CNN architectures and identifies the best-performing model for detecting masked and non-masked faces.

## Features

- Transfer Learning based image classification
- EfficientNetB0 benchmark
- ResNet50 benchmark
- Automated model comparison
- Data augmentation
- Early stopping and learning rate scheduling
- Classification performance evaluation
- Automated HTML report generation

## Dataset

Dataset location:

```text
datasets/Face_mask_detection
```

Supported structures:

```text
Face_mask_detection/
├── train/
├── test/
```

or

```text
Face_mask_detection/
└── data/
```

## Image Processing

- Image Size: 128 x 128
- RGB Images
- Data Augmentation Enabled
- Validation Split: 20%
- Test Split Support

## Models Evaluated

### EfficientNetB0

- Transfer Learning
- Frozen Backbone
- Dropout: 0.2
- Softmax Output

### ResNet50

- Transfer Learning
- Frozen Backbone
- Dropout: 0.5
- Softmax Output

## Training Configuration

- Epochs: 25
- Batch Size: 32
- Optimizer: Adam
- Learning Rate: 0.001
- Loss: Categorical Crossentropy
- Early Stopping Enabled
- Learning Rate Reduction Enabled

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Classification Metrics
- TensorFlow Evaluation Metrics

## Benchmark Workflow

```text
Dataset
   ↓
Data Augmentation
   ↓
EfficientNetB0 Training
   ↓
ResNet50 Training
   ↓
Performance Evaluation
   ↓
Model Ranking
   ↓
Best Model Selection
```

## Generated Outputs

- Dataset Summary
- Model Ranking Table
- Best Model Selection
- Training History
- Model Summaries
- Performance Metrics

## Generated Report

```text
reports/face_mask_transfer_learning_report.html
```

## Execution

```bash
python face_mask_transfer_learning.py
```

## Project Structure

```text
project/
├── face_mask_transfer_learning.py
├── datasets/
│   └── Face_mask_detection/
├── reports/
│   └── face_mask_transfer_learning_report.html
└── README.md
```

## Use Cases

- Public safety monitoring
- Workplace compliance
- Healthcare environments
- Real-time vision systems
- Transfer learning research
