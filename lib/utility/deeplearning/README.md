# Deep Learning Framework

Reusable deep learning utilities for training, evaluation, visualization, and experimentation.

## Highlights

- Framework-oriented design with common abstractions
- TensorFlow implementation with modular wrappers and trainers
- Reusable data loaders (tabular, image, text, and autoencoder flows)
- Optimization factories (loss, optimizer, regularizer, scheduler, callbacks)
- Visualization helpers (training curves, ROC, confusion matrix, reconstructions)
- Hybrid sklearn + TensorFlow ensemble utilities

## Folder Guide

- abstractions/: Base interfaces for model wrappers, trainers, evaluators, and data loaders
- config/: Shared training configuration objects
- evaluation/: Task-specific evaluators
- frameworks/: Framework implementations (TensorFlow currently)
- optimization/: Factory-based optimization components
- preprocessing/: Framework-independent preprocessing utilities
- visualization/: Plotly-based reporting helpers

## Typical Flow

1. Load and preprocess data
2. Build model wrapper
3. Configure training via DeepLearningConfig
4. Train/evaluate through framework utility
5. Render metrics and visual artifacts to reports
