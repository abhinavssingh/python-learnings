# Configuration

Centralized training configuration for deep learning experiments.

## Files

- deep_learning_config.py

## Purpose

DeepLearningConfig keeps training settings in one place so model wrappers, trainers, and pipelines can reuse the same configuration.

## Common Fields

- Training: epochs, batch_size, verbose
- Optimization: optimizer, learning_rate
- Loss: loss
- Regularization/initialization: regularizer, dropout_rate, initializer
- Callbacks: early_stopping, patience, reduce_lr, checkpoint paths, tensorboard/csv logging

## Design Note

The config object is framework-neutral and is interpreted by framework-specific trainers/factories.
