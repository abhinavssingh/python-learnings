# TensorFlow Data Layer

Data preparation helpers for multiple TensorFlow training scenarios.

## Files

- tf_data_loader.py: Generic TensorFlow dataset loader wrapper
- tf_dataset_builder.py: tf.data.Dataset creation and batching utilities
- tf_image_classification_data_loader.py: Image folder classification loaders
- tf_text_classification_data_loader.py: Text tokenization, sequence, and padding bundle builder
- tf_autoencoder_data_loader.py: NPZ autoencoder data loading + noise injection

## Supported Workflows

- Tabular to tf.data pipelines
- Image classification train/val/test generators
- Text classification sequence pipelines
- Denoising autoencoder noisy/clean pair preparation

## Output Types

- tf.data.Dataset
- Structured dataclass bundles (for text/autoencoder loaders)
