# TensorFlow Models

Model wrappers that expose a consistent interface for TensorFlow experiments.

## Base Wrappers

- tensorflow_model_wrapper.py
- functional_wrapper.py
- sequential_wrapper.py

## Model Families

- dense/: MLP variants for tabular inputs
- cnn/: Convolutional and transfer-learning wrappers
- sequence/: LSTM/Transformer/CNN-LSTM wrappers

## Wrapper Pattern

BaseModelWrapper -> TensorFlowModelWrapper -> Functional/Sequential wrapper -> Concrete architecture
