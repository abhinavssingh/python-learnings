# TensorFlow Framework

TensorFlow implementation layer for the deeplearning utilities.

## Components

- data/: TensorFlow-ready loaders and dataset builders
- models/: Wrapper-based model definitions (dense/cnn/sequence)
- training/: Trainer and normalized training history
- pipelines/: Reusable comparative experiment pipelines
- ensemble/: Hybrid sklearn + TensorFlow voting classifiers
- tensorflow_model_utility.py: High-level compile/train/evaluate/predict facade

## Internal Flow

1. Model wrapper builds a tf.keras.Model
2. TensorFlowModelUtility delegates compilation/training to TensorFlowTrainer
3. Optimization factories resolve optimizer/loss/callbacks
4. Metrics/history are returned in standardized formats
