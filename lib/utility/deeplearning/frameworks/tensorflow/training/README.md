# TensorFlow Training

Training engine and history normalization utilities.

## Files

- tensorflow_trainer.py
- tensorflow_training_history.py

## Responsibilities

- Compile model from generic config
- Fit model with optional validation and callbacks
- Evaluate and predict with normalized outputs
- Return standardized training history wrapper

## Integration

TensorFlowModelUtility -> TensorFlowTrainer -> Optimization factories/callbacks
