# TensorFlow Models

TensorFlow model wrappers.

## Base Classes

tensorflow_model_wrapper.py

functional_wrapper.py

sequential_wrapper.py

## Model Categories

dense/
cnn/
sequence/

## Design

BaseModelWrapper
↓
TensorFlowModelWrapper
↓
FunctionalWrapper / SequentialWrapper
↓
Concrete Model
