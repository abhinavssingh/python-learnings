# Abstractions

Framework-independent interfaces.

These contracts are implemented by TensorFlow and PyTorch modules.

## Files

base_model_wrapper.py
Model abstraction.

base_trainer.py
Training abstraction.

base_data_loader.py
Data loading abstraction.

base_evaluator.py
Evaluation abstraction.

## Purpose

Enable:

BaseModelWrapper
├── TensorFlowModelWrapper
└── PyTorchModelWrapper
