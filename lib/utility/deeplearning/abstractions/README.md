# Abstractions

Framework-independent contracts used across deep learning implementations.

## Files

- base_model_wrapper.py: Standard model wrapper interface
- base_trainer.py: Training/evaluation/prediction interface
- base_data_loader.py: Dataset loader/preparation interface
- base_evaluator.py: Metric evaluation interface
- base_persistence.py: Model artifact persistence interface

## Why It Exists

- Keeps framework code modular and replaceable
- Enables common orchestration across model types
- Makes adding new frameworks or trainers predictable

## Extension Pattern

Concrete framework classes should inherit these base contracts and provide framework-specific behavior while preserving method signatures.
