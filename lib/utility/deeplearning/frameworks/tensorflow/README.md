# TensorFlow Implementation

TensorFlow specific implementation of the deep learning framework.

## Components

data/
Dataset construction and loading

models/
TensorFlow model wrappers

training/
Model training

TensorFlowModelUtility.py
High-level TensorFlow orchestration class

## Architecture

BaseModelWrapper
↓
TensorFlowModelWrapper
↓
Concrete Model Wrapper

TensorFlowModelUtility
↓
TensorFlowTrainer
