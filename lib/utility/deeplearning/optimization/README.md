# Optimization

Optimization configuration and factories.

## Files

optimizer_factory.py
loss_factory.py
scheduler_factory.py
regularizer_factory.py
initializer_factory.py
callbacks_factory.py

## Purpose

Create framework-specific optimization objects from generic configuration.

Example

optimizer = OptimizerFactory.create(
framework="tensorflow",
optimizer_name="adam"
)

Supported Frameworks

- TensorFlow
- PyTorch
