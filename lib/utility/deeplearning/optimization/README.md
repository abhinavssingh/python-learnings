# Optimization

Factory utilities for optimization-related components.

## Files

- optimizer_factory.py
- loss_factory.py
- scheduler_factory.py
- regularizer_factory.py
- initializer_factory.py
- callbacks_factory.py

## Purpose

Translate framework-neutral config options into framework-specific objects.

## Example

```python
optimizer = OptimizerFactory.create(
	framework="tensorflow",
	optimizer_name="adam",
	learning_rate=0.001,
)
```

## Current Coverage

- TensorFlow: actively implemented
- PyTorch: placeholders/partial support depending on component
