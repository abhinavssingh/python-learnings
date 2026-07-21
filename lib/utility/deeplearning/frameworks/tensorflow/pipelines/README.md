# Deep Learning Pipelines

This folder contains reusable TensorFlow pipeline implementations for benchmarking, experimentation, hyperparameter tuning, and architecture comparison.

The pipelines are built on top of the framework components:

- TensorFlowModelUtility
- TensorFlowTrainer
- Model Wrappers
- Optimization Factories
- Evaluation Framework
- ExperimentResult

These pipelines provide a consistent way to evaluate model architectures and training configurations under identical conditions.

---

# Pipeline Architecture

```text
Dataset
    │
    ▼
Model Wrapper
    │
    ▼
TensorFlowModelUtility
    │
    ▼
TensorFlowTrainer
    │
    ▼
Evaluation
    │
    ▼
ExperimentResult
    │
    ▼
DataFrame / Visualization
```

---

# Available Pipelines

## tf_model_pipeline.py

Baseline training pipeline.

### Purpose

Train and evaluate a single model.

### Use Cases

- Model validation
- Baseline benchmarking
- Production training

### Example

```python
MLPWrapper
    ↓
TensorFlowModelUtility
    ↓
Training
    ↓
Evaluation
```

---

## tf_architecture_pipeline.py

Compares multiple model architectures.

### Purpose

Identify the best-performing architecture.

### Supported Models

- MLP
- MixedMLP
- ParallelMLP
- CNN
- ResNet
- LSTM
- Transformer

### Example Output

```text
Architecture      Accuracy

MLP               0.94
CNN               0.96
ResNet            0.97
Transformer       0.95
```

---

## tf_optimizer_pipeline.py

Compares optimizer performance.

### Optimizers

- Adam
- AdamW
- SGD
- RMSProp
- Adagrad

### Example Output

```text
Optimizer      Accuracy

Adam           0.95
AdamW          0.96
SGD            0.92
```

---

## tf_model_optimizer_pipeline.py

Cross-comparison of architectures and optimizers.

### Purpose

Find the best architecture + optimizer combination.

### Example

```text
Model            Optimizer

MLP              Adam
MLP              SGD
CNN              Adam
CNN              SGD
```

### Example Output

```text
Architecture      Optimizer      Accuracy

CNN               Adam           0.97
ResNet            AdamW          0.98
MLP               Adam           0.95
```

---

## tf_activation_pipeline.py

Compares activation functions.

### Activations

- ReLU
- Tanh
- Sigmoid
- ELU
- SELU
- GELU

### Example Output

```text
Activation      Accuracy

ReLU            0.95
GELU            0.96
Tanh            0.92
```

---

## tf_initializer_pipeline.py

Compares weight initialization strategies.

### Initializers

- He Normal
- He Uniform
- Glorot Normal
- Glorot Uniform
- Lecun Normal
- Lecun Uniform

### Example Output

```text
Initializer          Accuracy

He Normal            0.96
Glorot Uniform       0.94
```

---

## tf_learning_rate_pipeline.py

Compares learning rates.

### Examples

```text
0.1
0.01
0.001
0.0001
0.00001
```

### Example Output

```text
Learning Rate      Accuracy

0.001              0.96
0.0001             0.95
0.01               0.94
```

---

## tf_batch_size_pipeline.py

Compares batch sizes.

### Examples

```text
16
32
64
128
256
```

### Example Output

```text
Batch Size         Accuracy

32                 0.96
64                 0.95
128                0.94
```

---

## tf_regularization_pipeline.py

Compares regularization strategies.

### Supported

- None
- L1
- L2
- L1_L2

### Example Output

```text
Regularization     Accuracy

L2                 0.96
L1_L2              0.95
None               0.93
```

---

## tf_dropout_pipeline.py

Compares dropout rates.

### Examples

```text
0.0
0.1
0.2
0.3
0.5
```

### Example Output

```text
Dropout Rate       Accuracy

0.3                0.96
0.2                0.95
0.1                0.94
```

---

# Common Output

All pipelines return a Pandas DataFrame.

Example:

```python
results_df = pipeline.run(
    X_train,
    y_train,
    X_test,
    y_test
)
```

Output:

```text
-------------------------------------------------------
Model      Accuracy    Loss    Training_Time
-------------------------------------------------------
CNN         0.97       0.11       18.4
ResNet      0.98       0.09       35.2
MLP         0.95       0.14       10.7
-------------------------------------------------------
```

---

# Future Pipelines

Planned additions:

- tf_scheduler_pipeline.py
- tf_callbacks_pipeline.py
- tf_loss_pipeline.py
- tf_kfold_pipeline.py
- tf_embedding_pipeline.py
- tf_transfer_learning_pipeline.py
- tf_ensemble_pipeline.py

---

# Design Philosophy

Each pipeline should:

- Change only one variable at a time
- Keep training conditions consistent
- Produce reproducible results
- Return standardized ExperimentResult objects
- Support future TensorFlow and PyTorch benchmarking

This ensures fair and meaningful comparisons across models, optimizers, hyperparameters, and training strategies.
