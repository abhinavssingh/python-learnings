# TensorFlow Pipelines

Reusable benchmarking pipelines built on top of TensorFlowModelUtility.

## Available Pipelines

- tf_model_pipeline.py: Single-model train/evaluate flow
- tf_architecture_pipeline.py: Compare model wrappers
- tf_optimizer_pipeline.py: Compare optimizers
- tf_activation_pipeline.py: Compare activation functions
- tf_initializer_pipeline.py: Compare initialization strategies
- tf_learning_rate_pipeline.py: Compare learning rates
- tf_batch_size_pipeline.py: Compare batch sizes
- tf_dropout_pipeline.py: Compare dropout rates
- tf_regularization_pipeline.py: Compare regularization strategies

## Pipeline Pattern

Dataset -> Model Wrapper -> TensorFlowModelUtility -> Metrics/History -> DataFrame output

## Output Contract

Most pipelines return a Pandas DataFrame containing:

- model/config descriptor columns
- training time
- evaluation metrics

## Design Rules

- Change one variable per comparison
- Keep training conditions consistent
- Return standardized, report-friendly output
