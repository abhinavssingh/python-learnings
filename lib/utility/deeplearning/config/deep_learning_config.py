from dataclasses import dataclass


@dataclass
class DeepLearningConfig:

    # Training
    epochs: int = 30
    batch_size: int = 32

    # Optimizer
    optimizer: str = "adam"
    learning_rate: float = 0.001

    # Validation
    validation_split: float = 0.2

    # Regularization
    dropout_rate: float = 0.3

    # Early Stopping
    early_stopping: bool = True
    patience: int = 5

    # Learning Rate Scheduling
    reduce_lr: bool = True
    reduce_lr_factor: float = 0.5
    reduce_lr_patience: int = 3

    # Execution
    verbose: int = 1

    # Loss Function
    loss: str = "binary_crossentropy"
