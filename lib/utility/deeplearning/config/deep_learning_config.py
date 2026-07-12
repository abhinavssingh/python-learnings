from dataclasses import dataclass


@dataclass
class DeepLearningConfig:
    """
    Configuration for Deep Learning experiments.
    """

    # Training
    epochs: int = 100
    batch_size: int = 32

    # Optimizer
    optimizer: str = "adam"
    learning_rate: float = 0.001

    # Validation
    validation_split: float = 0.2

    # Early Stopping
    early_stopping: bool = True
    patience: int = 10

    # Network
    dropout_rate: float = 0.3

    # Verbose
    verbose: int = 1
