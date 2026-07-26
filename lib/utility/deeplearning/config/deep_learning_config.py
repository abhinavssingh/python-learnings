from dataclasses import dataclass


@dataclass
class DeepLearningConfig:

    # ======================================================
    # Training
    # ======================================================

    epochs: int = 30
    batch_size: int = 32
    verbose: int = 1

    # ======================================================
    # Optimization
    # ======================================================

    optimizer: str = "adam"
    learning_rate: float = 0.001

    # ======================================================
    # Loss
    # ======================================================

    loss: str = "binary_crossentropy"

    # ======================================================
    # Validation
    # ======================================================

    validation_split: float = 0.2

    # ======================================================
    # Model Hyperparameters
    # ======================================================

    activation: str = "relu"

    initializer: str = "he_normal"

    regularizer: str | None = None

    regularizer_l1: float = 0.01
    regularizer_l2: float = 0.01

    dropout_rate: float = 0.0

    # ======================================================
    # Early Stopping
    # ======================================================

    early_stopping: bool = True
    patience: int = 5

    early_stopping_monitor: str = "val_loss"

    # ======================================================
    # Learning Rate Scheduling
    # ======================================================

    reduce_lr: bool = True

    reduce_lr_factor: float = 0.5

    reduce_lr_patience: int = 3

    reduce_lr_monitor: str = "val_loss"

    scheduler: str | None = None

    # ======================================================
    # Checkpointing
    # ======================================================

    checkpoint_filepath: str | None = None

    checkpoint_monitor: str = "val_loss"

    # ======================================================
    # TensorBoard
    # ======================================================

    tensorboard_log_dir: str | None = None

    # ======================================================
    # CSV Logging
    # ======================================================

    csv_log_file: str | None = None
