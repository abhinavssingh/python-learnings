from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExperimentConfig:

    # Validation
    validation_strategy: str = "holdout"
    n_splits: int = 5
    shuffle: bool = True
    random_state: int = 42

    # Classification Specific
    imbalance_strategy: str | None = None
    imbalance_params: dict[str, Any] = field(default_factory=dict)

    # Optional Extensions
    tuning_enabled: bool = False
    feature_selection_enabled: bool = False
    explainability_enabled: bool = False
