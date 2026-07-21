from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExperimentResult:

    model_name: str

    model_type: str = ""

    training_time: float = 0.0

    metrics: dict = field(default_factory=dict)

    parameters: dict = field(default_factory=dict)

    history: Any = None

    artifacts: dict = field(default_factory=dict)
