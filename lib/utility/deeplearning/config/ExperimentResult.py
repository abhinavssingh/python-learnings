from dataclasses import dataclass


@dataclass
class ExperimentResult:

    model_name: str

    train_score: float = None

    test_score: float = None

    accuracy: float = None

    precision: float = None

    recall: float = None

    f1_score: float = None

    training_time: float = None
