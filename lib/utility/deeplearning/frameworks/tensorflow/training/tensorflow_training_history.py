from dataclasses import dataclass, field

import pandas as pd


@dataclass
class TensorFlowTrainingHistory:

    history: dict = field(default_factory=dict)

    @property
    def loss(self):
        return self.history.get("loss", [])

    @property
    def val_loss(self):
        return self.history.get("val_loss", [])

    @property
    def accuracy(self):
        return self.history.get("accuracy", [])

    @property
    def val_accuracy(self):
        return self.history.get("val_accuracy", [])

    @property
    def epochs(self):
        return len(self.loss)

    @classmethod
    def from_keras_history(cls, history):
        return cls(history=history.history)

    def to_dict(self) -> dict:
        return self.history

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.history)

        if not df.empty:
            df.insert(
                0,
                "epoch",
                range(1, len(df) + 1),
            )

        return df

    def to_json(self):
        return self.to_dataframe().to_json(
            orient="records"
        )

    def __len__(self):
        return len(self.history)

    def __getitem__(self, key):
        return self.history[key]

    def items(self):
        return self.history.items()

    def keys(self):
        return self.history.keys()

    def values(self):
        return self.history.values()
