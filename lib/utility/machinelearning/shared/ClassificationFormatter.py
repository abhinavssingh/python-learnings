import numpy as np
import pandas as pd


class ClassificationFormatter:

    @staticmethod
    def confusion_matrix(cm):

        if cm is None:
            return None

        # ✅ multilabel
        if isinstance(cm, (list, tuple)):
            return {
                f"label_{i}": pd.DataFrame(
                    m,
                    index=[f"Actual {j}" for j in range(len(m))],
                    columns=[f"Pred {j}" for j in range(len(m[0]))],
                )
                for i, m in enumerate(cm)
            }

        return pd.DataFrame(
            cm,
            index=[f"Actual {i}" for i in range(len(cm))],
            columns=[f"Pred {i}" for i in range(len(cm[0]))],
        )

    @staticmethod
    def roc_curve(roc_data):

        if not roc_data:
            return None

        fpr = roc_data.get("fpr")
        tpr = roc_data.get("tpr")

        if isinstance(fpr, (list, np.ndarray)):
            return pd.DataFrame({"fpr": fpr, "tpr": tpr})

        if isinstance(fpr, dict):
            return {
                f"class_{k}": pd.DataFrame({
                    "fpr": fpr[k],
                    "tpr": tpr[k]
                })
                for k in fpr
            }

        return None

    @staticmethod
    def pr_curve(pr_data):

        if not pr_data:
            return None

        return pd.DataFrame({
            "precision": pr_data.get("precision"),
            "recall": pr_data.get("recall")
        })

    @staticmethod
    def classification_report(report):

        if not report:
            return None

        return pd.DataFrame(report).transpose()
