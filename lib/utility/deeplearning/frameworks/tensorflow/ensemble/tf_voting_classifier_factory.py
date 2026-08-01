from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from .tf_keras_classifier import TFKerasClassifier


class TFVotingClassifierFactory:
    """
    Factory utilities for soft-voting ensembles that combine
    classical sklearn estimators with a TensorFlow Keras classifier.
    """

    @staticmethod
    def create_base_estimators(
        random_state: int = 42,
        rf_n_estimators: int = 100,
        keras_epochs: int = 25,
        keras_batch_size: int = 32,
        keras_verbose: int = 0,
    ) -> dict:
        log_clf = LogisticRegression(
            solver="lbfgs",
            random_state=random_state,
            max_iter=1000,
        )

        rnd_clf = RandomForestClassifier(
            n_estimators=rf_n_estimators,
            random_state=random_state,
        )

        svm_clf = SVC(
            gamma="scale",
            random_state=random_state,
            probability=True,
        )

        keras_clf = TFKerasClassifier(
            random_state=random_state,
            epochs=keras_epochs,
            batch_size=keras_batch_size,
            verbose=keras_verbose,
        )

        return {
            "log_clf": log_clf,
            "rnd_clf": rnd_clf,
            "svm_clf": svm_clf,
            "keras_clf": keras_clf,
        }

    @staticmethod
    def create_soft_voting_classifier(
        random_state: int = 42,
        rf_n_estimators: int = 100,
        keras_epochs: int = 25,
        keras_batch_size: int = 32,
        keras_verbose: int = 0,
        flatten_transform: bool = True,
    ) -> VotingClassifier:
        estimators = TFVotingClassifierFactory.create_base_estimators(
            random_state=random_state,
            rf_n_estimators=rf_n_estimators,
            keras_epochs=keras_epochs,
            keras_batch_size=keras_batch_size,
            keras_verbose=keras_verbose,
        )

        voting = VotingClassifier(
            estimators=[
                ("lr", estimators["log_clf"]),
                ("rf", estimators["rnd_clf"]),
                ("svc", estimators["svm_clf"]),
                ("keras", estimators["keras_clf"]),
            ],
            voting="soft",
            flatten_transform=flatten_transform,
        )

        return voting
