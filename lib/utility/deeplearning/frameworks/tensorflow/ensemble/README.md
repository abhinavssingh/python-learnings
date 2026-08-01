# TensorFlow Voting Ensemble Utilities

This module provides sklearn-compatible soft-voting ensembles that combine:

- LogisticRegression
- RandomForestClassifier
- SVC (probability=True)
- TensorFlow Keras neural network classifier (sklearn estimator style)

## Files

- tf_keras_classifier.py
- tf_voting_classifier_factory.py

## Factory Defaults

- LogisticRegression(solver="lbfgs", random_state=42)
- RandomForestClassifier(n_estimators=100, random_state=42)
- SVC(gamma="scale", probability=True, random_state=42)
- VotingClassifier(voting="soft", flatten_transform=True)

## Usage

```python
from lib.utility.deeplearning.frameworks.tensorflow.ensemble.tf_voting_classifier_factory import TFVotingClassifierFactory

voting = TFVotingClassifierFactory.create_soft_voting_classifier(
    random_state=42,
    rf_n_estimators=100,
    keras_epochs=25,
    keras_batch_size=32,
)

voting.fit(X_train, y_train)
pred = voting.predict(X_test)
proba = voting.predict_proba(X_test)
```
