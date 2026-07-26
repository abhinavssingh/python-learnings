"""
Neural Network Fundamentals

Learning Path:

1. Activation Functions
2. Single Neuron
3. Single Perceptron
4. Gradient Descent
5. Forward Propagation
6. Backpropagation
7. Multi-Layer Perceptron

Generates:
    neural_network_fundamentals_report.html
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from lib.html import HtmlBuilder
from lib.utility.reports.report_utils import ReportUtils as ru

# ==========================================================
# ACTIVATION FUNCTIONS
# ==========================================================


def step_function(x):
    return np.where(x >= 0, 1, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0, x)


# ==========================================================
# SINGLE PERCEPTRON
# ==========================================================

class SinglePerceptron:

    def __init__(
        self,
        learning_rate=0.1,
        epochs=20
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def activation(self, z):
        return np.where(z >= 0, 1, 0)

    def fit(self, X, y):

        n_features = X.shape[1]

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):

            for xi, target in zip(X, y):

                linear_output = (
                    np.dot(xi, self.weights)
                    + self.bias
                )

                prediction = self.activation(
                    linear_output
                )

                update = (
                    self.learning_rate
                    * (target - prediction)
                )

                self.weights += update * xi
                self.bias += update

    def predict(self, X):

        linear_output = (
            np.dot(X, self.weights)
            + self.bias
        )

        return self.activation(
            linear_output
        )


# ==========================================================
# GRADIENT DESCENT
# ==========================================================

def gradient_descent_demo():

    x = np.array(
        [1, 2, 3, 4, 5]
    )

    y = np.array(
        [2, 4, 6, 8, 10]
    )

    weight = 0.0

    learning_rate = 0.01

    history = []

    for epoch in range(100):

        predictions = weight * x

        error = predictions - y

        gradient = (
            2 * np.mean(error * x)
        )

        weight -= (
            learning_rate * gradient
        )

        loss = np.mean(
            error ** 2
        )

        history.append(
            {
                "epoch": epoch + 1,
                "weight": round(weight, 4),
                "loss": round(loss, 4),
            }
        )

    return pd.DataFrame(history)


# ==========================================================
# FORWARD PROPAGATION
# ==========================================================

def forward_propagation_demo():

    X = np.array([[1, 0]])

    W1 = np.array(
        [
            [0.5, 0.2],
            [0.3, 0.8]
        ]
    )

    b1 = np.array([[0.1, 0.1]])

    z1 = np.dot(X, W1) + b1

    a1 = sigmoid(z1)

    return {
        "Input": X.tolist(),
        "W1": W1.tolist(),
        "b1": b1.tolist(),
        "Z1": z1.tolist(),
        "A1": a1.tolist(),
    }


# ==========================================================
# BACKPROPAGATION
# ==========================================================

def backpropagation_demo():

    y_true = np.array([[1]])

    y_pred = np.array([[0.7]])

    error = y_pred - y_true

    derivative = (
        y_pred * (1 - y_pred)
    )

    gradient = (
        error * derivative
    )

    return {
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "error": error.tolist(),
        "gradient": gradient.tolist(),
    }


# ==========================================================
# MULTI LAYER PERCEPTRON
# ==========================================================

def mlp_demo():

    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        random_state=42,
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            64,
            activation="relu",
            input_shape=(20,)
        ),
        tf.keras.layers.Dense(
            32,
            activation="relu"
        ),
        tf.keras.layers.Dense(
            1,
            activation="sigmoid"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=32,
        verbose=0,
    )

    predictions = (
        model.predict(
            X_test,
            verbose=0
        ) > 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return {
        "accuracy": round(accuracy, 4),
        "history": history.history,
        "predictions": predictions[
            :20
        ].flatten().tolist(),
    }


# ==========================================================
# MAIN
# ==========================================================

def main():

    builder = HtmlBuilder()

    # Activation Functions

    x = np.linspace(-5, 5, 11)

    activation_df = pd.DataFrame({

        "x": x,

        "step": step_function(x),

        "sigmoid": np.round(
            sigmoid(x),
            4
        ),

        "tanh": np.round(
            tanh(x),
            4
        ),

        "relu": np.round(
            relu(x),
            4
        ),
    })

    # Perceptron

    X_and = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
    )

    y_and = np.array(
        [0, 0, 0, 1]
    )

    perceptron = SinglePerceptron()

    perceptron.fit(
        X_and,
        y_and,
    )

    perceptron_predictions = (
        perceptron.predict(
            X_and
        )
    )

    perceptron_df = pd.DataFrame({
        "Input": X_and.tolist(),
        "Actual": y_and,
        "Predicted": perceptron_predictions,
    })

    # Gradient Descent

    gd_df = gradient_descent_demo()

    # Forward Propagation

    forward_results = (
        forward_propagation_demo()
    )

    # Backpropagation

    backprop_results = (
        backpropagation_demo()
    )

    # MLP

    mlp_results = mlp_demo()

    # ======================================================
    # REPORT
    # ======================================================

    html_doc = builder.build_page(
        "Neural Network Fundamentals Report",

        builder.grid([

            builder.card(
                "Activation Functions",
                builder.render_dataframe(
                    activation_df
                ),
            ),

            builder.card(
                "Single Perceptron",
                builder.render_dataframe(
                    perceptron_df
                ),
            ),

            builder.card(
                "Gradient Descent",
                builder.render_dataframe(
                    gd_df.tail(20)
                ),
            ),

            builder.card(
                "Forward Propagation",
                builder.render_dict(
                    forward_results
                ),
            ),

            builder.card(
                "Backpropagation",
                builder.render_dict(
                    backprop_results
                ),
            ),

            builder.card(
                "MLP Accuracy",
                builder.render_dict(
                    {
                        "accuracy":
                            mlp_results[
                                "accuracy"
                            ]
                    }
                ),
            ),

            builder.card(
                "MLP Predictions",
                builder.render_dict(
                    {
                        "predictions":
                            mlp_results[
                                "predictions"
                            ]
                    }
                ),
            ),

        ])
    )

    ru.save_html_report(
        __file__,
        "neural_network_fundamentals_report.html",
        html_doc,
        subfolder="reports",
        open_in_browser=True,
    )


if __name__ == "__main__":
    main()
