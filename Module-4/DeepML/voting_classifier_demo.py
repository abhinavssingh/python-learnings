import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from lib.html import HtmlBuilder
from lib.utility.dataframe.data_loader import DataLoader as dl
from lib.utility.deeplearning.frameworks.tensorflow.ensemble.tf_voting_classifier_factory import (
    TFVotingClassifierFactory,
)
from lib.utility.reports.report_utils import (
    ReportUtils as ru,
)

DATASET_FILE = "Churn_Modeling.csv"
TARGET_COLUMN = "Exited"
DROP_COLUMNS = ["RowNumber", "CustomerId", "Surname"]


def main():
    builder = HtmlBuilder()
    df = dl.read_dataset(DATASET_FILE, optimize=True, handle_unnamed="drop", return_report=False,)

    df = df.drop(columns=DROP_COLUMNS, errors="ignore")

    y = df[TARGET_COLUMN]
    X = pd.get_dummies(df.drop(columns=[TARGET_COLUMN]), drop_first=True,)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y,)

    voting = TFVotingClassifierFactory.create_soft_voting_classifier(
        random_state=42,
        rf_n_estimators=100,
        keras_epochs=10,
        keras_batch_size=32,
        keras_verbose=0,
        flatten_transform=True,
    )

    voting.fit(X_train, y_train)
    y_pred = voting.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    html_doc = builder.build_page(
        "TensorFlow Voting Classifier Report",
        builder.grid([
            builder.card(
                "Voting Classifier Results",
                builder.render_dataframe(
                    pd.DataFrame({
                        "accuracy": [accuracy]
                    }),
                ),
            ),
            builder.card(
                "Best Voting Classifier",
                builder.render_dict(
                    {"accuracy": accuracy}
                ),
            ),
        ])
    )

    ru.save_html_report(
        __file__,
        "voting_classifier_report.html",
        html_doc,
        subfolder="reports",
        open_in_browser=True,
    )
    print("Soft voting classifier accuracy:", round(float(accuracy), 4))


if __name__ == "__main__":
    main()
