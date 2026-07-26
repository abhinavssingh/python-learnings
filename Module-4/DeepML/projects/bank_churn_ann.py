import io

import pandas as pd

from lib.html import HtmlBuilder
from lib.utility.dataframe.data_loader import DataLoader as dl
from lib.utility.deeplearning.config.deep_learning_config import (
    DeepLearningConfig,
)
from lib.utility.deeplearning.evaluation.classification_evaluator import (
    ClassificationEvaluator,
)
from lib.utility.deeplearning.frameworks.tensorflow.models.dense.mlp_wrapper import (
    MLPWrapper,
)
from lib.utility.deeplearning.frameworks.tensorflow.tensorflow_model_utility import (
    TensorFlowModelUtility,
)
from lib.utility.deeplearning.preprocessing.data_preprocessor import (
    DataPreprocessor,
)
from lib.utility.reports.report_utils import (
    ReportUtils as ru,
)


def main():

    builder = HtmlBuilder()
    content = []

    # ======================================================
    # LOAD DATASET
    # ======================================================

    df = dl.read_dataset(
        "Churn_Modeling.csv", optimize=True, handle_unnamed="drop", return_report=False
    )

    # ======================================================
    # PREPROCESS
    # ======================================================

    result = DataPreprocessor.prepare_classification_data(
        df=df,
        target_column="Exited",
        drop_columns=[
            "RowNumber",
            "CustomerId",
            "Surname",
        ],
        label_encode_columns=[
            "Gender",
        ],
        one_hot_columns=[
            "Geography",
        ],
        scale_numeric=True,
        test_size=0.2,
        random_state=0,
    )

    X_train = result.X_train
    X_test = result.X_test

    y_train = result.y_train
    y_test = result.y_test

    # ======================================================
    # CONFIG
    # ======================================================

    config = DeepLearningConfig(
        epochs=10,
        batch_size=10,
        optimizer="adam",
        learning_rate=0.001,
        loss="binary_crossentropy",
    )

    # ======================================================
    # MODEL
    # ======================================================

    model = MLPWrapper(
        input_dim=X_train.shape[1],
        output_dim=1,
        hidden_layers=[6],
        activation="relu",
        output_activation="sigmoid",
    )

    # ======================================================
    # TRAIN
    # ======================================================

    utility = TensorFlowModelUtility(model_wrapper=model, config=config)

    utility.compile()

    history = utility.train(X_train=X_train, y_train=y_train,)

    metrics = utility.evaluate(X_test=X_test, y_test=y_test,)

    predictions = utility.predict(X_test)

    predictions = (predictions > 0.5).astype(int)

    stream = io.StringIO()
    utility.get_model().summary(print_fn=lambda x: stream.write(x + "\n"))
    model_summary = stream.getvalue()

    # ======================================================
    # EVALUATION
    # ======================================================

    evaluator = ClassificationEvaluator()

    evaluation_result = evaluator.evaluate(y_true=y_test, y_pred=predictions)

    # ======================================================
    # TASK B
    # ======================================================

    customer = pd.DataFrame(
        [
            {
                "CreditScore": 600,
                "Geography": "France",
                "Gender": "Male",
                "Age": 40,
                "Tenure": 3,
                "Balance": 60000,
                "NumOfProducts": 2,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 50000,
            }
        ]
    )

    customer_x = DataPreprocessor.transform_new_data(
        customer,
        preprocessor=result.preprocessor,
        label_encoders=result.label_encoders,
    )

    customer_probability = utility.predict(customer_x)[0][0]

    customer_prediction = customer_probability > 0.5

    # ======================================================
    # REPORT
    # ======================================================

    report_data = {
        "Customer Response": "Leave" if customer_prediction else "Stay",
        "Probability": round(float(customer_probability), 4),
    }

    content.append(
        builder.full_width_card(
            "Original Bank Churn Data",
            builder.render_dataframe_collapsible(df, initial_rows=15)
        ))

    content.append(
        builder.full_width_card(
            "Model Summary",
            builder.render_pre(model_summary)
        ))

    content.append(
        builder.grid([
            builder.card("Customer Test Data", builder.render_dataframe(customer)),
            builder.card("Model Metrics", builder.render_dict(metrics),),
            builder.card("Classification Metrics", builder.render_dict(evaluation_result),),
            builder.card("Training History", builder.render_dataframe(history.to_dataframe()),),
            builder.card("Customer Prediction", builder.render_dict(report_data)),
        ]))

    html_doc = builder.build_page(
        "Bank Churn Prediction using ANN",
        "\n".join(content)
    )
    ru.save_html_report(
        __file__,
        "bank_churn_ann_report.html",
        html_doc,
        subfolder="reports",
        open_in_browser=True,
    )


if __name__ == "__main__":
    main()
