import io
from pathlib import Path

import numpy as np
import pandas as pd

from lib.html import HtmlBuilder
from lib.utility.deeplearning.config.deep_learning_config import DeepLearningConfig
from lib.utility.deeplearning.evaluation.classification_evaluator import (
    ClassificationEvaluator,
)
from lib.utility.deeplearning.frameworks.tensorflow.data.tf_image_classification_data_loader import (
    TFImageClassificationDataBundle,
    TFImageClassificationDataLoader,
)
from lib.utility.deeplearning.frameworks.tensorflow.models.cnn.efficientnet_wrapper import (
    EfficientNetB0Wrapper,
)
from lib.utility.deeplearning.frameworks.tensorflow.models.cnn.resnet_wrapper import (
    ResNetWrapper,
)
from lib.utility.deeplearning.frameworks.tensorflow.tensorflow_model_utility import (
    TensorFlowModelUtility,
)
from lib.utility.reports.report_utils import ReportUtils as ru


def _load_face_mask_data(
    dataset_root: str,
    image_size: tuple[int, int],
    batch_size: int,
) -> TFImageClassificationDataBundle:
    root = Path(dataset_root)

    train_dir = root / "train"
    test_dir = root / "test"

    if train_dir.exists() and test_dir.exists():
        return TFImageClassificationDataLoader.from_train_test_directories(
            train_directory=train_dir,
            test_directory=test_dir,
            image_size=image_size,
            batch_size=batch_size,
            validation_split=0.2,
            train_augmentation=True,
            seed=42,
        )

    data_dir = root / "data"

    if data_dir.exists():
        return TFImageClassificationDataLoader.from_single_directory(
            directory=data_dir,
            image_size=image_size,
            batch_size=batch_size,
            validation_split=0.2,
            test_split=0.1,
            train_augmentation=True,
            seed=42,
        )

    return TFImageClassificationDataLoader.from_single_directory(
        directory=root,
        image_size=image_size,
        batch_size=batch_size,
        validation_split=0.2,
        test_split=0.1,
        train_augmentation=True,
        seed=42,
    )


def _model_summary_text(model) -> str:
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    return stream.getvalue()


def _train_and_evaluate_model(
    name: str,
    model_wrapper,
    config: DeepLearningConfig,
    data_bundle: TFImageClassificationDataBundle,
) -> dict:
    utility = TensorFlowModelUtility(
        model_wrapper=model_wrapper,
        config=config,
    )

    utility.compile(metrics=["accuracy"])

    history = utility.train(
        X_train=data_bundle.train_generator,
        y_train=None,
        validation_data=data_bundle.validation_generator,
    )

    data_bundle.test_generator.reset()
    tf_metrics = utility.evaluate(
        X_test=data_bundle.test_generator,
        y_test=None,
    )

    data_bundle.test_generator.reset()
    probabilities = utility.predict(data_bundle.test_generator)

    y_true = data_bundle.test_generator.classes
    y_pred = np.argmax(probabilities, axis=1)

    evaluator = ClassificationEvaluator()
    cls_metrics = evaluator.evaluate(
        y_true=y_true,
        y_pred=y_pred,
    )

    result_metrics = {
        "model": name,
        **{f"tf_{k}": float(v) for k, v in tf_metrics.items()},
        **{f"cls_{k}": float(v) for k, v in cls_metrics.items()},
    }

    return {
        "name": name,
        "metrics": result_metrics,
        "history_df": history.to_dataframe(),
        "summary": _model_summary_text(utility.get_model()),
    }


def main():
    dataset_root = "datasets/Face_mask_detection"
    image_size = (128, 128)

    config = DeepLearningConfig(
        epochs=25,
        batch_size=32,
        optimizer="adam",
        learning_rate=0.001,
        loss="categorical_crossentropy",
        early_stopping=True,
        patience=5,
        early_stopping_monitor="val_loss",
        reduce_lr=True,
        reduce_lr_factor=0.5,
        reduce_lr_patience=3,
        reduce_lr_monitor="val_loss",
    )

    data_bundle = _load_face_mask_data(
        dataset_root=dataset_root,
        image_size=image_size,
        batch_size=config.batch_size,
    )

    num_classes = len(data_bundle.class_indices)

    efficientnet_model = EfficientNetB0Wrapper(
        input_shape=(image_size[0], image_size[1], 3),
        num_classes=num_classes,
        trainable=False,
        dropout_rate=0.2,
        output_activation="softmax",
    )

    resnet_model = ResNetWrapper(
        input_shape=(image_size[0], image_size[1], 3),
        num_classes=num_classes,
        trainable=False,
        dropout_rate=0.5,
        output_activation="softmax",
    )

    experiment_results = [
        _train_and_evaluate_model(
            name="EfficientNetB0",
            model_wrapper=efficientnet_model,
            config=config,
            data_bundle=data_bundle,
        ),
        _train_and_evaluate_model(
            name="ResNet50",
            model_wrapper=resnet_model,
            config=config,
            data_bundle=data_bundle,
        ),
    ]

    metrics_df = pd.DataFrame([result["metrics"] for result in experiment_results])

    # Sort by TensorFlow accuracy to identify the strongest architecture.
    ranking_df = metrics_df.sort_values(
        by="tf_accuracy",
        ascending=False,
    ).reset_index(drop=True)

    best_model = ranking_df.iloc[0]["model"]

    dataset_info = {
        "dataset_root": str(Path(dataset_root).resolve()),
        "classes": ", ".join(data_bundle.class_names),
        "num_classes": num_classes,
        "train_samples": data_bundle.train_samples,
        "validation_samples": data_bundle.validation_samples,
        "test_samples": data_bundle.test_samples,
        "image_size": f"{image_size[0]}x{image_size[1]}",
    }

    builder = HtmlBuilder()
    content = []

    content.append(
        builder.full_width_card(
            "Dataset Summary",
            builder.render_dict(dataset_info),
        )
    )

    content.append(
        builder.full_width_card(
            "Model Ranking",
            builder.render_dataframe(ranking_df, max_visible_rows=10),
        )
    )

    content.append(
        builder.card(
            "Best Model",
            builder.render_dict({"selected_model": best_model}),
        )
    )

    for result in experiment_results:
        content.append(
            builder.full_width_card(
                f"{result['name']} - Training History",
                builder.render_dataframe(result["history_df"], max_visible_rows=30),
            )
        )

        content.append(
            builder.full_width_card(
                f"{result['name']} - Model Summary",
                builder.render_pre(result["summary"], max_visible_lines=60),
            )
        )

    html_doc = builder.build_page(
        "Face Mask Detection - Transfer Learning Benchmark",
        "\n".join(content),
    )

    ru.save_html_report(
        __file__,
        "face_mask_transfer_learning_report.html",
        html_doc,
        subfolder="reports",
        open_in_browser=True,
    )


if __name__ == "__main__":
    main()
