"""Tourism analytics, collaborative recommendations, and optional image classifier."""

import argparse
import base64
import html
import io
import os
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import classification_report, pairwise_distances

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import init  # noqa: E402,F401

from lib.utility.genai.config import GenAIConfig  # noqa: E402
from lib.utility.genai.loaders import TabularLoader  # noqa: E402
from lib.utility.genai.reports import GenAIReport  # noqa: E402

TOURISM_FILENAMES = {
    "users": "user.csv",
    "places": "tourism_with_id.xlsx",
    "ratings": "tourism_rating.csv",
}
IMAGE_ARCHIVE = Path("Capstone 2") / "Part 1" / "dataset_hist_structures 2.zip"
TRAIN_DIR_NAMES = {"stuctures_dataset", "structures_dataset"}
TEST_DIR_NAMES = {"dataset_test"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RATING_SEED = 42


def load_tourism_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    paths = {name: data_dir / filename for name, filename in TOURISM_FILENAMES.items()}
    missing_files = [str(path) for path in paths.values() if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Required tourism dataset file(s) not found: {', '.join(missing_files)}")

    users = TabularLoader(paths["users"]).read_frame()
    places = TabularLoader(paths["places"]).read_frame()
    ratings = TabularLoader(paths["ratings"]).read_frame()
    schemas = {
        "users": (users, {"User_Id", "Location", "Age"}),
        "places": (places, {"Place_Id", "Place_Name", "Description", "Category", "City", "Rating"}),
        "ratings": (ratings, {"User_Id", "Place_Id", "Place_Ratings"}),
    }
    for name, (frame, required) in schemas.items():
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} dataset is missing required columns: {', '.join(sorted(missing))}")

    quality = {
        "users_rows": len(users),
        "places_rows": len(places),
        "rating_rows": len(ratings),
        "user_duplicate_rows": int(users.duplicated().sum()),
        "place_duplicate_rows": int(places.duplicated().sum()),
        "rating_duplicate_rows": int(ratings.duplicated().sum()),
        "rating_rows_with_missing_values": int(ratings.isna().any(axis=1).sum()),
        "rating_pairs_aggregated": 0,
    }

    users = users.drop_duplicates().drop_duplicates(subset=["User_Id"], keep="first").copy()
    places = places.drop(columns=[c for c in places if c.startswith("Unnamed:")], errors="ignore")
    places = places.drop_duplicates().drop_duplicates(subset=["Place_Id"], keep="first").copy()
    ratings = ratings.drop_duplicates().copy()
    for column in ("User_Id", "Age"):
        users[column] = pd.to_numeric(users[column], errors="coerce")
    for column in ("Place_Id", "Rating", "Price"):
        if column in places:
            places[column] = pd.to_numeric(places[column], errors="coerce")
    for column in ("User_Id", "Place_Id", "Place_Ratings"):
        ratings[column] = pd.to_numeric(ratings[column], errors="coerce")

    users = users.dropna(subset=["User_Id", "Age", "Location"])
    places = places.dropna(subset=["Place_Id", "Place_Name", "Category", "City"])
    ratings = ratings.dropna(subset=["User_Id", "Place_Id", "Place_Ratings"])
    users = users.loc[users["Age"].between(1, 110)]
    ratings = ratings.loc[ratings["Place_Ratings"].between(1, 5)]
    user_ids, place_ids = set(users["User_Id"]), set(places["Place_Id"])
    ratings = ratings.loc[ratings["User_Id"].isin(user_ids) & ratings["Place_Id"].isin(place_ids)]

    before_pairs = len(ratings)
    ratings = ratings.groupby(["User_Id", "Place_Id"], as_index=False)["Place_Ratings"].mean()
    quality["rating_pairs_aggregated"] = before_pairs - len(ratings)
    if ratings.empty or users.empty or places.empty:
        raise ValueError("No valid users, places, or ratings remain after quality checks.")
    return users, places, ratings, quality


def make_place_item_matrix(ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return user×place ratings and user-mean-centered values for item-CF."""
    matrix = ratings.pivot(index="User_Id", columns="Place_Id", values="Place_Ratings")
    centered = matrix.sub(matrix.mean(axis=1), axis=0).fillna(0.0)
    return matrix, centered


def item_similarity(centered: pd.DataFrame) -> pd.DataFrame:
    """Cosine similarity between places based on mean-centered user ratings."""
    vectors = centered.to_numpy(dtype=float).T
    similarity = 1.0 - pairwise_distances(vectors, metric="cosine")
    similarity = np.nan_to_num(similarity, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(similarity, 0.0)
    return pd.DataFrame(similarity, index=centered.columns, columns=centered.columns)


def recommend_for_user(
    user_id: int,
    matrix: pd.DataFrame,
    centered: pd.DataFrame,
    similarity: pd.DataFrame,
    places: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    if user_id not in matrix.index:
        raise ValueError(f"User_Id {user_id} has no ratings in the cleaned dataset.")
    rated = matrix.loc[user_id].dropna()
    if rated.empty:
        raise ValueError(f"User_Id {user_id} has no usable rating history.")
    user_centered = centered.loc[user_id]
    scores = similarity.dot(user_centered)
    denominators = similarity.abs().dot(matrix.loc[user_id].notna().astype(float))
    scores = scores.div(denominators.replace(0, np.nan))
    scores = scores.where(matrix.loc[user_id].isna()).dropna()

    if scores.empty:
        popular = places.set_index("Place_Id")["Rating"].sort_values(ascending=False)
        scores = popular.loc[~popular.index.isin(rated.index)].head(top_n)
    selected = scores.sort_values(ascending=False).head(top_n)
    result = places.loc[places["Place_Id"].isin(selected.index)].copy()
    result["predicted_preference"] = result["Place_Id"].map(selected)
    columns = ["Place_Id", "Place_Name", "Category", "City", "Rating", "predicted_preference"]
    return result[columns].sort_values("predicted_preference", ascending=False).reset_index(drop=True)


def recommend_similar_places(
    place_id: int, similarity: pd.DataFrame, places: pd.DataFrame, top_n: int = 10
) -> pd.DataFrame:
    if place_id not in similarity.index:
        raise ValueError(f"Place_Id {place_id} has no collaborative ratings.")
    nearest = similarity.loc[place_id].sort_values(ascending=False).head(top_n)
    recommendations = places.loc[places["Place_Id"].isin(nearest.index)].copy()
    recommendations["collaborative_similarity"] = recommendations["Place_Id"].map(nearest)
    columns = [
        "Place_Id", "Place_Name", "Category", "City", "Rating", "collaborative_similarity"
    ]
    return recommendations[columns].sort_values(
        "collaborative_similarity", ascending=False
    ).reset_index(drop=True)


def evaluate_recommender(
    ratings: pd.DataFrame, places: pd.DataFrame, top_n: int = 10
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Deterministic per-user holdout evaluation using hit rate, recall and NDCG."""
    counts = ratings.groupby("User_Id").size()
    eligible = ratings.loc[ratings["User_Id"].isin(counts[counts >= 2].index)]
    test = eligible.groupby("User_Id", group_keys=False).sample(n=1, random_state=RATING_SEED)
    held_out_pairs = pd.MultiIndex.from_frame(test[["User_Id", "Place_Id"]])
    train_pairs = pd.MultiIndex.from_frame(eligible[["User_Id", "Place_Id"]])
    train = eligible.loc[~train_pairs.isin(held_out_pairs)]
    train_matrix, train_centered = make_place_item_matrix(train)
    similarity = item_similarity(train_centered)
    catalog = places.loc[places["Place_Id"].isin(train_matrix.columns)]
    relevant = (
        test.loc[test["Place_Ratings"] >= 4]
        .groupby("User_Id")["Place_Id"]
        .apply(set)
        .to_dict()
    )
    outcomes = []
    for user_id, relevant_items in relevant.items():
        if not relevant_items or user_id not in train_matrix.index:
            continue
        ranked = recommend_for_user(
            user_id, train_matrix, train_centered, similarity, catalog, top_n=top_n
        )["Place_Id"].tolist()
        hits = [1 if place_id in relevant_items else 0 for place_id in ranked]
        hit_count = sum(hits)
        dcg = sum(hit / np.log2(rank + 2) for rank, hit in enumerate(hits))
        ideal_hits = min(len(relevant_items), top_n)
        idcg = sum(1 / np.log2(rank + 2) for rank in range(ideal_hits))
        outcomes.append({
            "User_Id": user_id,
            "relevant_test_places": len(relevant_items),
            "hits_at_k": hit_count,
            "recall_at_k": hit_count / len(relevant_items),
            "ndcg_at_k": dcg / idcg if idcg else 0.0,
            "hit_rate_at_k": float(hit_count > 0),
        })
    frame = pd.DataFrame(outcomes)
    if frame.empty:
        summary = {"users_evaluated": 0, "Recall@K": 0.0, "NDCG@K": 0.0, "HitRate@K": 0.0}
    else:
        summary = {
            "users_evaluated": int(len(frame)),
            "Recall@K": float(frame["recall_at_k"].mean()),
            "NDCG@K": float(frame["ndcg_at_k"].mean()),
            "HitRate@K": float(frame["hit_rate_at_k"].mean()),
        }
    return frame, summary


def build_tourism_report(
    script_file: str,
    users: pd.DataFrame,
    places: pd.DataFrame,
    ratings: pd.DataFrame,
    quality: dict,
    evaluation: dict,
    recommendations: pd.DataFrame,
) -> None:
    report = GenAIReport("Tourism Analytics and Collaborative Recommendations")
    rated_places = ratings.merge(
        places[["Place_Id", "Place_Name", "Category", "City"]],
        on="Place_Id",
        how="inner",
        validate="many_to_one",
    )
    user_analytics = users.copy()
    user_analytics["age_group"] = pd.cut(
        user_analytics["Age"], bins=[0, 17, 24, 34, 44, 54, 110],
        labels=["under 18", "18–24", "25–34", "35–44", "45–54", "55+"],
    )
    age_counts = user_analytics["age_group"].value_counts().sort_index().rename_axis(
        "age_group"
    ).reset_index(name="users")
    origins = user_analytics.assign(
        origin_city=user_analytics["Location"].astype(str).str.split(",", n=1).str[0].str.strip()
    ).groupby("origin_city", as_index=False).size().sort_values("size", ascending=False).head(15)
    categories = places.groupby("Category", as_index=False).agg(
        place_count=("Place_Id", "nunique"),
        average_place_rating=("Rating", "mean"),
    ).sort_values("place_count", ascending=False)
    city_categories = places.groupby(["City", "Category"], as_index=False).size()
    city_specialties = city_categories.loc[city_categories.groupby("City")["size"].idxmax()].sort_values(
        "City"
    )
    nature = places.loc[
        places["Category"].astype(str).str.contains(r"nature|cagar\s*alam", case=False, na=False)
    ]
    nature_cities = (
        nature.groupby("City", as_index=False).agg(
            nature_places=("Place_Id", "nunique"),
            mean_rating=("Rating", "mean"),
        ).sort_values(["mean_rating", "nature_places"], ascending=False)
    )
    city_attractions = places.groupby("City", as_index=False).agg(
        attractions=("Place_Id", "nunique")
    ).sort_values("attractions", ascending=False).head(15)
    place_popularity = rated_places.groupby(
        ["Place_Id", "Place_Name", "Category", "City"], as_index=False
    ).agg(
        average_user_rating=("Place_Ratings", "mean"),
        rating_count=("Place_Ratings", "size"),
    )
    place_popularity["bayesian_rating"] = (
        place_popularity["rating_count"] * place_popularity["average_user_rating"]
        + 10 * ratings["Place_Ratings"].mean()
    ) / (place_popularity["rating_count"] + 10)
    top_places = place_popularity.sort_values(
        ["bayesian_rating", "rating_count"], ascending=False
    ).head(15)
    city_rank = place_popularity.groupby("City", as_index=False).agg(
        rated_places=("Place_Id", "nunique"),
        average_user_rating=("average_user_rating", "mean"),
        rating_count=("rating_count", "sum"),
    ).sort_values(["average_user_rating", "rating_count"], ascending=False)
    category_popularity = rated_places.groupby("Category", as_index=False).agg(
        ratings=("Place_Ratings", "size"),
        mean_user_rating=("Place_Ratings", "mean"),
    ).sort_values("ratings", ascending=False)

    report.grid([
        report.kv("Dataset summary", {
            "users": len(users),
            "places": len(places),
            "unique user-place ratings": len(ratings),
            "rating scale": f"{ratings['Place_Ratings'].min():.0f}–{ratings['Place_Ratings'].max():.0f}",
            "places with ratings": ratings["Place_Id"].nunique(),
        }),
        report.kv("Collaborative-filtering holdout (K=10)", evaluation),
    ])
    report.full_table("Data-quality checks", [
        {"check": name, "count": value} for name, value in quality.items()
    ], rows=len(quality))
    report.plots([
        (px.bar(age_counts, x="age_group", y="users", title="Tourist age distribution"), "Tourist demographics"),
        (px.bar(origins.sort_values("size"), x="size", y="origin_city", orientation="h",
                title="Top visitor origins"), "Visitor origins"),
        (px.bar(categories, x="Category", y="place_count", title="Places by category"), "Tourism categories"),
        (px.bar(nature_cities.head(15), x="City", y="nature_places",
                color="mean_rating", title="Nature destinations by city"), "Nature tourism"),
        (px.bar(top_places.sort_values("bayesian_rating"), x="bayesian_rating", y="Place_Name",
                color="rating_count", orientation="h", title="Top attractions (support-adjusted)"),
         "Top-rated attractions"),
        (px.bar(category_popularity.head(15), x="Category", y="ratings",
                color="mean_user_rating", title="Popularity by user ratings"), "Category popularity"),
        (px.bar(city_attractions, x="City", y="attractions", title="Attractions by city"),
         "City-wise attractions"),
    ])
    report.full_table("City specialties (most common place category)", city_specialties, rows=50)
    report.full_table("Cities with the most listed attractions", city_attractions, rows=15)
    report.full_table("Best cities for nature tourism", nature_cities, rows=30)
    report.full_table("Cities with highest-rated attractions", city_rank, rows=30)
    report.full_table("Top attractions by support-adjusted user rating", top_places, rows=15)
    report.full_table("Personalized recommendations for first user", recommendations, rows=10)
    report.full_text(
        "Interpretation and limitations",
        "Recommendations are based on user-item collaborative filtering with cosine similarity over "
        "user-mean-centered ratings. A deterministic, per-user stratified holdout is used for Recall@10, "
        "NDCG@10, and HitRate@10; the source contains no timestamps, so this is not a temporal evaluation. "
        "Nature tourism is identified from 'nature' or the dataset's Indonesian 'Cagar Alam' category. "
        "The separate historical-"
        "structure image-classification workstream is available through the optional --task images mode.",
    )
    report.save(script_file, "tourism_ai_report.html")


def run_tourism(data_dir: Path, artifact_dir: Path, script_file: str) -> tuple:
    users, places, ratings, quality = load_tourism_data(data_dir)
    matrix, centered = make_place_item_matrix(ratings)
    similarity = item_similarity(centered)
    evaluation_rows, evaluation = evaluate_recommender(ratings, places)
    default_user = int(ratings["User_Id"].value_counts().index[0])
    personalized = recommend_for_user(default_user, matrix, centered, similarity, places)
    default_place = int(ratings["Place_Id"].value_counts().index[0])
    similar = recommend_similar_places(default_place, similarity, places)

    output = artifact_dir / "tourism_ai"
    output.mkdir(parents=True, exist_ok=True)
    ratings.to_csv(output / "cleaned_ratings.csv", index=False)
    personalized.to_csv(output / f"recommendations_user_{default_user}.csv", index=False)
    similar.to_csv(output / f"similar_places_{default_place}.csv", index=False)
    evaluation_rows.to_csv(output / "recommendation_evaluation_by_user.csv", index=False)
    joblib.dump(
        {
            "user_place_ratings": matrix,
            "centered_user_place_ratings": centered,
            "item_similarity": similarity,
            "places": places,
            "recommendation_method": "user-mean-centered item-item cosine similarity",
        },
        output / "tourism_recommender.joblib",
    )
    build_tourism_report(
        script_file, users, places, ratings, quality, evaluation, personalized
    )
    return users, places, ratings, matrix, centered, similarity, evaluation


def _safe_extract_image_archive(archive: Path, destination: Path) -> None:
    marker = destination / ".extraction_complete"
    if marker.is_file():
        return
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            member = PurePosixPath(info.filename)
            if info.is_dir() or not member.parts or "__MACOSX" in member.parts:
                continue
            if member.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            output = (destination / Path(*member.parts)).resolve()
            if os.path.commonpath([str(root), str(output)]) != str(root):
                raise ValueError(f"Unsafe path in image archive: {info.filename}")
            output.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as source, output.open("wb") as target:
                shutil.copyfileobj(source, target)
    marker.write_text("complete", encoding="utf-8")


def _image_gallery_html(train_dir: Path, class_names: list[str], per_class: int = 8) -> str:
    from PIL import Image

    cards = []
    for class_name in class_names:
        paths = sorted(
            path for path in (train_dir / class_name).rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )[:per_class]
        images = []
        for path in paths:
            with Image.open(path) as source:
                image = source.convert("RGB")
                image.thumbnail((180, 140))
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=75)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            images.append(
                f'<img src="data:image/jpeg;base64,{encoded}" alt="{html.escape(class_name)}" '
                'style="width:100%;max-width:180px;height:140px;object-fit:cover;border-radius:6px">'
            )
        cards.append(
            '<section style="margin:12px 0"><h3>'
            f'{html.escape(class_name)} ({len(images)} samples)</h3>'
            '<div style="display:flex;flex-wrap:wrap;gap:8px">'
            + "".join(images)
            + "</div></section>"
        )
    return "".join(cards)


def run_images(data_root: Path, artifact_dir: Path, script_file: str) -> None:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "Image classification requires TensorFlow, which is not installed in this Python environment. "
            "Install a TensorFlow version compatible with your Python/OS, then rerun with --task images. "
            "The tourism analytics and recommendations task runs without TensorFlow."
        ) from exc

    archive = data_root / IMAGE_ARCHIVE
    if not archive.is_file():
        raise FileNotFoundError(f"Historical structures image archive not found: {archive}")
    image_root = artifact_dir / "tourism_ai" / "image_data"
    _safe_extract_image_archive(archive, image_root)
    train_dir = next(
        (path for path in image_root.rglob("*") if path.is_dir() and path.name.lower() in TRAIN_DIR_NAMES),
        None,
    )
    test_dir = next(
        (path for path in image_root.rglob("*") if path.is_dir() and path.name.lower() in TEST_DIR_NAMES),
        None,
    )
    if train_dir is None or test_dir is None:
        raise FileNotFoundError("Image archive must contain Stuctures_Dataset and Dataset_test folders.")

    image_size = int(os.getenv("TOURISM_IMAGE_SIZE", "160"))
    batch_size = int(os.getenv("TOURISM_IMAGE_BATCH_SIZE", "32"))
    epochs = int(os.getenv("TOURISM_IMAGE_EPOCHS", "8"))
    seed = int(os.getenv("TOURISM_IMAGE_SEED", "42"))
    if image_size < 64 or batch_size < 1 or epochs < 1:
        raise ValueError("Image size must be at least 64; batch size and epochs must be positive.")

    base = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=(image_size, image_size), batch_size=batch_size,
        label_mode="int", shuffle=False,
    )
    class_names = list(base.class_names)
    if len(class_names) < 2:
        raise ValueError(f"Expected at least two training classes, found: {class_names}")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, validation_split=0.2, subset="training", seed=seed,
        image_size=(image_size, image_size), batch_size=batch_size, label_mode="int",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, validation_split=0.2, subset="validation", seed=seed,
        image_size=(image_size, image_size), batch_size=batch_size, label_mode="int",
    )
    if list(train_ds.class_names) != class_names or list(val_ds.class_names) != class_names:
        raise ValueError("Training/validation image class indexes do not match.")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, image_size=(image_size, image_size), batch_size=batch_size,
        label_mode="int", shuffle=False,
    )
    test_class_names = list(test_ds.class_names)
    class_indices = {name: i for i, name in enumerate(class_names)}
    unknown = sorted(set(test_class_names) - set(class_indices))
    if unknown:
        raise ValueError(f"Test data includes classes absent from training: {unknown}")
    test_label_map = tf.constant([class_indices[name] for name in test_class_names], dtype=tf.int32)
    test_ds = test_ds.map(lambda images, labels: (images, tf.gather(test_label_map, labels)))
    train_counts = {
        path.name: sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)
        for path in train_dir.iterdir() if path.is_dir()
    }
    test_counts = {
        path.name: sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)
        for path in test_dir.iterdir() if path.is_dir()
    }

    output = artifact_dir / "tourism_ai"
    output.mkdir(parents=True, exist_ok=True)
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.1),
    ], name="augmentation")

    def make_model(augment: bool):
        inputs = tf.keras.Input(shape=(image_size, image_size, 3))
        x = augmentation(inputs) if augment else inputs
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        backbone = tf.keras.applications.MobileNetV2(
            input_shape=(image_size, image_size, 3), include_top=False, weights="imagenet"
        )
        backbone.trainable = False
        x = backbone(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    report = GenAIReport("Historical Structure Image Classification")
    report.grid([
        report.table("Training image counts", [
            {"class": name, "images": train_counts.get(name, 0)} for name in class_names
        ]),
        report.table("Test image counts", [
            {"class": name, "images": test_counts.get(name, 0)} for name in test_class_names
        ]),
    ])
    report.full("Training samples (up to 8 images per class)", _image_gallery_html(train_dir, class_names))
    curves = []
    best_validation_accuracy = -1.0
    for use_augmentation in (False, True):
        variant = "with_augmentation" if use_augmentation else "without_augmentation"
        model = make_model(use_augmentation)
        callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )
        history = model.fit(
            train_ds, validation_data=val_ds, epochs=epochs,
            callbacks=[callback], verbose=2,
        )
        test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
        true_labels = np.concatenate([labels.numpy() for _, labels in test_ds])
        predicted_labels = np.argmax(model.predict(test_ds, verbose=0), axis=1)
        class_report = classification_report(
            true_labels,
            predicted_labels,
            labels=list(range(len(class_names))),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        model_path = output / f"historical_structures_{variant}.keras"
        model.save(model_path)
        report.kv(f"Test results: {variant}", {
            "test loss": round(float(test_loss), 4),
            "test accuracy": round(float(test_accuracy), 4),
            "best validation accuracy": round(max(history.history["val_accuracy"]), 4),
            "final train-validation accuracy gap": round(
                float(history.history["accuracy"][-1] - history.history["val_accuracy"][-1]), 4
            ),
            "epochs completed": len(history.epoch),
            "saved model": str(model_path),
        })
        report.full_table(
            f"Per-class test metrics: {variant}",
            [
                {"class": name, **class_report[name]}
                for name in class_names
            ],
            rows=len(class_names),
        )
        confusion = tf.math.confusion_matrix(
            true_labels, predicted_labels, num_classes=len(class_names)
        ).numpy()
        report.full_plot(
            px.imshow(
                confusion,
                x=class_names,
                y=class_names,
                text_auto=True,
                labels={"x": "Predicted class", "y": "Actual class", "color": "images"},
                title=f"Test confusion matrix — {variant}",
                aspect="auto",
            ),
            f"Test confusion matrix: {variant}",
        )
        curves.extend([
            (px.line(
                pd.DataFrame({
                    "epoch": range(1, len(history.history["accuracy"]) + 1),
                    "training accuracy": history.history["accuracy"],
                    "validation accuracy": history.history["val_accuracy"],
                }).melt(id_vars="epoch", var_name="series", value_name="accuracy"),
                x="epoch", y="accuracy", color="series", title=f"Accuracy — {variant}",
            ), f"Training curve: {variant}"),
            (px.line(
                pd.DataFrame({
                    "epoch": range(1, len(history.history["loss"]) + 1),
                    "training loss": history.history["loss"],
                    "validation loss": history.history["val_loss"],
                }).melt(id_vars="epoch", var_name="series", value_name="loss"),
                x="epoch", y="loss", color="series", title=f"Loss — {variant}",
            ), f"Loss curve: {variant}"),
        ])
        if max(history.history["val_accuracy"]) > best_validation_accuracy:
            best_validation_accuracy = max(history.history["val_accuracy"])
            model.save(output / "historical_structures_best.keras")
    report.plots(curves)
    report.full_text(
        "Image-classification notes",
        "Both models use an ImageNet-pretrained MobileNetV2 with the convolutional backbone frozen, "
        "a dense classification head, dropout, and early stopping. The second run adds training-only "
        "random flips, rotations, and zoom. The supplied held-out test folder omits the 'portal' class; "
        "test metrics cover only classes present in that held-out folder.",
    )
    report.save(script_file, "historical_structures_report.html")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("tourism", "images", "all"), default="tourism")
    parser.add_argument("--user-id", type=int, help="Print personalized place recommendations after tourism run.")
    parser.add_argument("--place-id", type=int, help="Print places similar by collaborative ratings.")
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()
    if args.top_n < 1:
        parser.error("--top-n must be positive.")

    config = GenAIConfig.from_env()
    data_root = Path(os.getenv("GENAI_TOURISM_DATA_DIR", str(config.dataset_dir))).expanduser()
    artifact_dir = config.artifact_dir

    if args.task in {"tourism", "all"}:
        result = run_tourism(data_root / "Capstone 2" / "Part 2", artifact_dir, __file__)
        if args.user_id is not None:
            _, places, _, matrix, centered, similarity, _ = result
            print(recommend_for_user(args.user_id, matrix, centered, similarity, places, args.top_n).to_string(index=False))
        if args.place_id is not None:
            _, places, _, _, _, similarity, _ = result
            print(recommend_similar_places(args.place_id, similarity, places, args.top_n).to_string(index=False))
    if args.task in {"images", "all"}:
        run_images(data_root, artifact_dir, __file__)


if __name__ == "__main__":
    main()
