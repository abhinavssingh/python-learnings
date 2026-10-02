"""Tesla safety analytics and pretrained vehicle detection capstone."""

import argparse
import html
import os
import random
import re
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import init  # noqa: E402,F401

from lib.utility.genai.config import GenAIConfig  # noqa: E402
from lib.utility.genai.reports import GenAIReport  # noqa: E402

DEFAULT_IMAGE_ARCHIVE = Path("Capstone 1") / "Part 1" / "Images.zip"
DEFAULT_SAFETY_CSV = Path("Capstone 1") / "Part 2" / "Tesla - Deaths.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VEHICLE_LABELS = {"car", "motorcycle", "bus", "truck"}
SAFETY_COLUMNS = {
    "case": "Case #",
    "date": "Date",
    "country": "Country",
    "state": "State",
    "description": "Description",
    "deaths": "Deaths",
    "driver": "Tesla driver",
    "occupant": "Tesla occupant",
    "other_vehicle": "Other vehicle",
    "cyclists_pedestrians": "Cyclists/ Peds",
    "tesla_cyclists_pedestrians": "TSLA+cycl / peds",
    "model": "Model",
    "autopilot_claimed": "Autopilot claimed",
    "verified_autopilot_deaths": "Verified Tesla Autopilot Deaths",
    "verified_autopilot_plus_sgo": (
        "Verified Tesla Autopilot Deaths + All Deaths Reported to NHTSA SGO"
    ),
}
COUNT_FIELDS = (
    "deaths",
    "driver",
    "occupant",
    "other_vehicle",
    "cyclists_pedestrians",
    "tesla_cyclists_pedestrians",
    "autopilot_claimed",
    "verified_autopilot_deaths",
    "verified_autopilot_plus_sgo",
)


def _clean_text(series: pd.Series, default: str = "Unknown") -> pd.Series:
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.replace({"": pd.NA, "-": pd.NA, "—": pd.NA, "nan": pd.NA})
    return cleaned.fillna(default)


def load_safety_data(csv_path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Tesla safety CSV not found: {csv_path}")
    raw = pd.read_csv(csv_path)
    raw.columns = raw.columns.astype(str).str.strip()
    missing_columns = set(SAFETY_COLUMNS.values()) - set(raw.columns)
    if missing_columns:
        raise ValueError(f"Tesla CSV is missing required columns: {', '.join(sorted(missing_columns))}")

    quality = {
        "source_rows": len(raw),
        "exact_duplicate_rows": int(raw.duplicated().sum()),
        "fully_empty_rows": int(raw.isna().all(axis=1).sum()),
    }
    data = raw.drop_duplicates().dropna(how="all").copy()
    data["date"] = pd.to_datetime(data[SAFETY_COLUMNS["date"]], errors="coerce")
    data["case_id"] = pd.to_numeric(data[SAFETY_COLUMNS["case"]], errors="coerce")
    quality["rows_without_case_or_valid_date"] = int(
        (data["case_id"].isna() | data["date"].isna()).sum()
    )
    data = data.loc[data["case_id"].notna() & data["date"].notna()].copy()
    quality["duplicate_case_ids_removed"] = int(data["case_id"].duplicated().sum())
    data = data.drop_duplicates(subset="case_id", keep="first")
    for output_name, source_column in SAFETY_COLUMNS.items():
        if output_name in {"case", "date"}:
            continue
        data[output_name] = _clean_text(data[source_column])

    for field in COUNT_FIELDS:
        normalized = data[field].astype("string").str.strip().replace(
            {"-": "0", "—": "0", "": pd.NA}
        )
        numeric = pd.to_numeric(normalized, errors="coerce")
        quality[f"{field}_invalid_numeric_values"] = int(numeric.isna().sum())
        data[field] = numeric.fillna(0).clip(lower=0)
    quality["clean_accident_records"] = len(data)
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["month_name"] = data["date"].dt.strftime("%b")
    data["day_of_week"] = data["date"].dt.day_name()
    data["state"] = data["state"].replace({"Unknown": "Not recorded"})
    data["country"] = data["country"].replace({"Unknown": "Not recorded"})
    data["model"] = data["model"].replace({"Unknown": "Not recorded"})
    data["autopilot_claimed_event"] = data["autopilot_claimed"].gt(0)
    data["verified_autopilot_event"] = data["verified_autopilot_deaths"].gt(0)
    data["verified_or_sgo_event"] = data["verified_autopilot_plus_sgo"].gt(0)
    data["multiple_victim_categories"] = (
        data[["driver", "occupant", "other_vehicle", "cyclists_pedestrians"]].gt(0).sum(axis=1) > 1
    )
    description = data["description"].astype(str).str.lower()
    data["collision_with_other_vehicle"] = (
        data["other_vehicle"].gt(0)
        | description.str.contains(
            r"\b(?:hit|struck|collid|crash(?:ed)? into|rear[- ]end|t-bone|head[- ]on|semi|truck|motorcycle|vehicle)\b",
            regex=True,
            na=False,
        )
    )
    data["tesla_related_deaths"] = data["driver"] + data["occupant"]
    return data.reset_index(drop=True), quality


def _top_or_empty(frame: pd.DataFrame, group: str, value: str, name: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[group, name])
    return frame.groupby(group, dropna=False)[value].sum().sort_values(ascending=False).head(20).reset_index(
        name=name
    )


def build_safety_report(
    records: pd.DataFrame, quality: dict[str, int], script_file: str
) -> pd.DataFrame:
    report = GenAIReport("Tesla Autopilot Safety Records — Exploratory Analysis")
    annual = records.groupby("year", as_index=False).agg(
        events=("case_id", "nunique"),
        reported_deaths=("deaths", "sum"),
        verified_autopilot_events=("verified_autopilot_event", "sum"),
        autopilot_claimed_events=("autopilot_claimed_event", "sum"),
        verified_or_sgo_events=("verified_or_sgo_event", "sum"),
    ).sort_values("year")
    monthly = records.groupby(records["date"].dt.to_period("M").astype(str)).agg(
        events=("case_id", "nunique"), reported_deaths=("deaths", "sum")
    ).reset_index(names="month")
    countries = _top_or_empty(records, "country", "deaths", "reported_deaths")
    states = _top_or_empty(records.loc[records["country"].eq("USA")], "state", "deaths", "reported_deaths")
    model_counts = records.groupby("model").agg(
        events=("case_id", "nunique"),
        reported_deaths=("deaths", "sum"),
        verified_autopilot_events=("verified_autopilot_event", "sum"),
    ).reset_index().sort_values("events", ascending=False)

    victim_summary = pd.DataFrame([
        {"victim_category": "Tesla drivers", "reported_count": records["driver"].sum()},
        {"victim_category": "Tesla occupants", "reported_count": records["occupant"].sum()},
        {"victim_category": "Other vehicle occupants", "reported_count": records["other_vehicle"].sum()},
        {"victim_category": "Cyclists / pedestrians", "reported_count": records["cyclists_pedestrians"].sum()},
    ])
    autopilot_counts = pd.DataFrame([
        {"measure": "Autopilot claimed", "events": int(records["autopilot_claimed_event"].sum())},
        {"measure": "Verified Autopilot deaths", "events": int(records["verified_autopilot_event"].sum())},
        {"measure": "Verified + NHTSA SGO records", "events": int(records["verified_or_sgo_event"].sum())},
    ])
    collision_summary = pd.DataFrame([
        {"measure": "Records with other vehicle fatalities", "events": int(records["other_vehicle"].gt(0).sum())},
        {"measure": "Records mentioning / indicating vehicle collision", "events": int(records["collision_with_other_vehicle"].sum())},
        {"measure": "Records with multiple victim categories", "events": int(records["multiple_victim_categories"].sum())},
        {"measure": "Total listed fatalities", "events": int(records["deaths"].sum())},
        {"measure": "Mean reported deaths per accident", "events": round(float(records["deaths"].mean()), 2)},
    ])
    report.grid([
        report.kv("Dataset overview", {
            "source rows": quality["source_rows"],
            "clean accident cases": len(records),
            "date range": f"{records['date'].min().date()} – {records['date'].max().date()}",
            "countries": records["country"].nunique(),
            "US states / territories listed": records.loc[records["country"].eq("USA"), "state"].nunique(),
            "total listed fatalities": int(records["deaths"].sum()),
        }),
        report.table("Data cleaning checks", [
            {"check": name.replace("_", " "), "rows": count} for name, count in quality.items()
        ], rows=len(quality)),
    ])
    report.plots([
        (px.line(annual, x="year", y="events", markers=True, title="Recorded events by year"), "Events by year"),
        (px.line(annual, x="year", y="reported_deaths", markers=True, title="Reported deaths by year"),
         "Fatalities by year"),
        (px.line(monthly, x="month", y="events", title="Recorded events over time"), "Events by month"),
        (px.bar(countries.sort_values("reported_deaths"), x="reported_deaths", y="country",
                orientation="h", title="Reported deaths by country"), "Country distribution"),
        (px.bar(states.sort_values("reported_deaths"), x="reported_deaths", y="state",
                orientation="h", title="Reported deaths by US state"), "US state distribution"),
        (px.bar(model_counts.head(15).sort_values("events"), x="events", y="model",
                orientation="h", color="verified_autopilot_events", title="Events by listed Tesla model"),
         "Tesla model records"),
        (px.bar(victim_summary, x="victim_category", y="reported_count", title="Reported victim categories"),
         "Fatalities by victim category"),
        (px.bar(autopilot_counts, x="measure", y="events", title="Autopilot claim and verification records"),
         "Autopilot verification"),
    ])
    report.full_table("Yearly event and Autopilot-verification summary", annual, rows=len(annual))
    report.full_table("Fatalities by victim category", victim_summary, rows=len(victim_summary))
    report.full_table("Tesla models", model_counts, rows=20)
    report.full_table("Collision and multiple-victim-category summary", collision_summary, rows=len(collision_summary))
    report.full_table("Countries with most reported deaths", countries, rows=20)
    report.full_table("US states with most reported deaths", states, rows=20)
    report.full_text(
        "Interpretation limits",
        "This CSV is an incident/death record compilation, not exposure-normalized crash-rate data. "
        "It does not provide vehicle miles traveled, fleet size, consistent Autopilot use verification, "
        "or complete model/year coverage. Reported and verified/SGO fields are analyzed separately; "
        "they are not proof that Autopilot caused an event. Collision mentions are keyword-assisted "
        "signals, not a manually adjudicated collision classification. Blank, dash, and unparseable "
        "fatality cells are represented as zero after the case-level rows have been validated. "
        "The source file's exact contents and cleaned records are saved alongside the report.",
    )
    report.save(script_file, "tesla_safety_report.html")
    return annual


def run_safety(csv_path: Path, artifact_dir: Path, script_file: str) -> tuple[pd.DataFrame, dict]:
    records, quality = load_safety_data(csv_path)
    output = artifact_dir / "safety_analysis"
    output.mkdir(parents=True, exist_ok=True)
    records.to_csv(output / "cleaned_tesla_death_records.csv", index=False)
    annual = build_safety_report(records, quality, script_file)
    annual.to_csv(output / "annual_safety_summary.csv", index=False)
    pd.DataFrame([{"check": key, "count": value} for key, value in quality.items()]).to_csv(
        output / "data_quality_summary.csv", index=False
    )
    return records, quality


def safe_extract_images(archive: Path, destination: Path) -> list[Path]:
    if not archive.is_file():
        raise FileNotFoundError(f"Vehicle image archive not found: {archive}")
    marker = destination / ".extraction_complete"
    destination.mkdir(parents=True, exist_ok=True)
    if not marker.is_file():
        root = destination.resolve()
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                member = PurePosixPath(info.filename)
                if info.is_dir() or member.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                if "__MACOSX" in member.parts:
                    continue
                output = (destination / Path(*member.parts)).resolve()
                if os.path.commonpath([str(root), str(output)]) != str(root):
                    raise ValueError(f"Unsafe path in vehicle image archive: {info.filename}")
                output.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as source, output.open("wb") as target:
                    shutil.copyfileobj(source, target)
        marker.write_text("complete", encoding="utf-8")

    images = []
    corrupt = []
    for path in destination.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            images.append(path)
        except (OSError, UnidentifiedImageError):
            corrupt.append(path)
    if corrupt:
        raise ValueError(
            f"Found {len(corrupt)} unreadable image(s), for example: {corrupt[0]}"
        )
    if not images:
        raise ValueError(f"No usable images found in {destination}.")
    return sorted(images)


def create_image_manifests(
    images: list[Path], artifact_dir: Path, seed: int = 42
) -> pd.DataFrame:
    """Create reproducible train/validation/test manifests for unlabeled images."""
    shuffled = list(images)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * 0.70)
    validation_end = train_end + int(total * 0.15)
    assignments = [
        ("train", shuffled[:train_end]),
        ("validation", shuffled[train_end:validation_end]),
        ("test", shuffled[validation_end:]),
    ]
    rows = []
    object_root = artifact_dir / "object_detection"
    for split, split_images in assignments:
        split_dir = object_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        rows.extend(
            {"image_path": str(path.resolve()), "split": split, "annotations_available": False}
            for path in split_images
        )
        pd.DataFrame([row for row in rows if row["split"] == split]).to_csv(
            split_dir / "images_manifest.csv", index=False
        )
    return pd.DataFrame(rows)


def load_vehicle_detector():
    try:
        import torch
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_Weights,
            fasterrcnn_resnet50_fpn,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Pretrained vehicle detection requires compatible torch and torchvision packages. "
            "Install Module-5/GENAI/projects/requirements-autonomous.txt."
        ) from exc

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights).to("cpu").eval()
    return torch, model, weights.transforms(), weights.meta["categories"]


def detect_vehicles(
    image_path: Path, model, transform, categories: list[str], torch, threshold: float = 0.5
) -> tuple[Image.Image, list[dict]]:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    tensor = transform(image)
    with torch.inference_mode():
        prediction = model([tensor])[0]

    draw = ImageDraw.Draw(image)
    detections = []
    for box, label_id, score in zip(
        prediction["boxes"].cpu().tolist(),
        prediction["labels"].cpu().tolist(),
        prediction["scores"].cpu().tolist(),
    ):
        label = categories[label_id]
        if label.lower() not in VEHICLE_LABELS or score < threshold:
            continue
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        draw.rectangle((x1, y1, x2, y2), outline="red", width=max(2, image.width // 250))
        caption = f"{label} {score:.2f}"
        try:
            bounds = draw.textbbox((x1, y1), caption)
            draw.rectangle(bounds, fill="red")
        except AttributeError:
            pass
        draw.text((x1, y1), caption, fill="white", font=ImageFont.load_default())
        detections.append({
            "image": str(image_path),
            "label": label,
            "confidence": float(score),
            "x_min": x1,
            "y_min": y1,
            "x_max": x2,
            "y_max": y2,
        })
    return image, detections


def run_detection(
    archive: Path,
    artifact_dir: Path,
    script_file: str,
    image_path: Path | None = None,
    sample_count: int = 4,
    threshold: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    if not 0 < threshold <= 1:
        raise ValueError("Detection confidence threshold must be greater than 0 and at most 1.")
    if sample_count < 1:
        raise ValueError("Detection sample count must be positive.")
    output = artifact_dir / "object_detection"
    image_root = output / "images"
    images = safe_extract_images(archive, image_root)
    manifest = create_image_manifests(images, artifact_dir, seed)
    samples = [image_path] if image_path else random.Random(seed).sample(images, min(sample_count, len(images)))
    for sample in samples:
        if not sample.is_file():
            raise FileNotFoundError(f"Inference image not found: {sample}")
        try:
            with Image.open(sample) as image:
                image.verify()
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"Unreadable inference image {sample}: {exc}") from exc

    torch, model, transform, categories = load_vehicle_detector()
    all_detections = []
    report = GenAIReport("Pretrained Vehicle Object Detection")
    report.grid([
        report.kv("Inference setup", {
            "images in supplied archive": len(images),
            "inference images": len(samples),
            "detector": "Faster R-CNN ResNet-50 FPN (COCO pretrained)",
            "device": "CPU",
            "confidence threshold": threshold,
            "custom training": "not run; archive has no labels or boxes",
        }),
        report.table("Unlabeled image split manifests", [
            {"split": name, "images": int((manifest["split"] == name).sum())}
            for name in ("train", "validation", "test")
        ]),
    ])
    cards = []
    for index, sample in enumerate(samples):
        annotated, detections = detect_vehicles(sample, model, transform, categories, torch, threshold)
        all_detections.extend(detections)
        target = output / "inference" / f"{index:03d}_{sample.stem}_detections.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        annotated.save(target, quality=90)
        from io import BytesIO
        import base64

        buffer = BytesIO()
        annotated.thumbnail((700, 500))
        annotated.save(buffer, format="JPEG", quality=80)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        caption = f"{len(detections)} vehicle(s): " + ", ".join(
            f"{item['label']} ({item['confidence']:.2f})" for item in detections
        )
        cards.append(
            f'<div><h3>{html.escape(sample.name)}</h3><p>{html.escape(caption)}</p>'
            f'<img alt="vehicle detections" src="data:image/jpeg;base64,{encoded}" '
            'style="max-width:100%;height:auto"></div>'
        )
    detections_frame = pd.DataFrame(
        all_detections,
        columns=["image", "label", "confidence", "x_min", "y_min", "x_max", "y_max"],
    )
    detections_frame.to_csv(output / "detections.csv", index=False)
    report.full("Annotated vehicle-detection samples", "".join(cards) or "<p>No sample images processed.</p>")
    report.full_table("Detected vehicle boxes", detections_frame, rows=100)
    report.full_text(
        "Evaluation limitation",
        "The supplied Images.zip contains images only; it has no class labels or bounding-box annotations. "
        "Therefore this run uses a pretrained COCO detector for inference and visualization, and does not "
        "train or compute ground-truth detection metrics. The train/validation/test CSVs are image manifests "
        "only and must not be treated as labeled splits. Add vehicle bounding-box annotations before "
        "supervised training or accuracy/mAP evaluation.",
    )
    report.save(script_file, "vehicle_detection_report.html")
    return detections_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("safety", "detect", "all"), default="safety")
    parser.add_argument("--safety-csv", type=Path)
    parser.add_argument("--image-archive", type=Path)
    parser.add_argument("--image", type=Path, help="Run detection on one image instead of random archive samples.")
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(os.getenv(
        "GENAI_AUTONOMOUS_DATA_DIR",
        str(GenAIConfig.from_env().dataset_dir),
    )).expanduser()
    artifact_dir = GenAIConfig.from_env().artifact_dir / "autonomous_driving"
    if args.task in {"safety", "all"}:
        csv_path = args.safety_csv or data_root / DEFAULT_SAFETY_CSV
        run_safety(csv_path, artifact_dir, __file__)
    if args.task in {"detect", "all"}:
        archive = args.image_archive or data_root / DEFAULT_IMAGE_ARCHIVE
        run_detection(
            archive, artifact_dir, __file__,
            image_path=args.image,
            sample_count=args.sample_count,
            threshold=args.confidence,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
