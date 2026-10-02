# Autonomous Driving Capstone

This project now lives under the capstone area. It provides a reproducible
workflow for Tesla incident/fatality exploratory analysis and vehicle-object
detection inference.

## Install and run

Install the CPU-compatible object-detection packages, then run either task:

```powershell
.\.venv\Scripts\python.exe -m pip install -r Module-5\GENAI\capstone\requirements-autonomous.txt
.\.venv\Scripts\python.exe Module-5\GENAI\capstone\autonomous_driving.py --task safety
.\.venv\Scripts\python.exe Module-5\GENAI\capstone\autonomous_driving.py --task detect
```

`safety` is the default and uses
`datasets/GEN AI/Capstone 1/Part 2/Tesla - Deaths.csv`. It cleans summary/footer
rows, normalizes missing values, drops duplicate cases, and creates visual
summaries of dates, years, countries, US states, victim groups, collision
signals, Tesla models, and claimed/verified Autopilot fields.

`detect` extracts
`datasets/GEN AI/Capstone 1/Part 1/Images.zip`, validates images, creates
reproducible train/validation/test **manifests**, then runs pretrained
Torchvision Faster R-CNN (COCO) on four sample images. First use downloads the
pretrained detector weights from PyTorch. Select an image, archive, or dataset
root with:

```powershell
.\.venv\Scripts\python.exe Module-5\GENAI\capstone\autonomous_driving.py --task detect --image C:\path\to\traffic.jpg
.\.venv\Scripts\python.exe Module-5\GENAI\capstone\autonomous_driving.py --task detect --sample-count 8 --confidence 0.65
$env:GENAI_AUTONOMOUS_DATA_DIR = "D:\datasets\GEN AI"
```

The supplied `Images.zip` has 5,626 JPEG images but **no class labels or
bounding-box annotations**. Consequently it cannot support supervised training
or ground-truth detection-accuracy/mAP evaluation. The detector task performs
pretrained inference and visualization only; the split CSVs are image manifests,
not labeled datasets. Add annotated boxes before claiming model training or
evaluation.

## Outputs

All machine-readable artifacts are under
`saved_models/genai/autonomous_driving/`:

- `safety_analysis/cleaned_tesla_death_records.csv`
- `safety_analysis/annual_safety_summary.csv`
- `safety_analysis/data_quality_summary.csv`
- `object_detection/{train,validation,test}/images_manifest.csv`
- `object_detection/inference/` (annotated sample images)
- `object_detection/detections.csv`

HTML reports are saved alongside the script in `capstone/reports/`:
`tesla_safety_report.html` and `vehicle_detection_report.html`.
Set `GENAI_OPEN_REPORTS=0` to avoid opening the reports automatically.

### Safety-data interpretation

These are reported incident records, not exposure-normalized crash rates.
They do not provide fleet miles, reliable vehicle-use denominators, or
consistent Autopilot verification. “Autopilot claimed”, “verified”, and NHTSA
SGO-related fields are reported separately and should not be interpreted as
proof of causation. Collision text matching is an exploratory signal, not a
manually verified label.
