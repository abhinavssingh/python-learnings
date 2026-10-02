# Capstone Work

This folder contains the capstone projects and applied data-science solutions used for the GenAI track. It includes the autonomous-driving work that was moved here from the older `projects/` area, alongside the restaurant forecasting and tourism analysis projects.

## Capstone projects

### Autonomous driving analysis and object detection

- `Autonomous_Driving_Capstone_Requirements.md` describes the Tesla incident analysis and object-detection work.
- `autonomous_driving.py` loads the safety dataset and runs pretrained object detection inference for selected images.
- `AUTONOMOUS_DRIVING_README.md` documents setup, command-line options, and output files.
- `requirements-autonomous.txt` holds the extra dependencies for the detection workflow.
- `tesla_safety_analysis.ipynb` and `vehicle_detection.ipynb` are notebook companions for exploration and diagnostics.

Example runs:

```powershell
python Module-5/GENAI/capstone/autonomous_driving.py --task safety
python Module-5/GENAI/capstone/autonomous_driving.py --task detect
```

The work is stored in the capstone area instead of `projects/`.

### Restaurant sales forecasting

- `Sales_Forecasting_Requirements.md` describes the expected ingestion, data validation, feature engineering, model comparison, and future-forecast deliverables.
- `sales_forecasting.py` loads restaurant, item, and sales CSV files, joins and validates them, analyzes sales, compares Linear Regression, Random Forest, and XGBoost, and forecasts daily revenue recursively by restaurant.
- `SALES_FORECASTING_README.md` documents data paths, configuration, outputs, and implementation limits.

Example run from the repository root:

```powershell
python Module-5/GENAI/capstone/sales_forecasting.py
```

The default source files are under `datasets/GEN AI/Capstone 3/`. The script uses a time-ordered validation window (28 days by default) and forecasts 365 days ahead. You can change those windows with `GENAI_SALES_VALIDATION_DAYS` and `GENAI_SALES_HORIZON_DAYS`. Inspect the generated report and saved metrics before deciding which model performs best.

### Tourism analysis and recommendations

- `Tourism_AI_Requirements.md` specifies two workstreams: historical-structure image classification and tourism analysis/recommendations.
- `tourism_ai.py` implements the tourism-data and collaborative-filtering workflow by default, and offers an optional historical-structure image task.
- `TOURISM_AI_README.md` gives the full setup and output details.

Example runs:

```powershell
python Module-5/GENAI/capstone/tourism_ai.py
python Module-5/GENAI/capstone/tourism_ai.py --user-id 1
python Module-5/GENAI/capstone/tourism_ai.py --place-id 1 --top-n 5
python Module-5/GENAI/capstone/tourism_ai.py --task images
```

The default analytics task reads the tourism user, place, and rating datasets under `datasets/GEN AI/Capstone 2/Part 2/`. The image task is optional and requires a compatible TensorFlow environment and pretrained MobileNetV2 weights; consult the project guide first.

## Learner workflow

1. Read the requirement document and turn each requirement into a checkable outcome.
2. Inspect input schemas, missing values, duplicates, and assumptions before modeling.
3. Reproduce the default run and open its HTML report.
4. Trace how features and validation splits are constructed. For forecasting, preserve time order to avoid training on future observations.
5. Compare the saved predictions and metrics with the business question, not just a single headline score.
6. Modify one assumption at a time and document the effect.

## Important limitations

The supplied restaurant items do not include a category column, so category-level results must not be invented. Tourism recommendation metrics depend on the rating data and holdout procedure. The image workflow is a distinct optional task; use the project-specific environment notes. All generated reports and artifacts are described in the linked project READMEs.
