# Tourism Analytics and AI Recommendations

`tourism_ai.py` covers tourism data preparation, demographic and destination
analysis (including the dataset's `Cagar Alam` nature category),
collaborative-filtering recommendations, and an optional historical
structure transfer-learning workflow from the tagged capstone requirements.

## Tourism analytics and recommendations

This project now lives in the capstone folder. Run it from the repository root:

```powershell
.\.venv\Scripts\python.exe Module-5\GENAI\capstone\tourism_ai.py
```

The default task reads `user.csv`, `tourism_with_id.xlsx`, and
`tourism_rating.csv` from `datasets/GEN AI/Capstone 2/Part 2/`. It cleans
duplicates and invalid records, evaluates item-item collaborative filtering
with deterministic per-user holdout metrics, then writes an HTML report,
cleaned ratings, recommendation examples, evaluation metrics, and a reusable
model bundle.

Request recommendations for a specific user or destination:

```powershell
.\.venv\Scripts\python.exe Module-5\GENAI\capstone\tourism_ai.py --user-id 1
.\.venv\Scripts\python.exe Module-5\GENAI\capstone\tourism_ai.py --place-id 1 --top-n 5
```

Configure the source dataset root with `GENAI_TOURISM_DATA_DIR`. The script
uses the GenAI utility's `GenAIConfig`, `TabularLoader`, and `GenAIReport`;
recommendation modelling is classical collaborative filtering and does not
require an LLM.

## Historical-structure image classification (optional)

The image workflow reads `Capstone 2/Part 1/dataset_hist_structures 2.zip`,
safely extracts the labeled training and test folders into
`saved_models/genai/tourism_ai/image_data/`, then trains MobileNetV2 transfer
learning classifiers with and without training-time augmentation.

```powershell
.\.venv\Scripts\python.exe Module-5\GENAI\capstone\tourism_ai.py --task images
```

This task requires TensorFlow compatible with the current Python/OS and an
internet connection on first run to fetch ImageNet MobileNetV2 weights. It is
not installed in the repository's current Python 3.14 environment, so the
default tourism task does not import it. Configure the image training with
`TOURISM_IMAGE_EPOCHS`, `TOURISM_IMAGE_BATCH_SIZE`, `TOURISM_IMAGE_SIZE`, and
`TOURISM_IMAGE_SEED`. The provided test split lacks the `portal` class, which
the image report calls out.

## Outputs

- `Module-5/GENAI/capstone/reports/tourism_ai_report.html`
- `Module-5/GENAI/capstone/reports/historical_structures_report.html` (image task)
- `saved_models/genai/tourism_ai/` (recommender, metrics, recommendation CSVs,
  and optional image models)

Set `GENAI_OPEN_REPORTS=0` to save the HTML reports without opening them.
