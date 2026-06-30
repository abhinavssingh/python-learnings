# ML Spotify Song Cohort Pipeline

## Script

- `Module-3/ML/projects/ml_spotify.py`

## Overview

This pipeline performs song cohort analysis for Rolling Stones Spotify tracks and produces recommendation-oriented insights:

- Data quality and refinement
- Album recommendation analysis
- Popularity driver analysis
- Feature engineering
- PCA-based dimensionality reduction
- Clustering model comparison and cohort definition
- HTML report generation

## Input Dataset

- `datasets/rolling_stones_spotify.csv`

## Workflow Sections

1. **Data Quality**

- DataFrame profiling
- Missing-value report
- Duplicate row/song-id analysis
- Outlier analysis using `DataFrameHelper.find_iqr_outliers` for:
  - `popularity`
  - `duration_ms`
  - `tempo`
  - `loudness`
- Dataset refinement and type standardization

2. **Album Recommendation Analysis**

- Computes popular song count per album
- Ranks albums and identifies top 2 recommended albums

3. **Feature Engineering**

- `release_year`, `song_age`, `album_age`
- `popularity_category`
- `tempo_category`
- `duration_min`, `duration_category`

4. **Popularity Analysis**

- Correlation with popularity across audio features
- Popularity trend by release year
- Scatter plots with trendlines

5. **PCA Analysis**

- Standardizes features
- Computes explained variance and cumulative variance
- Chooses minimum components that retain >= 90% variance

6. **Clustering Analysis**

- KMeans search for `k=2..8` with evaluation:
  - Inertia
  - Silhouette
  - Davies-Bouldin
  - Calinski-Harabasz
- Compares models:
  - KMeans
  - Agglomerative
  - DBSCAN
- Selects best model by silhouette (when valid)

7. **Cohort Definition**

- Creates cluster-level summary metrics
- Assigns business-friendly cohort names, e.g.:
  - High-Energy Popular Rock Tracks
  - Acoustic Classics
  - Dance-Oriented Tracks
  - Live Performance Songs
  - Balanced Deep Cuts

8. **Report Output**

- Unified HTML report with tables, charts, and executive summary

## Output

- Report file:
  - `Module-3/ML/projects/reports/ml_spotify_song_cohort_pipeline_report.html`

## How To Run

From workspace root:

```powershell
c:/IHFC/python-learnings/.venv/Scripts/python.exe -m Module-3.ML.projects.ml_spotify
```

Alternative:

```powershell
py -m Module-3.ML.projects.ml_spotify
```

## Notes

- Outlier logic is centralized through `lib/utility/dataframe/df_helper.py`.
- If clustering metrics are null for a model, verify number of discovered clusters (e.g., DBSCAN may return 0 or 1 cluster).
