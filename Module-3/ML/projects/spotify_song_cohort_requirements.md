# Spotify Song Cohort Analysis – Requirements Document

## 1. Project Overview

### Problem Scenario
Customers expect highly personalized recommendations across digital platforms such as Spotify, Netflix, and e-commerce websites. To improve engagement and content discovery, Spotify aims to create cohorts of songs with similar characteristics.

This project focuses on analyzing Spotify data for Rolling Stones albums and songs. Using exploratory data analysis, feature engineering, dimensionality reduction, and clustering techniques, the objective is to identify meaningful song groups that can support recommendation strategies.

---

## 2. Business Objective

As a Data Scientist, the objective is to:

- Perform data quality assessment and cleaning.
- Conduct exploratory data analysis (EDA).
- Understand the factors that influence song popularity.
- Identify album-level popularity trends.
- Apply dimensionality reduction techniques.
- Discover song cohorts using clustering algorithms.
- Define and interpret song clusters.
- Generate insights to support music recommendation systems.

---

## 3. Dataset Description

### Data Source
Spotify API dataset containing Rolling Stones albums and tracks.

### Key Assumption
Each song has a unique Spotify ID.

### Data Dictionary

| Column | Description |
|----------|-------------|
| name | Song name |
| album | Album name |
| release_date | Album release date |
| track_number | Track order in album |
| id | Unique Spotify song identifier |
| uri | Spotify URI |
| acousticness | Probability that the track is acoustic |
| danceability | Suitability of a track for dancing |
| energy | Perceived intensity and activity level |
| instrumentalness | Probability that the track contains no vocals |
| liveness | Probability that the track was performed live |
| loudness | Average loudness in decibels |
| speechiness | Presence of spoken words |
| tempo | Tempo in beats per minute |
| valence | Musical positivity score |
| popularity | Popularity score (0-100) |
| duration_ms | Song duration in milliseconds |

---

# 4. Functional Requirements

## FR-1 Initial Data Inspection and Data Cleaning

### Objective
Assess data quality and prepare the dataset for analysis.

### FR-1.1 Dataset Inspection

#### Tasks
- Load the dataset.
- Identify data structure and schema.
- Review data types.
- Verify record counts.

### Deliverables
- Dataset summary report.
- Data profiling output.

---

### FR-1.2 Missing Value Analysis

#### Tasks
- Identify missing values.
- Calculate missing value percentages.
- Recommend treatment strategies.

### Deliverables
- Missing value report.
- Data quality assessment.

---

### FR-1.3 Duplicate Analysis

#### Tasks
- Detect duplicate records.
- Detect duplicate song IDs.
- Remove duplicates where appropriate.

### Deliverables
- Duplicate analysis report.

---

### FR-1.4 Outlier Analysis

#### Tasks
- Analyze numerical distributions.
- Identify extreme values.
- Review abnormal popularity, duration, tempo, and loudness values.

### Deliverables
- Outlier report.
- Outlier treatment recommendations.

---

### FR-1.5 Data Refinement

#### Tasks
- Remove invalid records.
- Handle missing values.
- Standardize data types.
- Create clean analytical dataset.

### Deliverables
- Refined dataset.

---

## FR-2 Exploratory Data Analysis and Feature Engineering

### Objective
Understand song characteristics and popularity drivers.

---

### FR-2.1 Album Recommendation Analysis

#### Objective
Identify albums that should be recommended based on song popularity.

#### Tasks
- Calculate popular song counts per album.
- Define popularity thresholds.
- Rank albums by popular songs.
- Visualize album performance.

#### Deliverables
- Album ranking table.
- Top 2 recommended albums.
- Supporting visualizations.
- Business recommendations.

---

### FR-2.2 Song Feature Exploration

#### Tasks
Analyze distributions for:

- acousticness
- danceability
- energy
- instrumentalness
- liveness
- loudness
- speechiness
- tempo
- valence
- popularity
- duration_ms

#### Recommended Visualizations
- Histograms
- Box plots
- Density plots
- Violin plots

#### Deliverables
- Feature distribution report.
- Observed patterns and insights.

---

### FR-2.3 Popularity Analysis

#### Objective
Understand what drives song popularity.

#### Tasks
- Compare popularity against audio features.
- Study popularity trends over time.
- Evaluate popularity versus album release date.
- Identify influential music characteristics.

#### Recommended Visualizations
- Scatter plots
- Trend lines
- Correlation heatmaps
- Pair plots

#### Deliverables
- Popularity driver analysis.
- Key findings and recommendations.

---

### FR-2.4 Feature Engineering

#### Tasks
Create analytical features such as:

- Song age
- Album age
- Popularity categories
- Tempo categories
- Duration categories

#### Deliverables
- Engineered feature dataset.

---

### FR-2.5 Correlation Analysis

#### Tasks
- Calculate numerical feature correlations.
- Identify highly correlated variables.
- Detect redundant features.

#### Deliverables
- Correlation matrix.
- Correlation heatmap.
- Feature relationship insights.

---

## FR-3 Dimensionality Reduction Analysis

### Objective
Reduce feature complexity while preserving meaningful information.

---

### FR-3.1 PCA Analysis

#### Tasks
- Standardize numerical features.
- Apply Principal Component Analysis (PCA).
- Calculate explained variance.
- Determine optimal number of components.

#### Deliverables
- Explained variance chart.
- PCA component analysis.
- Reduced feature representation.

---

### FR-3.2 Dimensionality Reduction Insights

#### Tasks
Explain:

- Why dimensionality reduction is needed.
- Impact on clustering quality.
- Feature compression benefits.
- Information retention trade-offs.

#### Deliverables
- Dimensionality reduction summary.
- PCA observations.
- Business interpretation.

---

## FR-4 Cluster Analysis

### Objective
Create meaningful song cohorts.

---

### FR-4.1 Determine Optimal Number of Clusters

#### Tasks
Evaluate cluster counts using:

- Elbow Method
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score

#### Deliverables
- Cluster selection report.
- Optimal cluster recommendation.

---

### FR-4.2 Apply Clustering Algorithms

#### Candidate Algorithms
- K-Means
- Hierarchical Clustering
- DBSCAN
- Gaussian Mixture Models (Optional)

#### Tasks
- Train clustering models.
- Compare clustering quality.
- Select best-performing clustering solution.

#### Deliverables
- Model comparison report.
- Best clustering model.

---

### FR-4.3 Cluster Visualization

#### Tasks
Visualize clusters using:

- PCA scatter plots
- Cluster distribution plots
- Feature comparison charts

#### Deliverables
- Cluster visualizations.
- Song cohort maps.

---

### FR-4.4 Cluster Definition and Interpretation

#### Tasks
Define each cluster using:

- Popularity
- Energy
- Danceability
- Acousticness
- Tempo
- Valence
- Instrumentalness

#### Deliverables
For each cluster provide:

- Cohort name
- Cluster size
- Dominant characteristics
- Musical profile
- Recommendation insights

Example:

- High-Energy Popular Rock Tracks
- Acoustic Classics
- Dance-Oriented Tracks
- Live Performance Songs

---

# 5. Technical Requirements

## Data Preparation

- Handle missing values.
- Remove duplicates.
- Standardize numerical features.
- Encode categorical features where required.

## Modeling

- Use standardized numerical features.
- Apply PCA before clustering when appropriate.
- Compare multiple clustering techniques.

## Evaluation Metrics

### Clustering Metrics

- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score
- Inertia (K-Means)

---

# 6. Reporting Requirements

The final solution should include:

1. Data Quality Report
2. Exploratory Data Analysis Report
3. Popularity Analysis Report
4. Feature Engineering Summary
5. PCA Analysis Report
6. Clustering Analysis Report
7. Cluster Definitions
8. Recommendation Insights
9. Executive Summary

---

# 7. Expected Deliverables

## Analytical Deliverables

- Clean dataset
- Feature engineered dataset
- EDA visualizations
- Correlation analysis
- PCA outputs
- Cluster visualizations

## Business Deliverables

- Top recommended albums
- Song cohort definitions
- Popularity insights
- Music recommendation observations

---

# 8. Success Criteria

The project will be considered successful when:

- Data quality issues are identified and resolved.
- Key popularity drivers are explained.
- Top albums are identified using data-driven analysis.
- PCA findings are documented.
- Optimal number of clusters is determined.
- Song cohorts are generated and interpreted.
- Actionable recommendation insights are produced.
- Results can support Spotify recommendation use cases.
