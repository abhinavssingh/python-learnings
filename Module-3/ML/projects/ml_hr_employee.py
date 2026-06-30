"""
HR Employee Turnover Pipeline (Modular & Framework-Aligned)
============================================================

Unified ML workflow leveraging the framework architecture:
- UnsupervisedModelUtility: KMeans + DBSCAN clustering (left employees)
- ClassificationModelUtility: Logistic Regression, RandomForest, GradientBoosting (turnover prediction)
- VisualizerEngine: Unified visualization of all models + artifacts
- ResultBuilder: Standardized result schema across experiments
- Framework handles: SMOTE imbalance, CV evaluation, metric computation, artifact generation

Key Improvements:
✓ Modular section-based design (data quality → clustering → classification → risk scoring)
✓ Eliminates manual metric computation (framework handles it)
✓ Single visualization engine for all models
✓ Clean separation of domain logic from framework operations
✓ Loosely coupled components following README architecture
"""

import time

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split

from lib.html import HtmlBuilder, PlotRenderer
from lib.utility.dataframe.data_loader import DataLoader as dl
from lib.utility.dataframe.df_helper import DataFrameHelper as dfh
from lib.utility.machinelearning.facade.ClassificationModelUtility import ClassificationModelUtility
from lib.utility.machinelearning.facade.UnsupervisedModelUtility import UnsupervisedModelUtility
from lib.utility.machinelearning.visualization.core.VisualizerEngine import VisualizerEngine
from lib.utility.reports.report_utils import ReportUtils as ru

# ============================================================================
# DOMAIN LOGIC: Retention Risk Functions
# ============================================================================


def _risk_zone(probability):
    """Map turnover probability to retention risk zones."""
    if probability < 0.20:
        return "Safe Zone"
    if probability <= 0.60:
        return "Low Risk Zone"
    if probability <= 0.90:
        return "Medium Risk Zone"
    return "High Risk Zone"


def _risk_recommendation(zone):
    """HR retention recommendations by risk zone."""
    recommendations = {
        "Safe Zone": "Continue engagement, recognition, and career development programs.",
        "Low Risk Zone": "Schedule periodic check-ins and monitor emerging concerns.",
        "Medium Risk Zone": "Manager intervention, career planning, workload and compensation review.",
        "High Risk Zone": "Immediate HR retention interview and compensation/role redesign assessment.",
    }
    return recommendations.get(zone, "No recommendation available")


def _cluster_pattern(row):
    """Interpret clustering results by satisfaction + evaluation levels."""
    sat = row["avg_satisfaction"]
    eva = row["avg_evaluation"]

    if sat < 0.45 and eva >= 0.70:
        return "High performers with low satisfaction"
    if sat < 0.45 and eva < 0.70:
        return "Disengaged and lower performing"
    if sat >= 0.45 and eva >= 0.70:
        return "Engaged high performers"
    return "Moderately satisfied with lower evaluation"


# ============================================================================
# SECTION 1: DATA LOADING & QUALITY ASSESSMENT
# ============================================================================

def section_data_quality(df):
    """
    Generate data quality summary and missing value analysis.

    Returns dict with:
    - df_info: DataFrame structure info
    - missing_report: Missing value summary
    - quality_summary: Row/column/duplicate counts
    """
    df_info = dfh.get_dataframe_info_str(df)

    missing_counts = df.isna().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)

    missing_report_df = pd.DataFrame({
        "column": df.columns,
        "missing_count": missing_counts.values,
        "missing_pct": missing_pct.values,
    }).sort_values("missing_pct", ascending=False)

    quality_summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "total_missing_values": int(missing_counts.sum()),
    }

    return {
        "df_info": df_info,
        "missing_report": missing_report_df,
        "quality_summary": quality_summary,
    }


# ============================================================================
# SECTION 2: UNSUPERVISED CLUSTERING (Left Employees Only)
# ============================================================================

def section_unsupervised_clustering(df):
    """
    Cluster employees who left using satisfaction_level + last_evaluation.

    Framework handles: Model registry, pipeline construction, metric computation,
    artifact generation (silhouette scores, cluster labels, etc.)

    Domain logic: Interpret clusters as behavioral patterns

    Returns dict with:
    - um: UnsupervisedModelUtility instance (contains results + artifacts)
    - cluster_data: Cluster assignments + summary
    - visualizations: Cluster scatter plot
    """
    # Extract cluster features
    required_cols = ["satisfaction_level", "last_evaluation", "left"]
    employee_subset = df[required_cols].copy()
    left_employees = employee_subset[employee_subset["left"] == 1].copy()

    X_left = left_employees[["satisfaction_level", "last_evaluation"]].copy()

    # Initialize unsupervised utility
    # Framework builds reusable preprocessor once
    um = UnsupervisedModelUtility(X=X_left)
    um.prepare_data()

    # Run clustering models (Framework: registry → wrapper → evaluation)
    um.run_experiment("KMeans")      # Default n_clusters=3
    um.run_experiment("DBSCAN")

    # Extract KMeans labels for domain logic
    kmeans_labels = um.get_labels("KMeans")
    left_cluster_df = left_employees.copy().reset_index(drop=True)
    left_cluster_df["cluster"] = pd.Series(kmeans_labels).astype(int)

    # Compute cluster behavioral patterns
    cluster_summary = (
        left_cluster_df
        .groupby("cluster", as_index=False)
        .agg(
            employee_count=("cluster", "count"),
            avg_satisfaction=("satisfaction_level", "mean"),
            avg_evaluation=("last_evaluation", "mean"),
        )
    )
    cluster_summary["avg_satisfaction"] = cluster_summary["avg_satisfaction"].round(4)
    cluster_summary["avg_evaluation"] = cluster_summary["avg_evaluation"].round(4)
    cluster_summary["behavioral_pattern"] = cluster_summary.apply(_cluster_pattern, axis=1)

    # Cluster scatter visualization
    cluster_scatter_fig = px.scatter(
        left_cluster_df,
        x="satisfaction_level",
        y="last_evaluation",
        color=left_cluster_df["cluster"].astype(str),
        title="KMeans Employee Clusters (Left = 1)",
        labels={
            "color": "Cluster",
            "satisfaction_level": "Satisfaction Level",
            "last_evaluation": "Last Evaluation",
        },
    )

    return {
        "um": um,
        "employee_subset": employee_subset,
        "left_cluster_df": left_cluster_df,
        "cluster_summary": cluster_summary,
        "cluster_scatter_fig": cluster_scatter_fig,
    }


# ============================================================================
# SECTION 3: CLASSIFICATION PIPELINE (Turnover Prediction)
# ============================================================================

def section_classification_pipeline(df):
    """
    Classification workflow using framework.

    Framework handles:
    - Data preprocessing (categorical encoding)
    - Train/test split
    - SMOTE imbalance (classification only)
    - Model registry + wrapper pipeline construction
    - Cross-validation + metric evaluation
    - Artifact generation (ROC, PR, confusion matrix, etc.)
    - ResultBuilder standardization

    Domain logic: Apply risk zones to predictions (done in risk_scoring section)

    Returns dict with:
    - cls_util: ClassificationModelUtility instance
    - results_df: Standardized results (framework ResultBuilder schema)
    - best_model: Best performing model info
    - feature/split info
    """

    # Extract features and target
    y = df["left"].copy()
    X = df.drop(columns=["left"]).copy()

    # Encoding strategy
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Manual train/test split (framework API constraint)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=123, stratify=y
    )

    # Initialize classification utility with SMOTE imbalance handling
    cls_util = ClassificationModelUtility(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        imbalance_config={
            "type": "smote",
            "params": {"random_state": 123},
        },
    )
    cls_util.prepare_data()  # Framework builds reusable preprocessor

    # Run baseline models
    # Framework: registry → wrapper → pipeline (preprocessor → SMOTE → model)
    # → fit → evaluate → ResultBuilder standardization
    model_names = [
        "LogisticRegression",
        "RandomForestClassifier",
        "GradientBoosting",
    ]

    for model_name in model_names:
        cls_util.run_experiment(model_name)

    # Get standardized results (ResultBuilder schema)
    results_df = cls_util.get_results_df()
    best_model = cls_util.get_best_model(metric="recall_weighted")

    return {
        "cls_util": cls_util,
        "results_df": results_df,
        "best_model": best_model,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "encoded_feature_count": X_encoded.shape[1],
    }


# ============================================================================
# SECTION 4: RETENTION RISK SCORING & HR RECOMMENDATIONS
# ============================================================================

def section_risk_scoring(cls_util, best_model, X_test, y_test):
    """
    Apply domain logic to best model predictions.

    Extracts probabilities from best model and applies:
    - Risk zone classification (Safe/Low/Medium/High)
    - HR retention recommendations
    - Segment summary for HR action planning

    Returns dict with:
    - risk_scoring_df: Predictions + risk zones + recommendations
    - risk_segment_summary: Summary by risk zone
    """
    risk_scoring_df = pd.DataFrame()
    risk_segment_summary = pd.DataFrame()

    if best_model and "experiment" in best_model:
        best_exp_id = best_model["experiment"]
        if best_exp_id in cls_util.trained_models:
            best_wrapper = cls_util.trained_models[best_exp_id]["wrapper"]
            best_proba = best_wrapper.predict_proba(X_test)
            best_scores = (
                best_proba[:, 1]
                if best_proba is not None and best_proba.ndim > 1
                else np.zeros(len(X_test))
            )

            # Build risk scoring dataset
            risk_scoring_df = X_test.copy()
            risk_scoring_df["left_actual"] = y_test.values
            risk_scoring_df["turnover_probability"] = np.round(best_scores, 4)
            risk_scoring_df["risk_zone"] = risk_scoring_df["turnover_probability"].apply(_risk_zone)
            risk_scoring_df["recommendation"] = risk_scoring_df["risk_zone"].apply(
                _risk_recommendation
            )

            # HR segment summary
            risk_segment_summary = (
                risk_scoring_df
                .groupby("risk_zone", as_index=False)
                .agg(
                    employee_count=("risk_zone", "count"),
                    avg_probability=("turnover_probability", "mean"),
                )
            )
            risk_segment_summary["avg_probability"] = (
                risk_segment_summary["avg_probability"].round(4)
            )

    return {
        "risk_scoring_df": risk_scoring_df,
        "risk_segment_summary": risk_segment_summary,
    }


# ============================================================================
# SECTION 5: EDA VISUALIZATIONS
# ============================================================================

def section_eda_visualizations(df):
    """
    Generate exploratory data analysis visualizations.

    Returns dict with histogram, correlation, and pair plot figures
    """
    hist_fig_1 = px.histogram(
        df, x="salary", opacity=0.7, barmode="overlay",
        title="Salary Distribution"
    )

    hist_fig_2 = px.histogram(
        df, x="satisfaction_level", marginal="box", opacity=0.7, barmode="overlay",
        title="Satisfaction Level Distribution"
    )

    hist_fig_3 = px.histogram(
        df, x="last_evaluation", marginal="box", opacity=0.7, barmode="overlay",
        title="Last Evaluation Distribution"
    )

    hist_fig_4 = px.histogram(
        df, x="average_montly_hours", marginal="box", opacity=0.7, barmode="overlay",
        title="Average Monthly Hours Distribution"
    )

    hist_fig_5 = px.histogram(
        df, x="number_project", color="left", barmode="group",
        title="Employee Project Count by Attrition"
    )

    # Correlation matrix
    num_columns = df.select_dtypes(include=['number']).columns
    corr_fig = px.imshow(
        df[num_columns].corr(),
        text_auto=True,
        color_continuous_scale='RdBu',
        title='Correlation Matrix'
    )

    label_map = {
        "number_project": "No of Projects",
        "average_montly_hours": "Avg Mnthly Hours",
        "last_evaluation": "Last Evaluation",
        "satisfaction_level": "Satisfaction Level",
        "time_spend_company": "No of Years",
        "Work_accident": "Work Accident",
        "promotion_last_5years": "Promo LstY5rs",
        "left": "Left Company",
        "salary": "Salary"
    }

    corr_fig.update_xaxes(
        ticktext=[label_map.get(col, col) for col in num_columns],
        tickvals=list(range(len(num_columns)))
    )
    corr_fig.update_yaxes(
        ticktext=[label_map.get(col, col) for col in num_columns],
        tickvals=list(range(len(num_columns)))
    )

    # Pair plot
    pair_plot_fig = px.scatter_matrix(
        df[["number_project", "average_montly_hours", "last_evaluation"]],
        title='Pair Plot of Numerical Features'
    )

    return {
        "hist_salary": hist_fig_1,
        "hist_satisfaction": hist_fig_2,
        "hist_evaluation": hist_fig_3,
        "hist_hours": hist_fig_4,
        "hist_projects": hist_fig_5,
        "corr_matrix": corr_fig,
        "pair_plot": pair_plot_fig,
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Unified ML Pipeline: Unsupervised Clustering + Classification-based Risk Scoring

    Workflow:
    1. Load data + quality assessment
    2. Unsupervised clustering on left employees (KMeans, DBSCAN)
    3. Classification on turnover prediction (LogisticRegression, RF, GradientBoosting)
    4. Apply domain logic: risk zones + HR recommendations
    5. Unified visualization via VisualizerEngine
    6. HTML report generation
    """
    print("Running HR Employee Pipeline (Framework-Based)...")
    start_time = time.perf_counter()

    content = []
    builder = HtmlBuilder()
    plot_renderer = PlotRenderer()

    # ================================================================
    # LOAD DATA
    # ================================================================
    df, report = dl.read_dataset(
        "HR_comma_sep.csv",
        optimize=True,
        handle_unnamed="drop",
        return_report=True
    )

    # ================================================================
    # SECTION 1: DATA QUALITY
    # ================================================================
    print("[1/5] Data Quality Assessment...")
    quality_data = section_data_quality(df)

    content.append(
        builder.full_width_card(
            "Original HR Employee Data",
            builder.render_dataframe_collapsible(df, initial_rows=15)
        )
    )

    content.append(builder.grid([
        builder.card("Dataframe Info", builder.render_pre(quality_data["df_info"])),
        builder.card("Dataframe Description", builder.render_dict(df.describe().to_dict())),
        builder.card("Dataframe Optimization Report", builder.render_pre(report)),
        builder.card("Data Quality Summary", builder.render_dict(quality_data["quality_summary"])),
        builder.card("Missing Value Report", builder.render_dataframe(quality_data["missing_report"])),
    ]))

    # ================================================================
    # SECTION 2: UNSUPERVISED CLUSTERING
    # ================================================================
    print("[2/5] Unsupervised Clustering (Left Employees)...")
    cluster_data = section_unsupervised_clustering(df)
    um = cluster_data["um"]

    # Get framework results (standardized schema)
    unsupervised_results_df = um.get_results_df()
    unsupervised_plot_data = um.get_plot_data()

    # Visualization via VisualizerEngine (framework)
    viz = VisualizerEngine(um.results, unsupervised_plot_data)
    # dashboard = viz.render_all()

    content.append(builder.grid([
        builder.card(
            "Selected Columns for Clustering",
            builder.render_dataframe_collapsible(cluster_data["employee_subset"], initial_rows=15)
        ),
        builder.card(
            "KMeans Cluster Assignments (Left Employees)",
            builder.render_dataframe_collapsible(cluster_data["left_cluster_df"], initial_rows=20)
        ),
        builder.card(
            "Cluster Summary & Behavioral Patterns",
            builder.render_dataframe(cluster_data["cluster_summary"])
        ),
        builder.card("Unsupervised Results (Framework StandardizedSchema)",
                     builder.render_dataframe(unsupervised_results_df)),
    ]))

    # Framework visualizations
    content.append(builder.chart_grid([
        # plot_renderer.plot_to_card(dashboard.get("comparison"), "Unsupervised Model Comparison"),
        # plot_renderer.plot_to_card(dashboard.get("ranking"), "Unsupervised Model Ranking"),
        # plot_renderer.plot_to_card(dashboard.get("best_model"), "Best Unsupervised Model"),
        # plot_renderer.plot_to_card(dashboard.get("distribution"), "Metric Distribution"),
        plot_renderer.plot_to_card(cluster_data["cluster_scatter_fig"], "KMeans Cluster Visualization"),
    ]))

    # ================================================================
    # SECTION 3: CLASSIFICATION PIPELINE
    # ================================================================
    print("[3/5] Classification Pipeline (Turnover Prediction)...")
    class_data = section_classification_pipeline(df)
    cls_util = class_data["cls_util"]

    # Get framework standardized results
    class_results_df = class_data["results_df"]
    best_model = class_data["best_model"]

    content.append(builder.grid([
        builder.card("Preprocessing Summary", builder.render_dict({
            "numeric_columns": class_data["numeric_cols"],
            "categorical_columns": class_data["categorical_cols"],
            "encoded_features": class_data["encoded_feature_count"],
            "train_shape": list(class_data["X_train"].shape),
            "test_shape": list(class_data["X_test"].shape),
        })),
        builder.card(
            "Classification Results (Framework Standardized Schema)",
            builder.render_dataframe(class_results_df.drop(columns=["artifacts"], errors="ignore"))
        ),
        builder.card("Best Model Summary", builder.render_dict(best_model or {})),
    ]))

    # Framework classification visualizations
    viz_classification = VisualizerEngine(cls_util.results, cls_util.artifacts)
    dashboard_classification = viz_classification.render_all()

    content.append(builder.chart_grid([
        plot_renderer.plot_to_card(dashboard_classification.get("comparison"),
                                   "Classification Model Comparison"),
        plot_renderer.plot_to_card(dashboard_classification.get("ranking"),
                                   "Model Ranking (Recall-Weighted)"),
        plot_renderer.plot_to_card(dashboard_classification.get("best_model"),
                                   "Best Model Performance"),
        plot_renderer.plot_to_card(dashboard_classification.get("distribution"),
                                   "Metric Distribution"),
        *[
            plot_renderer.plot_to_card(fig, f"Classification {title.title()}")
            for title, fig in dashboard_classification.get("task_specific", {}).items()
        ],
    ]))

    # ================================================================
    # SECTION 4: RETENTION RISK SCORING (Domain Logic)
    # ================================================================
    print("[4/5] Retention Risk Scoring & Recommendations...")
    risk_data = section_risk_scoring(
        cls_util,
        best_model,
        class_data["X_test"],
        class_data["y_test"]
    )

    metric_rationale = (
        "Recall is prioritized for turnover prediction because false negatives "
        "(employees likely to leave but predicted as stay) are more costly for retention planning."
    )

    content.append(builder.grid([
        builder.card("Metric Selection Rationale", builder.render_pre(metric_rationale)),
        builder.card("Risk Segment Summary (HR Action Planning)",
                     builder.render_dataframe(risk_data["risk_segment_summary"])),
        builder.card(
            "Turnover Probability Scoring (Sample Predictions)",
            builder.render_dataframe_collapsible(risk_data["risk_scoring_df"].head(50), initial_rows=15)
        ),
    ]))

    # ================================================================
    # SECTION 5: EDA VISUALIZATIONS
    # ================================================================
    print("[5/5] EDA Visualizations...")
    eda_figs = section_eda_visualizations(df)

    content.append(builder.chart_grid([
        plot_renderer.plot_to_card(eda_figs["hist_salary"], "Salary Distribution"),
        plot_renderer.plot_to_card(eda_figs["hist_satisfaction"], "Satisfaction Level Distribution"),
        plot_renderer.plot_to_card(eda_figs["hist_evaluation"], "Last Evaluation Distribution"),
        plot_renderer.plot_to_card(eda_figs["hist_hours"], "Average Monthly Hours Distribution"),
        plot_renderer.plot_to_card(eda_figs["hist_projects"], "Project Count by Attrition"),
        plot_renderer.plot_to_card(eda_figs["corr_matrix"], "Correlation Matrix"),
        plot_renderer.plot_to_card(eda_figs["pair_plot"], "Pair Plot of Features"),
    ]))

    # ================================================================
    # FINAL HTML REPORT
    # ================================================================
    html_doc = builder.build_page(
        "ML HR Employee Pipeline Report",
        "\n".join(content)
    )

    output_path = ru.save_html_report(
        __file__,
        "ml_hr_employee_pipeline_report.html",
        html_doc,
        subfolder="reports",
        open_in_browser=True
    )

    print(f"✓ Report saved: {output_path}")

    end_time = time.perf_counter()
    print(f"✓ Execution time: {end_time - start_time:.2f}s")


if __name__ == "__main__":
    main()
