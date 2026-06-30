import time

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from lib.html import HtmlBuilder, PlotRenderer
from lib.utility.dataframe.data_loader import DataLoader as dl
from lib.utility.dataframe.df_helper import DataFrameHelper as dfh
from lib.utility.reports.report_utils import ReportUtils as ru

AUDIO_FEATURES = [
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo",
    "valence",
    "duration_ms",
    "popularity",
]


def _cohort_name(row):
    if row["avg_popularity"] >= 65 and row["avg_energy"] >= 0.65:
        return "High-Energy Popular Rock Tracks"
    if row["avg_acousticness"] >= 0.50:
        return "Acoustic Classics"
    if row["avg_danceability"] >= 0.62 and row["avg_tempo"] >= 120:
        return "Dance-Oriented Tracks"
    if row["avg_liveness"] >= 0.40:
        return "Live Performance Songs"
    return "Balanced Deep Cuts"


def _safe_cluster_metrics(X, labels):
    unique = set(labels)
    n_clusters = len(unique - {-1})

    result = {
        "n_clusters": int(n_clusters),
        "silhouette_score": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
    }

    if n_clusters < 2:
        return result

    try:
        result["silhouette_score"] = float(silhouette_score(X, labels))
        result["davies_bouldin"] = float(davies_bouldin_score(X, labels))
        result["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
    except Exception:
        pass

    return result


def section_data_quality(df):
    df_info = dfh.get_dataframe_info_str(df)

    missing_counts = df.isna().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)
    missing_report = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": missing_counts.values,
            "missing_pct": missing_pct.values,
        }
    ).sort_values("missing_pct", ascending=False)

    duplicate_rows = int(df.duplicated().sum())
    duplicate_song_ids = int(df.duplicated(subset=["id"]).sum())

    key_outlier_cols = ["popularity", "duration_ms", "tempo", "loudness"]
    outlier_flags_df = dfh.find_iqr_outliers(df, key_outlier_cols)
    outlier_rows = []
    for col in key_outlier_cols:
        flag_col = f"{col}_outlier"
        outlier_count = int(outlier_flags_df[flag_col].sum())
        outlier_rows.append(
            {
                "feature": col,
                "outlier_count": outlier_count,
                "outlier_pct": round((outlier_count / len(df)) * 100, 2),
            }
        )
    outlier_report = pd.DataFrame(outlier_rows)

    refined_df = df.copy()
    refined_df = refined_df.drop_duplicates()
    refined_df = refined_df.drop_duplicates(subset=["id"])
    refined_df = refined_df.dropna(subset=["id", "name", "album", "release_date"])

    for col in AUDIO_FEATURES + ["track_number"]:
        refined_df[col] = pd.to_numeric(refined_df[col], errors="coerce")
        refined_df[col] = refined_df[col].fillna(refined_df[col].median())

    refined_df["release_date"] = pd.to_datetime(refined_df["release_date"], errors="coerce")
    refined_df = refined_df.dropna(subset=["release_date"])

    quality_summary = {
        "rows_before": int(df.shape[0]),
        "rows_after": int(refined_df.shape[0]),
        "columns": int(refined_df.shape[1]),
        "duplicate_rows_found": duplicate_rows,
        "duplicate_song_ids_found": duplicate_song_ids,
        "total_missing_values_before": int(missing_counts.sum()),
        "total_missing_values_after": int(refined_df.isna().sum().sum()),
    }

    return {
        "df_info": df_info,
        "missing_report": missing_report,
        "outlier_report": outlier_report,
        "quality_summary": quality_summary,
        "refined_df": refined_df,
    }


def section_album_recommendation(df, popularity_threshold=70):
    album_ranking = (
        df.groupby("album", as_index=False)
        .agg(
            song_count=("id", "count"),
            avg_popularity=("popularity", "mean"),
            median_popularity=("popularity", "median"),
            popular_song_count=("popularity", lambda s: int((s >= popularity_threshold).sum())),
        )
        .sort_values(["popular_song_count", "avg_popularity"], ascending=False)
    )

    album_ranking["avg_popularity"] = album_ranking["avg_popularity"].round(2)
    album_ranking["median_popularity"] = album_ranking["median_popularity"].round(2)

    top_2_albums = album_ranking.head(2)

    fig_album_rank = px.bar(
        album_ranking.head(10),
        x="album",
        y="popular_song_count",
        color="avg_popularity",
        title=f"Top Albums by Popular Song Count (threshold >= {popularity_threshold})",
        labels={"popular_song_count": "Popular Song Count", "album": "Album"},
    )
    fig_album_rank.update_layout(xaxis_tickangle=-35)

    return {
        "album_ranking": album_ranking,
        "top_2_albums": top_2_albums,
        "fig_album_rank": fig_album_rank,
        "popularity_threshold": popularity_threshold,
    }


def section_feature_engineering(df):
    engineered_df = df.copy()

    # Avoid pandas cut/index issues on float16 after dataset optimization.
    engineered_df["popularity"] = engineered_df["popularity"].astype("float32")
    engineered_df["tempo"] = engineered_df["tempo"].astype("float32")
    engineered_df["duration_ms"] = engineered_df["duration_ms"].astype("float32")

    current_year = pd.Timestamp.today().year
    engineered_df["release_year"] = engineered_df["release_date"].dt.year
    engineered_df["song_age"] = (current_year - engineered_df["release_year"]).clip(lower=0)

    engineered_df["popularity_category"] = pd.cut(
        engineered_df["popularity"],
        bins=[-1, 30, 60, 80, 100],
        labels=["Low", "Moderate", "High", "Very High"],
    )

    engineered_df["tempo_category"] = pd.cut(
        engineered_df["tempo"],
        bins=[0, 90, 120, 160, np.inf],
        labels=["Slow", "Medium", "Fast", "Very Fast"],
    )

    engineered_df["duration_min"] = engineered_df["duration_ms"] / 60000
    engineered_df["duration_category"] = pd.cut(
        engineered_df["duration_min"],
        bins=[0, 3.0, 5.0, np.inf],
        labels=["Short", "Medium", "Long"],
    )

    engineered_df["album_age"] = engineered_df.groupby("album")["song_age"].transform("median")

    return engineered_df


def section_popularity_analysis(df):
    corr_series = (
        df[AUDIO_FEATURES]
        .corr(numeric_only=True)["popularity"]
        .drop("popularity")
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )
    popularity_drivers = pd.DataFrame(
        {"feature": corr_series.index, "correlation_with_popularity": corr_series.values}
    )

    release_trend = (
        df.groupby(df["release_date"].dt.year, as_index=False)["popularity"]
        .mean()
        .rename(columns={"release_date": "release_year", "popularity": "avg_popularity"})
    )
    release_trend.columns = ["release_year", "avg_popularity"]

    fig_popularity_energy = px.scatter(
        df,
        x="energy",
        y="popularity",
        color="album",
        hover_data=["name", "tempo", "danceability"],
        title="Popularity vs Energy",
        trendline="ols",
    )

    fig_popularity_dance = px.scatter(
        df,
        x="danceability",
        y="popularity",
        color="album",
        hover_data=["name", "tempo", "valence"],
        title="Popularity vs Danceability",
        trendline="ols",
    )

    fig_release_trend = px.line(
        release_trend,
        x="release_year",
        y="avg_popularity",
        markers=True,
        title="Popularity Trend by Release Year",
    )

    corr_fig = px.imshow(
        df[AUDIO_FEATURES].corr(numeric_only=True),
        text_auto=True,
        color_continuous_scale="RdBu",
        title="Audio Feature Correlation Matrix",
    )

    return {
        "popularity_drivers": popularity_drivers,
        "fig_popularity_energy": fig_popularity_energy,
        "fig_popularity_dance": fig_popularity_dance,
        "fig_release_trend": fig_release_trend,
        "fig_corr": corr_fig,
    }


def section_pca(df):
    pca_features = [
        "acousticness",
        "danceability",
        "energy",
        "instrumentalness",
        "liveness",
        "loudness",
        "speechiness",
        "tempo",
        "valence",
        "duration_ms",
    ]

    X = df[pca_features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca_full = PCA()
    pca_full.fit(X_scaled)
    explained_var = pca_full.explained_variance_ratio_
    cum_var = np.cumsum(explained_var)

    n_components = int(np.argmax(cum_var >= 0.90) + 1)

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    pca_cols = [f"PC{i + 1}" for i in range(n_components)]
    pca_df = pd.DataFrame(X_pca, columns=pca_cols, index=df.index)

    explained_df = pd.DataFrame(
        {
            "component": list(range(1, len(explained_var) + 1)),
            "explained_variance_ratio": explained_var,
            "cumulative_variance": cum_var,
        }
    )

    fig_explained_var = px.line(
        explained_df,
        x="component",
        y="cumulative_variance",
        markers=True,
        title="PCA Cumulative Explained Variance",
    )

    return {
        "pca_df": pca_df,
        "explained_df": explained_df,
        "fig_explained_var": fig_explained_var,
        "n_components": n_components,
        "pca_features": pca_features,
    }


def section_clustering(df, pca_df):
    X = pca_df.values

    k_eval_rows = []
    for k in range(2, 9):
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(X)

        k_eval_rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette_score": float(silhouette_score(X, labels)),
                "davies_bouldin": float(davies_bouldin_score(X, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
            }
        )

    k_eval_df = pd.DataFrame(k_eval_rows)
    best_k = int(k_eval_df.sort_values("silhouette_score", ascending=False).iloc[0]["k"])

    models = {
        "KMeans": KMeans(n_clusters=best_k, random_state=42, n_init=20),
        "Agglomerative": AgglomerativeClustering(n_clusters=best_k),
        "DBSCAN": DBSCAN(eps=0.8, min_samples=12),
    }

    model_eval_rows = []
    labels_store = {}
    for model_name, model in models.items():
        labels = model.fit_predict(X)
        labels_store[model_name] = labels
        metrics = _safe_cluster_metrics(X, labels)
        model_eval_rows.append(
            {
                "model": model_name,
                **metrics,
            }
        )

    model_eval_df = pd.DataFrame(model_eval_rows)
    best_row = (
        model_eval_df.dropna(subset=["silhouette_score"]).sort_values("silhouette_score", ascending=False)
    )
    best_model_name = best_row.iloc[0]["model"] if not best_row.empty else "KMeans"

    best_labels = labels_store[best_model_name]

    clustered_df = df.copy()
    clustered_df["cluster"] = best_labels
    clustered_df = clustered_df.join(pca_df[["PC1", "PC2"]], how="left")

    cluster_summary = (
        clustered_df.groupby("cluster", as_index=False)
        .agg(
            song_count=("id", "count"),
            avg_popularity=("popularity", "mean"),
            avg_energy=("energy", "mean"),
            avg_danceability=("danceability", "mean"),
            avg_acousticness=("acousticness", "mean"),
            avg_liveness=("liveness", "mean"),
            avg_tempo=("tempo", "mean"),
            avg_valence=("valence", "mean"),
            avg_instrumentalness=("instrumentalness", "mean"),
        )
        .sort_values("song_count", ascending=False)
    )

    for col in [
        "avg_popularity",
        "avg_energy",
        "avg_danceability",
        "avg_acousticness",
        "avg_liveness",
        "avg_tempo",
        "avg_valence",
        "avg_instrumentalness",
    ]:
        cluster_summary[col] = cluster_summary[col].round(3)

    cluster_summary["cohort_name"] = cluster_summary.apply(_cohort_name, axis=1)

    fig_elbow = px.line(
        k_eval_df,
        x="k",
        y="inertia",
        markers=True,
        title="Elbow Method (KMeans Inertia)",
    )

    fig_silhouette = px.line(
        k_eval_df,
        x="k",
        y="silhouette_score",
        markers=True,
        title="Silhouette by Cluster Count (KMeans)",
    )

    fig_cluster_scatter = px.scatter(
        clustered_df,
        x="PC1",
        y="PC2",
        color=clustered_df["cluster"].astype(str),
        hover_data=["name", "album", "popularity"],
        title=f"Song Cohorts via {best_model_name} (PCA space)",
        labels={"color": "Cluster"},
    )

    return {
        "k_eval_df": k_eval_df,
        "model_eval_df": model_eval_df,
        "best_k": best_k,
        "best_model_name": best_model_name,
        "clustered_df": clustered_df,
        "cluster_summary": cluster_summary,
        "fig_elbow": fig_elbow,
        "fig_silhouette": fig_silhouette,
        "fig_cluster_scatter": fig_cluster_scatter,
    }


def main():
    print("Running Spotify Song Cohort Analysis Pipeline...")
    start_time = time.perf_counter()

    content = []
    builder = HtmlBuilder()
    plot_renderer = PlotRenderer()

    # ---------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------
    df, report = dl.read_dataset(
        "rolling_stones_spotify.csv",
        optimize=True,
        handle_unnamed="drop",
        return_report=True,
    )

    # --------------------------------------------------
    # SECTION 1: DATA QUALITY
    # --------------------------------------------------
    quality = section_data_quality(df)
    refined_df = quality["refined_df"]

    # --------------------------------------------------
    # SECTION 2: EDA + FEATURE ENGINEERING
    # --------------------------------------------------
    album_data = section_album_recommendation(refined_df)
    engineered_df = section_feature_engineering(refined_df)
    popularity_data = section_popularity_analysis(engineered_df)

    hist_popularity = px.histogram(
        engineered_df,
        x="popularity",
        opacity=0.75,
        barmode="overlay",
        title="Song Popularity Distribution",
        nbins=20,
    )

    hist_tempo = px.histogram(
        engineered_df,
        x="tempo",
        opacity=0.75,
        title="Tempo Distribution",
        nbins=25,
    )

    violin_energy = px.violin(
        engineered_df,
        x="popularity_category",
        y="energy",
        box=True,
        title="Energy by Popularity Category",
    )

    # --------------------------------------------------
    # SECTION 3: PCA
    # --------------------------------------------------
    pca_data = section_pca(engineered_df)

    # --------------------------------------------------
    # SECTION 4: CLUSTERING
    # --------------------------------------------------
    cluster_data = section_clustering(engineered_df, pca_data["pca_df"])

    top_albums_text = {
        "recommended_albums": album_data["top_2_albums"]["album"].tolist(),
        "popularity_threshold": album_data["popularity_threshold"],
        "best_cluster_model": cluster_data["best_model_name"],
        "optimal_k_kmeans": cluster_data["best_k"],
        "pca_components_for_90pct_variance": pca_data["n_components"],
    }

    content.append(
        builder.full_width_card(
            "Original Spotify Data",
            builder.render_dataframe_collapsible(df, initial_rows=15),
        )
    )

    content.append(
        builder.grid(
            [
                builder.card("Dataframe Info", builder.render_pre(quality["df_info"])),
                builder.card("Data Quality Summary", builder.render_dict(quality["quality_summary"])),
                builder.card("Dataframe Optimization Report", builder.render_pre(report)),
                builder.card("Missing Value Report", builder.render_dataframe(quality["missing_report"])),
                builder.card("Outlier Analysis (IQR)", builder.render_dataframe(quality["outlier_report"])),
                builder.card("Album Recommendation Summary", builder.render_dict(top_albums_text)),
                builder.card("Top 2 Recommended Albums", builder.render_dataframe(album_data["top_2_albums"])),
                builder.card(
                    "Album Ranking",
                    builder.render_dataframe_collapsible(album_data["album_ranking"], initial_rows=12),
                ),
                builder.card("Popularity Driver Correlations", builder.render_dataframe(popularity_data["popularity_drivers"])),
                builder.card("Engineered Dataset Sample", builder.render_dataframe_collapsible(engineered_df.head(30), initial_rows=12)),
                builder.card("PCA Explained Variance", builder.render_dataframe(pca_data["explained_df"])),
                builder.card("Cluster Model Comparison", builder.render_dataframe(cluster_data["model_eval_df"])),
                builder.card("Cluster Definitions", builder.render_dataframe(cluster_data["cluster_summary"])),
            ]
        )
    )

    content.append(
        builder.chart_grid(
            [
                plot_renderer.plot_to_card(hist_popularity, "Popularity Histogram"),
                plot_renderer.plot_to_card(hist_tempo, "Tempo Histogram"),
                plot_renderer.plot_to_card(violin_energy, "Energy vs Popularity Category"),
                plot_renderer.plot_to_card(album_data["fig_album_rank"], "Top Albums by Popular Songs"),
                plot_renderer.plot_to_card(popularity_data["fig_popularity_energy"], "Popularity vs Energy"),
                plot_renderer.plot_to_card(popularity_data["fig_popularity_dance"], "Popularity vs Danceability"),
                plot_renderer.plot_to_card(popularity_data["fig_release_trend"], "Popularity Trend by Release Year"),
                plot_renderer.plot_to_card(popularity_data["fig_corr"], "Audio Feature Correlation Heatmap"),
                plot_renderer.plot_to_card(pca_data["fig_explained_var"], "PCA Cumulative Explained Variance"),
                plot_renderer.plot_to_card(cluster_data["fig_elbow"], "Elbow Method"),
                plot_renderer.plot_to_card(cluster_data["fig_silhouette"], "Silhouette Curve"),
                plot_renderer.plot_to_card(cluster_data["fig_cluster_scatter"], "Song Cohort Map (PCA)"),
            ]
        )
    )

    executive_summary = {
        "data_quality_status": "Dataset refined with duplicate and missing value handling",
        "top_albums_for_recommendation": album_data["top_2_albums"]["album"].tolist(),
        "key_popularity_drivers": popularity_data["popularity_drivers"].head(3)["feature"].tolist(),
        "dimensionality_reduction_insight": f"{pca_data['n_components']} principal components retain at least 90% variance",
        "selected_clustering_model": cluster_data["best_model_name"],
        "song_cohorts_identified": cluster_data["cluster_summary"]["cohort_name"].tolist(),
        "recommendation_use_case": "Use cohort tags to drive similarity-based song recommendations",
    }

    content.append(
        builder.full_width_card(
            "Executive Summary",
            builder.render_dict(executive_summary),
        )
    )

    # --------------------------------------------------
    # FINAL HTML
    # --------------------------------------------------
    html_doc = builder.build_page(
        "ML Spotify Song Cohort Analysis Report",
        "\n".join(content),
    )

    output_path = ru.save_html_report(
        __file__,
        "ml_spotify_song_cohort_pipeline_report.html",
        html_doc,
        subfolder="reports",
        open_in_browser=True,
    )

    print(f"Wrote report to: {output_path}")

    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.6f} seconds")


if __name__ == "__main__":
    main()
