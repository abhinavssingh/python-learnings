import time

import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA

from lib.html import HtmlBuilder, PlotRenderer
from lib.utility.dataframe.data_loader import DataLoader as dl
from lib.utility.dataframe.df_helper import DataFrameHelper as dfh
from lib.utility.machinelearning.facade.UnsupervisedModelUtility import UnsupervisedModelUtility
from lib.utility.machinelearning.pipeline.CustomImputer import CustomImputer
from lib.utility.machinelearning.pipeline.OutlierHandler import OutlierHandler
from lib.utility.reports.report_utils import ReportUtils as ru


def main():

    print("Running Unsupervised ML Pipeline Report...")

    start_time = time.perf_counter()

    content = []
    builder = HtmlBuilder()
    plotRenderer = PlotRenderer()

    # ========================================================
    # ✅ LOAD DATA
    # ========================================================
    df, report = dl.read_dataset(
        "adultcensusincome.csv",
        optimize=False,
        handle_unnamed="drop",
        return_report=True
    )

    df_info = dfh.get_dataframe_info_str(df)

    # ========================================================
    # ✅ INIT UTILITY
    # ========================================================
    imputer = CustomImputer(num_strategy="median")
    outlier = OutlierHandler(method="iqr", factor=1.5)

    um = UnsupervisedModelUtility(
        df,
        imputer=imputer,
        outlier_handler=outlier
    )

    # ========================================================
    # ✅ PREPARE DATA (CONSISTENCY ✅)
    # ========================================================
    um.prepare_data()

    # ========================================================
    # ✅ RUN MODELS
    # ========================================================
    um.run_experiment("KMeans")
    um.run_experiment("DBSCAN")

    results_df = um.get_results_df()

    # ========================================================
    # ✅ GET PROCESSED DATA
    # ========================================================
    X_processed = um.preprocessor.transform(um.X)

    # ========================================================
    # ✅ PCA (VISUALIZATION ONLY)
    # ========================================================
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_processed)

    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])

    pca_fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        title="PCA Projection (2D)"
    )

    # ========================================================
    # ✅ SAFE LABEL EXTRACTION ✅
    # ========================================================
    kmeans_labels = um.get_labels("KMeans")
    dbscan_labels = um.get_labels("DBSCAN")

    kmeans_fig = None
    dbscan_fig = None

    if kmeans_labels is not None:
        kmeans_fig = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            color=[str(x) for x in kmeans_labels],
            title="KMeans Clusters"
        )

    if dbscan_labels is not None:
        dbscan_fig = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            color=[str(x) for x in dbscan_labels],
            title="DBSCAN Clusters"
        )

    # ========================================================
    # ✅ ELBOW CURVE
    # ========================================================
    from sklearn.cluster import KMeans

    inertia = []

    for k in range(2, 10):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_processed)
        inertia.append(km.inertia_)

    elbow_fig = px.line(
        x=list(range(2, 10)),
        y=inertia,
        title="Elbow Method"
    )

    # ========================================================
    # ✅ DASHBOARD
    # ========================================================
    content.append(builder.grid([
        builder.card("Data Info", builder.render_pre(df_info)),
        builder.card("Processed Data", builder.render_dataframe(pd.DataFrame(X_processed).head())),
        builder.card("Unsupervised Results", builder.render_dataframe(results_df)),
    ]))

    content.append(builder.chart_grid([
        plotRenderer.plot_to_card(pca_fig, "PCA Projection"),
        plotRenderer.plot_to_card(elbow_fig, "Elbow Curve"),
        plotRenderer.plot_to_card(kmeans_fig, "KMeans Clusters"),
        plotRenderer.plot_to_card(dbscan_fig, "DBSCAN Clusters")
    ]))

    # ========================================================
    # ✅ HTML REPORT
    # ========================================================
    html_doc = builder.build_page(
        "Unsupervised ML Pipeline Report",
        "\n".join(content)
    )

    output_path = ru.save_html_report(
        __file__,
        "unsupervised_ml_pipeline_report.html",
        html_doc,
        subfolder="reports",
        open_in_browser=True
    )

    print(f"Wrote report to: {output_path}")

    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    main()
