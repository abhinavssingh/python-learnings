import time

import plotly.express as px

from lib.html import HtmlBuilder, PlotRenderer
from lib.utility.dataframe.data_loader import DataLoader as dl
from lib.utility.dataframe.df_helper import DataFrameHelper as dfh
from lib.utility.reports.report_utils import ReportUtils as ru


def main():

    print("Running Spotify Data Pipeline...")
    start_time = time.perf_counter()

    content = []
    builder = HtmlBuilder()
    plotRenderer = PlotRenderer()

    # ---------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------
    df, report = dl.read_dataset(
        "rolling_stones_spotify.csv",
        optimize=True,
        handle_unnamed="drop",
        return_report=True
    )

    df_info = dfh.get_dataframe_info_str(df)

    # --------------------------------------------------
    # EDA
    # --------------------------------------------------

    hist_box_fig_1 = px.histogram(df, x="popularity", opacity=0.7, barmode="overlay",
                                  title="Popularity Histogram Box Graph", hover_data=df.columns)

    content.append(
        builder.full_width_card(
            "Original Spotify Data",
            builder.render_dataframe_collapsible(df, initial_rows=15)
        )
    )

    content.append(builder.grid([
        builder.card("Dataframe Info", builder.render_pre(df_info)),
        builder.card("Dataframe Description", builder.render_dict(df.describe().to_dict())),
        builder.card("Dataframe Optimization Report", builder.render_pre(report))
    ]))

    content.append(builder.chart_grid([
        plotRenderer.plot_to_card(hist_box_fig_1, "Popularity Histogram Box Graph"),
    ]))

    # --------------------------------------------------
    # ✅ FINAL HTML
    # --------------------------------------------------
    html_doc = builder.build_page(
        "ML Spotify Data Pipeline Report",
        "\n".join(content)
    )

    output_path = ru.save_html_report(
        __file__,
        "ml_spotify_data_pipeline_report.html",
        html_doc,
        subfolder="reports",
        open_in_browser=True
    )

    print(f"Wrote report to: {output_path}")

    # --------------------------------------------------
    # ✅ EXECUTION TIME
    # --------------------------------------------------
    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.6f} seconds")


if __name__ == "__main__":
    main()
