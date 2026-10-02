import os
from typing import Any

import pandas as pd

from lib.html import HtmlBuilder, PlotRenderer
from lib.utility.reports.report_utils import ReportUtils as ru


class GenAIReport:
    """
    Thin helper over HtmlBuilder / PlotRenderer so learning scripts stay short.

        report = GenAIReport("N-grams")
        report.grid([report.text("Input", text), report.table("Counts", df)])
        report.plots([(fig, "Bigrams")])
        report.save(__file__, "ngrams_report.html")

    Set the environment variable GENAI_OPEN_REPORTS=0 to stop reports
    opening in the browser (useful when running every script with run.py).
    """

    def __init__(self, title: str):
        self.title = title
        self.builder = HtmlBuilder()
        self.plot_renderer = PlotRenderer()
        self.sections: list[str] = []

    # ------------------------------------------------------
    # Card factories (return HTML strings)
    # ------------------------------------------------------

    def text(self, title: str, text: Any, max_lines: int = 18) -> str:
        return self.builder.card(title, self.builder.render_pre(str(text), max_visible_lines=max_lines))

    def table(self, title: str, df: pd.DataFrame | list[dict], rows: int = 10) -> str:
        frame = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
        return self.builder.card(title, self.builder.render_dataframe(frame, max_visible_rows=rows))

    def kv(self, title: str, data: dict[str, Any], rows: int = 12) -> str:
        return self.builder.card(title, self.builder.render_kv([(k, v) for k, v in data.items()], max_visible_rows=rows))

    def html(self, title: str, raw_html: str) -> str:
        return self.builder.card(title, raw_html)

    # ------------------------------------------------------
    # Layout
    # ------------------------------------------------------

    def grid(self, cards: list[str], columns: int = 2) -> "GenAIReport":
        self.sections.append(self.builder.grid(cards, columns))
        return self

    def full(self, title: str, content_html: str) -> "GenAIReport":
        self.sections.append(self.builder.full_width_card(title, content_html))
        return self

    def full_text(self, title: str, text: Any, max_lines: int = 25) -> "GenAIReport":
        return self.full(title, self.builder.render_pre(str(text), max_visible_lines=max_lines))

    def full_table(self, title: str, df: pd.DataFrame | list[dict], rows: int = 15) -> "GenAIReport":
        frame = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
        return self.full(title, self.builder.render_dataframe(frame, max_visible_rows=rows))

    def plots(self, figures: list[tuple[Any, str]]) -> "GenAIReport":
        cards = [self.plot_renderer.plot_to_card(fig, title) for fig, title in figures]
        self.sections.append(self.builder.chart_grid(cards))
        return self

    def full_plot(self, figure: Any, title: str) -> "GenAIReport":
        self.sections.append(self.plot_renderer.plot_to_full_width_card(figure, title))
        return self

    # ------------------------------------------------------
    # Output
    # ------------------------------------------------------

    def render(self) -> str:
        page = self.builder.build_page(self.title, "\n".join(self.sections))
        # The shared page template makes bare "(...)" a MathJax delimiter; GenAI reports contain no LaTeX, so opt out
        # (MathJax 3.2 honours "mathjax_ignore" on descendants of <body>, not on <body> itself).
        start = page.find(">", page.find("<body")) + 1
        end = page.rfind("</body>")
        return f'{page[:start]}<div class="mathjax_ignore tex2jax_ignore">{page[start:end]}</div>{page[end:]}'

    def save(self, script_file: str, filename: str, open_in_browser: bool | None = None):
        if open_in_browser is None:
            open_in_browser = os.getenv("GENAI_OPEN_REPORTS", "1") != "0"
        path = ru.save_html_report(script_file, filename, self.render(), subfolder="reports", open_in_browser=open_in_browser)
        print(f"Report saved: {path}")
        return path
