
"""
arrays_html.py

Reusable helpers to render NumPy array info as HTML (cards laid out in columns),
optionally display directly in Jupyter/Colab, or save to an .html file.

Functions
---------
- array_info_html(arr, title="Array Info") -> str
- arrays_report_html(items, page_title="Array Report") -> str
- display_array_info(arr, title="Array Info") -> None  (Jupyter only)
- array_to_html_table(arr, max_rows=20, max_cols=20) -> str (2D arrays)

Usage
-----
from arrays_html import array_info_html, arrays_report_html
html_str = array_info_html(np.arange(6).reshape(2,3), title="Demo")

report = arrays_report_html([
    ("A", np.arange(6).reshape(2,3)),
    ("B", np.linspace(0,1,12).reshape(3,4))
], page_title="My Report")

You can save the returned HTML with:
with open("report.html", "w", encoding="utf-8") as f:
    f.write(report)
"""

from __future__ import annotations
import html
from typing import Iterable, List, Sequence, Tuple

import numpy as np

__all__ = [
    "array_info_html",
    "arrays_report_html",
    "display_array_info",
    "array_to_html_table",
    "arrays_table_html",
    "arrays_index_report_html",

]


def _array_preview(arr: np.ndarray, max_items: int = 24) -> str:
    """Return a safe, short string preview of the array for HTML display."""
    # For string/object arrays, avoid quotes in preview for readability
    if arr.dtype.kind in {"U", "S", "O"}:
        formatter = {"all": lambda x: str(x)}
    else:
        formatter = None
    s = np.array2string(
        arr,
        max_line_width=80,
        threshold=max_items,
        separator=", ",
        formatter=formatter,
    )
    return html.escape(s)


def array_info_html(arr: np.ndarray, title: str = "Array Info") -> str:
    """Return a standalone HTML string with key NumPy array properties."""
    nbytes = getattr(arr, "nbytes", arr.size * arr.itemsize)
    return f"""
    <div class="col">
      <div class="card">
        <div class="card-title">{html.escape(title)}</div>
        <div class="mono block">{_array_preview(arr)}</div>
        <table class="kv">
          <tr><th>Size (elements)</th><td>{arr.size}</td></tr>
          <tr><th>Shape</th><td>{html.escape(str(tuple(arr.shape)))}</td></tr>
          <tr><th>Dimensions (ndim)</th><td>{arr.ndim}</td></tr>
          <tr><th>Type</th><td>{html.escape(str(type(arr)))}</td></tr>
          <tr><th>Data Type (dtype)</th><td>{html.escape(str(arr.dtype))}</td></tr>
          <tr><th>Item Size (bytes)</th><td>{arr.itemsize}</td></tr>
          <tr><th>Total Memory</th><td>{nbytes} bytes</td></tr>
        </table>
      </div>
    </div>
    """


def arrays_report_html(items, page_title="Array Report"):
    """
    Build a responsive, column-wise HTML report for a list of (title, array).
    items: sequence of (title: str, arr: np.ndarray)
    """
    # Plain string (NOT an f-string). This avoids brace-escaping issues.
    style = """
    <style>
      :root {
        --bg: #ffffff;
        --fg: #111827;
        --muted: #6b7280;
        --border: #e5e7eb;
      }
      * { box-sizing: border-box; }
      body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, Arial, sans-serif;
             margin: 0; padding: 24px; background: var(--bg); color: var(--fg); }
      .title { font-size: 22px; font-weight: 700; margin-bottom: 16px; }
      .grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 16px;
      }
      @media (max-width: 1100px) {
        .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      }
      @media (max-width: 700px) {
        .grid { grid-template-columns: 1fr; }
      }
      .card {
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px;
        background: #fff;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
      }
      .card-title { font-weight: 600; margin-bottom: 10px; }
      .mono {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 12px; background: #0b1020; color: #e5e7eb;
        border-radius: 8px; padding: 10px 12px; overflow-x: auto;
      }
      .block { white-space: pre; margin-bottom: 12px; }
      .kv { width: 100%; border-collapse: collapse; font-size: 14px; }
      .kv th, .kv td { text-align: left; padding: 6px 8px; vertical-align: top; }
      .kv th { width: 48%; color: var(--muted); font-weight: 500; }
      .kv tr:nth-child(odd) td, .kv tr:nth-child(odd) th { background: #fafafa; }
      .footer { margin-top: 12px; color: var(--muted); font-size: 12px; }
      .col { display: block; }
    </style>
    """

    # Build all cards first (safe regular string join)
    cards = "\n".join(array_info_html(arr, title) for title, arr in items)

    # Now assemble the full page (only this is an f-string)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"/>{style}
<title>{html.escape(page_title)}</title></head>
<body>
  <div class="title">{html.escape(page_title)}</div>
  <div class="grid">
    {cards}
  </div>
  <div class="footer">Rendered {len(items)} arrays.</div>
</body></html>
"""



def display_array_info(arr: np.ndarray, title: str = "Array Info") -> None:
    """Display the array info card inline in Jupyter/Colab notebooks."""
    try:
        from IPython.display import HTML, display  # type: ignore
    except Exception:
        raise RuntimeError("display_array_info requires IPython/Jupyter environment")
    html_fragment = arrays_report_html([(title, arr)], page_title=title)
    display(HTML(html_fragment))


def array_to_html_table(arr: np.ndarray, max_rows: int = 20, max_cols: int = 20) -> str:
    """Render a 2D NumPy array as an HTML table (with optional clipping)."""
    if arr.ndim != 2:
        raise ValueError("array_to_html_table only supports 2D arrays.")

    rows = min(arr.shape[0], max_rows)
    cols = min(arr.shape[1], max_cols)

    header = "".join(f"<th>Col {j}</th>" for j in range(cols))
    body_rows: List[str] = []
    for i in range(rows):
        cells = "".join(f"<td>{html.escape(str(arr[i, j]))}</td>" for j in range(cols))
        body_rows.append(f"<tr><th>Row {i}</th>{cells}</tr>")
    body = "".join(body_rows)

    clipped = (arr.shape[0] > rows) or (arr.shape[1] > cols)
    note = "<em>Table truncated for display.</em>" if clipped else ""

    return f"""
    <style>
      table.arr2d {{ border-collapse: collapse; }}
      .arr2d th, .arr2d td {{ border: 1px solid #ddd; padding: 6px 8px; font-family: monospace; }}
      .arr2d thead th {{ background: #f3f4f6; }}
      .arr2d tbody tr:nth-child(odd) {{ background: #fafafa; }}
    </style>
    <table class="arr2d">
      <thead><tr><th></th>{header}</tr></thead>
      <tbody>{body}</tbody>
    </table>
    {note}
    """

def _format_value_html(value: np.ndarray) -> str:
    """
    Internal: reuse the same preview style for arrays.
    """
    return f'<div class="mono block">{_array_preview(value)}</div>'


def arrays_table_html(pairs: Sequence[Tuple[str, np.ndarray]]) -> str:
    """
    Render a two-column HTML table from (label: str, array: np.ndarray) pairs.
    Returns an HTML fragment (no <html> wrapper), matching your style.
    """
    rows = []
    for label, arr in pairs:
        rows.append(
            f'<tr><th>{html.escape(str(label))}</th><td>{_format_value_html(arr)}</td></tr>'
        )
    rows_html = "\n".join(rows)
    return f"""
    <table class="kv" style="margin-top:8px;">
      <tbody>
        {rows_html}
      </tbody>
    </table>
    """


def arrays_index_report_html(pairs: Sequence[Tuple[str, np.ndarray]], page_title: str = "Pairs Report") -> str:
    """
    Build a full, standalone HTML page showing the (str, array) pairs in a table.
    Uses the same CSS+look as arrays_report_html for consistency.
    """
    # Keep CSS as a plain triple-quoted string (not an f-string)
    style = """
    <style>
      :root {
        --bg: #ffffff;
        --fg: #111827;
        --muted: #6b7280;
        --border: #e5e7eb;
      }
      * { box-sizing: border-box; }
      body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, Arial, sans-serif;
             margin: 0; padding: 24px; background: var(--bg); color: var(--fg); }
      .title { font-size: 22px; font-weight: 700; margin-bottom: 16px; }
      .mono {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 12px; background: #0b1020; color: #e5e7eb;
        border-radius: 8px; padding: 10px 12px; overflow-x: auto;
      }
      .block { white-space: pre; margin-bottom: 12px; }
      .kv { width: 100%; border-collapse: collapse; font-size: 14px; }
      .kv th, .kv td { text-align: left; padding: 6px 8px; vertical-align: top; }
      .kv th { width: 28%; color: var(--muted); font-weight: 500; }
      .kv tr:nth-child(odd) td, .kv tr:nth-child(odd) th { background: #fafafa; }
      .footer { margin-top: 12px; color: var(--muted); font-size: 12px; }
    </style>
    """
    table_html = arrays_table_html(pairs)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"/>{style}
<title>{html.escape(page_title)}</title></head>
<body>
  <div class="title">{html.escape(page_title)}</div>
  {table_html}
  <div class="footer">Rendered {len(pairs)} pairs.</div>
</body></html>
"""