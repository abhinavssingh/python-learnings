
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
from typing import Iterable, List, Sequence, Tuple, Mapping, Any
import uuid
import numpy as np

try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:  # pandas not installed or not needed
    _HAS_PANDAS = False


__all__ = [
    "array_info_html",
    "arrays_report_html",
    "display_array_info",
    "array_to_html_table",
    "arrays_table_html",
    "arrays_index_report_html",

]

def _array_preview_any(obj: Any, max_items: int = 24) -> str:
    """
    Return a safe, short string preview for ndarray/Series/DataFrame/array-like.
    """
    # Convert to an ndarray for a compact preview
    if _HAS_PANDAS:
        if isinstance(obj, pd.Series):
            arr = obj.to_numpy()
        elif isinstance(obj, pd.DataFrame):
            arr = obj.to_numpy()
        else:
            arr = np.asarray(obj)
    else:
        arr = np.asarray(obj)

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

def _obj_info(obj: Any) -> dict:
    """
    Normalize ndarray/Series/DataFrame/array-like into a common info dict.
    Keys: size, shape, ndim, type_str, dtype_str, itemsize, nbytes, preview_html
    """
    # NumPy array
    if isinstance(obj, np.ndarray):
        arr = obj
        size = arr.size
        shape = tuple(arr.shape)
        ndim = arr.ndim
        type_str = str(type(obj))
        dtype_str = str(arr.dtype)
        itemsize = arr.itemsize
        nbytes = getattr(arr, "nbytes", size * itemsize)
        preview_html = f'<div class="mono block">{_array_preview_any(arr)}</div>'
        return locals()

    # pandas Series
    if _HAS_PANDAS and isinstance(obj, pd.Series):
        arr = obj.to_numpy()
        size = obj.size
        shape = (obj.size,)
        ndim = 1
        type_str = "pandas.Series"
        dtype_str = str(obj.dtype)
        itemsize = arr.itemsize
        nbytes = arr.nbytes  # data only; excludes index
        preview_html = f'<div class="mono block">{_array_preview_any(arr)}</div>'
        return locals()

    # pandas DataFrame
    if _HAS_PANDAS and isinstance(obj, pd.DataFrame):
        arr = obj.to_numpy()
        size = arr.size
        shape = tuple(obj.shape)
        ndim = obj.ndim  # should be 2
        type_str = "pandas.DataFrame"
        # Concise dtypes summary: col1:dt1, col2:dt2 ...
        dtype_str = ", ".join(f"{c}:{dt}" for c, dt in obj.dtypes.items())
        itemsize = arr.itemsize if arr.size else 0
        nbytes = arr.nbytes  # data only; excludes index
        preview_html = f'<div class="mono block">{_array_preview_any(arr)}</div>'
        return locals()

    # Fallback: try to coerce anything else to ndarray
    arr = np.asarray(obj)
    size = arr.size
    shape = tuple(arr.shape)
    ndim = arr.ndim
    type_str = str(type(obj))
    dtype_str = str(arr.dtype)
    itemsize = arr.itemsize
    nbytes = arr.nbytes
    preview_html = f'<div class="mono block">{_array_preview_any(arr)}</div>'
    return locals()

def _series_to_html_table(s, max_rows: int = 50, collapse_threshold: int = 6) -> str:
    """
    Render a pandas Series in native table format (Index | Value),
    with a 'Show all / Show less' toggle when length > collapse_threshold.
    """
    if not _HAS_PANDAS:
        raise RuntimeError("pandas is not available but a Series was passed.")
    import pandas as pd
    if not isinstance(s, pd.Series):
        raise TypeError("Expected a pandas Series.")

    n = len(s)
    rows_to_show = min(n, max_rows)

    # Build table rows (mark those beyond collapse_threshold as 'extra')
    body_rows = []
    for i, (idx, val) in enumerate(s.iloc[:rows_to_show].items()):
        extra_cls = " extra" if i >= collapse_threshold else ""
        body_rows.append(
            f"<tr class='series-row{extra_cls}'>"
            f"<th>{html.escape(str(idx))}</th>"
            f"<td class='mono'>{html.escape(str(val))}</td>"
            f"</tr>"
        )

    body_html = "\n".join(body_rows)

    # Unique container id so multiple Series on the same page don’t conflict
    uid = f"series-{uuid.uuid4().hex}"
    needs_collapse = n > collapse_threshold

    # Button only if we need collapsing
    if needs_collapse:
        hidden_count = n - collapse_threshold
        toggle_btn = (
            f"<button type='button' class='series-toggle' "
            f"data-total='{n}' aria-expanded='false'>"
            f"Show all {hidden_count}</button>"
        )
        collapsed_class = "series-collapsed"
    else:
        toggle_btn = ""
        collapsed_class = "series-expanded"

    # Styles + script kept inside fragment so each card is self-contained
    # (If you prefer, you can move these to the page-level style/script once.)
    return f"""
    <style>
      /* Color theme (tweak these) */
      :root {{
        --series-head-fg: #1e3a8a;  /* header text */
        --series-head-bg: #eef2ff;  /* header background */
        --series-cell-fg: #334155;  /* cell text */
      }}

      /* Table look */
      table.series2col {{ border-collapse: collapse; width: 100%; }}
      .series2col th, .series2col td {{
          border: 1px solid #ddd; padding: 6px 8px; text-align: left;
      }}
      .series2col thead th {{ background: var(--series-head-bg); color: var(--series-head-fg); }}
      .series2col tbody td {{ color: var(--series-cell-fg); }}
      .series2col tbody tr:nth-child(odd) {{ background: #fafafa; }}

      /* The global .mono style is dark; override in cells so they aren't black */
      .series2col td.mono {{
        background: transparent !important;
        color: var(--series-cell-fg) !important;
        border-radius: 0 !important;
        padding: 6px 8px !important;
        overflow: visible !important;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
                     "Liberation Mono", "Courier New", monospace;
        font-size: 12px;
      }}

      /* Collapse behavior: hide rows > threshold when collapsed */
      #{uid}.series-collapsed .series-row.extra {{ display: none; }}

      /* Optional: style the toggle button */
      #{uid} .series-toggle {{
        margin-top: 8px;
        background: #1f2937; color: #fff; border: none;
        border-radius: 6px; padding: 6px 10px; cursor: pointer;
      }}
      #{uid} .series-toggle:hover {{ filter: brightness(1.1); }}
    </style>

    <div id="{uid}" class="series2col-container {collapsed_class}">
      <table class="series2col">
        <thead>
          <tr><th>Index</th><th>Value</th></tr>
        </thead>
        <tbody>
          {body_html}
        </tbody>
      </table>
      {toggle_btn}
    </div>

    <script>
    (function() {{
      var root = document.getElementById("{uid}");
      if (!root) return;
      var btn = root.querySelector(".series-toggle");
      if (!btn) return;

      btn.addEventListener("click", function() {{
        var expanded = root.classList.toggle("series-expanded");
        if (expanded) {{
          root.classList.remove("series-collapsed");
          btn.setAttribute("aria-expanded", "true");
          btn.textContent = "Show less";
        }} else {{
          root.classList.add("series-collapsed");
          btn.setAttribute("aria-expanded", "false");
          var total = btn.getAttribute("data-total") || "";
          var threshold = {collapse_threshold};
          var hidden = Math.max(0, total - threshold);
          btn.textContent = hidden ? ("Show all " + hidden) : "Show all";
        }}
      }});

      // Ensure initial state consistent
      if (root.classList.contains("series-collapsed")) {{
        btn.setAttribute("aria-expanded", "false");
      }} else {{
        btn.setAttribute("aria-expanded", "true");
        btn.textContent = "Show less";
      }}
    }})();
    </script>
    """

def array_info_html(obj, title: str = "Array Info") -> str:
    """
    Return a standalone HTML string with key properties and a preview.

    - pandas.Series → native table: Index | Value, plus size/dtype/name info
    - np.ndarray / other → your existing array preview card
    """
    # --- Special case: pandas Series (native 2-column view) ---
    if _HAS_PANDAS:
        import pandas as pd  # local to keep dependency optional
        if isinstance(obj, pd.Series):
            s = obj
            # Compute summary safely
            size = s.size
            dtype_str = str(s.dtype)
            name_str = "" if s.name is None else f" (name: {html.escape(str(s.name))})"
            # Memory (data buffer only)
            try:
                nbytes = s.memory_usage(deep=False) - s.index.nbytes
                # If deep memory is preferred, swap to deep=True (costlier):
                # nbytes = s.memory_usage(deep=True) - s.index.nbytes
            except Exception:
                nbytes = s.to_numpy().nbytes

            # Native preview table
            table_html = _series_to_html_table(s)

            return f"""
            <div class="col">
              <div class="card">
                <div class="card-title">{html.escape(title)}</div>
                {table_html}
                <table class="kv" style="margin-top:10px">
                  <tr><th>Size (elements)</th><td>{size}</td></tr>
                  <tr><th>Type</th><td>pandas.Series{name_str}</td></tr>
                  <tr><th>Data Type (dtype)</th><td>{html.escape(dtype_str)}</td></tr>
                  <tr><th>Total Memory (data)</th><td>{int(nbytes)} bytes</td></tr>
                </table>
              </div>
            </div>
            """
            
    # --- Default: keep your existing ndarray/array-like card ---
    arr = np.asarray(obj)
    nbytes = getattr(arr, "nbytes", arr.size * arr.itemsize)
    return f"""
    <div class="col">
      <div class="card">
        <div class="card-title">{html.escape(title)}</div>
        <div class="mono block">{_array_preview_any(arr)}</div>
        <table class="kv">
          <tr><th>Size (elements)</th><td>{arr.size}</td></tr>
          <tr><th>Shape</th><td>{html.escape(str(tuple(arr.shape)))}</td></tr>
          <tr><th>Dimensions (ndim)</th><td>{arr.ndim}</td></tr>
          <tr><th>Type</th><td>{html.escape(str(type(obj)))}</td></tr>
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
    cards = "\n".join(array_info_html(obj, title) for title, obj in items)

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


def array_to_html_table(obj: Any, max_rows: int = 20, max_cols: int = 20) -> str:
    """
    Render values as an HTML table.
    - DataFrame: values shown as-is (2D)
    - Series: shown as a single-column table
    - ndarray/array-like: must be 2D; otherwise raise
    """
    if _HAS_PANDAS and isinstance(obj, pd.DataFrame):
        df = obj
        arr = df.to_numpy()
        col_labels = list(map(str, df.columns))
        row_labels = [str(i) for i in df.index]
    elif _HAS_PANDAS and isinstance(obj, pd.Series):
        s = obj
        # Convert to a 2D shape (n, 1) for display
        df = s.to_frame(name=s.name if s.name is not None else "value")
        arr = df.to_numpy()
        col_labels = list(map(str, df.columns))
        row_labels = [str(i) for i in df.index]
    else:
        arr = np.asarray(obj)
        if arr.ndim != 2:
            raise ValueError("array_to_html_table only supports 2D arrays (or Series/DataFrame).")
        # generic labels
        col_labels = [f"Col {j}" for j in range(arr.shape[1])]
        row_labels = [f"Row {i}" for i in range(arr.shape[0])]

    rows = min(arr.shape[0], max_rows)
    cols = min(arr.shape[1], max_cols)

    header = "".join(f"<th>{html.escape(str(col_labels[j]))}</th>" for j in range(cols))
    body_rows: List[str] = []
    for i in range(rows):
        cells = "".join(f"<td>{html.escape(str(arr[i, j]))}</td>" for j in range(cols))
        body_rows.append(f"<tr><th>{html.escape(str(row_labels[i]))}</th>{cells}</tr>")
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

def _html_kv_table(pairs: List[Tuple[str, Any]], caption: str | None = None) -> str:
    rows = "\n".join(
        f"<tr><th>{html.escape(str(k))}</th><td class='mono'>{html.escape(str(v))}</td></tr>"
        for k, v in pairs
    )
    cap = f"<caption style='caption-side:bottom;text-align:left;color:#6b7280;font-size:12px'>{html.escape(caption)}</caption>" if caption else ""
    return f"""
    <style>
      table.kv2 {{ border-collapse: collapse; width: 100%; }}
      .kv2 th, .kv2 td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; vertical-align: top; }}
      .kv2 th {{ width: 28%; background: #f3f4f6; color: #374151; font-weight: 600; }}
      .kv2 tbody tr:nth-child(odd) td {{ background: #fafafa; }}
      .kv2 td.mono {{
        background: transparent !important; color: #334155 !important; border-radius: 0 !important;
        overflow: visible !important; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
                     "Liberation Mono", "Courier New", monospace; font-size: 12px;
      }}
    </style>
    <table class="kv2">{cap}<tbody>
      {rows}
    </tbody></table>
    """


def _nested_dict_to_matrix(d: Mapping[str, Any]) -> tuple[list[str], list[str], list[list[Any]]]:
    """
    Accepts nested dict in either orientation:
      {col: {stat: value}}  OR  {stat: {col: value}}
    Returns (row_labels, col_labels, matrix).
    Heuristic: choose the orientation whose inner dict keys look like 'stats' set.
    """
    if not d:
        return [], [], []

    # Normalize keys to str for labels
    def str_keys(x): return {str(k): v for k, v in x.items()}

    outer = str_keys(d)

    # Heuristic: if most inner keys are typical stats, treat inner=stats
    # Typical stats keys from pandas.describe
    stats_like = {"count", "mean", "std", "min", "25%", "50%", "75%", "max"}
    def inner_keys_set(m): 
        try:
            return set(next(iter(m.values())).keys())
        except Exception:
            return set()

    inner = inner_keys_set(outer)

    # Try orientation A: {col: {stat: value}}
    a_inner_stats_ratio = len(inner & stats_like) / (len(inner) or 1)

    # Try orientation B by transposing the interpretation.
    # Build a candidate reversed mapping {stat: {col: value}} if possible.
    # Only compute scores; actual matrix build comes later.
    # To get inner keys for B, look at keys of each inner dict and union them.
    inner_for_b = set()
    for v in outer.values():
        if isinstance(v, Mapping):
            inner_for_b |= set(map(str, v.keys()))
    b_inner_stats_ratio = len(inner_for_b & stats_like) / (len(inner_for_b) or 1)

    orientation = "col_to_stat" if a_inner_stats_ratio >= b_inner_stats_ratio else "stat_to_col"

    if orientation == "col_to_stat":
        col_labels = sorted(outer.keys())
        # Collect all stats to ensure full rectangular matrix
        stats = set()
        for v in outer.values():
            if isinstance(v, Mapping):
                stats |= set(map(str, v.keys()))
        row_labels = [s for s in ["count","mean","std","min","25%","50%","75%","max"] if s in stats] + \
                     sorted(stats - stats_like)
        matrix: list[list[Any]] = []
        for r in row_labels:
            row = []
            for c in col_labels:
                inner = outer.get(c, {})
                val = None
                if isinstance(inner, Mapping):
                    val = inner.get(r, None)
                row.append(val)
            matrix.append(row)
        return row_labels, col_labels, matrix
    else:
        # orientation 'stat_to_col' — swap roles
        row_labels = sorted(outer.keys())
        cols = set()
        for v in outer.values():
            if isinstance(v, Mapping):
                cols |= set(map(str, v.keys()))
        col_labels = sorted(cols)
        matrix: list[list[Any]] = []
        for r in row_labels:
            inner = outer.get(r, {})
            row = []
            for c in col_labels:
                val = None
                if isinstance(inner, Mapping):
                    val = inner.get(c, None)
                row.append(val)
            matrix.append(row)
        return row_labels, col_labels, matrix


def _dict_to_html_table(d: Mapping[str, Any], caption: str | None = None) -> str:
    """
    Render dicts:
      - Flat dict -> 2-column key/value table
      - Nested dict -> grid with row/column headers
    """
    # Detect nested dict: any value is a Mapping
    is_nested = any(isinstance(v, Mapping) for v in d.values())

    if not is_nested:
        pairs = [(str(k), d[k]) for k in d.keys()]
        return _html_kv_table(pairs, caption=caption)

    # Nested: turn into matrix grid
    row_labels, col_labels, matrix = _nested_dict_to_matrix(d)

    # Build header and body
    header = "".join(f"<th>{html.escape(str(c))}</th>" for c in col_labels)
    body_rows = []
    for i, r in enumerate(row_labels):
        cells = "".join(
            f"<td class='mono'>{html.escape(str(matrix[i][j]))}</td>"
            for j in range(len(col_labels))
        )
        body_rows.append(f"<tr><th>{html.escape(str(r))}</th>{cells}</tr>")
    body = "\n".join(body_rows)

    cap = f"<caption style='caption-side:bottom;text-align:left;color:#6b7280;font-size:12px'>{html.escape(caption)}</caption>" if caption else ""

    return f"""
    <style>
      table.dictgrid {{ border-collapse: collapse; width: 100%; }}
      .dictgrid th, .dictgrid td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
      .dictgrid thead th {{ background: #f3f4f6; color: #374151; }}
      .dictgrid tbody tr:nth-child(odd) {{ background: #fafafa; }}
      .dictgrid td.mono {{
        background: transparent !important; color: #334155 !important; border-radius: 0 !important;
        overflow: visible !important; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
                     "Liberation Mono", "Courier New", monospace; font-size: 12px;
      }}
    </style>
    <table class="dictgrid">
      {cap}
      <thead><tr><th></th>{header}</tr></thead>
      <tbody>{body}</tbody>
    </table>
    """

def _format_value_html(value) -> str:
    # pandas-specific rich rendering
    if _HAS_PANDAS:
        import pandas as pd
        if isinstance(value, pd.Series):
            return _series_to_html_table(value, max_rows=30)
        if isinstance(value, pd.DataFrame):
            # Use your existing 2D array renderer for compactness
            return array_to_html_table(value, max_rows=20, max_cols=20)

    # Mapping (dict, OrderedDict, etc.) — special handling
    try:
        from collections.abc import Mapping as _MappingABC
    except Exception:
        _MappingABC = Mapping  # fallback

    if isinstance(value, _MappingABC):
        return _dict_to_html_table(value, caption="mapping")

    # Fallback: generic compact preview
    return f'<div class="mono block">{_array_preview_any(value)}</div>'

def arrays_table_html(pairs: Sequence[Tuple[str, Any]]) -> str:
    rows = []
    for label, obj in pairs:
        rows.append(
            f'<tr><th>{html.escape(str(label))}</th><td>{_format_value_html(obj)}</td></tr>'
        )
    rows_html = "\n".join(rows)
    return f"""
    <table class="kv" style="margin-top:8px;">
      <tbody>
        {rows_html}
      </tbody>
    </table>
    """

def arrays_index_report_html(pairs: Sequence[Tuple[str, Any]], page_title: str = "Pairs Report") -> str:
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