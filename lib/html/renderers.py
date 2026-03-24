import html

import numpy as np

try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False


# --- NumPy Arrays ---
def render_array(arr: np.ndarray) -> str:
    preview = html.escape(np.array2string(arr, threshold=40))
    rows = [
        ("Size (elements)", arr.size),
        ("Shape", str(arr.shape)),
        ("Dimensions (ndim)", arr.ndim),
        ("Data Type (dtype)", str(arr.dtype)),
        ("Item Size (bytes)", arr.itemsize),
        ("Total Memory", f"{arr.nbytes} bytes"),
    ]
    return f"""
<pre class="bg-slate-900 text-white p-3 rounded-lg overflow-x-auto text-xs">{preview}</pre>
{render_kv(rows)}
"""


# --- Pandas Series ---
def render_series(s):
    # Handle list / numpy array gracefully
    if isinstance(s, (list, tuple, np.ndarray)):
        rows = "\n".join(
            f"<tr><td class='border px-3 py-1'>{i}</td>"
            f"<td class='border px-3 py-1'>{html.escape(str(v))}</td></tr>"
            for i, v in enumerate(s)
        )
        return f"""
<table class="table-auto border-collapse w-full text-sm">
  <thead>
    <tr class="bg-slate-100">
      <th class="border px-3 py-1 text-left">Index</th>
      <th class="border px-3 py-1 text-left">Value</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>
"""

    # Handle pandas Series (correct original behavior)
    if _HAS_PANDAS:
        import pandas as pd
        if isinstance(s, pd.Series):
            rows = "\n".join(
                f"<tr><td class='border px-3 py-1'>{html.escape(str(i))}</td>"
                f"<td class='border px-3 py-1'>{html.escape(str(v))}</td></tr>"
                for i, v in s.items()
            )
            return f"""
<table class="table-auto border-collapse w-full text-sm">
  <thead>
    <tr class="bg-slate-100">
      <th class="border px-3 py-1 text-left">Index</th>
      <th class="border px-3 py-1 text-left">Value</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>
"""

    return f"<div class='text-red-600'>Cannot render series-like object: {type(s)}</div>"


# --- Pandas DataFrame ---
def render_dataframe(df):
    """
    Render a pandas DataFrame or dataframe‑like object (list of dicts, dict of lists)
    using Tailwind table utilities for a clean, consistent UI.
    """

    # --- Normalize input to pandas DataFrame ---
    if _HAS_PANDAS:
        import pandas as pd

        # If already a DataFrame
        if isinstance(df, pd.DataFrame):
            pass

        # List of dicts
        elif isinstance(df, list) and all(isinstance(r, dict) for r in df):
            df = pd.DataFrame(df)

        # Dict of lists
        elif isinstance(df, dict) and all(isinstance(v, (list, tuple)) for v in df.values()):
            df = pd.DataFrame(df)

        else:
            return (
                f"<div class='text-red-600 text-sm'>"
                f"Cannot render DataFrame-like object of type {type(df)}</div>"
            )

        # Empty DF
        if df.empty:
            return "<div class='text-slate-500 italic text-sm'>Empty DataFrame</div>"

        # --- Build column header ---
        header_cells = "".join(
            f"<th class='border px-3 py-1 font-medium bg-slate-100 text-left'>"
            f"{html.escape(str(col))}</th>"
            for col in df.columns
        )

        # --- Build table rows ---
        body_rows = []
        for idx, row in df.iterrows():
            cells = "".join(
                f"<td class='border px-3 py-1'>{html.escape(str(v))}</td>"
                for v in row.values
            )
            body_rows.append(
                f"<tr>"
                f"<td class='border px-3 py-1 font-medium bg-slate-50'>{html.escape(str(idx))}</td>"
                f"{cells}"
                f"</tr>"
            )

        body_html = "\n".join(body_rows)

        # --- Return final Tailwind DataFrame ---
        return f"""
<div class="overflow-x-auto">
<table class="table-auto border-collapse w-full text-sm">
  <thead>
    <tr>
      <th class="border px-3 py-1 bg-slate-200 text-left">Index</th>
      {header_cells}
    </tr>
  </thead>
  <tbody>
    {body_html}
  </tbody>
</table>
</div>
"""

    # --- Pandas not installed ---
    return (
        f"<div class='text-red-600 text-sm'>"
        f"pandas not installed — cannot render DataFrame</div>"
    )


# --- Python Dict ---
def render_dict(d: dict, title: str | None = None) -> str:
    """
    Render a Python dict as a clean Tailwind key-value table with borders.
    """

    def clean_value(v):
        if isinstance(v, (np.generic,)):
            return v.item()
        if isinstance(v, (list, tuple, np.ndarray)):
            return ", ".join(str(clean_value(x)) for x in v)
        if isinstance(v, dict):
            return render_dict(v)
        return str(v)

    rows = "\n".join(
        f"""
        <tr class="odd:bg-gray-50">
            <th class="border border-slate-300 bg-gray-100 text-left font-medium text-slate-700 px-3 py-1">
                {html.escape(str(k))}
            </th>
            <td class="border border-slate-300 px-3 py-1 text-slate-800">
                {clean_value(v)}
            </td>
        </tr>
        """
        for k, v in d.items()
    )

    header = f"<h3 class='text-md font-semibold mb-2'>{title}</h3>" if title else ""

    return f"""
{header}
<table class="table-auto border-collapse w-full text-sm">
    <tbody>
        {rows}
    </tbody>
</table>
"""

# --- Generic Key–Value Renderer ---


def render_kv(rows: list[tuple[str, str]]) -> str:
    html_rows = "\n".join(
        f"<tr class='odd:bg-gray-50'><th class='text-left text-slate-500 font-medium pr-4'>{k}</th><td>{v}</td></tr>"
        for k, v in rows
    )
    return f"""
<table class="w-full text-sm border-collapse">
  {html_rows}
</table>
"""
