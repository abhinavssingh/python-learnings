import html

import numpy as np

try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False


# --- NumPy Arrays ---
def render_array(arr: np.ndarray) -> str:
    preview = html.escape(
        np.array2string(arr, threshold=40, max_line_width=80)
    )

    rows = [
        ("Size (elements)", arr.size),
        ("Shape", str(arr.shape)),
        ("Dimensions (ndim)", arr.ndim),
        ("Data Type (dtype)", str(arr.dtype)),
        ("Item Size (bytes)", arr.itemsize),
        ("Total Memory", f"{arr.nbytes} bytes"),
    ]

    return f"""
<pre class="rounded-lg overflow-x-auto text-xs leading-tight
            p-4
            bg-slate-900 text-slate-100
            dark:bg-gray-200 dark:text-slate-900">
{preview}
</pre>

{render_kv(rows)}
"""


# --- Pandas Series ---
def render_series(s):
    # Handle list / numpy array gracefully
    if isinstance(s, (list, tuple, np.ndarray)):
        rows = "\n".join(
            f"<tr class='odd:bg-gray-50 dark:odd:bg-slate-900'>"
            f"<td class='border border-slate-300 dark:border-slate-700 text-slate-800 dark:text-slate-100 px-4 py-2'>{i}</td>"
            f"<td class='border border-slate-300 dark:border-slate-700 text-slate-800 dark:text-slate-100 px-4 py-2'>{html.escape(str(v))}</td></tr>"
            for i, v in enumerate(s)
        )
        return f"""
<table class="table-auto border-collapse w-full text-sm">
  <thead>
    <tr class="odd:bg-gray-50 dark:odd:bg-slate-900">

<th class="
border border-slate-300 dark:border-slate-700
           bg-gray-100 dark:bg-slate-700
           text-slate-700 dark:text-slate-100
           px-4 py-2 font-medium text-left
">
Index</th>

<th class="
border border-slate-300 dark:border-slate-700
           bg-gray-100 dark:bg-slate-700
           text-slate-700 dark:text-slate-100
           px-4 py-2 font-medium text-left
">
Value</th>
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
                f"<tr class='odd:bg-gray-50 dark:odd:bg-slate-900'>"
                f"<td class='border border-slate-300 dark:border-slate-700 text-slate-800 dark:text-slate-100 px-4 py-2'>{
                    html.escape(
                        str(i))}</td>"
                f"<td class='border border-slate-300 dark:border-slate-700 text-slate-800 dark:text-slate-100 px-4 py-2'>{
                    html.escape(
                        str(v))}</td></tr>"
                for i, v in s.items()
            )
            return f"""
<table class="table-auto border-collapse w-full text-sm">
  <thead>
    <tr class="odd:bg-gray-50 dark:odd:bg-slate-900">
      <th class="
border border-slate-300 dark:border-slate-700
           bg-gray-100 dark:bg-slate-700
           text-slate-700 dark:text-slate-100
           px-4 py-2 font-medium text-left
">
Index</th>
      <th class="
border border-slate-300 dark:border-slate-700
           bg-gray-100 dark:bg-slate-700
           text-slate-700 dark:text-slate-100
           px-4 py-2 font-medium text-left
">
Value</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>
"""

    return f"<div class='text-red-600'>Cannot render series-like object: {type(s)}</div>"


# --- Pandas DataFrame ---
def render_dataframe(df):
    if _HAS_PANDAS:
        import pandas as pd

        # Normalize:
        if isinstance(df, pd.DataFrame):
            pass
        elif isinstance(df, list) and all(isinstance(x, dict) for x in df):
            df = pd.DataFrame(df)
        elif isinstance(df, dict) and all(isinstance(v, (list, tuple)) for v in df.values()):
            df = pd.DataFrame(df)
        else:
            return f"<div class='text-red-600'>Invalid DataFrame-like object</div>"

        # Empty handling
        if df.empty:
            return "<div class='italic text-slate-500'>Empty DataFrame</div>"

        # Build header
        header = "".join(
            f"<th class='border border-slate-300 dark:border-slate-700 bg-gray-100 dark:bg-slate-700 text-slate-700 dark:text-slate-100 px-4 py-2"
            f"font-medium text-left'>{html.escape(str(c))}</th>"
            for c in df.columns
        )

        # Build rows
        rows = ""
        for idx, row in df.iterrows():
            cells = "".join(
                f"<td class='border border-slate-300 dark:border-slate-700 text-slate-800 dark:text-slate-100 px-4 py-2"
                f"text-slate-800 dark:text-slate-100 px-4 py-2'>{html.escape(str(v))}</td>"
                for v in row.values
            )
            rows += f"""
<tr class="odd:bg-gray-50 dark:odd:bg-slate-900">
    <td class="
border border-slate-300 dark:border-slate-700
           text-slate-800 dark:text-slate-100
           px-4 py-2
">
        {html.escape(str(idx))}
    </td>
    {cells}
</tr>
"""

        return f"""
<div class="overflow-x-auto">
<table class="table-auto border-collapse w-full text-sm">
    <thead><tr class="odd:bg-gray-50 dark:odd:bg-slate-900"><th class="
border border-slate-300 dark:border-slate-700
           bg-gray-100 dark:bg-slate-700
           text-slate-700 dark:text-slate-100
           px-4 py-2 font-medium text-left
">Index</th>{header}</tr></thead>
    <tbody>{rows}</tbody>
</table>
</div>
"""


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
        <tr class="odd:bg-gray-50 dark:odd:bg-slate-900">


<th class="
border border-slate-300 dark:border-slate-700
           bg-gray-100 dark:bg-slate-700
           text-slate-700 dark:text-slate-100
           px-4 py-2 font-medium text-left
">


                {html.escape(str(k))}
            </th>
            <td class="border border-slate-300 dark:border-slate-700 text-slate-800 dark:text-slate-200 px-3 py-2">
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


def render_kv(rows):
    """
    Tailwind-styled key-value table for metadata (NumPy / general use).
    """
    html_rows = "\n".join(
        f"""
        <tr class="odd:bg-gray-50 dark:odd:bg-slate-900">
            <th class="border border-slate-300 dark:border-slate-700
                       bg-gray-100 dark:bg-slate-700
                       text-slate-700 dark:text-slate-200
                       font-medium text-left px-4 py-2">
                {html.escape(str(k))}
            </th>
            <td class="border border-slate-300 dark:border-slate-700
                       text-slate-800 dark:text-slate-200
                       px-4 py-2">
                {html.escape(str(v))}
            </td>
        </tr>
        """
        for k, v in rows
    )

    return f"""
<table class="table-auto border-collapse w-full text-sm mt-2">
    <tbody>
        {html_rows}
    </tbody>
</table>
"""
