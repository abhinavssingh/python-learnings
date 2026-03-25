import uuid
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
def render_series(s, max_visible_rows: int = 5):
    """
    Render a pandas Series or list-like object.
    Adds vertical scrolling if the number of rows exceeds max_visible_rows.
    """

    def build_rows(iterable):
        return "\n".join(
            f"""
            <tr class="odd:bg-gray-50 dark:odd:bg-slate-900">
                <td class="border border-slate-300 dark:border-slate-700
                           text-slate-800 dark:text-slate-100
                           px-4 py-2 font-medium">
                    {html.escape(str(i))}
                </td>
                <td class="border border-slate-300 dark:border-slate-700
                           text-slate-800 dark:text-slate-100
                           px-4 py-2">
                    {html.escape(str(v))}
                </td>
            </tr>
            """
            for i, v in iterable
        )

    # --- Handle list / tuple / numpy array ---
    if isinstance(s, (list, tuple, np.ndarray)):
        row_count = len(s)
        rows = build_rows(enumerate(s))

    # --- Handle pandas Series ---
    elif _HAS_PANDAS:
        import pandas as pd
        if isinstance(s, pd.Series):
            row_count = len(s)
            rows = build_rows(s.items())
        else:
            return f"<div class='text-red-600'>Cannot render series-like object: {type(s)}</div>"
    else:
        return f"<div class='text-red-600'>Cannot render series-like object: {type(s)}</div>"

    # Enable scrolling only if series is large
    scroll_wrapper_class = (
        "max-h-64 overflow-y-auto pr-1"
        if row_count > max_visible_rows
        else ""
    )

    return f"""
<div class="{scroll_wrapper_class}">
  <table class="table-auto border-collapse w-full text-sm">
    <thead>
      <tr>
        <th class="border border-slate-300 dark:border-slate-700
                   bg-gray-100 dark:bg-slate-700
                   text-slate-700 dark:text-slate-100
                   px-4 py-2 font-medium text-left">
          Index
        </th>
        <th class="border border-slate-300 dark:border-slate-700
                   bg-gray-100 dark:bg-slate-700
                   text-slate-700 dark:text-slate-100
                   px-4 py-2 font-medium text-left">
          Value
        </th>
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
</div>
"""


# --- Pandas DataFrame ---
def render_dataframe(df, max_visible_rows: int = 5):
    """
    Render a pandas DataFrame or DataFrame-like structure.
    Adds vertical scrolling when row count exceeds max_visible_rows.
    """

    if not _HAS_PANDAS:
        return "<div class='text-red-600'>pandas not installed</div>"

    import pandas as pd

    # -------- Normalize input --------
    if isinstance(df, pd.DataFrame):
        pass
    elif isinstance(df, list) and all(isinstance(x, dict) for x in df):
        df = pd.DataFrame(df)
    elif isinstance(df, dict) and all(isinstance(v, (list, tuple)) for v in df.values()):
        df = pd.DataFrame(df)
    else:
        return (
            "<div class='text-red-600 text-sm'>"
            "Invalid DataFrame-like object</div>"
        )

    # Empty DF
    if df.empty:
        return "<div class='italic text-slate-500'>Empty DataFrame</div>"

    row_count = len(df)

    # -------- Header --------
    header = "".join(
        f"""
        <th class="border border-slate-300 dark:border-slate-700
                   bg-gray-100 dark:bg-slate-700
                   text-slate-700 dark:text-slate-100
                   px-4 py-2 font-medium">
            {html.escape(str(c))}
        </th>
        """
        for c in df.columns
    )

    # -------- Rows --------
    rows = ""
    for idx, row in df.iterrows():
        cells = "".join(
            f"""
            <td class="border border-slate-300 dark:border-slate-700
                       text-slate-800 dark:text-slate-200
                       px-4 py-2">
                {html.escape(str(v))}
            </td>
            """
            for v in row.values
        )

        rows += f"""
<tr class="odd:bg-gray-50 dark:odd:bg-slate-900">
    <td class="border border-slate-300 dark:border-slate-700
               bg-gray-100 dark:bg-slate-800
               text-slate-800 dark:text-slate-100
               px-4 py-2 font-medium">
        {html.escape(str(idx))}
    </td>
    {cells}
</tr>
"""

    # -------- Conditional scroll wrapper --------
    scroll_wrapper_class = (
        "max-h-72 overflow-y-auto pr-1"
        if row_count > max_visible_rows
        else ""
    )

    # -------- Final HTML --------
    return f"""
<div class="{scroll_wrapper_class} overflow-x-auto">
  <table class="table-auto border-collapse w-full text-sm">
      <thead>
          <tr>
              <th class="border border-slate-300 dark:border-slate-700
                         bg-gray-100 dark:bg-slate-700
                         text-slate-700 dark:text-slate-100
                         px-4 py-2">
                  Index
              </th>
              {header}
          </tr>
      </thead>
      <tbody>
          {rows}
      </tbody>
  </table>
</div>
"""


# --- Python Dict ---
def render_dict(d: dict, title: str | None = None, max_visible_rows: int = 1) -> str:
    """
    Render a Python dict as a Tailwind key-value table.
    If the dict is large, enable vertical scrolling to avoid card height explosion.
    """

    def clean_value(v):
        if isinstance(v, (np.generic,)):
            return v.item()
        if isinstance(v, (list, tuple, np.ndarray)):
            return ", ".join(str(clean_value(x)) for x in v)
        if isinstance(v, dict):
            return render_dict(v)
        return str(v)

    row_count = len(d)

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
            <td class="
                border border-slate-300 dark:border-slate-700
                text-slate-800 dark:text-slate-200
                px-4 py-2
            ">
                {clean_value(v)}
            </td>
        </tr>
        """
        for k, v in d.items()
    )

    header = (
        f"<h3 class='text-md font-semibold mb-2 text-slate-800 dark:text-slate-100'>{title}</h3>"
        if title
        else ""
    )

    # Enable scroll if dictionary is large
    scroll_wrapper_class = (
        "max-h-72 overflow-y-auto pr-1"
        if row_count > max_visible_rows
        else ""
    )

    return f"""
{header}
<div class="{scroll_wrapper_class}">
  <table class="table-auto border-collapse w-full text-sm">
      <tbody>
          {rows}
      </tbody>
  </table>
</div>
"""

# --- Generic Key–Value Renderer ---


def render_kv(rows, max_visible_rows: int = 5):
    """
    Tailwind-styled key-value table for metadata (NumPy / general use).
    Adds vertical scrolling when the number of rows exceeds max_visible_rows.
    """

    row_count = len(rows)

    html_rows = "\n".join(
        f"""
        <tr class="odd:bg-gray-50 dark:odd:bg-slate-900">
            <th class="
                border border-slate-300 dark:border-slate-700
                bg-gray-100 dark:bg-slate-700
                text-slate-700 dark:text-slate-200
                font-medium text-left px-4 py-2
            ">
                {html.escape(str(k))}
            </th>
            <td class="
                border border-slate-300 dark:border-slate-700
                text-slate-800 dark:text-slate-200
                px-4 py-2
            ">
                {html.escape(str(v))}
            </td>
        </tr>
        """
        for k, v in rows
    )

    # Enable vertical scroll only for large metadata sets
    scroll_wrapper_class = (
        "max-h-64 overflow-y-auto pr-1"
        if row_count > max_visible_rows
        else ""
    )

    return f"""
<div class="{scroll_wrapper_class}">
    <table class="table-auto border-collapse w-full text-sm mt-2">
        <tbody>
            {html_rows}
        </tbody>
    </table>
</div>
"""

# --- Preformatted Text Renderer ---- For code snippets, error messages, or any preformatted text.


def render_pre(text: str) -> str:
    return f"""
<pre class="rounded-lg overflow-x-auto text-xs leading-tight p-4
            bg-slate-900 text-slate-100
            dark:bg-gray-200 dark:text-slate-900">
{text}
</pre>
"""

# --- Collapsible DataFrame Renderer --- Renders a DataFrame with "Show more" / "Show less" controls to manage large tables without overwhelming the card's layout.


def render_dataframe_collapsible(df, initial_rows: int = 15) -> str:
    if not _HAS_PANDAS:
        return "<div class='text-red-600'>pandas not installed</div>"

    import pandas as pd
    import uuid

    if not isinstance(df, pd.DataFrame):
        return "<div class='text-red-600'>Not a DataFrame</div>"

    uid = f"df_{uuid.uuid4().hex[:8]}"

    # Header
    header = "".join(
        f"""
        <th class="border border-slate-300 dark:border-slate-700
                   bg-gray-100 dark:bg-slate-700
                   px-4 py-2 text-left font-medium text-slate-700 dark:text-slate-200">
            {html.escape(str(col))}
        </th>
        """
        for col in df.columns
    )

    # Rows
    body = []
    for row_idx, (df_index, row) in enumerate(df.iterrows()):
        cells = "".join(
            f"""
            <td class="border border-slate-300 dark:border-slate-700
                       px-4 py-2 text-slate-800 dark:text-slate-200">
                {html.escape(str(v))}
            </td>
            """
            for v in row.values
        )
        hidden_style = "" if row_idx < initial_rows else 'style="display:none"'
        body.append(

            f"""
    <tr class="odd:bg-gray-50 dark:odd:bg-slate-900"
        data-row="{uid}"
        data-hidden="{1 if row_idx >= initial_rows else 0}"
        {hidden_style}>
        <td class="border border-slate-300 dark:border-slate-700
           px-4 py-2 font-medium text-left
           text-slate-800 dark:text-slate-100
           bg-gray-100 dark:bg-slate-800">
            {html.escape(str(df_index))}
        </td>
        {cells}
    </tr>
    """

        )

    rows_html = "".join(body)

    return f"""
<div class="space-y-4">

  <!-- Filter input -->
  <input type="text"
         oninput="filterRows('{uid}', this.value)"
         placeholder="Filter rows..."
         class="w-full px-4 py-2 rounded-md border border-slate-300
                dark:border-slate-600 bg-white dark:bg-slate-800
                text-slate-900 dark:text-slate-100">


<!-- Header: row info + Show less -->
<div class="flex justify-between items-center">
  <div id="{uid}-info"
       class="text-sm text-slate-600 dark:text-slate-300">
    Showing first {initial_rows} of {len(df)} rows
  </div>

  <button id="{uid}-showless"
          onclick="toggleRows('{uid}', false)"
          style="display:none"
          class="px-3 py-1.5 text-sm rounded-md
                 bg-slate-600 text-white hover:bg-slate-700
                 dark:bg-slate-500 dark:hover:bg-slate-600">
    Show less
  </button>
</div>



  <!-- Table -->
  <div class="overflow-x-auto">
    <table class="table-auto border-collapse w-full text-sm">
      <thead>
        <tr>
          <th class="border border-slate-300 dark:border-slate-700
                     bg-gray-100 dark:bg-slate-700
                     px-4 py-2 text-left font-medium">
            Index
          </th>
          {header}
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
  </div>

  <!-- Controls -->
  <div class="flex justify-center gap-3 pt-2">
    <button onclick="toggleRows('{uid}', true)"
            class="px-4 py-2 text-sm rounded-md
                   bg-slate-600 text-white hover:bg-slate-700">
      Show more
    </button>
  </div>

</div>
"""
