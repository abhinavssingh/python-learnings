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
    uid = f"series_{uuid.uuid4().hex[:8]}"

    def build_rows(iterable):
        return "\n".join(
            f"""
            <tr class="odd:bg-gray-50 dark:odd:bg-slate-900">
              <td class="
border border-slate-300 dark:border-slate-700
    text-slate-900 dark:text-slate-100
    bg-white dark:bg-slate-800
    px-4 py-2
">
                {html.escape(str(i))}
              </td>
              <td class="
border border-slate-300 dark:border-slate-700
    text-slate-900 dark:text-slate-100
    bg-white dark:bg-slate-800
    px-4 py-2
">
                {html.escape(str(v))}
              </td>
            </tr>
            """
            for i, v in iterable
        )

    if isinstance(s, (list, tuple, np.ndarray)):
        rows = build_rows(enumerate(s))
        row_count = len(s)
    elif _HAS_PANDAS and isinstance(s, pd.Series):
        rows = build_rows(s.items())
        row_count = len(s)
    else:
        return "<div class='text-red-600'>Cannot render series</div>"

    has_scroll = row_count > max_visible_rows
    scroll_class = "max-h-64 overflow-y-auto pr-1" if has_scroll else ""

    table = f"""
    <table class="table-auto border-collapse w-full text-sm">
      <thead>
        <tr>
          <th class="
border border-slate-300 dark:border-slate-700
    text-slate-900 dark:text-slate-100
    bg-gray-100 dark:bg-slate-700
    px-4 py-2 font-medium
">Index</th>
          <th class="
border border-slate-300 dark:border-slate-700
    text-slate-900 dark:text-slate-100
    bg-gray-100 dark:bg-slate-700
    px-4 py-2
">Value</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
    """

    return f"""
<div class="{scroll_class}">
  {table}
</div>
{(
        f'''
<button onclick="openModalFromTemplate('{uid}')"
        class="mt-2 text-sm px-3 py-1.5 rounded bg-slate-600 text-white hover:bg-slate-700">
  View details
</button>

<template id="{uid}">
  {table}
</template>
'''
        if has_scroll else ""
    )}
"""


# --- Pandas DataFrame ---
def render_dataframe(df, max_visible_rows: int = 5):
    if not _HAS_PANDAS:
        return "<div class='text-red-600'>pandas not installed</div>"

    if not isinstance(df, pd.DataFrame):
        return "<div class='text-red-600'>Invalid DataFrame</div>"

    if df.empty:
        return "<div class='italic text-slate-500'>Empty DataFrame</div>"

    uid = f"df_{uuid.uuid4().hex[:8]}"
    row_count = len(df)
    has_scroll = row_count > max_visible_rows

    # ---- Header (unchanged from old version) ----
    header = "".join(
        f"""
        <th class="border border-slate-300 dark:border-slate-700
                   bg-gray-100 dark:bg-slate-700
                   text-slate-900 dark:text-slate-100
                   px-4 py-2 font-medium">
            {html.escape(str(c))}
        </th>
        """
        for c in df.columns
    )

    # ---- Rows (restore striping + old backgrounds) ----
    rows = ""
    for idx, row in df.iterrows():
        cells = "".join(
            f"""
            <td class="border border-slate-300 dark:border-slate-600
                       text-slate-900 dark:text-slate-100
                       px-4 py-2">
                {html.escape(str(v))}
            </td>
            """
            for v in row.values
        )

        rows += f"""
<tr class="odd:bg-gray-50 dark:odd:bg-slate-900">
    <td class="border border-slate-300 dark:border-slate-600
               bg-gray-100 dark:bg-slate-800
               text-slate-900 dark:text-slate-100
               px-4 py-2 font-medium">
        {html.escape(str(idx))}
    </td>
    {cells}
</tr>
"""

    table = f"""
<table class="table-auto border-collapse w-full text-sm">
    <thead>
        <tr>
            <th class="border border-slate-300 dark:border-slate-700
                       bg-gray-100 dark:bg-slate-700
                       text-slate-900 dark:text-slate-100
                       px-4 py-2 font-medium">
                Index
            </th>
            {header}
        </tr>
    </thead>
    <tbody>
        {rows}
    </tbody>
</table>
"""

    scroll_class = "max-h-72 overflow-y-auto pr-1" if has_scroll else ""

    return f"""
<div class="{scroll_class} overflow-x-auto">
  {table}
</div>
{(
        f'''
<button onclick="openModalFromTemplate('{uid}')"
        class="mt-2 text-sm px-3 py-1.5 rounded
               bg-slate-600 text-white hover:bg-slate-700">
  View details
</button>

<template id="{uid}">
  {table}
</template>
'''
        if has_scroll else ""
    )}
"""


# --- Python Dict ---
def render_dict(d: dict, title: str | None = None, max_visible_rows: int = 1):
    uid = f"dict_{uuid.uuid4().hex[:8]}"
    row_count = len(d)
    has_scroll = row_count > max_visible_rows

    def clean_value(v):
        # NumPy scalar → Python scalar
        if isinstance(v, (np.generic,)):
            return v.item()

    # Plain Python scalar
        if isinstance(v, (int, float, bool, str)):
            return v

    # Lists / arrays
        if isinstance(v, (list, tuple, np.ndarray)):
            return ", ".join(str(clean_value(x)) for x in v)

    # ✅ FIX: Nested dict → render recursively
        if isinstance(v, dict):
            return render_dict(v)

    # Fallback
        return str(v)

    rows = "\n".join(
        f"""
        <tr class="odd:bg-gray-50 dark:odd:bg-slate-900">
          <th class="
border border-slate-300 dark:border-slate-700
    text-slate-900 dark:text-slate-100
    bg-gray-100 dark:bg-slate-700
    px-4 py-2 font-medium
">
            {html.escape(str(k))}
          </th>
          <td class="
border border-slate-300 dark:border-slate-700
    text-slate-900 dark:text-slate-100
    bg-white dark:bg-slate-800
    px-4 py-2
">
            {clean_value(v)}
          </td>
        </tr>
        """
        for k, v in d.items()
    )

    scroll_class = "max-h-72 overflow-y-auto pr-1" if has_scroll else ""
    header = f"<h3 class='text-md font-semibold mb-2'>{title}</h3>" if title else ""

    return f"""
{header}
<div class="{scroll_class}">
  <table class="table-auto border-collapse w-full text-sm">
    <tbody>{rows}</tbody>
  </table>
</div>
{(
        f'''
<button onclick="openModalFromTemplate('{uid}')"
        class="mt-2 text-sm px-3 py-1.5 rounded bg-slate-600 text-white hover:bg-slate-700">
  View details
</button>

<template id="{uid}">
  <table class="table-auto border-collapse w-full text-sm">
    <tbody>{rows}</tbody>
  </table>
</template>
'''
        if has_scroll else ""
    )}
"""
# --- Generic Key–Value Renderer ---


def render_kv(rows, max_visible_rows: int = 5):
    uid = f"kv_{uuid.uuid4().hex[:8]}"
    row_count = len(rows)
    has_scroll = row_count > max_visible_rows

    html_rows = "\n".join(
        f"""
        <tr class="odd:bg-gray-50 dark:odd:bg-slate-900">
          <th class="
              border border-slate-300 dark:border-slate-600
              bg-gray-100 dark:bg-slate-700
              text-slate-900 dark:text-slate-100
              px-4 py-2 font-medium text-left
          ">
              {html.escape(str(k))}
          </th>
          <td class="
              border border-slate-300 dark:border-slate-600
              bg-white dark:bg-slate-800
              text-slate-900 dark:text-slate-100
              px-4 py-2
          ">
              {html.escape(str(v))}
          </td>
        </tr>
        """
        for k, v in rows
    )

    scroll_class = "max-h-64 overflow-y-auto pr-1" if has_scroll else ""

    return f"""
<div class="{scroll_class}">
  <table class="table-auto border-collapse w-full text-sm mt-2">
    <tbody>{html_rows}</tbody>
  </table>
</div>
{(
        f'''
<button onclick="openModalFromTemplate('{uid}')"
        class="mt-2 text-sm px-3 py-1.5 rounded bg-slate-600 text-white hover:bg-slate-700">
  View details
</button>

<template id="{uid}">
  <table class="table-auto border-collapse w-full text-sm">
    <tbody>{html_rows}</tbody>
  </table>
</template>
'''
        if has_scroll else ""
    )}
"""
# --- Preformatted Text Renderer ---- For code snippets, error messages, or any preformatted text.


def render_pre(text: str, max_visible_lines: int = 15) -> str:
    uid = f"pre_{uuid.uuid4().hex[:8]}"

    # Normalize line endings and count lines
    lines = text.splitlines()
    line_count = len(lines)

    has_scroll = line_count > max_visible_lines

    # Inline (possibly truncated) view
    inline_pre = f"""
<pre class="
    rounded-lg
    text-xs
    leading-tight
    p-4
    bg-slate-900
    text-slate-100
    dark:bg-gray-200
    dark:text-slate-900
    {'max-h-80 overflow-y-auto' if has_scroll else ''}
">
{html.escape(text)}
</pre>
"""

    # Full (modal) view – no height restriction
    modal_pre = f"""
<pre class="
    rounded-lg
    text-xs
    leading-tight
    p-4
    bg-slate-900
    text-slate-100
    dark:bg-gray-200
    dark:text-slate-900
">
{html.escape(text)}
</pre>
"""

    return f"""
{inline_pre}
{(
        f'''
<button onclick="openModalFromTemplate('{uid}')"
        class="mt-2 text-sm px-3 py-1.5 rounded
               bg-slate-600 text-white hover:bg-slate-700">
  View details
</button>

<template id="{uid}">
  {modal_pre}
</template>
'''
        if has_scroll else ""
    )}
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
        <th class="
border border-slate-300 dark:border-slate-700
    text-slate-900 dark:text-slate-100
    bg-gray-100 dark:bg-slate-700
    px-4 py-2 font-medium
">
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
            <td class="
border border-slate-300 dark:border-slate-700
    text-slate-900 dark:text-slate-100
    bg-white dark:bg-slate-800
    px-4 py-2
">
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
        <td class="
border border-slate-300 dark:border-slate-700
    text-slate-900 dark:text-slate-100
    bg-white dark:bg-slate-800
    px-4 py-2
">
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
              <th class="
border border-slate-300 dark:border-slate-700
    text-slate-900 dark:text-slate-100
    bg-gray-100 dark:bg-slate-700
    px-4 py-2 font-medium
">
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
