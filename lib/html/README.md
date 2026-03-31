# 📊 Modular Tailwind HTML Rendering System – README

This project provides a **clean, modular, and scalable HTML reporting system** for:

- NumPy arrays
- Pandas Series
- Pandas DataFrames (small & large)
- Dictionary-based metadata and summaries
- Interactive reports with Light/Dark theme
- Large CSV / Excel data previews with filtering
- **On-demand modal (detail) views for truncated content**

The system is designed to **avoid performance issues**, maintain **good UX**, and scale from **small tutorials to large datasets**.

---

## 🧱 Architecture Overview

### File Structure
```
lib/
├── html/
│   ├── __init__.py          # Exports only HtmlBuilder (single import point)
│   ├── builder.py           # HtmlBuilder class - unified OOP interface
│   ├── base.py              # PageBuilder class - page shell, CSS, JS (theme, modal)
│   ├── components.py        # ComponentsBuilder class - cards, grids, layouts
│   ├── renderers.py         # RenderersBuilder class - data visualization
│   └── theme.css            # Tailwind CSS build
│
├── report_utils.py          # Save/open report helpers
└── logger.py                # Centralized logging helper
```

### Class-Based Delegation Pattern
**HtmlBuilder** is the unified interface that internally delegates to three specialized builders:
- **PageBuilder** – Builds complete HTML5 pages with Tailwind CSS, theme toggle, modals
- **ComponentsBuilder** – Creates layout components (cards, grids, full-width sections)
- **RenderersBuilder** – Renders data (arrays, series, dataframes, dicts, preformatted text)

**Benefits:**
- Single import point: `from lib.html import HtmlBuilder`
- Clean separation of concerns
- No code duplication across modules
- Fully object-oriented design

---

## 🎯 Design Principles

1. **Separation of concerns**
   - Rendering logic ≠ layout ≠ page shell
2. **Right renderer for the right data size**
3. **Avoid dumping large data blindly**
4. **Interactive UX for large datasets**
5. **Safe for HTML & Python f-strings**
6. **Consistent Light/Dark theming**
7. **Progressive disclosure of detail (inline → modal)**

---

## 🚀 Quick Start

### Modern OOP API (Recommended)
```python
from lib.html import HtmlBuilder
import numpy as np

builder = HtmlBuilder()

# Use all three builder classes through unified interface
array_card = builder.card("NumPy Array", builder.render_array(np.array([[1, 2], [3, 4]])))
dict_card = builder.card("Stats", builder.render_dict({"mean": 42, "std": 5}))

page = builder.build_page(
    "Report Title",
    builder.grid([array_card, dict_card, ...])
)
```

### Legacy Function-Based API (Backward Compatible)
```python
from lib.html.components import card, grid
from lib.html.renderers import render_array, render_dict

# Old function-based API still works
card("Title", render_array(arr))
```

---

## 🧩 Components (via ComponentsBuilder)

### ✅ `builder.card(title, content)` or `card(title, content)`
**Use when:**
- Content is *small or medium*
- Fits naturally inside a grid
- Summary metrics, arrays, small tables

**Examples:**
- NumPy array preview
- Pandas `Series.head()`
- Small DataFrame

```python
# OOP API
builder.card("Original Array", builder.render_array(arr))

# Legacy API (backward compatible)
card("Original Array", render_array(arr))
```

---

### ✅ `builder.full_width_card(title, content)` or `full_width_card(title, content)`
**Use when:**
- Content is *large or interactive*
- Horizontal scrolling is expected
- Filtering or expansion controls are present

**Examples:**
- Collapsible DataFrame
- Filterable CSV / Excel preview

```python
# OOP API
builder.full_width_card(
    "Housing Dataset – Interactive Preview",
    builder.render_dataframe_collapsible(df)
)

# Legacy API
full_width_card(
    "Housing Dataset – Interactive Preview",
    render_dataframe_collapsible(df)
)
```

---

### ✅ `builder.grid(cards, columns=3)` or `grid(cards, columns=3)`
**Use when:**
- Displaying multiple **small cards**
- Dashboards, comparisons, summaries

```python
# OOP API
builder.grid([
    builder.card("Mean Price", builder.render_dict(stats)),
    builder.card("Array Shape", builder.render_array(arr)),
    builder.card("Top Values", builder.render_series(s))
])

# Legacy API
grid([
    card("Mean Price", render_dict(stats)),
    card("Array Shape", render_array(arr)),
    card("Top Values", render_series(s))
])
```

---

## 📐 Renderers (via RenderersBuilder)

### ✅ `builder.render_array(arr: np.ndarray)` or `render_array(arr: np.ndarray)`
**Purpose:** NumPy array visualization

**Best for:**
- Small to medium arrays
- Teaching / demonstrations
- Metadata inspection

**Renders:**
- Pretty array preview (`<pre>`)
- Shape, dtype, memory usage (key–value table)

**Detail behavior:**
- Inline view always shown
- Uses key–value renderer for metadata
- Modal view is enabled **only if metadata table scrolls**

✅ Use with `card()`

---

### ✅ `builder.render_series(series)` or `render_series(series)`
**Purpose:** Pandas Series display

**Best for:**
- Pandas Series
- Aggregated results
- Label/value inspection

**Renders:**
- Index | Value table

**Detail behavior:**
- If number of rows ≤ threshold → inline table only
- If number of rows > threshold:
  - Inline table with vertical scroll
  - **“View details” button appears**
  - Full Series available in modal

✅ Use with `card()`

---

### ✅ `builder.render_dict(data: dict)` or `render_dict(data: dict)`
**Purpose:** Metadata & summaries

**Best for:**
- Statistics
- Configuration data
- Pandas `.describe().to_dict()` output

**Renders:**
- Key–Value table
- Supports **nested dictionaries** (recursive rendering)

**Detail behavior:**
- Small dictionaries → inline table only
- Large dictionaries →
  - Inline table with vertical scroll
  - **“View details” modal for full structure**

✅ Use with `card()`

---

### ✅ `builder.render_pre(text: str)` or `render_pre(text: str)`
**Purpose:** Monospaced raw text output

**Best for:**
- `df.info()`
- Logs
- Debug output
- Multiline generated text

```python
buf = io.StringIO()
df.info(buf=buf)
card("DataFrame Info", render_pre(buf.getvalue()))
```

✅ Use with `card()`

---

### ✅ `builder.render_dataframe(df)` or `render_dataframe(df)`
**Purpose:** Small to medium DataFrame rendering

**Use when:**
- DataFrame is modest in size
- Displaying `.head()`, `.tail()`, or small computed results

**Detail behavior:**
- If rows ≤ threshold → inline table only
- If rows > threshold →
  - Inline table with vertical scroll
  - **“View details” button shown**
  - Full DataFrame accessible via modal

❌ Not suitable for very large datasets

✅ Use with `card()`

---

### ✅ `builder.render_dataframe_collapsible(df, initial_rows=15)` or `render_dataframe_collapsible(df, initial_rows=15)`
**Purpose:** Interactive DataFrame preview for large datasets

✅ **Recommended for large datasets**

**Features:**
- Show first N rows by default
- Show more / show less controls
- Debounced filter box (search all columns)
- Proper index handling
- Light/Dark theme support
- No external JS libraries

**Detail behavior:**
- Inline preview always visible
- Users progressively expand rows
- Modal not required because full dataset is already accessible interactively

✅ Use with `full_width_card()`

```python
# OOP API
builder.full_width_card(
    "Housing Dataset – Interactive Preview",
    builder.render_dataframe_collapsible(df, initial_rows=15)
)

# Legacy API
full_width_card(
    "Housing Dataset – Interactive Preview",
    render_dataframe_collapsible(df, initial_rows=15)
)
```

---

## 📘 Which renderer to use? (Quick Guide)

| Data type | Size | Renderer | Layout | Details (Modal) |
|---------|------|---------|-------|------------------|
| NumPy array | small/medium | `render_array` | `card` | Conditional |
| Pandas Series | small | `render_series` | `card` | ❌ |
| Pandas Series | large | `render_series` | `card` | ✅ |
| Metadata dict | small | `render_dict` | `card` | ❌ |
| Metadata dict | large | `render_dict` | `card` | ✅ |
| Small DataFrame | small | `render_dataframe` | `card` | ❌ |
| Medium DataFrame | medium | `render_dataframe` | `card` | ✅ |
| Large DataFrame | large | `render_dataframe_collapsible` | `full_width_card` | N/A |

---

## 🎨 Themes

- Supported: **Light & Dark**
- Theme toggled from page header
- All renderers explicitly set text and background colors
- Borders and index columns are visible in both themes
- Modal uses same HTML as inline view

---

## ✅ Best Practices

✅ Do
- **Prefer OOP API**: Use `HtmlBuilder` class for new code
- Mix OOP and legacy APIs: No issues with backward compatibility
- Use modal view only when content is truncated
- Use `builder.render_dataframe_collapsible()` for large datasets
- Separate grid sections from full-width sections
- Keep inline cards readable and compact

❌ Avoid
- Showing “View details” when no extra data exists
- Rendering entire CSV in grid cards
- Filtering without debouncing on large tables
- Relying on inherited text colors

---

## 🧠 Why this design works

- Prevents browser slowdowns
- Keeps grids aligned and stable
- Uses progressive disclosure for details
- Modal does not affect layout flow
- Scales cleanly from small demos to large datasets

---

## 🚀 Future Enhancements (Optional)

- ✅ Column sorting (inline or modal)
- ✅ Sticky headers & index (modal-only)
- ✅ Page size selector (15 / 50 / 100 / All)
- ✅ CSV / Excel download from modal
- ✅ Server-side pagination

---

## ✅ Summary

This system balances **clarity, performance, and extensibility**.

Key ideas:
- Inline views stay compact and readable
- Details are revealed **only when necessary** via modal windows
- Large datasets remain interactive without overwhelming the UI

You’re building this the right way.
