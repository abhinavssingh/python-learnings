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

```
lib/
├── html/
│   ├── base.py          # Page shell, CSS injection, JS helpers (theme, modal, filter)
│   ├── components.py    # Layout components (cards, grids, full-width sections)
│   ├── renderers.py     # Data renderers (arrays, series, dataframes)
│   └── theme.css        # Tailwind CSS build
│
├── report_utils.py      # Save/open report helpers
└── logger.py            # Centralized logging helper
```

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

## 🧩 Components (`components.py`)

### ✅ `card(title, content)`
**Use when:**
- Content is *small or medium*
- Fits naturally inside a grid
- Summary metrics, arrays, small tables

**Examples:**
- NumPy array preview
- Pandas `Series.head()`
- Small DataFrame

```python
card("Original Array", render_array(arr))
```

---

### ✅ `full_width_card(title, content)`
**Use when:**
- Content is *large or interactive*
- Horizontal scrolling is expected
- Filtering or expansion controls are present

**Examples:**
- Collapsible DataFrame
- Filterable CSV / Excel preview

```python
full_width_card(
    "Housing Dataset – Interactive Preview",
    render_dataframe_collapsible(df)
)
```

---

### ✅ `grid(cards, columns=3)`
**Use when:**
- Displaying multiple **small cards**
- Dashboards, comparisons, summaries

```python
grid([
    card("Mean Price", render_dict(stats)),
    card("Array Shape", render_array(arr)),
    card("Top Values", render_series(s))
])
```

---

## 📐 Renderers (`renderers.py`)

### ✅ `render_array(arr: np.ndarray)`
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

### ✅ `render_series(series)`
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

### ✅ `render_dict(data: dict)`
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

### ✅ `render_pre(text: str)`
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

### ✅ `render_dataframe(df)`
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

### ✅ `render_dataframe_collapsible(df, initial_rows=15)`
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
- Use modal view only when content is truncated
- Use `render_dataframe_collapsible()` for large datasets
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
