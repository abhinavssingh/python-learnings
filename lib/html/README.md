# 📊 Modular Tailwind HTML Rendering System – README

This project provides a **clean, modular, and scalable HTML reporting system** for:

- NumPy arrays
- Pandas Series
- Pandas DataFrames (small & large)
- Dictionary-based metadata and summaries
- Interactive reports with Light/Dark theme
- Large CSV / Excel data previews with filtering

The system is designed to **avoid performance issues**, maintain **good UX**, and scale from **small tutorials to large datasets**.

---

## 🧱 Architecture Overview

```
lib/
├── html/
│   ├── base.py          # Page shell, CSS injection, JS helpers
│   ├── components.py    # Layout components (cards, grids)
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

---

## 🧩 Components (`components.py`)

### ✅ `card(title, content)`
**Use when:**
- Content is *small or medium*
- Fits naturally inside a grid
- Summary metrics, arrays, small tables

**Examples:**
- NumPy array preview
- Pandas `Series.describe()`
- Small DataFrame (`df.head()`)

```python
card("Original Array", render_array(arr))
```

---

### ✅ `full_width_card(title, content)`
**Use when:**
- Content is *large or interactive*
- Tables need horizontal scrolling
- User interaction is expected

**Examples:**
- Large CSV / Excel preview
- Collapsible DataFrame
- Filterable datasets

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
- Shape, dtype, memory usage (tabular)

✅ Use with `card()`

---

### ✅ `render_series(series)`
**Purpose:** Pandas Series display

**Best for:**
- Pandas Series
- Small list-like data
- Aggregated results

**Renders:**
- Index | Value table
- Styled for light & dark mode

✅ Use with `card()`

---

### ✅ `render_dict(data: dict)`
**Purpose:** Metadata & summaries

**Best for:**
- Statistics
- Configuration data
- Aggregation results

**Renders:**
- Key-Value table with borders
- Handles nested dicts and NumPy types

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
**Purpose:** Small DataFrame rendering

**Use ONLY when:**
- `df.shape[0] <= 20–30 rows`
- `df.head()`, `df.describe()`

❌ Do NOT use for full CSV / Excel

✅ Use with `card()`


---

### ✅ `render_dataframe_collapsible(df, initial_rows=15)`
**Purpose:** Interactive DataFrame preview

✅ **Recommended for large datasets**

**Features:**
- Show first N rows by default
- Show more / show less
- Filter box (search all columns)
- Proper index handling
- Light/Dark theme support
- No external JS libraries

✅ Use with `full_width_card()`

```python
full_width_card(
    "Housing Dataset – Interactive Preview",
    render_dataframe_collapsible(df, initial_rows=15)
)
```

---

## 📘 Which renderer to use? (Quick Guide)

| Data type | Size | Renderer | Layout |
|---------|------|---------|-------|
NumPy array | small/medium | `render_array` | `card` |
Pandas Series | any | `render_series` | `card` |
Metadata dict | any | `render_dict` | `card` |
Raw text (`.info()`) | any | `render_pre` | `card` |
Small DataFrame | ≤ 20 rows | `render_dataframe` | `card` |
Interactive large DF | large | `render_dataframe_collapsible` | `full_width_card` |

---

## 🎨 Themes

- Supported: **Light & Dark**
- Theme toggled from page header
- All renderers are theme-aware
- Index column, borders, text contrast handled explicitly

---

## ✅ Best Practices

✅ Do
- Use `render_dataframe_collapsible()` for large data
- Separate grid sections from full-width sections
- Reset index in renderers for user clarity
- Keep large tables out of grids

❌ Avoid
- Rendering entire CSV without pagination/collapse
- Dumping 1000+ rows in grid cards
- Mixing summary cards with large tables
- Relying on implicit Tailwind text colors

---

## 🧠 Why this design works

- Prevents browser slowdowns
- Keeps HTML size reasonable
- Clear UX separation
- Easy future extension:
  - pagination
  - column sorting
  - export buttons
  - sticky headers

---

## 🚀 Future Enhancements (Optional)

- ✅ Column sorting
- ✅ Sticky headers & index
- ✅ Page size selector (15 / 50 / 100 / All)
- ✅ CSV / Excel download
- ✅ Server-side pagination

---

## ✅ Summary

This system is designed to scale from **learning notebooks** to **production-grade HTML dashboards**.

Using the **right renderer + right component** is the key to:
- Performance
- UX
- Maintainability

You’re building this exactly the right way.
