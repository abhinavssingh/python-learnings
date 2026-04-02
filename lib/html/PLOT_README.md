# Plot Rendering Utilities

This package provides **clean, class-based utilities** for converting Python plot objects to styled HTML with consistent, professional layouts.

**`PlotRenderer`** converts any plot object into HTML and wraps it in UI components (cards, grids, containers) using `ComponentsBuilder` for styling.

This design ensures correctness, extensibility, and consistent styling across the application.

---

## Core Design Principle

`PlotRenderer` supports **all three canonical ways** plots become HTML in Python:

| Method | Description | Libraries |
|------|-------------|----------|
| Native HTML | Plot object knows how to emit HTML | Plotly, Bokeh, Altair |
| Image → HTML | Plot rendered as PNG and embedded as `<img>` | Matplotlib, Seaborn |
| JSON Wrapper | Plotly charts wrapped with JavaScript context | Plotly |

This ensures type-safe conversion regardless of plot source.

---

## Module Overview

```
lib/html/
├── plotrenderer.py    # PlotRenderer: Convert any plot to styled HTML
└── components.py      # ComponentsBuilder: Cards, grids, containers
```

---

## PlotRenderer

`PlotRenderer` **converts any plot object to HTML** and wraps it in styled UI components.

### Supported Plot Types

- **Plotly** – `plotly.graph_objs.Figure` (interactive HTML with JavaScript context)
- **Bokeh** – Any object with `.to_html()` method
- **Altair** – Any object with `.to_html()` method
- **Matplotlib** – `matplotlib.figure.Figure` and axes objects (converted to base64 PNG)

### Methods

#### `plot_to_html(plot_obj) → tuple[str, str | None]`

Converts any supported plot object to HTML.

**Returns:**
- For Plotly: `(html_string, plotly_var_name)` – includes JavaScript context for interactivity
- For others: `(html_string, None)`

**Example:**
```python
renderer = PlotRenderer()
html, plotly_var = renderer.plot_to_html(fig)
```

---

#### `plot_to_card(plot_obj, title: str) → str`

Converts plot to HTML and wraps it in a styled card component.

Uses `ComponentsBuilder.chart_card()` for consistent styling.

**Example:**
```python
renderer = PlotRenderer()
html = renderer.plot_to_card(fig, "Sales Chart")
```

---

## Usage Flow

```
Prepare Figure (any library)
  ↓
PlotRenderer.plot_to_html() → detect type & convert
  ↓
PlotRenderer.plot_to_card() → wrap in styled component
  ↓
ComponentsBuilder → add to layout
  ↓
Write to HTML file
```

✅ Automatic type detection
✅ Consistent styling via ComponentsBuilder
✅ Support for Plotly interactivity
✅ Base64 PNG encoding for Matplotlib
