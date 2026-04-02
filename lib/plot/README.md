# Plot Rendering Utilities

This package provides **clean, layered utilities** for building heatmaps and rendering plots as HTML in a consistent, extensible way.

It is intentionally split into two responsibilities:

1. **`PlotUtility`** – creates heatmap **figures** using Seaborn or Plotly
2. **`PlotRenderer`** – converts plot objects into **HTML** and applies UI layout (cards, grids)

This separation ensures correctness, extensibility, and avoids common plotting errors.

---

## Core Design Principle

> **There are only three valid ways plots become HTML in Python**

| Method | Description | Libraries |
|------|-------------|----------|
| Native HTML | Plot object knows how to emit HTML | Plotly, Bokeh, Altair |
| Image → HTML | Plot rendered as PNG/SVG and embedded as `<img>` | Matplotlib, Seaborn |
| JS Spec → HTML | JSON spec rendered by JavaScript | Vega‑Lite, D3 wrappers |

This project strictly enforces this rule.

---

## Module Overview

```
lib/
├── plot/
│   ├── plotutility.py   # Create heatmap figures
│   └── plotbuilder.py   # Convert plots to HTML + UI layout
└── html/
    └── components.py    # Cards, grids, containers
```

---

## PlotUtility

`PlotUtility` **creates heatmap figures** only. It does not aggregate data, convert HTML, or handle layout.

### Supported Backends

- `seaborn` → static matplotlib heatmap (image-based)
- `plotly`  → interactive heatmap (native HTML)

### Example Usage

```python
from lib.plot.plotutility import PlotUtility

fig = PlotUtility.plot_heatmap(
    aggregated_df,
    index="Income Category",
    columns="Age Category",
    backend="plotly",
    title="Age vs Income Category"
)
```

Returned object:
- Plotly backend → `plotly.graph_objs.Figure`
- Seaborn backend → `matplotlib.figure.Figure`

---

## PlotRenderer

`PlotRenderer` **converts plot objects into HTML** and wraps them in UI components.

### Responsibilities

- Detect plot type safely
- Convert to HTML correctly
- Apply cards, grids, and containers

### Example Usage

```python
from lib.plot.plotbuilder import PlotRenderer

renderer = PlotRenderer()
html = renderer.plot_to_card(fig, "Age vs Income Heatmap")
```

---

## Correct Usage Flow (Very Important)

```
Raw Data
  ↓
Group / Aggregate (groupby + size)
  ↓
PlotUtility → create Figure
  ↓
PlotRenderer → convert to HTML
  ↓
ComponentsBuilder → layout
```

❌ Never pivot raw data
❌ Never send DataFrame to PlotRenderer
✅ Always send a Figure

---

## Common Errors Prevented

| Issue | Prevented By |
|----|----|
Heatmap shows only `1`s | Required aggregation before plotting |
Plot not visible | Explicit height + correct HTML rendering |
Wrong adapter | Strict plot-type detection |
Broken HTML | Single canonical conversion path |

---

## Extensibility

This architecture easily supports:

- Normalized heatmaps (percent view)
- Log-scaled colorbars
- Additional backends (Altair, Vega‑Lite)
- Export to static images

---

## Summary

✅ Clear separation of concerns
✅ No plotting magic
✅ Correct HTML semantics
✅ Production‑grade structure

This design mirrors how professional analytics dashboards are built.
