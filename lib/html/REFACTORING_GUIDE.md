# HTML Builder Refactoring Guide

## Overview

The `lib/html` module has been completely refactored from a **functional** approach to an **Object-Oriented Programming (OOPS)** approach. This refactoring consolidates all HTML generation, component creation, and data rendering into a single `HtmlBuilder` class.

## What Changed

### Before (Functional Approach)
```python
# Multiple imports required
from lib.html.base import build_html_page
from lib.html.components import card, grid, full_width_card, chart_card
from lib.html.renderers import render_array, render_dataframe, render_dict, render_kv

# Usage scattered across different imports
html = build_html_page(
    "Title",
    grid([
        card("Card 1", render_array(arr1)),
        card("Card 2", render_dict(data))
    ])
)
```

### After (OOPS Approach)
```python
# Single import only!
from lib.html import HtmlBuilder

# Everything accessed through one instance
builder = HtmlBuilder()
html = builder.build_page(
    "Title",
    builder.grid([
        builder.card("Card 1", builder.render_array(arr1)),
        builder.card("Card 2", builder.render_dict(data))
    ])
)
```

## Benefits

1. **Single Import**: Only import `HtmlBuilder` instead of 4+ different functions/modules
2. **Cleaner Code**: All methods are logically grouped in one class
3. **Better OOP Design**: Encapsulation - related functionality is bundled together
4. **Easier Maintenance**: All HTML logic in one place (`builder.py`)
5. **Scalability**: Easy to add new methods or features
6. **Reduced Import Complexity**: No more managing multiple imports across files

## New File Structure

```
lib/html/
├── __init__.py          # Exports HtmlBuilder
├── builder.py           # Main HtmlBuilder class (NEW)
├── base.py              # Kept for backward compatibility
├── components.py        # Kept for backward compatibility
├── renderers.py         # Kept for backward compatibility
├── theme.css            # Styling (unchanged)
├── theme.min.css        # Minified styling (unchanged)
└── README.md
```

## HtmlBuilder Class Methods

### Page & Layout Methods
- `build_page(title, body_html)` - Build complete HTML page
- `card(title, content)` - Create standard card
- `grid(cards, columns=3)` - Create responsive grid
- `full_width_card(title, content)` - Full-width card
- `chart_card(title, content)` - Chart-optimized card
- `chart_grid_2x2(cards)` - 2x2 chart layout
- `chart_container(content, height=300)` - Fixed-height container

### Data Rendering Methods
- `render_array(arr)` - Render NumPy arrays
- `render_series(s, max_visible_rows=5)` - Render Pandas Series or lists
- `render_dataframe(df, max_visible_rows=5)` - Render Pandas DataFrames
- `render_dict(d, title=None, max_visible_rows=1)` - Render dictionaries
- `render_kv(rows, max_visible_rows=5)` - Render key-value pairs
- `render_pre(text, max_visible_lines=15)` - Render preformatted text
- `render_dataframe_collapsible(df, initial_rows=15)` - Collapsible DataFrame

## Migration Examples

### Example 1: NumPy Report
**Before:**
```python
from lib.html.base import build_html_page
from lib.html.components import card, grid
from lib.html.renderers import render_array

html = build_html_page("Report", grid([
    card("Array", render_array(arr))
]))
```

**After:**
```python
from lib.html import HtmlBuilder

builder = HtmlBuilder()
html = builder.build_page("Report", builder.grid([
    builder.card("Array", builder.render_array(arr))
]))
```

### Example 2: Pandas Report
**Before:**
```python
from lib.html.base import build_html_page
from lib.html.components import card, grid, full_width_card
from lib.html.renderers import render_dataframe, render_dict

html = build_html_page("Data Report", grid([
    card("Data", render_dataframe(df)),
    card("Stats", render_dict(stats))
]))
```

**After:**
```python
from lib.html import HtmlBuilder

builder = HtmlBuilder()
html = builder.build_page("Data Report", builder.grid([
    builder.card("Data", builder.render_dataframe(df)),
    builder.card("Stats", builder.render_dict(stats))
]))
```

### Example 3: Mixed Components
**Before:**
```python
from lib.html.base import build_html_page
from lib.html.components import full_width_card, chart_grid_2x2, chart_card
from lib.html.renderers import render_pre

html = build_html_page("Dashboard", [
    full_width_card("Logs", render_pre(log_text)),
    chart_grid_2x2([
        chart_card("Chart 1", chart_html_1),
        chart_card("Chart 2", chart_html_2)
    ])
])
```

**After:**
```python
from lib.html import HtmlBuilder

builder = HtmlBuilder()
html = builder.build_page("Dashboard", 
    builder.full_width_card("Logs", builder.render_pre(log_text)) +
    builder.chart_grid_2x2([
        builder.chart_card("Chart 1", chart_html_1),
        builder.chart_card("Chart 2", chart_html_2)
    ])
)
```

## Path Setup

The refactored scripts use automatic path discovery:

```python
import sys
from pathlib import Path

# Automatically find project root
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from lib.html import HtmlBuilder
```

This ensures scripts work whether run directly or through the main runner.

## Backward Compatibility

The original files (`base.py`, `components.py`, `renderers.py`) are still present and functional. You can:
- Gradually migrate projects using the old import style
- Mix old and new approaches in the same project if needed

However, **new projects should use the `HtmlBuilder` class** for consistency.

## Testing Your Migration

After updating your files:

```python
# Your test file
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from lib.html import HtmlBuilder

builder = HtmlBuilder()
print("✓ HtmlBuilder imported successfully!")

# Test a simple page
html = builder.build_page(
    "Test", 
    builder.card("Test Card", "<p>It works!</p>")
)
print(f"✓ HTML generated: {len(html)} characters")
```

## Next Steps

1. Update all your report files to use `HtmlBuilder`
2. Test each report to ensure HTML generation works
3. Run your reports through the normal workflow
4. Remove old import patterns once migration is complete

---

For questions or issues, refer to the `builder.py` docstrings for detailed method documentation.
