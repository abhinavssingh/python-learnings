# Linear Regression Mathshelper Report Generator

This script generates an HTML report containing formulas and explanations related to **Linear Regression, Regularization, and Derivations** using a custom HTML builder and formula registry.

---

## 🚀 Features

- 📄 Generates a styled HTML report
- 🧮 Renders mathematical formulas dynamically
- 📚 Supports categorized formula selection
- 🌐 Automatically saves and opens report in browser
- 🧩 Modular design using reusable utilities

---

## 📦 Dependencies

Ensure the following modules are available in your project:

- `HtmlBuilder`
- `FORMULA_REGISTRY`
- `ReportUtils`

---

## 📂 Project Structure (Example)

```
lib/
├── html.py
├── mathshelper.py
├── utility/
│   └── reports/
│       └── report_utils.py
```

---

## 🧠 How It Works

### 1. Fetch Formulas

```python
FORMULA_REGISTRY.by_category([
    "Linear Regression",
    "Regularization",
    "Derivation"
])
```

- Retrieves formulas grouped by category

---

### 2. Render HTML Blocks

```python
formula.render(builder)
```

- Each formula is rendered into HTML using `HtmlBuilder`

---

### 3. Layout with Grid

```python
builder.grid([...])
```

- Organizes formulas into a structured grid layout

---

### 4. Build Complete HTML Page

```python
builder.build_page(title, content)
```

- Wraps content into a full HTML document

---

### 5. Save Report

```python
ru.save_html_report(...)
```

- Saves file locally
- Optionally opens it in a browser

---

## 📥 Usage

### Run Script

```bash
python your_script.py
```

---

## 📝 Output

- File: `ml_linear_regression_maths_report.html`
- Location: `/reports` directory (configurable)
- Automatically opens in browser

---

## ⚙️ Example Code

```python
builder = HtmlBuilder()
content = []

content.append(
    builder.grid([
        formula.render(builder)
        for formula in FORMULA_REGISTRY.by_category([
            "Linear Regression",
            "Regularization",
            "Derivation"
        ])
    ])
)

html_doc = builder.build_page(
    "Linear Regression Mathshelper Report",
    "
".join(content)
)

output_path = ru.save_html_report(
    __file__,
    "ml_linear_regression_maths_report.html",
    html_doc,
    subfolder="reports",
    open_in_browser=True
)
```

---

## ✅ Key Benefits

- Automates report generation
- Clean separation of rendering and logic
- Easily extendable with new formula categories
- Great for documentation and learning

---

## 📜 License

Free to use for learning and internal projects 🚀
