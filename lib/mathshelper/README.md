# mathshelper

The `mathshelper` package provides **reusable mathematical helper utilities** used across the project for **linear algebra, matrix operations, and mathematical reasoning**.

Unlike black‑box numerical computation, helpers in this folder are designed to:
- Compute results
- Explain *how* results are derived
- Support **step‑by‑step mathematical reporting** with LaTeX output

This module forms the **mathematical backbone** of the `Maths` section in *Module‑2*.

---

## 📌 Purpose of This Package

The main objectives of `mathshelper` are to:

- Encapsulate reusable mathematical logic
- Separate **math computation** from **presentation**
- Provide transparent, explainable computations
- Support educational and reporting‑based workflows
- Bridge mathematical theory with Python code

This package is **learning‑first**, not performance‑first.

---

## 📂 Package Overview

The `mathshelper` folder contains multiple helper modules, each focused on a specific mathematical responsibility.

### 🧮 Matrix‑Related Helpers

These helpers deal with **matrix and vector operations**, commonly required for:
- Linear algebra
- Machine learning foundations
- Scientific computing

Operations supported across helpers include:
- Dot product (matrix multiplication)
- Cross product (vector)
- Determinant
- Adjoint (adjugate)
- Inverse
- Row echelon / RREF
- Rank
- Eigenvalues and eigenvectors

Each operation is implemented in a way that can return:
- ✅ The final numerical result
- ✅ Intermediate steps (when applicable)
- ✅ LaTeX‑formatted explanations

---

### 📐 Step‑by‑Step Mathematics Helpers

Some helpers inside this package focus on:

- Breaking complex operations into smaller mathematical steps
- Producing readable **LaTeX math expressions**
- Making internal logic explicit for:
  - Learning
  - Debugging
  - Teaching
  - Report generation

These helpers are heavily used in HTML reports where **mathematical clarity** is required.

---

### ⚙️ Supporting Math Utilities

Other helper files in this folder provide:
- Computation support for higher‑level maths modules
- Reusable calculation patterns
- Small mathematical building blocks

These utilities are not always used directly but support:
- Matrix helpers
- Probability and statistics exercises
- Reporting logic in the `Maths` module

---

## 🖥 Integration with Reporting

The `mathshelper` package is commonly used alongside:

- `HtmlBuilder` – for structured HTML output
- `ReportUtils` – for saving and opening reports
- `PlotRenderer` – for combining math with visuals

Typical flow:
1. Compute using `mathshelper`
2. Capture steps + result
3. Render math with LaTeX
4. Export as an HTML report

---

## ▶️ Example Usage

```python
from lib.mathshelper.matrixhelper import MatrixHelper
import numpy as np

helper = MatrixHelper()

A = np.array([[1, 2], [3, 4]])
result = helper.inverse(A)

print(result["result"])        # Numerical output
print(result["steps_latex"])   # Mathematical steps in LaTeX
```

### Formula System – Mathematical Formula Registry & Auto‑Registration

This module provides a **scalable, extensible framework for managing mathematical formulas** used across statistics, probability, and distribution reports.

It enables **automatic registration, categorization, and rendering of LaTeX formulas** without hardcoding formulas into report logic, ensuring clean separation between **math, metadata, and presentation**.

---

## 🎯 Design Goals

The formula system is designed to:

- Centralize all mathematical formulas in one place
- Avoid hardcoding formulas in reports or analysis scripts
- Automatically register formulas across multiple modules
- Categorize formulas by domain (Probability, Distribution, Statistics, CDF)
- Support structured rendering in HTML reports
- Enable easy extension without touching rendering code

---

## 🧱 Core Components

### 1. `Formula` (Data Model)

The `Formula` class represents a **single mathematical formula** with associated metadata.

```python
@dataclass
class Formula:
    key: str
    title: str
    latex: str
    category: str
    subcategory: Optional[str] = None
```
Responsibilities

- Stores LaTeX representation of the formula
- Holds semantic metadata (title, category, subcategory)
- Knows how to render itself using a report builder
---

### 2. 2. FormulaRegistry
The FormulaRegistry acts as an in‑memory catalog of all registered formulas.
```
class FormulaRegistry:
    def add(self, formula: Formula)
    def get(self, key: str) -> Formula
    def by_category(self, category: str)
    def all(self)
```
Key Features

- Lookup by formula key
- Filter formulas by category
- Easily iterate over all formulas

---
### 3. FORMULA_METADATA
FORMULA_METADATA defines the semantic meaning of each formula.
It does not store LaTeX itself—only descriptive information.
```
FORMULA_METADATA = {
    "simple_prob_ltx": {
        "title": "Simple Probability (Discrete):",
        "category": "Probability",
        "subcategory": "Discrete",
    },
    ...
}
```
Why Metadata Matters

- Titles are human‑readable
- Categories enable filtering and grouping
- Subcategories provide finer classification
- LaTeX definitions stay separate (single responsibility)

### 4. LaTeX Constants Modules
Each constants module defines LaTeX strings, such as:
```
Pythonsimple_prob_ltx = r"P(X = x) = \frac{\text{Number of favorable outcomes}}{\text{Total outcomes}}"
```
---

## 🔄 Automatic Formula Registration
This function dynamically discovers and registers formulas:
```
def auto_register_formulas(registry, constants_modules):
    for module in constants_modules:
        for const_name, meta in FORMULA_METADATA.items():
            latex_value = getattr(module, const_name, None)
            if latex_value is None:
                continue

            registry.add(
                Formula(
                    key=const_name.replace("_ltx", ""),
                    title=meta["title"],
                    latex=latex_value,
                    category=meta["category"],
                    subcategory=meta.get("subcategory"),
                )
            )
```
### ✅ How Auto‑Registration Works

- Iterates through provided LaTeX constant modules
- Matches constants against FORMULA_METADATA
- Extracts LaTeX expressions dynamically
- Constructs Formula objects
- Registers them into FormulaRegistry


### ✅ Benefits of Auto‑Registration

- ✅ Zero manual wiring of formulas
- ✅ Adding a new formula requires:

    - One LaTeX constant
    - One metadata entry


- ✅ Reports pick up formulas automatically
- ✅ Consistent formatting everywhere
- ✅ No duplicate or out‑of‑sync formulas