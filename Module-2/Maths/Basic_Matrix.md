# Basic Matrix Operations – Linear Algebra Report

This script demonstrates **fundamental linear algebra operations on matrices and vectors** using **NumPy** and a custom `MatrixHelper` utility, and produces a **step‑by‑step HTML report** with mathematical explanations rendered in **LaTeX**.

It is designed as a **learning and teaching aid** for understanding matrix mathematics beyond black‑box NumPy calls.

---

## 📌 Purpose of This Script

The main objectives of this script are to:

- Demonstrate core **matrix and vector operations**
- Show both:
  - ✅ **Manual / step‑by‑step computation**
  - ✅ **NumPy built‑in equivalents**
- Visualize mathematical steps using **LaTeX**
- Generate a **comprehensive HTML report** for easy learning and review

---

## 🧱 Data Used

### ✅ Matrices
- **Matrix A (3×3)** – used throughout all operations
- **Matrix B (3×3)** – used for dot product
- **Matrix C (4×4)** – random matrix for determinant example

### ✅ Vectors
- **Vector U**
- **Vector V**

These are used for demonstrating **cross product**.

---

## ➕ Operations Demonstrated

### 1. Dot Product (Matrix Multiplication)
- Computes `A · B`
- Shows:
  - Step‑by‑step multiplication
  - Result comparison with `np.dot()`

---

### 2. Cross Product (Vectors)
- Computes `U × V`
- Displays:
  - Geometric vector result
  - NumPy validation using `np.cross()`

---

### 3. Determinant
- Calculates determinant of:
  - Matrix A (3×3)
  - Random Matrix C (4×4)
- Explains:
  - Expansion and reduction steps
  - Comparison with `np.linalg.det()`

---

### 4. Adjoint (Adjugate Matrix)
- Computes the adjoint of matrix A
- Shows:
  - Cofactor expansion
  - Matrix transformation steps

---

### 5. Inverse of a Matrix
- Computes `A⁻¹`
- Displays:
  - Mathematical derivation
  - NumPy comparison using `np.linalg.inv()`

⚠️ Demonstrates why **determinant ≠ 0** is required.

---

### 6. Row Echelon Form (RREF)
- Performs row operations on matrix A
- Shows transformation steps used in:
  - Solving linear systems
  - Rank calculation

---

### 7. Rank of a Matrix
- Calculates matrix rank using row‑reduction
- Compares result with `np.linalg.matrix_rank()`

---

### 8. Eigenvalues and Eigenvectors
- Computes eigenvalues and eigenvectors of matrix A
- Displays:
  - Characteristic equation steps
  - Final eigen pairs
- Includes formatted mathematical output

---

## 🧮 Teaching‑First Design

This script intentionally:
- **Does not rely only on NumPy**
- Shows **how results are derived**
- Helps build **mathematical intuition**

The custom `MatrixHelper` class provides:
- Results
- LaTeX steps
- Structured mathematical explanations

---

## 🖥 HTML Report Output

The script generates a **rich HTML report** containing:

- Input matrices and vectors
- Final computed results
- Side‑by‑side NumPy validations
- LaTeX‑formatted mathematical steps
- Structured cards for each concept

### Output File
**Generated HTML File:**
`reports/basic_matrix_operation_report.html`
---
## 🛠 Utilities Used

| Utility | Purpose |
|------|--------|
| `MatrixHelper` | Matrix Helper to get step by step computation  |
| `HtmlBuilder` | Structured HTML output |
| `ReportUtils` | Save & open HTML reports |

These utilities ensure **clear separation** between:
- Data handling
- Analysis
- Presentation

---