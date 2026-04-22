# Basic Probability Distribution Operations Report

This script demonstrates **fundamental probability and probability distribution concepts** using **NumPy, Pandas, SciPy, and Plotly**, and generates a **comprehensive interactive HTML report**.

It connects **probability theory with real data**, showing how empirical data can be used to estimate, visualize, and interpret **discrete and continuous probability distributions**.

---

## 📌 Purpose of the Script

The primary objectives of this script are to:

- Explain **basic probability concepts** using real data
- Demonstrate **discrete and continuous probability distributions**
- Estimate probability model parameters from data
- Visualize **PMF, PDF, KDE, and CDF**
- Explore **joint, marginal, and conditional probabilities**
- Generate a **rich, interactive HTML probability report**

---

## 🧱 Concepts Covered

---

### 1. Probability Foundations

The script demonstrates:
- Simple probability
- Marginal probability
- Joint probability
- Conditional probability
- Empirical vs theoretical probability

Using both **discrete** and **continuous** variables.

---

### 2. Dataset Overview

A synthetic dataset is generated containing:
- **X_discrete** → Discrete variable (dice outcomes: 1–6)
- **Y_continuous** → Continuous variable (normally distributed)

This allows side‑by‑side exploration of:
- Discrete distributions
- Continuous distributions

---

### 3. Discrete Probability Distributions

#### ✅ Discrete Uniform Distribution
- Empirical PMF from dice outcomes
- KDE overlay for density intuition

#### ✅ Bernoulli Distribution
- Success defined as `X_discrete == 1`
- Demonstrates binary probability modeling

#### ✅ Binomial Distribution
- Success event: `X_discrete == 3`
- Probability estimated from data
- Fixed number of trials
- Empirical intuition + theoretical PMF and CDF

#### ✅ Poisson Distribution
- Event: `X_discrete == 6`
- Lambda (λ) estimated from data
- Models event counts over intervals

---

### 4. Continuous Probability Distributions

#### ✅ Continuous Uniform Distribution
- Modeled using min–max range of Y
- Compared against empirical histogram

#### ✅ Exponential Distribution
- Parameter estimated from data mean
- Modeled for non‑negative continuous behavior

#### ✅ Normal (Gaussian) Distribution
- Parameters (μ, σ) estimated from data
- Histogram + PDF + KDE comparison
- Used for probability calculations and CDF examples

---

### 5. Kernel Density Estimation (KDE)

KDE is used throughout the script to:
- Smooth empirical distributions
- Compare real data to theoretical models
- Visualize underlying density clearly

---

### 6. Cumulative Distribution Functions (CDF)

The script demonstrates:
- Empirical CDF
- Fitted Normal CDF
- Comparison between empirical and theoretical CDFs
- Discrete and continuous CDF examples

---

### 7. Joint & Conditional Analysis

Demonstrates:
- Joint PMF via heatmaps
- Conditional PMF: `P(X | Y > threshold)`
- Approximate empirical joint CDF

These concepts are critical for:
- Multivariate probability
- Statistical modeling
- Machine learning features

---

### 8. Standard Normal (Z‑Score) Distribution

- Converts continuous data to Z‑scores
- Visualizes standard normal PDF
- Demonstrates probability computation using z‑values

---

## 📊 Visualizations Included

The HTML report includes:
- PMF + KDE plots
- PDF + KDE plots
- Histograms
- CDF comparisons
- Joint probability heatmaps
- Standard normal distribution plots

All plots are **interactive (Plotly)**.

---

## 📐 Mathematical Formula Integration

The report automatically renders probability and distribution formulas using:
- `FORMULA_REGISTRY`
- Categories such as:
  - Probability
  - Distribution
  - CDF

This bridges **math ↔ code ↔ visualization**.

---

## 📂 Output Details

**Generated HTML Report:**
`reports/basic_probability_distribution_operation_report.html`
---
## 🛠 Utilities Used

| Utility | Purpose |
|------|--------|
| `FORMULA_REGISTRY` | Automatic registration of formulas and rendering |
| `HtmlBuilder` | Structured HTML output |
| `PlotRenderer` | Structured Plot output |
| `ReportUtils` | Save & open HTML reports |

These utilities ensure **clear separation** between:
- Data handling
- Analysis
- Presentation

---