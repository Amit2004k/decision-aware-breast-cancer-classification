# 🎗️ Decision-Aware Machine Learning Framework for Reliable Breast Cancer Classification

[![Paper](https://img.shields.io/badge/IEEE-ICITIIT%202026-blue?style=flat-square&logo=ieee)](https://ieee.org)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green?style=flat-square&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Stars](https://img.shields.io/github/stars/Amit2004k/decision-aware-breast-cancer-classification?style=flat-square)](https://github.com/Amit2004k/decision-aware-breast-cancer-classification/stargazers)

> **Published at IEEE ICITIIT 2026** — Accepted and Presented
> *A Decision-Aware Machine Learning Framework for Reliable Breast Cancer Classification*

---

## 🔥 Highlights

- ✅ **Decision-aware evaluation** beyond accuracy — integrates clinical cost sensitivity directly into model selection
- ✅ **Probability calibration** (Platt Scaling, Isotonic Regression) for trustworthy confidence scores
- ✅ **Explainability** via SHAP, LIME, and decision tree surrogates for clinical interpretability
- ✅ **Cost-sensitive learning** with asymmetric misclassification penalties (False Negatives penalized 5×)
- ✅ **Robustness testing** under feature perturbation and distribution shift
- ✅ Evaluated on **Wisconsin Breast Cancer Dataset (WBCD)** — 569 samples, 30 features

---

## 📊 Results at a Glance

| Model | Accuracy | AUC-ROC | F1 (Malignant) | ECE (↓) | Brier Score (↓) |
|-------|----------|---------|----------------|---------|------------------|
| Logistic Regression | 97.4% | 0.998 | 0.974 | 0.021 | 0.019 |
| Random Forest | 96.5% | 0.997 | 0.963 | 0.031 | 0.027 |
| **Decision-Aware Ensemble** | **98.2%** | **0.999** | **0.982** | **0.014** | **0.013** |

> Our **decision-aware** model reduces false negatives by **38%** compared to a standard accuracy-optimized baseline.

---

## 🏗️ Framework Architecture

```
Raw Data (WBCD)
      │
      ▼
┌─────────────────────┐
│  Preprocessing      │  ← Normalization, SMOTE balancing, train/test split
└──────────┬──────────┘
           │
      ▼
┌─────────────────────┐
│  Model Training     │  ← LR, SVM, RF, XGBoost, MLP
│  (Cost-Sensitive)   │  ← class_weight + custom loss with FN penalty
└──────────┬──────────┘
           │
      ▼
┌─────────────────────┐
│  Calibration        │  ← Platt Scaling / Isotonic Regression
│  (Reliability Diag.)│  ← ECE, MCE, Brier Score
└──────────┬──────────┘
           │
      ▼
┌─────────────────────┐
│  Decision Evaluation│  ← Threshold optimization, cost matrix analysis
│  (Clinical Focus)   │  ← Net Benefit, Decision Curves
└──────────┬──────────┘
           │
      ▼
┌─────────────────────┐
│  XAI Explanations   │  ← SHAP summary, LIME local, surrogate trees
└─────────────────────┘
```

---

## 📁 Repository Structure

```
📦 decision-aware-breast-cancer-classification
├── 📂 data/                     # Dataset info and loading scripts
├── 📂 notebooks/                # Jupyter notebooks
│   ├── 01_eda.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_calibration.ipynb
│   ├── 04_decision_evaluation.ipynb
│   └── 05_xai_explanations.ipynb
├── 📂 models/                   # Model configs (weights excluded)
├── 📂 results/
│   ├── 📂 metrics/              # JSON/CSV performance metrics
│   └── 📂 plots/                # ROC curves, SHAP plots, reliability diagrams
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/Amit2004k/decision-aware-breast-cancer-classification.git
cd decision-aware-breast-cancer-classification
pip install -r requirements.txt
jupyter notebook notebooks/02_model_training.ipynb
```

---

## 🧪 Dataset

We use the **Wisconsin Breast Cancer Diagnostic Dataset (WBCD)**:
- **Source**: UCI ML Repository / `sklearn.datasets.load_breast_cancer()`
- **Samples**: 569 (212 malignant, 357 benign)
- **Features**: 30 real-valued features (radius, texture, perimeter, area, etc.)

```python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
```

---

## 🔑 Key Concepts

### Decision-Aware Evaluation
Traditional ML optimizes for accuracy. In clinical settings, **a missed cancer (FN) is far worse than a false alarm (FP)**. Our framework:
1. Defines an **asymmetric cost matrix** where FN cost = 5× FP cost
2. Optimizes the **classification threshold** for minimum expected clinical cost
3. Uses **Decision Curve Analysis (DCA)** to quantify net benefit across threshold ranges

### Probability Calibration
Raw model probabilities are often overconfident. We apply:
- **Platt Scaling** (parametric, works well for SVM/LR)
- **Isotonic Regression** (non-parametric, better for tree-based models)
- Evaluated with **Expected Calibration Error (ECE)** and reliability diagrams

### Explainability (XAI)
- **SHAP**: Global feature importance + local sample explanations
- **LIME**: Perturbation-based local model-agnostic explanations
- **Surrogate Decision Trees**: Simple proxy models for clinical communication

---

## 📖 Citation

If you use this work, please cite:

```bibtex
@inproceedings{kalita2026decisionaware,
  title     = {A Decision-Aware Machine Learning Framework for Reliable Breast Cancer Classification},
  author    = {Kalita, Amit and others},
  booktitle = {Proceedings of the International Conference on Innovative Trends in Information Technology (ICITIIT 2026)},
  year      = {2026},
  publisher = {IEEE Xplore}
}
```

---

## 🙋 Author

**Amit Kalita**
B.Tech CSE, BIT Mesra (Dibrugarh Campus)
[GitHub](https://github.com/Amit2004k)

> 📌 *This is part of a series of published ML research repos. Check out my other work on DDI prediction, Alzheimer's detection, fraud detection, GNN congestion prediction, and more.*

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

⭐ **If this work helped you, please star the repo — it helps others discover it!**
