# Data

This project uses the **Wisconsin Breast Cancer Diagnostic Dataset (WBCD)**.

## Loading the Dataset

No manual download needed — load directly from scikit-learn:

```python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X, y = data.data, data.target
print(data.DESCR)
```

## Dataset Info

- **Samples**: 569
- **Features**: 30 numeric features computed from digitized images of FNA of breast masses
- **Classes**: Malignant (212), Benign (357)
- **Source**: UCI ML Repository — https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)

## Feature Groups

For each of 10 base measurements, mean, SE, and worst values are provided → 30 features total:
radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension
