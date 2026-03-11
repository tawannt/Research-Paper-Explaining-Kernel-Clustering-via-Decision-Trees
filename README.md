# Explaining Kernel Clustering via Decision Trees

A Python implementation of methods for making kernel clustering algorithms interpretable through decision tree explanations. This project implements and compares various algorithms for explainable kernel clustering, including IMM, ExKMC, and Expand methods.

---
## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Algorithms](#algorithms)
- [Datasets](#datasets)
- [Code Examples](#code-examples)
- [Results and Evaluation](#results-and-evaluation)
- [References](#references)

---
## Overview

Kernel clustering methods like kernel k-means are powerful tools for identifying non-linear cluster structures in data. However, they operate in high-dimensional feature spaces, making their results difficult to interpret. This project implements several methods to explain kernel clustering results using simple, interpretable decision trees.

### Key Research Goals

1. **Explainability**: Transform complex kernel clustering results into human-interpretable decision trees
2. **Performance**: Maintain clustering quality while providing explanations
3. **Comparison**: Evaluate different explainability methods (IMM, ExKMC, Expand) on various datasets

---
## Features

- **Multiple Kernel Functions**
  - Radial Basis Function (RBF/Gaussian) kernel
  - Laplace kernel
  - Linear kernel

- **Clustering Algorithms**
  - Standard k-means
  - Kernel k-means

- **Explainability Methods**
  - IMM (Iterative Mistake Minimization)
  - Taylor IMM (for Gaussian kernels)
  - Kernel Matrix IMM
  - ExKMC (Explainable Kernel Means Clustering)
  - Expand (Expanded IMM)

- **Evaluation Metrics**
  - Adjusted Rand Index for clustering quality
  - Cost analysis for explainability trade-offs

---
## Project Structure

```
source/
├── kernel.py              # Kernel function definitions (RBF, Laplace, Linear)
├── kernel_kmeans.py       # Kernel k-means implementation
├── kernel_imm.py          # IMM explainability method
├── kernel_exkmc.py        # ExKMC explainability method
├── kernel_expand.py       # Expand explainability method
├── experiments.py         # Experiment runners and evaluation
├── utils.py              # Utility functions for data loading and visualization
├── main.ipynb            # Main notebook for running experiments
└── data/                 # Dataset files
    ├── aggregation.arff
    ├── flame.arff
    └── pathbased.arff
```

---
## Installation

### Prerequisites

- Python 3.7+
- NumPy
- scikit-learn
- pandas
- matplotlib

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Research-Paper-Explaining-Kernel-Clustering-via-Decision-Trees
```

2. Install required packages:
```bash
pip install numpy scikit-learn pandas matplotlib
```

3. Navigate to the source directory:
```bash
cd source
```

---
## Usage

### Quick Start

```python
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from kernel import rbf
from kernel_kmeans import kernelkmeans
from kernel_imm import imm
from utils import load_dataset

# Load a dataset
X, y_true = load_dataset("Pathbased")

# Perform kernel k-means clustering
gamma = 0.1
Kmat = pairwise_kernels(X, metric=rbf, gamma=gamma)
y_kkm = kernelkmeans(Kmat, n_clusters=3, algo='kernelkmeans', n_init=10, n_iter=100)

# Generate explainable decision tree
y_imm, threshold_cuts = imm(X, y_kkm, check_all_cuts=True)
```

### Running Experiments

Use the provided notebook or run experiments programmatically:

```python
from experiments import imm_experiments
from utils import load_dataset

# Load dataset
X, y_true = load_dataset("Flame")

# Run complete IMM experiments
gammas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
results = imm_experiments(X, y_true, gammas, n_init=10)
```

---
## Algorithms

### 1. Kernel K-means

Standard kernel k-means algorithm that operates in the kernel-induced feature space:

**Mathematical Foundation:**
- Minimizes: $\sum_{i=1}^{k} \sum_{x \in C_i} ||\phi(x) - \mu_i||^2$
- Where $\phi(x)$ is the feature map and $\mu_i$ is the cluster center in feature space

**Implementation:**
```python
def kernelkmeans(Kmat, n_clusters, algo='kernelkmeans', n_init=10, n_iter=100):
    """
    Perform kernel k-means clustering.
    
    Args:
        Kmat: Kernel matrix (n_samples, n_samples)
        n_clusters: Number of clusters
        algo: 'kernelkmeans' or 'kmeans'
        n_init: Number of random initializations
        n_iter: Maximum iterations
    
    Returns:
        y: Cluster assignments
    """
```

### 2. IMM (Iterative Mistake Minimization)

Creates an interpretable decision tree by recursively splitting the feature space to minimize misclassifications relative to kernel clustering results.

**Key Features:**
- Builds axis-aligned decision trees
- Minimizes mistakes at each split
- Supports multiple kernel types through different feature representations

**Variants:**
- **Standard IMM**: Works with original features
- **Taylor IMM**: Uses Taylor expansion for Gaussian kernels
- **Kernel Matrix IMM**: Uses kernel matrix rows as features

### 3. ExKMC (Explainable Kernel Means Clustering)

Optimizes splits based on the kernel k-means cost function rather than mistake counting.

**Advantages:**
- Directly optimizes clustering objective
- Better preserves cluster structure
- Works with interval splits

### 4. Expand

An extension of IMM that uses interval splits instead of single threshold cuts.

**Benefits:**
- More flexible split patterns
- Can capture complex decision boundaries
- Maintains interpretability

---
## Datasets

The project includes several benchmark datasets:

| Dataset | Dimensions | Samples | Clusters | Description |
|---------|-----------|---------|----------|-------------|
| **Pathbased** | 2 | 300 | 3 | Synthetic path-based structure |
| **Aggregation** | 2 | 788 | 7 | Aggregated point clusters |
| **Flame** | 2 | 240 | 2 | Flame-shaped clusters |
| **Iris** | 4 | 150 | 3 | Classic iris flower dataset |
| **Breast Cancer** | 30 | 569 | 2 | Wisconsin breast cancer dataset |

### Loading Datasets

```python
from utils import load_dataset

# Load any dataset
X, y_true = load_dataset("Pathbased")    # or "Aggregation", "Flame", "Iris", "Cancer"
```

---
## Code Examples

### Example 1: Basic Kernel Clustering

```python
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics.cluster import adjusted_rand_score
from kernel import rbf, laplace, linear
from kernel_kmeans import kernelkmeans

# Load data
from utils import load_dataset
X, y_true = load_dataset("Aggregation")

# Define kernel parameters
gamma = 0.5
n_clusters = len(np.unique(y_true))

# Compute kernel matrix
Kmat = pairwise_kernels(X, metric=rbf, gamma=gamma)

# Perform clustering
y_pred = kernelkmeans(Kmat, n_clusters, algo='kernelkmeans', n_init=10, n_iter=200)

# Evaluate
rand_score = adjusted_rand_score(y_true, y_pred)
print(f"Adjusted Rand Score: {rand_score:.4f}")
```

### Example 2: Comparing Different Kernels

```python
from kernel import rbf, laplace, linear

kernels = {
    'RBF': (rbf, 0.5),
    'Laplace': (laplace, 0.5),
    'Linear': (linear, None)
}

for name, (kernel_func, gamma) in kernels.items():
    if gamma:
        Kmat = pairwise_kernels(X, metric=kernel_func, gamma=gamma)
    else:
        Kmat = pairwise_kernels(X, metric=kernel_func)
    
    y_pred = kernelkmeans(Kmat, n_clusters, algo='kernelkmeans', n_init=10)
    score = adjusted_rand_score(y_true, y_pred)
    print(f"{name} Kernel - Rand Score: {score:.4f}")
```

### Example 3: Generating Explainable Trees

```python
from kernel_imm import imm, taylor_imm, kernelmatrix_imm
from kernel_exkmc import exkmc_build_on_imm
from kernel_expand import expand_build_on_imm

# First, get kernel k-means clustering
gamma = 0.5
Kmat = pairwise_kernels(X, metric=rbf, gamma=gamma)
y_kkm = kernelkmeans(Kmat, n_clusters, algo='kernelkmeans', n_init=10)

# Method 1: Taylor IMM (for Gaussian kernel)
y_taylor, cuts_taylor = taylor_imm(X, y_kkm, gamma, degree=5, check_all_cuts=True)
score_taylor = adjusted_rand_score(y_true, y_taylor)
print(f"Taylor IMM Rand Score: {score_taylor:.4f}")

# Method 2: Kernel Matrix IMM
y_kmat, cuts_kmat = kernelmatrix_imm(X, y_kkm, gamma, rbf, check_all_cuts=True)
score_kmat = adjusted_rand_score(y_true, y_kmat)
print(f"Kernel Matrix IMM Rand Score: {score_kmat:.4f}")

# Method 3: ExKMC
tree_exkmc = exkmc_build_on_imm(X, y_kkm, Kmat)
score_exkmc = adjusted_rand_score(y_true, tree_exkmc['y_pred'])
print(f"ExKMC Rand Score: {score_exkmc:.4f}")

# Method 4: Expand
tree_expand = expand_build_on_imm(X, y_kkm)
score_expand = adjusted_rand_score(y_true, tree_expand['y_pred'])
print(f"Expand Rand Score: {score_expand:.4f}")
```

### Example 4: Hyperparameter Selection

```python
from experiments import get_hyperparam

# Test multiple gamma values
gammas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]

# Find best kernel and gamma
best_gamma, best_kernel = get_hyperparam(X, y_true, gammas)

kernel_names = {0: 'RBF', 1: 'Laplace'}
print(f"Best kernel: {kernel_names[best_kernel]}")
print(f"Best gamma: {best_gamma}")
```

### Example 5: Visualizing Results

```python
import matplotlib.pyplot as plt
from utils import plot_result

# Get clustering results from different methods
y_kmeans, _ = kernelkmeans(X, n_clusters, algo='kmeans', n_init=10)
y_kkm = kernelkmeans(Kmat, n_clusters, algo='kernelkmeans', n_init=10)
y_kmeans_imm, _ = imm(X, y_kmeans)
y_imm, _ = taylor_imm(X, y_kkm, gamma, 5, check_all_cuts=True)
tree_exkmc = exkmc_build_on_imm(X, y_kkm, Kmat)
tree_expand = expand_build_on_imm(X, y_kkm)

# Plot comparison
plot_result(X, y_kmeans, y_kkm, tree_expand['y_pred'], 
           y_kmeans_imm, y_imm, tree_exkmc['y_pred'])
plt.tight_layout()
plt.show()
```

### Example 6: Cost Analysis

```python
from kernel_kmeans import kernelkmeanscost

# Compare costs between original and explained clustering
cost_original = np.sum(kernelkmeanscost(Kmat, y_kkm))
cost_taylor = np.sum(kernelkmeanscost(Kmat, y_taylor))
cost_kmat = np.sum(kernelkmeanscost(Kmat, y_kmat))

# Calculate price (cost ratio)
price_taylor = cost_taylor / cost_original
price_kmat = cost_kmat / cost_original

print(f"Original kernel k-means cost: {cost_original:.4f}")
print(f"Taylor IMM cost: {cost_taylor:.4f} (price: {price_taylor:.4f})")
print(f"Kernel Matrix IMM cost: {cost_kmat:.4f} (price: {price_kmat:.4f})")
```

---
## Results and Evaluation

### Evaluation Metrics

1. **Adjusted Rand Index (ARI)**
   - Measures clustering similarity to ground truth
   - Range: [-1, 1], where 1 is perfect agreement
   - Adjusted for chance

2. **Cost/Price Ratio**
   - Ratio of explained clustering cost to original kernel k-means cost
   - Lower is better (closer to 1.0 means minimal quality loss)

3. **Tree Complexity**
   - Number of splits in the decision tree
   - Fewer splits = more interpretable

### Expected Results

The methods typically achieve:
- **Kernel k-means**: Baseline clustering quality
- **IMM variants**: 80-95% of original clustering quality with simple trees
- **ExKMC**: Often better preserves cost structure
- **Expand**: Flexible boundary representation

### Running Full Experiments

```python
from experiments import imm_experiments

# Run comprehensive experiments
gammas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
results = imm_experiments(X, y_true, gammas, n_init=10)

# Access results
print("Results Dictionary Keys:", results.keys())
print(f"K-means Rand Score: {results['rand_kmeans']:.4f}")
print(f"Kernel K-means Rand Score: {results['rand_kkm']:.4f}")
print(f"Taylor IMM Rand Score: {results['rand_taylor_imm_on_kkm']:.4f}")
print(f"Kernel Matrix IMM Rand Score: {results['rand_kmat_imm_on_kkm']:.4f}")
```

---
## Algorithm Details

### Kernel Functions

#### RBF (Gaussian) Kernel
```python
def rbf(x, y, gamma):
    return np.exp(-gamma * np.sum((x - y)**2))
```
- Smooth, infinite-dimensional feature space
- Controlled by gamma (bandwidth parameter)

#### Laplace Kernel
```python
def laplace(x, y, gamma):
    return np.exp(-gamma * np.sum(np.abs(x - y)))
```
- L1 distance-based
- More robust to outliers

#### Linear Kernel
```python
def linear(x, y):
    return np.dot(x, y)
```
- Equivalent to standard k-means

### Decision Tree Split Strategies

**Threshold Cuts** (IMM, ExKMC):
- Split: $x_i < \theta$ vs $x_i \geq \theta$
- Simple binary decisions

**Interval Cuts** (Expand):
- Split: $\theta_1 \leq x_i \leq \theta_2$ vs outside interval
- More expressive boundaries

---
## References

This implementation is based on research in explainable kernel clustering. The original research explores methods for making kernel-based clustering results interpretable through decision trees.

Key concepts:
- Kernel methods in machine learning
- Decision tree explanations
- Clustering explainability
- Trade-offs between accuracy and interpretability

---
## Contributing

Contributions are welcome! Areas for improvement:
- Additional kernel functions
- New explainability methods
- More benchmark datasets
- Performance optimizations
- Visualization tools

---
## License

This project is for educational and research purposes.

---
## Authors

Research implementation for Data Mining course project, Falculty of Information Technology, University of Science. VNU-HCM (Third Year, First Term).

---

**Note**: This implementation prioritizes clarity and educational value over computational efficiency. For production use, consider optimizing the kernel matrix computations and caching intermediate results.
