# DBSCAN-Based Under-Sampling for Class Imbalance

> **Article:** Guzmán-Ponce, A., et al. (2020). *A Cluster-Based Under-Sampling Algorithm for Class-Imbalanced Data*. **MICAI 2020**, Springer LNAI 12469, pp. 299–310. [DOI: 10.1007/978-3-030-61705-9_25](https://doi.org/10.1007/978-3-030-61705-9_25)

## 📖 Algorithm Overview

This algorithm proposes a cluster-based under-sampling method that applies **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) **exclusively to the majority class** in an imbalanced binary dataset.

The key innovation is:
- DBSCAN's `ε` and `minPts` are **automatically estimated** from the class sizes (no manual tuning needed).
- Instances identified as **noise are removed**, cleaning borderline and outlier majority instances.
- The process **iterates** until the parameter estimates stabilise.

### Algorithm 1 — DBSCAN

```
Input:  D = {p1, …, pn}, ε, minPts
Output: cluster labels

for each unvisited pᵢ in D:
    mark pᵢ as visited
    nbhd ← Neighbors(ε, pᵢ)
    if |nbhd| < minPts:
        mark pᵢ as noise
    else:
        expand cluster from pᵢ using density-reachability
```

### Algorithm 2 — Under-Sampling Based on DBSCAN

```
Input:  D = {p1, …, pn}
Output: D' (noise-free, class-cleaned)

Split D into C⁻ (majority) and C⁺ (minority)
repeat
    Estimate ε and minPts
    for each unvisited pᵢ⁻ in C⁻:
        if pᵢ⁻ has fewer than minPts neighbours at distance ε:
            remove pᵢ⁻ from C⁻
until ε and minPts do not change
D' ← C⁺ ∪ C⁻
```

### Parameter Estimation (Eqs. 1 & 2)

| Parameter | Formula |
|-----------|---------|
| ε | `Σ dist(m, pᵢ⁻) / |C⁻|`  — avg Euclidean distance of majority instances to the class centroid |
| minPts | `(π·ε²) / ((4/3)·π·ε³) × |C⁺|` — proportional to minority class size |

## 🚀 Quick Start

```bash
# Install dependencies
pip install numpy matplotlib

# Run the demo
python dbscan_undersampling.py
```

## 📊 What the Demo Does

1. Builds a synthetic imbalanced dataset with two majority clusters + scattered noise (160 majority / 15 minority).
2. Applies the DBSCAN-based under-sampling iteratively until convergence.
3. Prints instance counts and IR before and after.
4. Saves a side-by-side scatter plot as **`dbscan_undersampling_result.png`**.



## 📐 Key Functions

| Function | Description |
|----------|-------------|
| `estimate_params(C_neg, C_pos)` | Computes ε and minPts (Eqs. 1 & 2) |
| `dbscan(D, ε, minPts)` | Standard DBSCAN (Algorithm 1) |
| `dbscan_undersampling(DS)` | Iterative under-sampling loop (Algorithm 2) |

## 📋 Dependencies

| Library | Purpose |
|---------|---------|
| `numpy` | Numerical computation |
| `matplotlib` | Visualisation |
