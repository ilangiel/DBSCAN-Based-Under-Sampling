"""
A Cluster-Based Under-Sampling Algorithm for Class-Imbalanced Data (DBSCAN-US)
===============================================================================
Reference: Guzmán-Ponce et al. (2020). HAIS 2020, Chapter 25,
           Springer LNAI 12469, pp. 299–310.

This demo implements the DBSCAN-based under-sampling method proposed in the paper:
  Algorithm 1 — DBSCAN  (standard algorithm applied to the majority class)
  Algorithm 2 — Under-sampling based on DBSCAN

The key idea: apply a modified DBSCAN (with ε and minPts computed from the
class sizes) **only on the majority class** to remove noisy instances that are
close to the decision boundary, thus producing a cleaner and more compact
majority class representation without making assumptions about the number
of clusters.
"""

import numpy as np
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────
def euclidean(a, b):
    return float(np.sqrt(np.sum((a - b) ** 2)))


# ─────────────────────────────────────────────────────────────────────────────
# Parameter estimation  (Eqs. 1 & 2 from the paper)
# ─────────────────────────────────────────────────────────────────────────────
def estimate_params(C_neg, C_pos):
    """
    Eq. 1 — epsilon: average distance from each majority instance to the centroid.
    Eq. 2 — minPts: ratio of circle area to sphere volume × |C+|.
    """
    m = C_neg.mean(axis=0)                                  # centroid
    n_neg = len(C_neg)
    epsilon = np.sum([euclidean(m, p) for p in C_neg]) / n_neg

    total_volume = (4 / 3) * np.pi * (epsilon ** 3)
    area = np.pi * (epsilon ** 2)
    minPts = max(1, int((area / total_volume) * len(C_pos)))
    return epsilon, minPts


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm 1 — DBSCAN  (paper's version applied to one class)
# ─────────────────────────────────────────────────────────────────────────────
def dbscan(D, epsilon, minPts):
    """
    Standard DBSCAN.  Returns a label array:
      >= 0 -> cluster id
      -1  -> noise (will be removed by the under-sampling algorithm)
    """
    n = len(D)
    labels = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    def neighbors(idx):
        return [j for j in range(n)
                if j != idx and euclidean(D[idx], D[j]) <= epsilon]

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        nbhd = neighbors(i)
        if len(nbhd) < minPts:
            labels[i] = -1          # noise
        else:
            labels[i] = cluster_id
            seeds = set(nbhd)
            while seeds:
                j = seeds.pop()
                if not visited[j]:
                    visited[j] = True
                    nbhd_j = neighbors(j)
                    if len(nbhd_j) >= minPts:
                        seeds.update(nbhd_j)
                if labels[j] == -1:
                    labels[j] = cluster_id
            cluster_id += 1

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm 2 — Under-sampling based on DBSCAN (main algorithm in the paper)
# ─────────────────────────────────────────────────────────────────────────────
def dbscan_undersampling(DS):
    """
    Remove noisy majority-class instances using DBSCAN until epsilon and minPts
    stabilise (iterative loop as described on p. 303 of the paper).

    Parameters
    ----------
    DS : ndarray  shape (n, d+1)  — last column is the binary class label

    Returns
    -------
    DS_clean : ndarray  with noise-free majority class + all minority instances
    """
    labels = DS[:, -1]
    features = DS[:, :-1]

    classes, counts = np.unique(labels, return_counts=True)
    minority_lbl = classes[np.argmin(counts)]
    majority_lbl = classes[np.argmax(counts)]

    C_pos = features[labels == minority_lbl]
    C_neg = features[labels == majority_lbl].copy()

    print(f"[DBSCAN-US] Original  → majority: {len(C_neg)},  "
          f"minority: {len(C_pos)}  (IR = {len(C_neg)/len(C_pos):.2f})")

    prev_eps = None
    for iteration in range(50):          # cap iterations for the demo
        if len(C_neg) == 0:
            break
        eps, minPts = estimate_params(C_neg, C_pos)

        if prev_eps is not None and abs(eps - prev_eps) < 1e-8:
            break                        # parameters have converged → stop

        lbl = dbscan(C_neg, eps, minPts)
        C_neg = C_neg[lbl >= 0]          # keep only non-noise instances
        prev_eps = eps

    print(f"[DBSCAN-US] After US  → majority: {len(C_neg)},  "
          f"minority: {len(C_pos)}  (IR = {len(C_neg)/len(C_pos):.2f})")

    pos_rows = np.column_stack([C_pos, np.full(len(C_pos), minority_lbl)])
    neg_rows = np.column_stack([C_neg, np.full(len(C_neg), majority_lbl)])
    return np.vstack([pos_rows, neg_rows])


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────
def create_dataset(seed=7):
    rng = np.random.default_rng(seed)
    # Majority class — two compact clusters + some scattered noise
    cl1 = rng.multivariate_normal([3, 3], [[0.5, 0], [0, 0.5]], 80)
    cl2 = rng.multivariate_normal([-1, 2], [[0.4, 0], [0, 0.4]], 60)
    noise = rng.uniform(-3, 6, (20, 2))
    majority = np.vstack([cl1, cl2, noise])

    # Minority class — one small cluster mixed with majority
    minority = rng.multivariate_normal([1, 1], [[0.2, 0], [0, 0.2]], 15)

    X = np.vstack([majority, minority])
    y = np.concatenate([np.zeros(len(majority)), np.ones(len(minority))])
    return np.column_stack([X, y])


if __name__ == "__main__":
    DS = create_dataset()
    DS_clean = dbscan_undersampling(DS)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle("DBSCAN-Based Under-Sampling for Class Imbalance", fontsize=12)

    for ax, data, title in zip(axes,
                               [DS, DS_clean],
                               ["Original (imbalanced)", "After DBSCAN-US"]):
        maj = data[data[:, -1] == 0, :2]
        minn = data[data[:, -1] == 1, :2]
        ax.scatter(maj[:, 0], maj[:, 1], c="#e74c3c", alpha=0.45,
                   s=40, label=f"Majority ({len(maj)})", edgecolors="none")
        ax.scatter(minn[:, 0], minn[:, 1], c="#3498db", s=80,
                   marker="^", label=f"Minority ({len(minn)})")
        ax.set_title(title)
        ax.legend()
        ax.set_xlabel("Feature 1")

    axes[0].set_ylabel("Feature 2")
    plt.tight_layout()
    plt.savefig("dbscan_undersampling_result.png", dpi=150)
    plt.show()
    print("\nPlot saved as dbscan_undersampling_result.png")
