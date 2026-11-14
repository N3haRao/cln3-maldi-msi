"""Lipizones: brain areas defined by their lipid make-up, found without looking at genotype.

Follows the idea of the Lipid Brain Atlas (Nature 2026): pixels are grouped by their whole
lipid profile, and the groups (lipizones) turn out to follow anatomy (white matter tracts,
cortical layers, hippocampal fields, ...). Their method is a deep hierarchy of binary splits
over 1.6 million pixels; here, with 12 sections, a simpler version:

  1. features: the peaks that pass the feature QC of 09 (fallback: Moran-selected peaks of 08)
  2. each peak divided by its median on that slide (both sections pooled, so the KO vs Control
     difference within a slide is kept), then log1p, then z-scored
  3. PCA, fitted on the same number of pixels from every section, so every animal (and both
     genotypes) shape the lipizones equally
  4. k-means with the finest number of lipizones, then the cluster centres are merged by Ward
     linkage into the coarser levels, so the levels nest (each fine lipizone sits inside one
     coarse lipizone)

Clustering itself is description, not a test. The genotype comparison is per animal: each
animal's share of pixels in a lipizone, and each animal's mean lipid levels inside it, then the
same paired test over the slides as in 09. Pixels are never treated as replicates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def scale_features(V: np.ndarray, slides: np.ndarray, per_slide: bool = True) -> np.ndarray:
    """Per-slide median scaling, log1p, then z-score per feature."""
    V = np.asarray(V, dtype=np.float64).copy()
    if per_slide:
        for s in np.unique(slides):
            r = slides == s
            med = np.median(V[r], axis=0)
            if np.any(med <= 0):
                # sparse peaks: fall back to the mean of the positive values
                pos = np.where(V[r] > 0, V[r], np.nan)
                with np.errstate(all="ignore"):
                    alt = np.nanmean(pos, axis=0)
                med = np.where(med > 0, med, np.nan_to_num(alt, nan=1.0))
            V[r] /= np.where(med > 0, med, 1.0)
    V = np.log1p(np.clip(V, 0, None))
    mu, sd = V.mean(axis=0), V.std(axis=0)
    return (V - mu) / np.where(sd > 0, sd, 1.0)


def balanced_sample(groups: np.ndarray, per_group: int, rng) -> np.ndarray:
    """Row indices: up to per_group rows from every group (here: every section)."""
    out = []
    for g in pd.unique(groups):
        idx = np.flatnonzero(groups == g)
        out.append(rng.choice(idx, min(per_group, len(idx)), replace=False))
    return np.sort(np.concatenate(out)) if out else np.array([], int)


def fit(Z: np.ndarray, train: np.ndarray, levels: list[int], n_pcs: int, seed: int = 0) -> dict:
    """PCA + k-means on the training rows, applied to all rows; Ward merges for coarser levels.

    Returns dict with labels (DataFrame-ready dict level -> array of 1-based ids per row),
    pcs (all rows x n_pcs), centres (finest), order (finest ids in dendrogram order),
    parent (level -> array mapping finest id to that level's id), pca."""
    levels = sorted({int(k) for k in levels})
    k_fine = levels[-1]
    n_pcs = int(min(n_pcs, Z.shape[1], len(train) - 1))
    pca = PCA(n_components=n_pcs, random_state=seed).fit(Z[train])
    P = pca.transform(Z)
    km = KMeans(n_clusters=k_fine, n_init=10, random_state=seed).fit(P[train])
    fine0 = km.predict(P)                                    # 0..k_fine-1
    C = km.cluster_centers_
    L = linkage(C, method="ward")
    order0 = leaves_list(L)                                  # dendrogram order of the centres
    # fine ids 1..k in dendrogram order, so neighbouring ids are similar lipizones
    rank = np.empty(k_fine, int)
    rank[order0] = np.arange(k_fine)
    fine = rank[fine0] + 1
    parent = {k_fine: np.arange(1, k_fine + 1)}
    for k in levels[:-1]:
        lab = fcluster(L, t=k, criterion="maxclust")         # per centre (original order)
        # renumber coarse ids by first appearance along the dendrogram
        seen, ren = {}, np.empty(k_fine, int)
        for pos, c0 in enumerate(order0):
            seen.setdefault(lab[c0], len(seen) + 1)
        for c0 in range(k_fine):
            ren[rank[c0]] = seen[lab[c0]]
        parent[k] = ren                                      # index: fine id - 1
    labels = {k: parent[k][fine - 1] for k in levels}
    return {"labels": labels, "pcs": P, "centres": C[order0], "parent": parent, "pca": pca,
            "levels": levels, "linkage": L}


def markers(Z: np.ndarray, fine: np.ndarray, feature_labels: list[str], top_n: int = 5) -> pd.DataFrame:
    """Mean z-score of every feature in every lipizone (lipizones x features)."""
    ids = np.unique(fine)
    M = np.vstack([Z[fine == i].mean(axis=0) for i in ids])
    return pd.DataFrame(M, index=ids, columns=feature_labels)


def composition(obs: pd.DataFrame, lab: np.ndarray, mask: np.ndarray | None = None) -> pd.DataFrame:
    """animals x lipizones: fraction of the animal's pixels (within mask) in each lipizone."""
    sel = np.ones(len(obs), bool) if mask is None else mask
    ct = pd.crosstab(obs["animal_id"].to_numpy()[sel], lab[sel])
    return ct.div(ct.sum(axis=1), axis=0)


def palette(n: int):
    """n distinguishable colours (tab20 + tab20b + tab20c, then repeats)."""
    import matplotlib.pyplot as plt
    cols = []
    for name in ("tab20", "tab20b", "tab20c"):
        cm = plt.get_cmap(name)
        cols += [cm(i) for i in range(cm.N)]
    return [cols[i % len(cols)] for i in range(n)]
