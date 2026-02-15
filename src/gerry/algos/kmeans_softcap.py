from __future__ import annotations
import numpy as np

def kmeans_softcap_labels(
    coords: np.ndarray,
    weight: np.ndarray,
    num_districts: int,
    pop_tolerance: float = 0.05,
    max_iter: int = 80,
    alpha: float = 10000.0,
    seed: int = 42,
) -> np.ndarray:
    """
    K-means-like assignment with:
      - hard cap: cluster_pop + weight[i] <= max_pop
      - soft penalty for deviation from ideal_pop
    Returns labels in [0..num_districts-1]
    """
    N = coords.shape[0]
    rng = np.random.default_rng(seed)

    total = float(weight.sum())
    ideal = total / num_districts
    max_pop = ideal * (1 + pop_tolerance)

    init_idx = rng.choice(N, size=num_districts, replace=False)
    centers = coords[init_idx].astype(float).copy()

    labels = np.full(N, -1, dtype=int)
    cluster_pop = np.zeros(num_districts, dtype=float)

    for it in range(max_iter):
        changed = False

        for i in range(N):
            best_c = -1
            best_score = float("inf")

            wi = float(weight[i])
            if wi <= 0:
                continue

            # If reassigning, temporarily remove from old cluster for scoring
            old = labels[i]
            if old != -1:
                cluster_pop[old] -= wi

            for c in range(num_districts):
                if cluster_pop[c] + wi > max_pop:
                    continue

                dist = np.linalg.norm(coords[i] - centers[c])
                # penalize being away from ideal population
                new_pop = cluster_pop[c] + wi
                penalty = alpha * ((new_pop - ideal) / ideal) ** 2
                score = dist + penalty

                if score < best_score:
                    best_score = score
                    best_c = c

            if best_c == -1:
                # fallback: put it back where it was (or smallest)
                if old != -1:
                    best_c = old
                else:
                    best_c = int(np.argmin(cluster_pop))

            if labels[i] != best_c:
                labels[i] = best_c
                changed = True

            cluster_pop[best_c] += wi

        # recompute centers
        for c in range(num_districts):
            members = coords[labels == c]
            if len(members) > 0:
                centers[c] = members.mean(axis=0)

        if not changed:
            break

    # any -1 labels (shouldn’t happen often)
    unassigned = np.where(labels == -1)[0]
    for i in unassigned:
        labels[i] = int(np.argmin(cluster_pop))

    return labels
