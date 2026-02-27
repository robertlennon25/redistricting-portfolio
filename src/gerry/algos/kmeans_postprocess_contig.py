from __future__ import annotations
from collections import deque, defaultdict
import numpy as np
import json
from pathlib import Path

def build_adj_idx(unit_ids: list[str], adj_json: dict[str, list[str]]) -> list[list[int]]:
    id_to_idx = {uid: i for i, uid in enumerate(unit_ids)}
    adj_idx: list[list[int]] = [[] for _ in unit_ids]
    for u, nbrs in adj_json.items():
        if u not in id_to_idx:
            continue
        i = id_to_idx[u]
        for v in nbrs:
            j = id_to_idx.get(v)
            if j is not None:
                adj_idx[i].append(j)
    return adj_idx

def district_components(labels: np.ndarray, adj_idx: list[list[int]], d: int) -> list[list[int]]:
    # components inside induced subgraph labels==d
    nodes = np.where(labels == d)[0]
    in_d = np.zeros(labels.shape[0], dtype=bool)
    in_d[nodes] = True

    seen = np.zeros(labels.shape[0], dtype=bool)
    comps: list[list[int]] = []

    for start in nodes:
        if seen[start]:
            continue
        q = deque([start])
        seen[start] = True
        comp = []
        while q:
            x = q.popleft()
            comp.append(x)
            for y in adj_idx[x]:
                if in_d[y] and not seen[y]:
                    seen[y] = True
                    q.append(y)
        comps.append(comp)

    # largest first
    comps.sort(key=len, reverse=True)
    return comps

def boundary_neighbor_districts(C: list[int], labels: np.ndarray, adj_idx: list[list[int]]) -> dict[int, int]:
    # returns {district_id: number_of_boundary_edges_into_it}
    counts = defaultdict(int)
    sC = set(C)
    for x in C:
        for y in adj_idx[x]:
            if y in sC:
                continue
            dy = int(labels[y])
            if dy >= 0:
                counts[dy] += 1
    return counts

def enforce_contiguity_by_reattach(
    labels: np.ndarray,
    weight: np.ndarray,
    adj_idx: list[list[int]],
    num_districts: int,
    eps: float = 0.10,
    max_passes: int = 10,
) -> np.ndarray:
    labels = labels.copy()
    total = float(weight.sum())
    ideal = total / num_districts
    min_pop = ideal * (1 - eps)
    max_pop = ideal * (1 + eps)

    # district pops
    pop = np.zeros(num_districts, dtype=float)
    for d in range(num_districts):
        pop[d] = float(weight[labels == d].sum())

    for _pass in range(max_passes):
        any_change = False

        for d in range(num_districts):
            comps = district_components(labels, adj_idx, d)
            if len(comps) <= 1:
                continue

            core = comps[0]
            islands = comps[1:]

            # Move smallest islands first (less likely to violate pop)
            islands.sort(key=len)

            for C in islands:
                popC = float(weight[C].sum())
                nbr_counts = boundary_neighbor_districts(C, labels, adj_idx)
                if not nbr_counts:
                    # isolated? shouldn't happen unless graph missing edges
                    continue

                best_e = None
                best_score = None

                for e, cut_edges in nbr_counts.items():
                    if e == d:
                        continue

                    # pop feasibility
                    if pop[d] - popC < min_pop:
                        continue
                    if pop[e] + popC > max_pop:
                        continue

                    # scoring: prefer stronger adjacency contact (more cut edges),
                    # and prefer keeping d closer to ideal too
                    score = (
                        -cut_edges
                        + 0.001 * abs((pop[e] + popC) - ideal)
                        + 0.001 * abs((pop[d] - popC) - ideal)
                    )

                    if best_score is None or score < best_score:
                        best_score = score
                        best_e = e

                if best_e is None:
                    # couldn't move whole island within pop bounds
                    # leave it for now (or handle with "peel" step below)
                    continue

                # perform move d -> best_e
                for x in C:
                    labels[x] = best_e
                pop[d] -= popC
                pop[best_e] += popC
                any_change = True

        if not any_change:
            break

    return labels