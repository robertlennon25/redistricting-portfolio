from __future__ import annotations
from collections import Counter, deque
import numpy as np

def greedy_packing_labels(
    dem: np.ndarray,
    rep: np.ndarray,
    weight: np.ndarray,
    adj: list[list[int]],
    num_districts: int,
    pop_tolerance: float,
    maximize: str = "dem",
) -> np.ndarray:
    """
    Returns labels array of shape (N,) with district ids [0..num_districts-1]
    """

    N = len(weight)
    total_pop = float(weight.sum())
    target = total_pop / num_districts
    min_pop = target * (1 - pop_tolerance)
    max_pop = target * (1 + pop_tolerance)

    valid = set(np.where(weight > 0)[0].tolist())
    unassigned = set(valid)
    labels = np.full(N, -1, dtype=int)

    # metrics
    def share(votes_arr):
        # return share per node, guarding weight=0
        s = np.full(N, -np.inf, dtype=float)
        mask = weight > 0
        s[mask] = votes_arr[mask] / weight[mask]
        return s

    if maximize.lower() == "gop":
        seed_metric = share(rep)
        frontier_metric = (rep - dem) / np.maximum(weight, 1e-9)
    else:
        seed_metric = share(dem)
        frontier_metric = (dem - rep) / np.maximum(weight, 1e-9)

    for d in range(num_districts):
        if not unassigned:
            break

        # pick best seed among unassigned
        seed = max(unassigned, key=lambda i: seed_metric[i])

        block = {seed}
        pop = float(weight[seed])
        unassigned.remove(seed)

        while pop < min_pop and unassigned:
            frontier = set()
            for n in block:
                for nbr in adj[n]:
                    if nbr in unassigned:
                        frontier.add(nbr)

            if not frontier:
                break

            best = max(frontier, key=lambda i: frontier_metric[i])
            block.add(best)
            pop += float(weight[best])
            unassigned.remove(best)

            if pop > max_pop:
                break

        for i in block:
            labels[i] = d

    # assign remaining nodes to neighbor majority district (or smallest)
    district_pops = Counter()
    for i in range(N):
        lab = labels[i]
        if lab != -1:
            district_pops[lab] += float(weight[i])

    # if some districts never created, initialize them
    for d in range(num_districts):
        district_pops.setdefault(d, 0.0)

    for i in range(N):
        if labels[i] != -1:
            continue
        nbr_labels = [labels[n] for n in adj[i] if labels[n] != -1]
        if nbr_labels:
            choice = Counter(nbr_labels).most_common(1)[0][0]
        else:
            choice = min(district_pops.keys(), key=lambda d: district_pops[d])
        labels[i] = choice
        district_pops[choice] += float(weight[i])

    return labels

def fix_contiguity(
    labels: np.ndarray,
    adj: list[list[int]],
    num_districts: int,
    max_passes: int = 8,
) -> np.ndarray:
    """
    Make districts more contiguous by reassigning nodes in smaller disconnected components
    to neighboring districts.

    Bounded runtime: at most `max_passes` global passes.
    """
    labels = labels.copy()
    N = len(labels)

    for _ in range(max_passes):
        changed = False

        for d in range(num_districts):
            members = np.where(labels == d)[0]
            if members.size <= 1:
                continue

            member_set = set(members.tolist())

            # Find connected components within this district using BFS
            seen = set()
            comps = []

            for start in members:
                if start in seen:
                    continue
                q = deque([int(start)])
                seen.add(int(start))
                comp = [int(start)]
                while q:
                    u = q.popleft()
                    for v in adj[u]:
                        if v in member_set and v not in seen:
                            seen.add(v)
                            q.append(v)
                            comp.append(v)
                comps.append(comp)

            if len(comps) <= 1:
                continue

            # Keep largest component, reassign others
            main = max(comps, key=len)
            main_set = set(main)

            for comp in comps:
                if comp is main:
                    continue
                for node in comp:
                    # reassign to the most common neighboring district (not d)
                    nbr_ds = [labels[n] for n in adj[node] if labels[n] != d]
                    if nbr_ds:
                        labels[node] = Counter(nbr_ds).most_common(1)[0][0]
                        changed = True

        if not changed:
            break

    return labels


def post_balance_pop(
    labels: np.ndarray,
    weight: np.ndarray,
    adj: list[list[int]],
    num_districts: int,
    pop_tolerance: float,
) -> np.ndarray:
    """
    Try to fix population imbalances with local moves while maintaining contiguity-ish.
    (Simple version: boundary swaps; still incremental-friendly.)
    """
    labels = labels.copy()
    total_pop = float(weight.sum())
    target = total_pop / num_districts
    min_pop = target * (1 - pop_tolerance)
    max_pop = target * (1 + pop_tolerance)

    def district_pops():
        pops = np.zeros(num_districts, dtype=float)
        for i, d in enumerate(labels):
            pops[d] += float(weight[i])
        return pops

    for _ in range(200):  # cap iterations so we don't loop forever
        pops = district_pops()
        low = [d for d in range(num_districts) if pops[d] < min_pop]
        high = [d for d in range(num_districts) if pops[d] > max_pop]
        if not low and not high:
            break

        moved = False

        # try to move a boundary node from high->low
        for d_hi in high:
            nodes_hi = np.where(labels == d_hi)[0]
            # boundary nodes: any node with neighbor in other district
            boundary = [i for i in nodes_hi if any(labels[n] != d_hi for n in adj[i])]
            for i in boundary:
                for n in adj[i]:
                    d_lo = labels[n]
                    if d_lo == d_hi:
                        continue
                    if d_lo not in low:
                        continue
                    new_hi = pops[d_hi] - weight[i]
                    new_lo = pops[d_lo] + weight[i]
                    if min_pop <= new_lo <= max_pop and min_pop <= new_hi <= max_pop:
                        labels[i] = d_lo
                        moved = True
                        break
                if moved:
                    break
            if moved:
                break

        if not moved:
            break

        labels = fix_contiguity(labels, adj, num_districts)

    return labels
