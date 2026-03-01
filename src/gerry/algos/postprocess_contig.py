from __future__ import annotations

from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


# ----------------------------
# Adjacency utilities
# ----------------------------

def build_adj_idx(unit_ids: list[str], adj_json: dict[str, list[str]]) -> list[list[int]]:
    id_to_idx = {uid: i for i, uid in enumerate(unit_ids)}
    adj_idx: list[list[int]] = [[] for _ in unit_ids]
    for u, nbrs in adj_json.items():
        i = id_to_idx.get(u)
        if i is None:
            continue
        for v in nbrs:
            j = id_to_idx.get(v)
            if j is not None:
                adj_idx[i].append(j)
    return adj_idx


def district_components(labels: np.ndarray, adj_idx: list[list[int]], d: int) -> list[list[int]]:
    """Connected components in the induced subgraph where labels == d. Largest-first."""
    nodes = np.where(labels == d)[0]
    if len(nodes) == 0:
        return []

    in_d = np.zeros(labels.shape[0], dtype=bool)
    in_d[nodes] = True

    seen = np.zeros(labels.shape[0], dtype=bool)
    comps: list[list[int]] = []

    for start in nodes:
        if seen[start]:
            continue
        q = deque([start])
        seen[start] = True
        comp: list[int] = []
        while q:
            x = q.popleft()
            comp.append(x)
            for y in adj_idx[x]:
                if in_d[y] and not seen[y]:
                    seen[y] = True
                    q.append(y)
        comps.append(comp)

    comps.sort(key=len, reverse=True)
    return comps


def boundary_neighbor_districts(C: list[int], labels: np.ndarray, adj_idx: list[list[int]]) -> dict[int, int]:
    """Counts boundary edges from component C into each neighboring district."""
    counts = defaultdict(int)
    sC = set(C)
    for x in C:
        for y in adj_idx[x]:
            if y in sC:
                continue
            dy = int(labels[y])
            if dy >= 0:
                counts[dy] += 1
    return dict(counts)


# ----------------------------
# Optional "no seat flips" constraint
# ----------------------------

@dataclass
class SeatGuard:
    dem: np.ndarray  # (N,)
    rep: np.ndarray  # (N,)
    # per-district running totals
    dem_tot: np.ndarray  # (K,)
    rep_tot: np.ndarray  # (K,)
    winner: np.ndarray   # (K,) bool True=Dem wins else Rep wins

    @staticmethod
    def from_arrays(labels: np.ndarray, dem_votes: np.ndarray, rep_votes: np.ndarray, K: int) -> "SeatGuard":
        dem_tot = np.zeros(K, dtype=float)
        rep_tot = np.zeros(K, dtype=float)
        for d in range(K):
            m = (labels == d)
            dem_tot[d] = float(dem_votes[m].sum())
            rep_tot[d] = float(rep_votes[m].sum())
        winner = dem_tot > rep_tot
        return SeatGuard(dem_votes, rep_votes, dem_tot, rep_tot, winner)

    def would_flip_if_move(self, nodes: Iterable[int], src: int, dst: int) -> bool:
        """Return True if moving nodes src->dst would flip winner in src or dst."""
        nodes = list(nodes)
        d_delta = float(self.dem[nodes].sum())
        r_delta = float(self.rep[nodes].sum())

        # src after removal
        dem_src = self.dem_tot[src] - d_delta
        rep_src = self.rep_tot[src] - r_delta
        win_src = dem_src > rep_src

        # dst after addition
        dem_dst = self.dem_tot[dst] + d_delta
        rep_dst = self.rep_tot[dst] + r_delta
        win_dst = dem_dst > rep_dst

        return (win_src != bool(self.winner[src])) or (win_dst != bool(self.winner[dst]))

    def apply_move(self, nodes: Iterable[int], src: int, dst: int) -> None:
        """Apply src->dst for nodes to internal totals/winners."""
        nodes = list(nodes)
        d_delta = float(self.dem[nodes].sum())
        r_delta = float(self.rep[nodes].sum())

        self.dem_tot[src] -= d_delta
        self.rep_tot[src] -= r_delta
        self.dem_tot[dst] += d_delta
        self.rep_tot[dst] += r_delta

        self.winner[src] = self.dem_tot[src] > self.rep_tot[src]
        self.winner[dst] = self.dem_tot[dst] > self.rep_tot[dst]


# ----------------------------
# Contiguity repair core (swallow islands; optional bridge fallback)
# ----------------------------

def enforce_contiguity_postprocess(
    labels: np.ndarray,
    weight: np.ndarray,
    adj_idx: list[list[int]],
    num_districts: int,
    *,
    eps: float = 0.15,
    max_passes: int = 10,
    # swallow vs bridge knobs
    enable_bridge: bool = True,
    max_bridge_len: int = 30,
    # seat constraint
    seat_guard: Optional[SeatGuard] = None,
) -> np.ndarray:
    """
    Make districts connected by moving disconnected components ("islands") to neighbors.
    Primary method: swallow island into adjacent district with best boundary contact & pop score.
    Fallback: optionally build a short bridge path into the district if swallow isn't feasible.

    eps: population tolerance used DURING repair (can be looser than final tolerance).
    """
    labels = labels.copy()

    total = float(weight.sum())
    ideal = total / num_districts
    min_pop = ideal * (1 - eps)
    max_pop = ideal * (1 + eps)

    pop = np.zeros(num_districts, dtype=float)
    for d in range(num_districts):
        pop[d] = float(weight[labels == d].sum())

    for _pass in range(max_passes):
        any_change = False

        for d in range(num_districts):
            comps = district_components(labels, adj_idx, d)
            if len(comps) <= 1:
                continue

            core = set(comps[0])
            islands = comps[1:]
            # smallest first tends to be easier/safer
            islands.sort(key=lambda C: float(weight[C].sum()))

            for C in islands:
                popC = float(weight[C].sum())
                nbr_counts = boundary_neighbor_districts(C, labels, adj_idx)
                if not nbr_counts:
                    continue

                # ----- Try swallow island into best neighbor -----
                best_e = None
                best_score = None

                for e, cut_edges in nbr_counts.items():
                    e = int(e)
                    if e == d:
                        continue

                    # population feasibility
                    if pop[d] - popC < min_pop:
                        continue
                    if pop[e] + popC > max_pop:
                        continue

                    # seat safety
                    if seat_guard is not None and seat_guard.would_flip_if_move(C, src=d, dst=e):
                        continue

                    # scoring:
                    # - prefer stronger boundary contact (more cut edges)
                    # - prefer improved pop balance
                    score = (
                        -cut_edges
                        + 0.002 * abs((pop[e] + popC) - ideal)
                        + 0.002 * abs((pop[d] - popC) - ideal)
                    )

                    if best_score is None or score < best_score:
                        best_score = score
                        best_e = e

                if best_e is not None:
                    # apply swallow
                    for x in C:
                        labels[x] = best_e
                    pop[d] -= popC
                    pop[best_e] += popC
                    if seat_guard is not None:
                        seat_guard.apply_move(C, src=d, dst=best_e)
                    any_change = True
                    continue

                # ----- If swallow not feasible, optionally attempt bridge -----
                if enable_bridge:
                    bridge = _find_bridge_path(labels, adj_idx, d, core, C, pop, ideal, seat_guard, max_bridge_len)
                    if bridge is not None and len(bridge) > 0:
                        # bridge nodes currently belong to other districts; we move them into d
                        # apply one-by-one so pop and seat constraints are tracked accurately
                        ok = True
                        for x in bridge:
                            src = int(labels[x])
                            if src == d:
                                continue
                            wx = float(weight[x])

                            if pop[src] - wx < min_pop or pop[d] + wx > max_pop:
                                ok = False
                                break
                            if seat_guard is not None and seat_guard.would_flip_if_move([x], src=src, dst=d):
                                ok = False
                                break

                            labels[x] = d
                            pop[src] -= wx
                            pop[d] += wx
                            if seat_guard is not None:
                                seat_guard.apply_move([x], src=src, dst=d)

                        if ok:
                            any_change = True
                            # after bridging, this district may become connected; move on
                            continue

        if not any_change:
            break

    return labels


def _find_bridge_path(
    labels: np.ndarray,
    adj_idx: list[list[int]],
    d: int,
    core: set[int],
    island: list[int],
    pop: np.ndarray,
    ideal: float,
    seat_guard: Optional[SeatGuard],
    max_len: int,
) -> Optional[list[int]]:
    """
    Find a short path of nodes to pull into district d connecting island -> core.

    Strategy: multi-source BFS from island boundary outward until touching core.
    Cost heuristic: prefer stepping through nodes in districts with LOWER pop (closer below ideal).
    """
    island_set = set(island)

    # seeds: boundary nodes of island
    seeds = []
    for x in island:
        for y in adj_idx[x]:
            if y not in island_set:
                seeds.append(y)
    if not seeds:
        return None

    # BFS with parent pointers; we cap depth to max_len
    # We treat nodes in d (core) as targets; nodes in island_set are start region.
    q = deque()
    parent = {}
    depth = {}

    # initialize frontier with neighbors of island (excluding nodes already in d-island)
    for s in seeds:
        if s in parent:
            continue
        parent[s] = -1
        depth[s] = 1
        q.append(s)

    def step_cost(node: int) -> float:
        src = int(labels[node])
        if src < 0:
            return 1.0
        # prefer stealing from underfull districts -> cheaper
        # (underfull => pop[src] < ideal)
        under = max(0.0, (ideal - pop[src]) / ideal)
        return 1.0 - 0.5 * under  # between ~0.5 and 1.0

    # We’ll do a simple BFS by layers, but choose next nodes with slight preference:
    # (not full Dijkstra to keep this small and reliable)
    while q:
        x = q.popleft()
        if depth.get(x, 10**9) > max_len:
            continue

        # reached core?
        if x in core or int(labels[x]) == d:
            # reconstruct path excluding the final core node
            path = []
            cur = x
            while cur != -1:
                path.append(cur)
                cur = parent[cur]
            path.reverse()

            # path starts at some neighbor of island and ends in core; we want all non-core nodes
            bridge = [n for n in path if (n not in core and int(labels[n]) != d)]
            return bridge

        # expand
        nbrs = adj_idx[x]
        # heuristic ordering: cheaper first
        nbrs = sorted(nbrs, key=step_cost)

        for y in nbrs:
            if y in island_set:
                continue
            if y in parent:
                continue
            parent[y] = x
            depth[y] = depth[x] + 1
            q.append(y)

    return None


# ----------------------------
# Light population rebalancing (boundary single-node moves)
# ----------------------------

def rebalance_population_local(
    labels: np.ndarray,
    weight: np.ndarray,
    adj_idx: list[list[int]],
    num_districts: int,
    *,
    max_moves: int = 800,
    seat_guard: Optional[SeatGuard] = None,
) -> np.ndarray:
    """
    Very small, safe postprocess: move boundary nodes from overfull -> underfull districts
    if it improves deviation and doesn't break contiguity or flip seats.

    This is conservative (single-node moves + contiguity check via "still connected" test).
    """
    labels = labels.copy()

    total = float(weight.sum())
    ideal = total / num_districts

    pop = np.zeros(num_districts, dtype=float)
    for d in range(num_districts):
        pop[d] = float(weight[labels == d].sum())

    def deviation(d: int) -> float:
        return abs(pop[d] - ideal)

    def is_boundary_node(i: int) -> bool:
        di = int(labels[i])
        for j in adj_idx[i]:
            if int(labels[j]) != di:
                return True
        return False

    def would_disconnect_if_removed(i: int, d: int) -> bool:
        """Conservative check: after removing node i from district d, is remaining still connected?"""
        nodes = np.where(labels == d)[0]
        if len(nodes) <= 1:
            return True
        nodes_set = set(nodes.tolist())
        nodes_set.discard(i)
        if not nodes_set:
            return True

        start = next(iter(nodes_set))
        q = deque([start])
        seen = {start}
        while q:
            x = q.popleft()
            for y in adj_idx[x]:
                if y in nodes_set and y not in seen:
                    seen.add(y)
                    q.append(y)
        return len(seen) != len(nodes_set)

    moves = 0

    # Simple loop: scan boundary nodes; try moves that improve deviation
    # Stop when we hit max_moves or no more improvements in a full pass.
    for _ in range(10):  # up to 10 sweeps
        improved = False

        boundary_nodes = [i for i in range(len(labels)) if is_boundary_node(i)]
        # prioritize heavy nodes in overfull districts
        boundary_nodes.sort(key=lambda i: (pop[int(labels[i])] - ideal, weight[i]), reverse=True)

        for i in boundary_nodes:
            if moves >= max_moves:
                return labels

            src = int(labels[i])
            wi = float(weight[i])

            # candidate targets: neighboring district ids
            targets = set(int(labels[j]) for j in adj_idx[i] if int(labels[j]) != src and int(labels[j]) >= 0)
            if not targets:
                continue

            # only consider moving from overfull -> underfull (or near)
            if pop[src] <= ideal:
                continue

            best_dst = None
            best_gain = 0.0

            for dst in targets:
                # compute deviation improvement
                before = deviation(src) + deviation(dst)
                after = abs((pop[src] - wi) - ideal) + abs((pop[dst] + wi) - ideal)
                gain = before - after
                if gain <= 0:
                    continue

                # contiguity safety: removing i shouldn't disconnect src
                if would_disconnect_if_removed(i, src):
                    continue

                # seat safety
                if seat_guard is not None and seat_guard.would_flip_if_move([i], src=src, dst=dst):
                    continue

                if gain > best_gain:
                    best_gain = gain
                    best_dst = dst

            if best_dst is None:
                continue

            # apply
            labels[i] = best_dst
            pop[src] -= wi
            pop[best_dst] += wi
            if seat_guard is not None:
                seat_guard.apply_move([i], src=src, dst=best_dst)

            moves += 1
            improved = True

        if not improved:
            break

    return labels