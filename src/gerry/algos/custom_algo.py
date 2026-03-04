from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import math
import random
from typing import Dict, List, Set, Tuple, Optional

import numpy as np


# -----------------------------
# Helpers (pure graph utilities)
# -----------------------------

def _district_connected(nodes: List[int], node_set: Set[int], adj: List[List[int]]) -> bool:
    """Check if the induced subgraph on node_set is connected. nodes is list(node_set) for speed."""
    if len(nodes) <= 1:
        return True
    start = nodes[0]
    seen = {start}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v in node_set and v not in seen:
                seen.add(v)
                q.append(v)
    return len(seen) == len(node_set)


def _connected_after_removal(d_nodes: Set[int], remove_node: int, adj: List[List[int]]) -> bool:
    """Check if district remains connected after removing remove_node."""
    if remove_node not in d_nodes:
        return True
    if len(d_nodes) <= 2:
        # removing one from size 2 leaves size 1 => connected
        return True

    # pick a start node in district excluding remove_node
    start = None
    for x in d_nodes:
        if x != remove_node:
            start = x
            break
    assert start is not None

    target_size = len(d_nodes) - 1
    seen = {start}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v == remove_node:
                continue
            if v in d_nodes and v not in seen:
                seen.add(v)
                q.append(v)

    return len(seen) == target_size


def _sigmoid(x: float) -> float:
    # stable-ish sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


# -----------------------------
# Config / Parameters
# -----------------------------

@dataclass
class AlgoParams:
    num_districts: int = 17
    pop_tolerance: float = 0.05

    # Phase 1
    seed_method: str = "farthest_graph"  # "farthest_graph" | "farthest_euclid" | "random"
    random_seed: int = 7

    # Phase 2 (annealing)
    iters: int = 25000
    k_sigmoid: float = 18.0
    anti_waste_lambda: float = 0.10
    target_win_share: float = 0.55

    # annealing schedule
    t0: float = 0.25
    t_min: float = 0.005

    # move sampling
    moves_per_iter: int = 1  # keep 1 for simplicity
    allow_soft_pop_early: bool = False
    soft_pop_lambda: float = 3.0  # penalty weight if allow_soft_pop_early=True
    soft_pop_until_frac: float = 0.35  # fraction of iters where pop can be slightly violated with penalty

    # move type
    allow_pair_moves: bool = True
    pair_move_attempts: int = 30  # attempts when single move blocked


def _params_from_cfg(cfg: dict) -> AlgoParams:
    p = AlgoParams()
    run_cfg = cfg.get("run", {})
    algo_cfg = cfg.get("algo", {}).get("custom", {})  # optional section

    # allow existing run keys to drive these (matches your project style)
    p.num_districts = int(algo_cfg.get("num_districts", run_cfg.get("num_districts", p.num_districts)))
    p.pop_tolerance = float(algo_cfg.get("pop_tolerance", run_cfg.get("pop_tolerance", p.pop_tolerance)))

    p.seed_method = str(algo_cfg.get("seed_method", p.seed_method))
    p.random_seed = int(algo_cfg.get("random_seed", p.random_seed))

    p.iters = int(algo_cfg.get("iters", p.iters))
    p.k_sigmoid = float(algo_cfg.get("k_sigmoid", p.k_sigmoid))
    p.anti_waste_lambda = float(algo_cfg.get("anti_waste_lambda", p.anti_waste_lambda))
    p.target_win_share = float(algo_cfg.get("target_win_share", p.target_win_share))

    p.t0 = float(algo_cfg.get("t0", p.t0))
    p.t_min = float(algo_cfg.get("t_min", p.t_min))

    p.allow_soft_pop_early = bool(algo_cfg.get("allow_soft_pop_early", p.allow_soft_pop_early))
    p.soft_pop_lambda = float(algo_cfg.get("soft_pop_lambda", p.soft_pop_lambda))
    p.soft_pop_until_frac = float(algo_cfg.get("soft_pop_until_frac", p.soft_pop_until_frac))

    p.allow_pair_moves = bool(algo_cfg.get("allow_pair_moves", p.allow_pair_moves))
    p.pair_move_attempts = int(algo_cfg.get("pair_move_attempts", p.pair_move_attempts))

    return p


# -----------------------------
# Phase 1: Seeding + Growth
# -----------------------------

def _graph_bfs_dists(start: int, adj: List[List[int]], N: int) -> np.ndarray:
    dist = np.full(N, -1, dtype=int)
    dist[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def _pick_seeds_farthest_graph(adj: List[List[int]], N: int, k: int, rng: random.Random) -> List[int]:
    seeds = [rng.randrange(N)]
    # maintain min distance to any seed
    min_d = _graph_bfs_dists(seeds[0], adj, N).astype(float)
    min_d[min_d < 0] = 0.0

    for _ in range(1, k):
        # pick farthest by graph distance
        idx = int(np.argmax(min_d))
        seeds.append(idx)
        d = _graph_bfs_dists(idx, adj, N).astype(float)
        d[d < 0] = 0.0
        min_d = np.minimum(min_d, d)

    return seeds


def _pick_seeds_farthest_euclid(xy: np.ndarray, k: int, rng: random.Random) -> List[int]:
    N = xy.shape[0]
    seeds = [rng.randrange(N)]
    min_d2 = np.sum((xy - xy[seeds[0]]) ** 2, axis=1)
    for _ in range(1, k):
        idx = int(np.argmax(min_d2))
        seeds.append(idx)
        d2 = np.sum((xy - xy[idx]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2)
    return seeds


def _phase1_seeded_growth(
    dem: np.ndarray,
    rep: np.ndarray,
    weight: np.ndarray,
    adj: List[List[int]],
    xy: np.ndarray,
    params: AlgoParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a feasible initial solution:
      - contiguous (by construction)
      - pop-balanced (heuristic, should land near tolerance)
    Returns:
      labels (N,), district_pops (K,), district_party_votes (K,)
    """
    rng = random.Random(params.random_seed)

    N = len(weight)
    K = params.num_districts
    total_pop = float(weight.sum())
    target = total_pop / K
    min_pop = target * (1 - params.pop_tolerance)
    max_pop = target * (1 + params.pop_tolerance)

    # seeds
    if params.seed_method == "random":
        seeds = rng.sample(range(N), K)
    elif params.seed_method == "farthest_euclid":
        seeds = _pick_seeds_farthest_euclid(xy, K, rng)
    else:
        seeds = _pick_seeds_farthest_graph(adj, N, K, rng)

    labels = np.full(N, -1, dtype=int)
    district_pops = np.zeros(K, dtype=float)
    district_dem = np.zeros(K, dtype=float)
    district_rep = np.zeros(K, dtype=float)

    # init each district with one seed
    frontiers: List[Set[int]] = [set() for _ in range(K)]
    unassigned: Set[int] = set(range(N))

    for d, s in enumerate(seeds):
        labels[s] = d
        unassigned.remove(s)
        district_pops[d] += float(weight[s])
        district_dem[d] += float(dem[s])
        district_rep[d] += float(rep[s])

    # initialize frontiers
    for d in range(K):
        members = np.where(labels == d)[0]
        for u in members:
            for v in adj[u]:
                if v in unassigned:
                    frontiers[d].add(v)

    # parallel growth: repeatedly assign one node to the district that needs pop most
    # heuristic: choose the district with smallest pop first (keeps balance)
    stuck_guard = 0
    while unassigned and stuck_guard < N * 10:
        stuck_guard += 1

        # pick district needing population (lowest pop)
        d = int(np.argmin(district_pops))

        # candidate frontier nodes for that district that keep under max_pop
        cand = [v for v in frontiers[d] if v in unassigned and district_pops[d] + float(weight[v]) <= max_pop]
        if not cand:
            # if no candidates fit for that district, try any district
            found = False
            for d2 in np.argsort(district_pops):
                cand2 = [v for v in frontiers[int(d2)] if v in unassigned and district_pops[int(d2)] + float(weight[v]) <= max_pop]
                if cand2:
                    d = int(d2)
                    cand = cand2
                    found = True
                    break
            if not found:
                # last resort: assign something somewhere even if it violates max_pop slightly
                # (rare if tolerance is reasonable). We'll repair in annealing.
                v = next(iter(unassigned))
                # choose a neighboring district if possible
                nbr_ds = {labels[n] for n in adj[v] if labels[n] != -1}
                d = int(min(nbr_ds, key=lambda dd: district_pops[dd]) if nbr_ds else int(np.argmin(district_pops)))
                cand = [v]

        # choose candidate that makes d closest to target
        v_best = min(cand, key=lambda v: abs((district_pops[d] + float(weight[v])) - target))
        labels[v_best] = d
        unassigned.remove(v_best)
        district_pops[d] += float(weight[v_best])
        district_dem[d] += float(dem[v_best])
        district_rep[d] += float(rep[v_best])

        # update frontier sets for district d (and maybe others)
        # remove assigned node from all frontiers; add its neighbors to d's frontier
        for dd in range(K):
            if v_best in frontiers[dd]:
                frontiers[dd].discard(v_best)
        for nbr in adj[v_best]:
            if nbr in unassigned:
                frontiers[d].add(nbr)

    return labels, district_pops, np.stack([district_dem, district_rep], axis=1)


# -----------------------------
# Phase 2: Annealing swaps
# -----------------------------

def _score_expected_seats(
    district_dem: np.ndarray,
    district_rep: np.ndarray,
    k_sigmoid: float,
    anti_waste_lambda: float,
    target_win_share: float,
    maximize: str,
) -> float:
    """
    Smooth seats objective + anti-waste:
      score = sum sigmoid(k*(shareA-0.5)) - lambda * sum (max(0, shareA - target_win_share))^2
    """
    if maximize == "rep":
        A = district_rep
        B = district_dem
    else:
        A = district_dem
        B = district_rep

    tot = A + B
    # avoid zero division
    share = np.where(tot > 0, A / tot, 0.0)

    seats = 0.0
    waste = 0.0
    for s in share:
        seats += _sigmoid(k_sigmoid * (float(s) - 0.5))
        over = max(0.0, float(s) - target_win_share)
        waste += over * over

    return float(seats - anti_waste_lambda * waste)


def _pop_penalty(district_pops: np.ndarray, min_pop: float, max_pop: float) -> float:
    # quadratic penalty for violations
    pen = 0.0
    for p in district_pops:
        if p < min_pop:
            d = min_pop - p
            pen += d * d
        elif p > max_pop:
            d = p - max_pop
            pen += d * d
    return pen


def _compute_boundary_nodes(labels: np.ndarray, adj: List[List[int]]) -> List[int]:
    N = len(labels)
    boundary = []
    for i in range(N):
        di = labels[i]
        for n in adj[i]:
            if labels[n] != di:
                boundary.append(i)
                break
    return boundary


def _anneal(
    labels: np.ndarray,
    dem: np.ndarray,
    rep: np.ndarray,
    weight: np.ndarray,
    adj: List[List[int]],
    district_pops: np.ndarray,
    district_votes: np.ndarray,  # shape (K,2): [dem, rep]
    params: AlgoParams,
    maximize: str,
) -> np.ndarray:
    rng = random.Random(params.random_seed + 101)

    K = params.num_districts
    total_pop = float(weight.sum())
    target = total_pop / K
    min_pop = target * (1 - params.pop_tolerance)
    max_pop = target * (1 + params.pop_tolerance)

    district_dem = district_votes[:, 0].copy()
    district_rep = district_votes[:, 1].copy()

    # initial score
    base_score = _score_expected_seats(
        district_dem, district_rep,
        k_sigmoid=params.k_sigmoid,
        anti_waste_lambda=params.anti_waste_lambda,
        target_win_share=params.target_win_share,
        maximize=maximize,
    )

    def temperature(t: int) -> float:
        # exponential cooling from t0 -> t_min
        frac = t / max(1, params.iters - 1)
        return params.t0 * ((params.t_min / params.t0) ** frac)

    for it in range(params.iters):
        if it % 1000 == 0: 
            #### print to show progress ####
            current_score = base_score
            # compute hard seats quickly
            if maximize == "rep":
                A = district_rep
                B = district_dem
            else:
                A = district_dem
                B = district_rep

            seats = sum(
                1 for d in range(len(A))
                if (A[d] + B[d] > 0) and (A[d] / (A[d] + B[d]) > 0.5)
            )
            print(f"[Anneal] iter={it}  seats={seats}  score={current_score:.4f}")
            #### end print block ####

        T = temperature(it)

        # boundary sampling
        boundary_nodes = _compute_boundary_nodes(labels, adj)
        if not boundary_nodes:
            break
        v = rng.choice(boundary_nodes)
        d_from = int(labels[v])

        # pick a neighboring district
        nbr_ds = {int(labels[n]) for n in adj[v] if int(labels[n]) != d_from}
        if not nbr_ds:
            continue
        d_to = rng.choice(list(nbr_ds))

        # check pop constraint (hard or soft)
        wv = float(weight[v])
        new_pop_from = district_pops[d_from] - wv
        new_pop_to = district_pops[d_to] + wv

        soft_phase = params.allow_soft_pop_early and (it < int(params.soft_pop_until_frac * params.iters))
        if not soft_phase:
            if not (min_pop <= new_pop_from <= max_pop and min_pop <= new_pop_to <= max_pop):
                # try pair move if enabled
                if not params.allow_pair_moves:
                    continue
                moved = _try_pair_move(
                    labels, dem, rep, weight, adj,
                    district_pops, district_dem, district_rep,
                    params, maximize, base_score,
                    rng, min_pop, max_pop,
                    soft_phase=False, T=T,
                )
                if moved is not None:
                    labels, district_pops, district_dem, district_rep, base_score = moved
                continue

        # connectivity check: removing v from d_from must keep d_from connected
        nodes_from = set(np.where(labels == d_from)[0].tolist())
        if not _connected_after_removal(nodes_from, v, adj):
            if not params.allow_pair_moves:
                continue
            moved = _try_pair_move(
                labels, dem, rep, weight, adj,
                district_pops, district_dem, district_rep,
                params, maximize, base_score,
                rng, min_pop, max_pop,
                soft_phase=soft_phase, T=T,
            )
            if moved is not None:
                labels, district_pops, district_dem, district_rep, base_score = moved
            continue

        # score delta (O(1) update on affected districts)
        dem_v = float(dem[v])
        rep_v = float(rep[v])

        # propose update
        # save old
        old_pop_from, old_pop_to = district_pops[d_from], district_pops[d_to]
        old_dem_from, old_dem_to = district_dem[d_from], district_dem[d_to]
        old_rep_from, old_rep_to = district_rep[d_from], district_rep[d_to]

        district_pops[d_from] = new_pop_from
        district_pops[d_to] = new_pop_to
        district_dem[d_from] = old_dem_from - dem_v
        district_dem[d_to] = old_dem_to + dem_v
        district_rep[d_from] = old_rep_from - rep_v
        district_rep[d_to] = old_rep_to + rep_v

        # compute new score
        new_score = _score_expected_seats(
            district_dem, district_rep,
            k_sigmoid=params.k_sigmoid,
            anti_waste_lambda=params.anti_waste_lambda,
            target_win_share=params.target_win_share,
            maximize=maximize,
        )

        if soft_phase:
            new_score -= params.soft_pop_lambda * _pop_penalty(district_pops, min_pop, max_pop)
            old_score_soft = base_score - params.soft_pop_lambda * _pop_penalty(
                np.array([old_pop_from if i == d_from else old_pop_to if i == d_to else district_pops[i]
                          for i in range(K)]),
                min_pop, max_pop
            )
            delta = new_score - old_score_soft
        else:
            delta = new_score - base_score

        accept = (delta >= 0) or (rng.random() < math.exp(-max(0.0, -delta) / max(1e-9, T)))

        if accept:
            labels[v] = d_to
            base_score = new_score if not soft_phase else base_score  # keep base_score as "seat score" baseline
        else:
            # revert
            district_pops[d_from], district_pops[d_to] = old_pop_from, old_pop_to
            district_dem[d_from], district_dem[d_to] = old_dem_from, old_dem_to
            district_rep[d_from], district_rep[d_to] = old_rep_from, old_rep_to

    return labels


def _try_pair_move(
    labels: np.ndarray,
    dem: np.ndarray,
    rep: np.ndarray,
    weight: np.ndarray,
    adj: List[List[int]],
    district_pops: np.ndarray,
    district_dem: np.ndarray,
    district_rep: np.ndarray,
    params: AlgoParams,
    maximize: str,
    base_score: float,
    rng: random.Random,
    min_pop: float,
    max_pop: float,
    soft_phase: bool,
    T: float,
):
    """
    Attempt a 2-precinct swap-like move:
      move v: A->B and u: B->A
    chosen along a boundary.
    Returns updated tuple or None.
    """
    K = params.num_districts

    boundary_nodes = _compute_boundary_nodes(labels, adj)
    if not boundary_nodes:
        return None

    for _ in range(params.pair_move_attempts):
        v = rng.choice(boundary_nodes)
        A = int(labels[v])
        nbr_ds = {int(labels[n]) for n in adj[v] if int(labels[n]) != A}
        if not nbr_ds:
            continue
        B = rng.choice(list(nbr_ds))

        # pick u from B that is adjacent to A somewhere (boundary in opposite direction)
        cand_u = [u for u in adj[v] if int(labels[u]) == B]
        if not cand_u:
            # broader: any boundary node in B adjacent to A
            cand_u = []
            nodes_B = np.where(labels == B)[0]
            for u2 in nodes_B:
                for n2 in adj[int(u2)]:
                    if int(labels[n2]) == A:
                        cand_u.append(int(u2))
                        break
        if not cand_u:
            continue
        u = rng.choice(cand_u)

        wv = float(weight[v])
        wu = float(weight[u])

        # proposed pops
        new_pop_A = district_pops[A] - wv + wu
        new_pop_B = district_pops[B] + wv - wu

        if not soft_phase:
            if not (min_pop <= new_pop_A <= max_pop and min_pop <= new_pop_B <= max_pop):
                continue

        # connectivity checks: remove v from A, remove u from B (after v/u swap)
        nodes_A = set(np.where(labels == A)[0].tolist())
        nodes_B = set(np.where(labels == B)[0].tolist())

        if not _connected_after_removal(nodes_A, v, adj):
            continue
        if not _connected_after_removal(nodes_B, u, adj):
            continue

        # apply proposal (temporarily)
        oldA_pop, oldB_pop = district_pops[A], district_pops[B]
        oldA_dem, oldB_dem = district_dem[A], district_dem[B]
        oldA_rep, oldB_rep = district_rep[A], district_rep[B]

        dv, rv = float(dem[v]), float(rep[v])
        du, ru = float(dem[u]), float(rep[u])

        district_pops[A] = new_pop_A
        district_pops[B] = new_pop_B

        # votes update:
        # A loses v gains u
        district_dem[A] = oldA_dem - dv + du
        district_rep[A] = oldA_rep - rv + ru
        # B gains v loses u
        district_dem[B] = oldB_dem + dv - du
        district_rep[B] = oldB_rep + rv - ru

        new_score = _score_expected_seats(
            district_dem, district_rep,
            k_sigmoid=params.k_sigmoid,
            anti_waste_lambda=params.anti_waste_lambda,
            target_win_share=params.target_win_share,
            maximize=maximize,
        )

        if soft_phase:
            new_score -= params.soft_pop_lambda * _pop_penalty(district_pops, min_pop, max_pop)

        delta = new_score - base_score

        accept = (delta >= 0) or (rng.random() < math.exp(-max(0.0, -delta) / max(1e-9, T)))

        if accept:
            labels[v] = B
            labels[u] = A
            return labels, district_pops, district_dem, district_rep, new_score
        else:
            # revert
            district_pops[A], district_pops[B] = oldA_pop, oldB_pop
            district_dem[A], district_dem[B] = oldA_dem, oldB_dem
            district_rep[A], district_rep[B] = oldA_rep, oldB_rep

    return None


# -----------------------------
# Public entry point
# -----------------------------

def run(pack, cfg: dict, maximize: str = "dem") -> np.ndarray:
    """
    Entry point called by runner.

    maximize: "dem" or "rep"
    returns labels: np.ndarray shape (N,) of district ids.
    """
    maximize = maximize.lower().strip()
    if maximize not in {"dem", "rep"}:
        maximize = "dem"

    params = _params_from_cfg(cfg)

    # pull data from pack
    dem = pack.dem.astype(float)
    rep = pack.rep.astype(float)
    weight = pack.weight.astype(float)
    adj = pack.adj

    # coordinates for euclid seeding (if present)
    # build_map_pack writes centroid_x/y in attributes.csv and load_map_pack should expose them if you added
    # if not available, fall back to zeros (euclid seeding won't be used unless configured)
    try:
        xy = np.stack([pack.centroid_x, pack.centroid_y], axis=1).astype(float)
    except Exception:
        xy = np.zeros((len(weight), 2), dtype=float)

    # Phase 1
    labels, district_pops, district_votes = _phase1_seeded_growth(dem, rep, weight, adj, xy, params)

    # Phase 2
    labels = _anneal(labels, dem, rep, weight, adj, district_pops, district_votes, params, maximize)

    # final sanity: ensure all assigned and labels in range
    labels = labels.astype(int)
    labels[labels < 0] = 0
    labels[labels >= params.num_districts] = params.num_districts - 1
    return labels