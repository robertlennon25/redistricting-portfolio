from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np


Party = str  # "dem" or "rep"


@dataclass
class HillclimbConfig:
    party: Party = "dem"
    pop_tolerance: float = 0.10
    max_steps: int = 50_000
    patience: int = 10_000  # stop if no improvement for this many accepted proposals
    seed: int = 42

    # move sampling
    boundary_sample_k: int = 600  # sample this many boundary nodes per step batch
    # objective weights
    seat_weight: float = 1.0
    margin_weight: float = 0.001  # tie-breaker signal


def build_adj_idx(unit_ids: List[str], adj_json: Dict[str, List[str]]) -> List[List[int]]:
    """
    Convert adjacency json keyed by unit_id -> list[unit_id] into adjacency list by index.
    """
    id_to_idx = {uid: i for i, uid in enumerate(unit_ids)}
    adj_idx: List[List[int]] = [[] for _ in unit_ids]
    for u, nbrs in adj_json.items():
        i = id_to_idx.get(u)
        if i is None:
            continue
        for v in nbrs:
            j = id_to_idx.get(v)
            if j is not None:
                adj_idx[i].append(j)
    return adj_idx


def compute_district_sums(
    labels: np.ndarray,
    num_districts: int,
    weight: np.ndarray,
    dem_votes: np.ndarray,
    rep_votes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pop = np.zeros(num_districts, dtype=float)
    dem = np.zeros(num_districts, dtype=float)
    rep = np.zeros(num_districts, dtype=float)
    for d in range(num_districts):
        mask = labels == d
        if np.any(mask):
            pop[d] = float(weight[mask].sum())
            dem[d] = float(dem_votes[mask].sum())
            rep[d] = float(rep_votes[mask].sum())
    return pop, dem, rep


def seats_won(dem_sum: np.ndarray, rep_sum: np.ndarray, party: Party) -> int:
    if party == "dem":
        return int(np.sum(dem_sum > rep_sum))
    if party == "rep":
        return int(np.sum(rep_sum > dem_sum))
    raise ValueError("party must be 'dem' or 'rep'")


def margin_score(dem_sum: np.ndarray, rep_sum: np.ndarray, party: Party) -> float:
    """
    Tie-breaker: sum of winning margins (only in won districts).
    """
    if party == "dem":
        wins = dem_sum > rep_sum
        return float(np.sum((dem_sum - rep_sum)[wins]))
    if party == "rep":
        wins = rep_sum > dem_sum
        return float(np.sum((rep_sum - dem_sum)[wins]))
    raise ValueError("party must be 'dem' or 'rep'")


def objective(dem_sum: np.ndarray, rep_sum: np.ndarray, party: Party, cfg: HillclimbConfig) -> float:
    return cfg.seat_weight * seats_won(dem_sum, rep_sum, party) + cfg.margin_weight * margin_score(dem_sum, rep_sum, party)


def is_connected_district_after_removal(
    node: int,
    district: int,
    labels: np.ndarray,
    adj_idx: List[List[int]],
    district_nodes_count: int,
) -> bool:
    """
    Check if district remains connected if `node` is removed from it.
    We do BFS from any remaining node in the district.
    district_nodes_count is the number of nodes currently in the district (before removal).

    Fast exits:
      - if district size <= 2: removing 1 node leaves size 1 or 0 => connected (or empty)
      - if node has 0 or 1 neighbors in district: removing it can't disconnect others
    """
    if district_nodes_count <= 2:
        return True

    in_d_neighbors = [nbr for nbr in adj_idx[node] if labels[nbr] == district]
    if len(in_d_neighbors) <= 1:
        return True

    # find a start node in district that isn't `node`
    start = None
    for nbr in in_d_neighbors:
        if nbr != node:
            start = nbr
            break
    if start is None:
        # node might be the only connector but no other node found -> treat as connected (conservative)
        return True

    # BFS within district excluding node
    seen = set([start])
    q = deque([start])
    while q:
        x = q.popleft()
        for y in adj_idx[x]:
            if y == node:
                continue
            if labels[y] != district:
                continue
            if y not in seen:
                seen.add(y)
                q.append(y)

    # We need to verify all district nodes except `node` are reachable.
    # Count reachable by scanning neighbors is expensive; instead do one pass counting.
    reachable = 0
    for i in range(labels.shape[0]):
        if labels[i] == district and i != node:
            if i in seen:
                reachable += 1
            else:
                return False
    return True


def move_is_feasible(
    node: int,
    src: int,
    dst: int,
    labels: np.ndarray,
    adj_idx: List[List[int]],
    pop: np.ndarray,
    dem_sum: np.ndarray,
    rep_sum: np.ndarray,
    weight: np.ndarray,
    dem_votes: np.ndarray,
    rep_votes: np.ndarray,
    min_pop: float,
    max_pop: float,
    district_counts: np.ndarray,
) -> bool:
    """
    Feasible if:
      - dst is adjacent to node (so dst stays connected when adding)
      - pop bounds respected for src/dst
      - src remains connected after removal
    """
    if src == dst:
        return False

    # adding to dst keeps dst connected if node touches dst
    touches_dst = any(labels[nbr] == dst for nbr in adj_idx[node])
    if not touches_dst and district_counts[dst] > 0:
        return False

    w = float(weight[node])

    if pop[src] - w < min_pop:
        return False
    if pop[dst] + w > max_pop:
        return False

    # src connectivity after removal
    if not is_connected_district_after_removal(node, src, labels, adj_idx, int(district_counts[src])):
        return False

    return True


def apply_move(
    node: int,
    src: int,
    dst: int,
    labels: np.ndarray,
    pop: np.ndarray,
    dem_sum: np.ndarray,
    rep_sum: np.ndarray,
    district_counts: np.ndarray,
    weight: np.ndarray,
    dem_votes: np.ndarray,
    rep_votes: np.ndarray,
):
    w = float(weight[node])
    dv = float(dem_votes[node])
    rv = float(rep_votes[node])

    labels[node] = dst

    pop[src] -= w
    pop[dst] += w

    dem_sum[src] -= dv
    dem_sum[dst] += dv

    rep_sum[src] -= rv
    rep_sum[dst] += rv

    district_counts[src] -= 1
    district_counts[dst] += 1


def boundary_nodes(labels: np.ndarray, adj_idx: List[List[int]]) -> np.ndarray:
    """
    Node is boundary if it has at least one neighbor in a different district.
    """
    N = labels.shape[0]
    out = np.zeros(N, dtype=bool)
    for i in range(N):
        li = labels[i]
        for j in adj_idx[i]:
            if labels[j] != li:
                out[i] = True
                break
    return np.where(out)[0]


def hillclimb_max_seats(
    *,
    labels_init: np.ndarray,
    adj_idx: List[List[int]],
    weight: np.ndarray,
    dem_votes: np.ndarray,
    rep_votes: np.ndarray,
    num_districts: int,
    cfg: HillclimbConfig,
) -> np.ndarray:
    """
    Hillclimb by single-node boundary moves that:
      - preserve contiguity (src remains connected; dst stays connected by adjacency)
      - preserve population bounds (within cfg.pop_tolerance)
      - greedily improve objective (maximize seats for cfg.party)

    Returns improved labels.
    """
    labels = labels_init.copy()
    rng = np.random.default_rng(cfg.seed)

    total_pop = float(weight.sum())
    ideal = total_pop / num_districts
    min_pop = ideal * (1 - cfg.pop_tolerance)
    max_pop = ideal * (1 + cfg.pop_tolerance)

    pop, dem_sum, rep_sum = compute_district_sums(labels, num_districts, weight, dem_votes, rep_votes)
    district_counts = np.array([(labels == d).sum() for d in range(num_districts)], dtype=int)

    best_obj = objective(dem_sum, rep_sum, cfg.party, cfg)
    no_improve = 0

    print(f"[hillclimb] Starting optimization for party='{cfg.party}'")
    print(f"[hillclimb] Initial seats: {seats_won(dem_sum, rep_sum, cfg.party)}")
    print(f"[hillclimb] Initial objective: {best_obj:.6f}")
    print(f"[hillclimb] Max steps: {cfg.max_steps}, Patience: {cfg.patience}")
    print("------------------------------------------------------")

    for step in range(cfg.max_steps):
        tested = 0
        feasible = 0
        improving = 0
        if (step + 1) % 500 == 0:
            seats_now = seats_won(dem_sum, rep_sum, cfg.party)
            print(
                f"[hillclimb] step={step+1} seats={seats_now} obj={best_obj:.6f} "
                f"tested={tested} feasible={feasible} improving={improving} "
                f"boundary_nodes={len(bnodes)}"
                 )
        bnodes = boundary_nodes(labels, adj_idx)
        if len(bnodes) == 0:
            print("[hillclimb] No boundary nodes left — stopping.")
            break

        if len(bnodes) > cfg.boundary_sample_k:
            cand_nodes = rng.choice(bnodes, size=cfg.boundary_sample_k, replace=False)
        else:
            cand_nodes = bnodes

        improved = False

        for node in cand_nodes:
            src = int(labels[node])
            nbr_districts = {int(labels[nbr]) for nbr in adj_idx[node] if int(labels[nbr]) != src}
            if not nbr_districts:
                continue

            dsts = list(nbr_districts)
            rng.shuffle(dsts)

            for dst in dsts:
                tested += 1
                if not move_is_feasible(
                    node=node,
                    src=src,
                    dst=dst,
                    labels=labels,
                    adj_idx=adj_idx,
                    pop=pop,
                    dem_sum=dem_sum,
                    rep_sum=rep_sum,
                    weight=weight,
                    dem_votes=dem_votes,
                    rep_votes=rep_votes,
                    min_pop=min_pop,
                    max_pop=max_pop,
                    district_counts=district_counts,
                ):
                    continue
                    
                feasible += 1
                dv = float(dem_votes[node])
                rv = float(rep_votes[node])

                dem_src_old, rep_src_old = dem_sum[src], rep_sum[src]
                dem_dst_old, rep_dst_old = dem_sum[dst], rep_sum[dst]

                dem_sum[src] = dem_src_old - dv
                rep_sum[src] = rep_src_old - rv
                dem_sum[dst] = dem_dst_old + dv
                rep_sum[dst] = rep_dst_old + rv

                new_obj = objective(dem_sum, rep_sum, cfg.party, cfg)

                dem_sum[src], rep_sum[src] = dem_src_old, rep_src_old
                dem_sum[dst], rep_sum[dst] = dem_dst_old, rep_dst_old

                if new_obj > best_obj + 1e-12:
                    improving += 1
                    apply_move(
                        node=node,
                        src=src,
                        dst=dst,
                        labels=labels,
                        pop=pop,
                        dem_sum=dem_sum,
                        rep_sum=rep_sum,
                        district_counts=district_counts,
                        weight=weight,
                        dem_votes=dem_votes,
                        rep_votes=rep_votes,
                    )

                    best_obj = new_obj
                    improved = True
                    no_improve = 0

                    if step % 500 == 0:
                        seats_now = seats_won(dem_sum, rep_sum, cfg.party)
                        print(f"[hillclimb] step={step} | seats={seats_now} | obj={best_obj:.6f}")

                    break

            if improved:
                break

        if not improved:
            no_improve += 1

        if no_improve >= cfg.patience:
            print(f"[hillclimb] Early stopping — no improvement for {cfg.patience} accepted attempts.")
            break

        if (step + 1) % 2000 == 0:
            seats_now = seats_won(dem_sum, rep_sum, cfg.party)
            print(f"[hillclimb] heartbeat step={step+1} | seats={seats_now} | obj={best_obj:.6f}")

    print("------------------------------------------------------")
    print(f"[hillclimb] Finished after {step+1} steps.")
    print(f"[hillclimb] Final seats: {seats_won(dem_sum, rep_sum, cfg.party)}")
    print(f"[hillclimb] Final objective: {best_obj:.6f}")
    return labels