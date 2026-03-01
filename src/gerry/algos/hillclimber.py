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
    max_steps: int = 2
    patience: int = 10_000
    seed: int = 45

    boundary_sample_k: int = 8000

    seat_weight: float = 1_000_000.0
    flip_weight: float = 1000.0       # NEW: pushes closest_loss toward 0
    margin_weight: float = 0.0   # optional polish; very small
    loss_weight: float = 0.01
  
         # invoke swap-assist if -50 <= closest_loss <= 0

       # when closest_loss >= -50, go into tie-break mode
    tiny_patience: int = 500         # how many tiny-mode iterations before attempting swap
    swap_max_a: int = 600
    swap_max_b: int = 600

    tiny_window: float = 50.0
    near_flip_window: float = 2000.0
    plateau_accept_prob: float = 0.2
    swap_max_a: int = 600
    swap_max_b: int = 600


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

def party_margin(dem_sum: np.ndarray, rep_sum: np.ndarray, party: str) -> np.ndarray:
    if party == "dem":
        return dem_sum - rep_sum
    if party == "rep":
        return rep_sum - dem_sum
    raise ValueError("party must be 'dem' or 'rep'")


def closest_loss_margin(dem_sum: np.ndarray, rep_sum: np.ndarray, party: str) -> float:
    m = party_margin(dem_sum, rep_sum, party)
    losers = m[m <= 0]
    if losers.size > 0:
        return float(losers.max())   # closest to 0 among losers (still <= 0)
    return float(m.min())            # all wins: smallest win margin               # positive, small is "closest"

def total_win_margin(dem_sum: np.ndarray, rep_sum: np.ndarray, party: Party) -> float:
    m = party_margin(dem_sum, rep_sum, party)
    # print("m min/max:", float(m.min()), float(m.max()))
    # print("#losers:", int(np.sum(m <= 0)))
    wins = m > 0
    return float(np.sum(m[wins]))

def loss_sum_margin(dem_sum, rep_sum, party):
    m = party_margin(dem_sum, rep_sum, party)
    losers = m[m <= 0]
    return float(losers.sum()) if losers.size > 0 else 0.0

def objective(dem_sum, rep_sum, party, cfg):
    seats = seats_won(dem_sum, rep_sum, party)
    cl = closest_loss_margin(dem_sum, rep_sum, party)   # <= 0 until all wins
    ls = loss_sum_margin(dem_sum, rep_sum, party)       # negative; want it toward 0
    return cfg.seat_weight * seats + cfg.flip_weight * cl + cfg.loss_weight * ls

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

   
        # If district was size district_nodes_count before removal,
    # then remaining nodes should be district_nodes_count - 1.
    return len(seen) == (district_nodes_count - 1)

def district_party_margin(dem_sum, rep_sum, party):
    return (dem_sum - rep_sum) if party == "dem" else (rep_sum - dem_sum)

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

def try_break_tie_with_swap(
    d_tie: int,
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
    party: str,
    rng: np.random.Generator,
    max_a: int = 200,
    max_b: int = 200,
) -> bool:
    """
    Try a 2-node swap to break a tied district:
      move a: src -> d_tie
      move b: d_tie -> src
    Returns True if a swap was applied.
    """

    # Candidate "a" nodes: outside tie that touch tie
    bnodes = boundary_nodes(labels, adj_idx)
    a_candidates = []
    for a in bnodes:
        a = int(a)
        src = int(labels[a])
        if src == d_tie:
            continue
        if any(labels[nbr] == d_tie for nbr in adj_idx[a]):
            a_candidates.append(a)

    if not a_candidates:
        return False

    rng.shuffle(a_candidates)
    a_candidates = a_candidates[:max_a]

    def vote_gain(node: int) -> float:
        dv = float(dem_votes[node])
        rv = float(rep_votes[node])
        return (dv - rv) if party == "dem" else (rv - dv)

    for a in a_candidates:
        src = int(labels[a])
        wa = float(weight[a])
        ga = vote_gain(a)

        # We want bringing a into tie to help the tie margin if possible
        # (still allow ga <= 0 if needed, but prefer positive)
        # We'll filter later via tie margin improvement.
        # Check src removal connectivity early (like move_is_feasible)
        if not is_connected_district_after_removal(a, src, labels, adj_idx, int(district_counts[src])):
            continue

        # After moving a in, tie pop increases by wa; might exceed max_pop
        pop_tie_new = pop[d_tie] + wa
        pop_src_new = pop[src] - wa
        if pop_src_new < min_pop:
            continue  # can't remove from src at all

        # Candidate "b" nodes: currently in tie, that touch src (so dst stays connected when adding b to src)
        b_candidates = []
        for b in bnodes:
            b = int(b)
            if int(labels[b]) != d_tie:
                continue
            if any(labels[nbr] == src for nbr in adj_idx[b]):
                b_candidates.append(b)

        if not b_candidates:
            continue

        rng.shuffle(b_candidates)
        b_candidates = b_candidates[:max_b]

        for b in b_candidates:
            wb = float(weight[b])

            # If tie would be overfull after adding a, b must offset it
            pop_tie_after = pop[d_tie] + wa - wb
            pop_src_after = pop[src] - wa + wb

            if pop_tie_after < min_pop or pop_tie_after > max_pop:
                continue
            if pop_src_after < min_pop or pop_src_after > max_pop:
                continue

            # tie connectivity after removing b
            if not is_connected_district_after_removal(b, d_tie, labels, adj_idx, int(district_counts[d_tie])):
                continue

            # src connectivity after removing a already checked; adding b keeps src connected because b touches src.
            # But also ensure b addition doesn't create weird disconnectedness: since it touches src, it's fine.

            # Check tie margin improvement
            # Current tie margin:
            m_before = party_margin(dem_sum, rep_sum, party)[d_tie]

            # Apply swap virtually
            dv_a = float(dem_votes[a]); rv_a = float(rep_votes[a])
            dv_b = float(dem_votes[b]); rv_b = float(rep_votes[b])

            # tie gets +a -b
            if party == "dem":
                m_after = m_before + (dv_a - rv_a) - (dv_b - rv_b)
            else:
                m_after = m_before + (rv_a - dv_a) - (rv_b - dv_b)

            # We want to break tie: m_after > 0 (or at least improve)
            if m_after <= m_before:
                continue

            # ---- Commit swap ----
            # Move b: tie -> src
            apply_move(
                node=b, src=d_tie, dst=src,
                labels=labels, pop=pop,
                dem_sum=dem_sum, rep_sum=rep_sum,
                district_counts=district_counts,
                weight=weight, dem_votes=dem_votes, rep_votes=rep_votes
            )
            # Move a: src -> tie  (note: a is still labeled src until we move it)
            apply_move(
                node=a, src=src, dst=d_tie,
                labels=labels, pop=pop,
                dem_sum=dem_sum, rep_sum=rep_sum,
                district_counts=district_counts,
                weight=weight, dem_votes=dem_votes, rep_votes=rep_votes
            )
            return True

    return False


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
    print("N =", len(labels), "K =", num_districts)
    print("unique labels =", len(set(labels.tolist())), "min/max =", int(labels.min()), int(labels.max()))
    print("adj edges total =", sum(len(nbrs) for nbrs in adj_idx))
    print("avg degree =", (sum(len(nbrs) for nbrs in adj_idx) / max(1, len(adj_idx))))
    for step in range(cfg.max_steps):
        tested = feasible = 0

        # Always compute boundary nodes FIRST (used by tie/tiny logic)
        bnodes = boundary_nodes(labels, adj_idx)
        if len(bnodes) == 0:
            print("[hillclimb] no boundary nodes; stopping", flush=True)
            break

        # recompute margins + closest loss + target district
        m = party_margin(dem_sum, rep_sum, cfg.party)
        losers = m[m <= 0]
        cl = float(losers.max()) if losers.size > 0 else float(m.min())

        if losers.size > 0:
            loser_ds = np.where(m <= 0)[0]
            d_target = int(loser_ds[np.argmax(m[loser_ds])])
        else:
            d_target = None

        # ---- TINY MODE LOCK: handle tie/-1 walls before anything else ----
        # tiny_mode triggers when we're very close to flipping the closest-losing seat
        tiny_mode = (d_target is not None and cl >= -cfg.tiny_window)

        if tiny_mode:
            print(f"[tiny] step={step+1} d_target={d_target} margin={cl:.1f}", flush=True)

            # (A) First: try swap assist on the target district (works for tie AND -1 walls)
            # You can attempt swap every step in tiny_mode, or every few steps.
            did_swap = try_break_tie_with_swap(
                d_tie=d_target,
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
                party=cfg.party,
                rng=rng,
                max_a=cfg.swap_max_a,
                max_b=cfg.swap_max_b,
            )
            if did_swap:
                best_obj = objective(dem_sum, rep_sum, cfg.party, cfg)
                no_improve = 0
                print("[tiny] swap applied", flush=True)
                # print progress line below
            else:
                # (B) If swap didn't work, try a single-node move INTO target that improves target margin
                best = None
                best_gain = 0.0

                for node in bnodes:
                    src = int(labels[node])
                    if src == d_target:
                        continue

                    if not any(labels[nbr] == d_target for nbr in adj_idx[node]):
                        continue

                    if not move_is_feasible(
                        node=node,
                        src=src,
                        dst=d_target,
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

                    dv = float(dem_votes[node])
                    rv = float(rep_votes[node])
                    gain = (dv - rv) if cfg.party == "dem" else (rv - dv)

                    # Only accept moves that help target
                    if gain > best_gain:
                        best_gain = gain
                        best = (int(node), int(src), int(d_target))

                if best is not None and best_gain > 0:
                    node_best, src_best, dst_best = best
                    apply_move(
                        node=node_best,
                        src=src_best,
                        dst=dst_best,
                        labels=labels,
                        pop=pop,
                        dem_sum=dem_sum,
                        rep_sum=rep_sum,
                        district_counts=district_counts,
                        weight=weight,
                        dem_votes=dem_votes,
                        rep_votes=rep_votes,
                    )
                    best_obj = objective(dem_sum, rep_sum, cfg.party, cfg)
                    no_improve = 0
                    print(f"[tiny] single-node gain move applied gain={best_gain:.1f}", flush=True)
                else:
                    # (C) Plateau shuffle: allow a sideways move that DOES NOT WORSEN target margin
                    # We'll use your best-of-batch but with a guard
                    best_move = None
                    best_delta = -1e18  # allow equal/sideways within tiny mode

                    # widen search in tiny mode
                    cand_nodes = (
                        rng.choice(bnodes, size=min(cfg.boundary_sample_k, len(bnodes)), replace=False)
                        if len(bnodes) > cfg.boundary_sample_k
                        else bnodes
                    )

                    m_target_before = float(m[d_target])

                    for node in cand_nodes:
                        src = int(labels[node])
                        nbr_districts = set(int(labels[nbr]) for nbr in adj_idx[node] if int(labels[nbr]) != src)
                        if not nbr_districts:
                            continue

                        for dst in nbr_districts:
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

                            # compute target margin after move (fast)
                            dv = float(dem_votes[node]); rv = float(rep_votes[node])
                            gain_party = (dv - rv) if cfg.party == "dem" else (rv - dv)

                            # effect on target district margin only if move involves target
                            m_target_after = m_target_before
                            if src == d_target:
                                m_target_after = m_target_before - gain_party
                            elif dst == d_target:
                                m_target_after = m_target_before + gain_party

                            # guard: do not worsen target margin in tiny mode
                            if m_target_after < m_target_before:
                                continue

                            # compute global delta objective
                            dem_src_old, rep_src_old = dem_sum[src], rep_sum[src]
                            dem_dst_old, rep_dst_old = dem_sum[dst], rep_sum[dst]

                            dem_sum[src] = dem_src_old - dv
                            rep_sum[src] = rep_src_old - rv
                            dem_sum[dst] = dem_dst_old + dv
                            rep_sum[dst] = rep_dst_old + rv

                            new_obj = objective(dem_sum, rep_sum, cfg.party, cfg)

                            dem_sum[src], rep_sum[src] = dem_src_old, rep_src_old
                            dem_sum[dst], rep_sum[dst] = dem_dst_old, rep_dst_old

                            delta = new_obj - best_obj
                            if delta > best_delta:
                                best_delta = delta
                                best_move = (int(node), int(src), int(dst))

                    # accept plateau moves with probability
                    accept = False
                    if best_move is not None:
                        if best_delta > 1e-12:
                            accept = True
                        else:
                            accept = (rng.random() < cfg.plateau_accept_prob)

                    if accept and best_move is not None:
                        node, src, dst = best_move
                        apply_move(
                            node=node, src=src, dst=dst,
                            labels=labels, pop=pop,
                            dem_sum=dem_sum, rep_sum=rep_sum,
                            district_counts=district_counts,
                            weight=weight, dem_votes=dem_votes, rep_votes=rep_votes
                        )
                        best_obj = objective(dem_sum, rep_sum, cfg.party, cfg)
                        no_improve = 0
                        print(f"[tiny] plateau shuffle applied delta={best_delta:.6f}", flush=True)
                    else:
                        no_improve += 1
                        print("[tiny] no acceptable move found", flush=True)

            # ---- Progress print (keep your current debug line) ----
            seats_now = seats_won(dem_sum, rep_sum, cfg.party)
            m2 = party_margin(dem_sum, rep_sum, cfg.party)
            losers2 = m2[m2 <= 0]
            cl2 = float(losers2.max()) if losers2.size > 0 else float(m2.min())
            ties2 = int(np.sum(m2 == 0))
            strict_losers2 = int(np.sum(m2 < 0))
            print(
                f"[hillclimb] step={step+1} seats={seats_now} closest_loss={cl2:.1f} "
                f"strict_losers={strict_losers2} ties={ties2} tested={tested} feasible={feasible} no_improve={no_improve}",
                flush=True
            )

            if no_improve >= cfg.patience:
                print(f"[hillclimb] stopping: no improvement for {cfg.patience} steps", flush=True)
                break

            # tiny mode handled this step fully
            continue

        # ---- NON-TINY MODE: your original two-phase search (near-flip target then generic) ----

        # ---- Phase 1: targeted "flip the next seat" search when near flip ----
        did_move = False
        if d_target is not None and cl >= -cfg.near_flip_window:
            best = None
            best_delta = -1e18

            for node in bnodes:
                src = int(labels[node])
                if src == d_target:
                    continue
                if not any(labels[nbr] == d_target for nbr in adj_idx[node]):
                    continue

                tested += 1
                if not move_is_feasible(
                    node=node,
                    src=src,
                    dst=d_target,
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

                dv = float(dem_votes[node]); rv = float(rep_votes[node])
                dem_src_old, rep_src_old = dem_sum[src], rep_sum[src]
                dem_dst_old, rep_dst_old = dem_sum[d_target], rep_sum[d_target]

                dem_sum[src] = dem_src_old - dv
                rep_sum[src] = rep_src_old - rv
                dem_sum[d_target] = dem_dst_old + dv
                rep_sum[d_target] = rep_dst_old + rv

                new_obj = objective(dem_sum, rep_sum, cfg.party, cfg)

                dem_sum[src], rep_sum[src] = dem_src_old, rep_src_old
                dem_sum[d_target], rep_sum[d_target] = dem_dst_old, rep_dst_old

                delta = new_obj - best_obj
                if delta > best_delta:
                    best_delta = delta
                    best = (int(node), int(src), int(d_target))

            if best is not None:
                accept = (best_delta > 1e-12) or (abs(best_delta) <= 1e-12 and rng.random() < cfg.plateau_accept_prob)
                if accept:
                    node, src, dst = best
                    apply_move(
                        node=node, src=src, dst=dst,
                        labels=labels, pop=pop,
                        dem_sum=dem_sum, rep_sum=rep_sum,
                        district_counts=district_counts,
                        weight=weight, dem_votes=dem_votes, rep_votes=rep_votes
                    )
                    best_obj = objective(dem_sum, rep_sum, cfg.party, cfg)
                    no_improve = 0
                    did_move = True

        # ---- Phase 2: generic best-of-batch if we didn’t move in targeted mode ----
        if not did_move:
            best_move = None
            best_delta = 0.0

            cand_nodes = (
                rng.choice(bnodes, size=min(cfg.boundary_sample_k, len(bnodes)), replace=False)
                if len(bnodes) > cfg.boundary_sample_k
                else bnodes
            )

            for node in cand_nodes:
                src = int(labels[node])
                nbr_districts = set(int(labels[nbr]) for nbr in adj_idx[node] if int(labels[nbr]) != src)
                if not nbr_districts:
                    continue

                for dst in nbr_districts:
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

                    dv = float(dem_votes[node]); rv = float(rep_votes[node])
                    dem_src_old, rep_src_old = dem_sum[src], rep_sum[src]
                    dem_dst_old, rep_dst_old = dem_sum[dst], rep_sum[dst]

                    dem_sum[src] = dem_src_old - dv
                    rep_sum[src] = rep_src_old - rv
                    dem_sum[dst] = dem_dst_old + dv
                    rep_sum[dst] = rep_dst_old + rv

                    new_obj = objective(dem_sum, rep_sum, cfg.party, cfg)

                    dem_sum[src], rep_sum[src] = dem_src_old, rep_src_old
                    dem_sum[dst], rep_sum[dst] = dem_dst_old, rep_dst_old

                    delta = new_obj - best_obj
                    if delta > best_delta + 1e-12:
                        best_delta = delta
                        best_move = (int(node), int(src), int(dst))

            accept = False
            if best_move is not None and best_delta > 1e-12:
                accept = True
            elif best_move is not None and abs(best_delta) <= 1e-12 and rng.random() < cfg.plateau_accept_prob:
                accept = True

            if accept and best_move is not None:
                node, src, dst = best_move
                apply_move(
                    node=node, src=src, dst=dst,
                    labels=labels, pop=pop,
                    dem_sum=dem_sum, rep_sum=rep_sum,
                    district_counts=district_counts,
                    weight=weight, dem_votes=dem_votes, rep_votes=rep_votes
                )
                best_obj = objective(dem_sum, rep_sum, cfg.party, cfg)
                no_improve = 0
            else:
                no_improve += 1

        # ---- Progress print every step (keep your current debug line) ----
        seats_now = seats_won(dem_sum, rep_sum, cfg.party)
        m2 = party_margin(dem_sum, rep_sum, cfg.party)
        losers2 = m2[m2 <= 0]
        cl2 = float(losers2.max()) if losers2.size > 0 else float(m2.min())
        ties2 = int(np.sum(m2 == 0))
        strict_losers2 = int(np.sum(m2 < 0))
        print(
            f"[hillclimb] step={step+1} seats={seats_now} closest_loss={cl2:.1f} "
            f"strict_losers={strict_losers2} ties={ties2} tested={tested} feasible={feasible} no_improve={no_improve}",
            flush=True
        )

        if no_improve >= cfg.patience:
            print(f"[hillclimb] stopping: no improvement for {cfg.patience} steps", flush=True)
            break

        ### end step loop (main hillclimb loop)
    print("------------------------------------------------------")
    print(f"[hillclimb] Finished after {step+1} steps.")
    print(f"[hillclimb] Final seats: {seats_won(dem_sum, rep_sum, cfg.party)}")
    print(f"[hillclimb] Final objective: {best_obj:.6f}")
    return labels