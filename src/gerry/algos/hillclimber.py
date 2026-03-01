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
    *,
    # NEW: anti-oscillation hooks (optional)
    tabu: dict | None = None,     # maps (node, src, dst) -> expire_step
    step: int | None = None,
    tabu_ttl: int = 15,
    # NEW: require at least this much improvement in tie margin
    min_margin_gain: float = 1e-9,
) -> tuple[bool, dict | None]:
    """
    Try a 2-node swap to break/improve a (near-)tied district:

      move a: src -> d_tie
      move b: d_tie -> src

    Returns:
      (did_swap, swap_meta)

    swap_meta (if did_swap) includes:
      {
        "a": int, "b": int,
        "src": int, "tie": int,
        "moves": [(node, old_label, new_label), ...]  # for rollback/tabu
      }
    """

    def is_tabu_move(node: int, src: int, dst: int) -> bool:
        if tabu is None or step is None:
            return False
        # forbid reversing a recent move: (node, dst, src)
        rev = (int(node), int(dst), int(src))
        return tabu.get(rev, -1) >= step

    def mark_tabu_move(node: int, src: int, dst: int) -> None:
        if tabu is None or step is None:
            return
        tabu[(int(node), int(src), int(dst))] = step + int(tabu_ttl)

    def vote_margin(d: int) -> float:
        # party-specific margin for district d
        if party == "dem":
            return float(dem_sum[d] - rep_sum[d])
        else:
            return float(rep_sum[d] - dem_sum[d])

    # Candidate "a" nodes: outside tie that touch tie
    bnodes = boundary_nodes(labels, adj_idx)
    a_candidates = []
    for a in bnodes:
        a = int(a)
        src = int(labels[a])
        if src == d_tie:
            continue
        if any(labels[nbr] == d_tie for nbr in adj_idx[a]):
            # tabu: if we would move a src->tie, ensure it's not reversing a recent tie->src
            if is_tabu_move(a, src, d_tie):
                continue
            a_candidates.append(a)

    if not a_candidates:
        return (False, None)

    rng.shuffle(a_candidates)
    a_candidates = a_candidates[:max_a]

    for a in a_candidates:
        src = int(labels[a])
        wa = float(weight[a])

        # connectivity: src must remain connected after removing a
        if not is_connected_district_after_removal(a, src, labels, adj_idx, int(district_counts[src])):
            continue

        # removing a from src must keep src within bounds
        pop_src_new = pop[src] - wa
        if pop_src_new < min_pop:
            continue

        # Candidate "b" nodes: in tie, touching src
        b_candidates = []
        for b in bnodes:
            b = int(b)
            if int(labels[b]) != d_tie:
                continue
            if any(labels[nbr] == src for nbr in adj_idx[b]):
                # tabu: if we would move b tie->src, ensure it's not reversing src->tie
                if is_tabu_move(b, d_tie, src):
                    continue
                b_candidates.append(b)

        if not b_candidates:
            continue

        rng.shuffle(b_candidates)
        b_candidates = b_candidates[:max_b]

        m_before = vote_margin(d_tie)

        dv_a = float(dem_votes[a]); rv_a = float(rep_votes[a])

        for b in b_candidates:
            wb = float(weight[b])

            # population after full swap
            pop_tie_after = pop[d_tie] + wa - wb
            pop_src_after = pop[src] - wa + wb

            if pop_tie_after < min_pop or pop_tie_after > max_pop:
                continue
            if pop_src_after < min_pop or pop_src_after > max_pop:
                continue

            # tie must remain connected after removing b
            if not is_connected_district_after_removal(b, d_tie, labels, adj_idx, int(district_counts[d_tie])):
                continue

            dv_b = float(dem_votes[b]); rv_b = float(rep_votes[b])

            # margin after swap on tie: +a - b
            if party == "dem":
                m_after = m_before + (dv_a - rv_a) - (dv_b - rv_b)
            else:
                m_after = m_before + (rv_a - dv_a) - (rv_b - dv_b)

            # require improvement (and a tiny minimum gain to avoid floating jitter)
            if m_after <= m_before + float(min_margin_gain):
                continue

            # ---- Commit swap (b first, then a) ----
            # Save moves for rollback
            moves = []

            # Move b: tie -> src
            old_b = int(labels[b])
            if old_b != d_tie:
                # label changed since candidate list was formed; skip
                continue
            apply_move(
                node=b, src=d_tie, dst=src,
                labels=labels, pop=pop,
                dem_sum=dem_sum, rep_sum=rep_sum,
                district_counts=district_counts,
                weight=weight, dem_votes=dem_votes, rep_votes=rep_votes
            )
            moves.append((int(b), int(d_tie), int(src)))
            mark_tabu_move(int(b), int(d_tie), int(src))

            # Move a: src -> tie
            old_a = int(labels[a])
            if old_a != src:
                # a moved since selection; rollback b and skip
                apply_move(
                    node=b, src=src, dst=d_tie,
                    labels=labels, pop=pop,
                    dem_sum=dem_sum, rep_sum=rep_sum,
                    district_counts=district_counts,
                    weight=weight, dem_votes=dem_votes, rep_votes=rep_votes
                )
                # note: we don't need to "unmark" tabu; it's fine to keep it conservative
                continue

            apply_move(
                node=a, src=src, dst=d_tie,
                labels=labels, pop=pop,
                dem_sum=dem_sum, rep_sum=rep_sum,
                district_counts=district_counts,
                weight=weight, dem_votes=dem_votes, rep_votes=rep_votes
            )
            moves.append((int(a), int(src), int(d_tie)))
            mark_tabu_move(int(a), int(src), int(d_tie))

            swap_meta = {
                "a": int(a),
                "b": int(b),
                "src": int(src),
                "tie": int(d_tie),
                "moves": moves,  # [(node, old, new), ...]
            }
            return (True, swap_meta)

    return (False, None)

def score_tuple(dem_sum, rep_sum, party, cfg):
    seats = seats_won(dem_sum, rep_sum, party)
    m = party_margin(dem_sum, rep_sum, party)
    losers = m[m <= 0]
    closest_loss = float(losers.max()) if losers.size > 0 else float(m.min())
    # maximize seats, then maximize closest_loss (less negative is better)
    return (int(seats), float(closest_loss))


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
    Hillclimb by single-node boundary moves with optional "tiny mode" near-flip helpers.

    Fixes added:
      - tabu (prevents immediate reversals / 2-cycles)
      - tiny-target cooldown (prevents hammering the same district)
      - accept tiny swap ONLY if it improves a stable score (seats, closest_loss) or improves objective
      - fingerprints guard (breaks out of oscillations if they still happen)
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

    # ----------------------------
    # Anti-oscillation state
    # ----------------------------
    TABU_TTL = int(getattr(cfg, "tabu_ttl", 15))
    TARGET_COOLDOWN = int(getattr(cfg, "tiny_target_cooldown", 10))
    MAX_STUCK_FINGERPRINTS = int(getattr(cfg, "max_stuck_fingerprints", 3))

    tabu: dict[tuple[int, int, int], int] = {}  # (node, src, dst) -> expire_step
    target_cooldown_until: dict[int, int] = {}  # district -> expire_step
    recent_fingerprints = deque(maxlen=8)

    def score_tuple() -> tuple[int, float]:
        """Stable score: maximize seats, then maximize closest_loss (less negative is better)."""
        seats = seats_won(dem_sum, rep_sum, cfg.party)
        m = party_margin(dem_sum, rep_sum, cfg.party)
        losers = m[m <= 0]
        closest_loss = float(losers.max()) if losers.size > 0 else float(m.min())
        return (int(seats), float(closest_loss))

    def is_tabu(node: int, src: int, dst: int, step: int) -> bool:
        """Block immediate reversals and recent repeated moves."""
        # forbid reversing (node, dst, src) if it's still active
        rev = (int(node), int(dst), int(src))
        return tabu.get(rev, -1) >= step

    def mark_tabu(node: int, src: int, dst: int, step: int) -> None:
        tabu[(int(node), int(src), int(dst))] = step + TABU_TTL

    def fingerprint(labels_arr: np.ndarray) -> int:
        # cheap-ish stable-ish hash to detect oscillations; don't need cryptographic stability
        # use a subset + mixing to keep fast
        n = labels_arr.shape[0]
        if n <= 0:
            return 0
        idx = np.linspace(0, n - 1, num=min(256, n), dtype=int)
        x = labels_arr[idx].astype(np.int64)
        # mix
        h = int(np.bitwise_xor.reduce((x + 1315423911) * 2654435761) & 0xFFFFFFFF)
        return h

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

        # Oscillation detection (very lightweight)
        fp = fingerprint(labels)
        recent_fingerprints.append(fp)
        if len(recent_fingerprints) == recent_fingerprints.maxlen:
            # if the last few are repeating, we're cycling; force a cooldown escape
            if len(set(list(recent_fingerprints)[-6:])) <= MAX_STUCK_FINGERPRINTS:
                # back off tiny mode aggressively for a bit
                # (prevents the 7<->8 seat loop you saw)
                for d in range(num_districts):
                    target_cooldown_until[d] = max(target_cooldown_until.get(d, -1), step + TARGET_COOLDOWN)
                print("[hillclimb] detected possible cycle; applying global tiny cooldown", flush=True)

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
        tiny_mode = (d_target is not None and cl >= -cfg.tiny_window)

        # Don't keep targeting same district every step
        if tiny_mode and (step < target_cooldown_until.get(int(d_target), -1)):
            tiny_mode = False

        if tiny_mode:
            print(f"[tiny] step={step+1} d_target={d_target} margin={cl:.1f}", flush=True)

            score_before = score_tuple()
            obj_before = best_obj

            # (A) Try swap assist on the target district
            did_swap, swap_meta = try_break_tie_with_swap(
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
                tabu=tabu,
                step=step,
                tabu_ttl=TABU_TTL,
            )

            # Accept swap ONLY if it improves stable score OR improves objective
            if did_swap and swap_meta is not None:
                score_after = score_tuple()
                obj_after = objective(dem_sum, rep_sum, cfg.party, cfg)

                accept_swap = (score_after > score_before) or (obj_after > obj_before + 1e-12)

                if accept_swap:
                    best_obj = obj_after
                    no_improve = 0
                    print(
                        f"[tiny] swap applied score {score_before}->{score_after} "
                        f"obj {obj_before:.6f}->{obj_after:.6f}",
                        flush=True,
                    )
                    target_cooldown_until[int(d_target)] = step + TARGET_COOLDOWN

                else:
                    # ✅ ROLLBACK swap using swap_meta
                    for node, old_lbl, new_lbl in reversed(swap_meta["moves"]):
                        apply_move(
                            node=int(node),
                            src=int(new_lbl),
                            dst=int(old_lbl),
                            labels=labels,
                            pop=pop,
                            dem_sum=dem_sum,
                            rep_sum=rep_sum,
                            district_counts=district_counts,
                            weight=weight,
                            dem_votes=dem_votes,
                            rep_votes=rep_votes,
                        )

                    no_improve += 1
                    target_cooldown_until[int(d_target)] = step + TARGET_COOLDOWN
                    print(
                        f"[tiny] swap rejected + rolled back score {score_before}->{score_after} "
                        f"obj {obj_before:.6f}->{obj_after:.6f}",
                        flush=True,
                    )
            else:
                # (B) Single-node move INTO target that improves target margin
                best = None
                best_gain = 0.0

                for node in bnodes:
                    src = int(labels[node])
                    if src == d_target:
                        continue

                    if not any(labels[nbr] == d_target for nbr in adj_idx[node]):
                        continue

                    # tabu reversal guard
                    if is_tabu(int(node), src, int(d_target), step):
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

                    if gain > best_gain:
                        best_gain = gain
                        best = (int(node), int(src), int(d_target))

                if best is not None and best_gain > 0:
                    node_best, src_best, dst_best = best

                    score_before = score_tuple()
                    obj_before = best_obj

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

                    mark_tabu(node_best, src_best, dst_best, step)

                    obj_after = objective(dem_sum, rep_sum, cfg.party, cfg)
                    score_after = score_tuple()
                    accept_move = (score_after > score_before) or (obj_after > obj_before + 1e-12)

                    if accept_move:
                        best_obj = obj_after
                        no_improve = 0
                        target_cooldown_until[int(d_target)] = step + TARGET_COOLDOWN
                        print(f"[tiny] single-node gain move applied gain={best_gain:.1f}", flush=True)

                    else:
                        # ✅ ROLLBACK the single-node move
                        apply_move(
                            node=node_best,
                            src=dst_best,
                            dst=src_best,
                            labels=labels,
                            pop=pop,
                            dem_sum=dem_sum,
                            rep_sum=rep_sum,
                            district_counts=district_counts,
                            weight=weight,
                            dem_votes=dem_votes,
                            rep_votes=rep_votes,
                        )

                        no_improve += 1
                        target_cooldown_until[int(d_target)] = step + TARGET_COOLDOWN
                        print("[tiny] single-node move rejected + rolled back", flush=True)

                    # accept only if stable score or objective improved
                    if (score_after > score_before) or (obj_after > obj_before + 1e-12):
                        best_obj = obj_after
                        no_improve = 0
                        target_cooldown_until[int(d_target)] = step + TARGET_COOLDOWN
                        print(f"[tiny] single-node gain move applied gain={best_gain:.1f}", flush=True)
                    else:
                        # rollback (we CAN rollback single node easily)
                        apply_move(
                            node=node_best,
                            src=dst_best,
                            dst=src_best,
                            labels=labels,
                            pop=pop,
                            dem_sum=dem_sum,
                            rep_sum=rep_sum,
                            district_counts=district_counts,
                            weight=weight,
                            dem_votes=dem_votes,
                            rep_votes=rep_votes,
                        )
                        no_improve += 1
                        target_cooldown_until[int(d_target)] = step + TARGET_COOLDOWN
                        print("[tiny] single-node move rejected (no net progress)", flush=True)

                else:
                    # (C) Plateau shuffle: sideways move that DOES NOT WORSEN target margin
                    best_move = None
                    best_delta = -1e18

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
                            # tabu reversal guard
                            if is_tabu(int(node), src, int(dst), step):
                                continue

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
                            gain_party = (dv - rv) if cfg.party == "dem" else (rv - dv)

                            m_target_after = m_target_before
                            if src == d_target:
                                m_target_after = m_target_before - gain_party
                            elif dst == d_target:
                                m_target_after = m_target_before + gain_party

                            if m_target_after < m_target_before:
                                continue

                            # compute delta objective
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

                    accept = False
                    if best_move is not None:
                        if best_delta > 1e-12:
                            accept = True
                        else:
                            accept = (rng.random() < cfg.plateau_accept_prob)

                    if accept and best_move is not None:
                        node, src, dst = best_move

                        score_before = score_tuple()
                        obj_before = best_obj

                        apply_move(
                            node=node, src=src, dst=dst,
                            labels=labels, pop=pop,
                            dem_sum=dem_sum, rep_sum=rep_sum,
                            district_counts=district_counts,
                            weight=weight, dem_votes=dem_votes, rep_votes=rep_votes
                        )
                        mark_tabu(node, src, dst, step)

                        obj_after = objective(dem_sum, rep_sum, cfg.party, cfg)
                        score_after = score_tuple()

                        # accept if it didn't worsen stable score and doesn't worsen objective too much
                        if (score_after >= score_before) and (obj_after >= obj_before - 1e-12):
                            best_obj = obj_after
                            no_improve = 0
                            target_cooldown_until[int(d_target)] = step + TARGET_COOLDOWN
                            print(f"[tiny] plateau shuffle applied delta={best_delta:.6f}", flush=True)
                        else:
                            # rollback
                            apply_move(
                                node=node, src=dst, dst=src,
                                labels=labels, pop=pop,
                                dem_sum=dem_sum, rep_sum=rep_sum,
                                district_counts=district_counts,
                                weight=weight, dem_votes=dem_votes, rep_votes=rep_votes
                            )
                            no_improve += 1
                            target_cooldown_until[int(d_target)] = step + TARGET_COOLDOWN
                            print("[tiny] plateau shuffle rejected (no net progress)", flush=True)
                    else:
                        no_improve += 1
                        target_cooldown_until[int(d_target)] = step + TARGET_COOLDOWN
                        print("[tiny] no acceptable move found", flush=True)

            # ---- Progress print ----
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

            continue  # tiny mode handled this step fully

        # ---- NON-TINY MODE ----

        did_move = False

        # ---- Phase 1: targeted near-flip search ----
        if d_target is not None and cl >= -cfg.near_flip_window:
            best = None
            best_delta = -1e18

            for node in bnodes:
                src = int(labels[node])
                if src == d_target:
                    continue
                if not any(labels[nbr] == d_target for nbr in adj_idx[node]):
                    continue

                # tabu reversal guard
                if is_tabu(int(node), src, int(d_target), step):
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
                    mark_tabu(node, src, dst, step)
                    best_obj = objective(dem_sum, rep_sum, cfg.party, cfg)
                    no_improve = 0
                    did_move = True

        # ---- Phase 2: generic best-of-batch ----
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
                    # tabu reversal guard
                    if is_tabu(int(node), src, int(dst), step):
                        continue

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
                mark_tabu(node, src, dst, step)
                best_obj = objective(dem_sum, rep_sum, cfg.party, cfg)
                no_improve = 0
            else:
                no_improve += 1

        # ---- Progress print ----
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

    print("------------------------------------------------------")
    print(f"[hillclimb] Finished after {step+1} steps.")
    print(f"[hillclimb] Final seats: {seats_won(dem_sum, rep_sum, cfg.party)}")
    print(f"[hillclimb] Final objective: {best_obj:.6f}")
    return labels