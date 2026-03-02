from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

Party = str  # "dem" or "rep"


# ----------------------------
# Config
# ----------------------------
@dataclass
class HillclimbConfig:
    party: Party = "dem"
    pop_tolerance: float = 0.10
    max_steps: int = 2_000
    patience: int = 10_000
    seed: int = 45

    boundary_sample_k: int = 8000

    # objective weights
    seat_weight: float = 1_000_000.0
    flip_weight: float = 1000.0
    loss_weight: float = 0.01

    # windows + acceptance
    tiny_window: float = 50.0
    near_flip_window: float = 2000.0
    plateau_accept_prob: float = 0.2

    # swap assist limits
    swap_max_a: int = 600
    swap_max_b: int = 600

    # anti-oscillation
    tabu_ttl: int = 15
    tiny_target_cooldown: int = 10

    # cycle detection (only when actually stuck)
    cycle_check_after: int = 25
    cycle_cooldown: int = 50

    # lock narrow wins (prevents "poisoned flips" from getting immediately undone)
    lock_margin: float = 200.0   # if 0 < margin <= lock_margin, lock district
    lock_ttl: int = 80           # lock duration in steps


# ----------------------------
# Core helpers
# ----------------------------
def build_adj_idx(unit_ids: List[str], adj_json: Dict[str, List[str]]) -> List[List[int]]:
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


def party_margin(dem_sum: np.ndarray, rep_sum: np.ndarray, party: Party) -> np.ndarray:
    if party == "dem":
        return dem_sum - rep_sum
    if party == "rep":
        return rep_sum - dem_sum
    raise ValueError("party must be 'dem' or 'rep'")


def closest_loss_margin(dem_sum: np.ndarray, rep_sum: np.ndarray, party: Party) -> float:
    m = party_margin(dem_sum, rep_sum, party)
    losers = m[m <= 0]
    if losers.size > 0:
        return float(losers.max())
    return float(m.min())


def loss_sum_margin(dem_sum: np.ndarray, rep_sum: np.ndarray, party: Party) -> float:
    m = party_margin(dem_sum, rep_sum, party)
    losers = m[m <= 0]
    return float(losers.sum()) if losers.size > 0 else 0.0


def objective(dem_sum: np.ndarray, rep_sum: np.ndarray, party: Party, cfg: HillclimbConfig) -> float:
    seats = seats_won(dem_sum, rep_sum, party)
    cl = closest_loss_margin(dem_sum, rep_sum, party)
    ls = loss_sum_margin(dem_sum, rep_sum, party)
    return cfg.seat_weight * seats + cfg.flip_weight * cl + cfg.loss_weight * ls


def score_tuple_static(dem_sum: np.ndarray, rep_sum: np.ndarray, party: Party) -> tuple[int, float]:
    seats = seats_won(dem_sum, rep_sum, party)
    m = party_margin(dem_sum, rep_sum, party)
    losers = m[m <= 0]
    closest_loss = float(losers.max()) if losers.size > 0 else float(m.min())
    return (int(seats), float(closest_loss))


def boundary_nodes(labels: np.ndarray, adj_idx: List[List[int]]) -> np.ndarray:
    N = labels.shape[0]
    out = np.zeros(N, dtype=bool)
    for i in range(N):
        li = labels[i]
        for j in adj_idx[i]:
            if labels[j] != li:
                out[i] = True
                break
    return np.where(out)[0]


def is_connected_district_after_removal(
    node: int,
    district: int,
    labels: np.ndarray,
    adj_idx: List[List[int]],
    district_nodes_count: int,
) -> bool:
    if district_nodes_count <= 2:
        return True

    in_d_neighbors = [nbr for nbr in adj_idx[node] if labels[nbr] == district]
    if len(in_d_neighbors) <= 1:
        return True

    start = None
    for nbr in in_d_neighbors:
        if nbr != node:
            start = nbr
            break
    if start is None:
        return True

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

    return len(seen) == (district_nodes_count - 1)


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
    if src == dst:
        return False

    touches_dst = any(labels[nbr] == dst for nbr in adj_idx[node])
    if not touches_dst and district_counts[dst] > 0:
        return False

    w = float(weight[node])

    if pop[src] - w < min_pop:
        return False
    if pop[dst] + w > max_pop:
        return False

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


# ----------------------------
# Swap assist (returns rollback meta + supports tabu + respects locks)
# ----------------------------
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
    party: Party,
    rng: np.random.Generator,
    max_a: int = 200,
    max_b: int = 200,
    *,
    tabu: dict | None = None,     # maps (node, src, dst) -> expire_step
    step: int | None = None,
    tabu_ttl: int = 15,
    min_margin_gain: float = 1e-9,
    locked_until: dict[int, int] | None = None,
) -> tuple[bool, dict | None]:

    def is_locked(d: int) -> bool:
        if locked_until is None or step is None:
            return False
        return step < locked_until.get(int(d), -1)

    def is_tabu_move(node: int, src: int, dst: int) -> bool:
        if tabu is None or step is None:
            return False
        rev = (int(node), int(dst), int(src))
        return tabu.get(rev, -1) >= step

    def mark_tabu_move(node: int, src: int, dst: int) -> None:
        if tabu is None or step is None:
            return
        tabu[(int(node), int(src), int(dst))] = step + int(tabu_ttl)

    def vote_margin(d: int) -> float:
        if party == "dem":
            return float(dem_sum[d] - rep_sum[d])
        else:
            return float(rep_sum[d] - dem_sum[d])

    if is_locked(d_tie):
        return (False, None)

    bnodes = boundary_nodes(labels, adj_idx)

    a_candidates = []
    for a in bnodes:
        a = int(a)
        src = int(labels[a])
        if src == d_tie:
            continue
        if is_locked(src):
            continue
        if any(labels[nbr] == d_tie for nbr in adj_idx[a]):
            if is_tabu_move(a, src, d_tie):
                continue
            a_candidates.append(a)

    if not a_candidates:
        return (False, None)

    rng.shuffle(a_candidates)
    a_candidates = a_candidates[:max_a]

    for a in a_candidates:
        src = int(labels[a])
        if is_locked(src) or is_locked(d_tie):
            continue

        wa = float(weight[a])

        if not is_connected_district_after_removal(a, src, labels, adj_idx, int(district_counts[src])):
            continue
        if pop[src] - wa < min_pop:
            continue

        b_candidates = []
        for b in bnodes:
            b = int(b)
            if int(labels[b]) != d_tie:
                continue
            if any(labels[nbr] == src for nbr in adj_idx[b]):
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

            pop_tie_after = pop[d_tie] + wa - wb
            pop_src_after = pop[src] - wa + wb

            if pop_tie_after < min_pop or pop_tie_after > max_pop:
                continue
            if pop_src_after < min_pop or pop_src_after > max_pop:
                continue

            if not is_connected_district_after_removal(b, d_tie, labels, adj_idx, int(district_counts[d_tie])):
                continue

            dv_b = float(dem_votes[b]); rv_b = float(rep_votes[b])

            if party == "dem":
                m_after = m_before + (dv_a - rv_a) - (dv_b - rv_b)
            else:
                m_after = m_before + (rv_a - dv_a) - (rv_b - dv_b)

            if m_after <= m_before + float(min_margin_gain):
                continue

            moves = []

            # commit b: tie -> src
            if int(labels[b]) != d_tie:
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

            # commit a: src -> tie
            if int(labels[a]) != src:
                # rollback b and skip
                apply_move(
                    node=b, src=src, dst=d_tie,
                    labels=labels, pop=pop,
                    dem_sum=dem_sum, rep_sum=rep_sum,
                    district_counts=district_counts,
                    weight=weight, dem_votes=dem_votes, rep_votes=rep_votes
                )
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

            swap_meta = {"a": int(a), "b": int(b), "src": int(src), "tie": int(d_tie), "moves": moves}
            return (True, swap_meta)

    return (False, None)


# ----------------------------
# Hillclimb
# ----------------------------
def hillclimb_max_seats(
    *,
    labels_init: np.ndarray,
    adj_idx: List[List[int]],
    weight: np.ndarray,
    dem_votes: np.ndarray,
    rep_votes: np.ndarray,
    num_districts: int,
    cfg: HillclimbConfig,
    on_frame: Optional[Callable[[int, np.ndarray, dict], Any]] = None,
    frame_every: int = 5,
) -> np.ndarray:
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

    TABU_TTL = int(cfg.tabu_ttl)
    TARGET_COOLDOWN = int(cfg.tiny_target_cooldown)

    tabu: dict[tuple[int, int, int], int] = {}
    target_cooldown_until: dict[int, int] = {}

    # locking narrow wins
    locked_until: dict[int, int] = {}  # district -> step until locked

    def update_locks(step: int) -> None:
        m_now = party_margin(dem_sum, rep_sum, cfg.party)
        for d in range(num_districts):
            md = float(m_now[d])
            if md > 0.0 and md <= float(cfg.lock_margin):
                locked_until[d] = max(locked_until.get(d, -1), step + int(cfg.lock_ttl))

    def is_locked(d: int, step: int) -> bool:
        return step < locked_until.get(int(d), -1)

    # cycle detection (only when stuck)
    best_score_seen = score_tuple_static(dem_sum, rep_sum, cfg.party)
    no_score_improve = 0
    cycle_cooldown_until = -1
    recent_fingerprints = deque(maxlen=10)

    def is_tabu(node: int, src: int, dst: int, step: int) -> bool:
        rev = (int(node), int(dst), int(src))
        return tabu.get(rev, -1) >= step

    def mark_tabu(node: int, src: int, dst: int, step: int) -> None:
        tabu[(int(node), int(src), int(dst))] = step + TABU_TTL

    def fingerprint(labels_arr: np.ndarray) -> int:
        return hash(labels_arr.tobytes())
    def _emit_frame(step: int):
        if on_frame is None:
            return
        if step % frame_every != 0:
            return
        stats = {
            "seats": int(seats_won(dem_sum, rep_sum, cfg.party)),
            "closest_loss": float(closest_loss_margin(dem_sum, rep_sum, cfg.party)),
            "objective": float(best_obj),
            "locked": int(sum(1 for d, until in locked_until.items() if step < until)) if "locked_until" in locals() else 0,
        }
        on_frame(step, labels, stats)
    ##SCREENSHOT INTITAL FRAME
    if on_frame is not None:
        stats0 = {
            "seats": int(seats_won(dem_sum, rep_sum, cfg.party)),
            "closest_loss": float(closest_loss_margin(dem_sum, rep_sum, cfg.party)),
            "objective": float(best_obj),
            "locked": 0,
        }
        on_frame(0, labels, stats0)

    print(f"[hillclimb] Starting optimization for party='{cfg.party}'")
    print(f"[hillclimb] Initial seats: {seats_won(dem_sum, rep_sum, cfg.party)}")
    print(f"[hillclimb] Initial objective: {best_obj:.6f}")
    print(f"[hillclimb] Max steps: {cfg.max_steps}, Patience: {cfg.patience}")
    print("------------------------------------------------------")

    for step in range(cfg.max_steps):
        tested = feasible = 0

        # Track score improvement for cycle gating
        cur_score = score_tuple_static(dem_sum, rep_sum, cfg.party)
        if cur_score > best_score_seen:
            best_score_seen = cur_score
            no_score_improve = 0
        else:
            no_score_improve += 1

        # Cycle detection ONLY when stuck
        recent_fingerprints.append(fingerprint(labels))
        if (
            step >= cycle_cooldown_until
            and no_score_improve >= int(cfg.cycle_check_after)
            and len(recent_fingerprints) >= 6
        ):
            last6 = list(recent_fingerprints)[-6:]
            period2 = (last6[0] == last6[2] == last6[4]) and (last6[1] == last6[3] == last6[5])
            period3 = (last6[0] == last6[3]) and (last6[1] == last6[4]) and (last6[2] == last6[5])
            if period2 or period3:
                for d in range(num_districts):
                    target_cooldown_until[d] = max(target_cooldown_until.get(d, -1), step + TARGET_COOLDOWN)
                cycle_cooldown_until = step + int(cfg.cycle_cooldown)
                print(
                    f"[hillclimb] detected likely cycle (period={'2' if period2 else '3'}) "
                    f"after {no_score_improve} non-improving steps; applying tiny cooldown",
                    flush=True,
                )

        # boundary nodes
        bnodes = boundary_nodes(labels, adj_idx)
        if len(bnodes) == 0:
            print("[hillclimb] no boundary nodes; stopping", flush=True)
            break

        # recompute target district + locks
        m = party_margin(dem_sum, rep_sum, cfg.party)
        update_locks(step)

        losers = m[m <= 0]
        cl = float(losers.max()) if losers.size > 0 else float(m.min())

        if losers.size > 0:
            loser_ds = np.where(m <= 0)[0]
            d_target = int(loser_ds[np.argmax(m[loser_ds])])
        else:
            d_target = None

        # ---- TINY MODE ----
        tiny_mode = (d_target is not None and cl >= -cfg.tiny_window)
        if tiny_mode and (step < target_cooldown_until.get(int(d_target), -1)):
            tiny_mode = False
        if tiny_mode and d_target is not None and is_locked(int(d_target), step):
            tiny_mode = False

        if tiny_mode:
            print(f"[tiny] step={step+1} d_target={d_target} margin={cl:.1f}", flush=True)

            score_before = score_tuple_static(dem_sum, rep_sum, cfg.party)
            obj_before = best_obj

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
                locked_until=locked_until,
            )

            if did_swap and swap_meta is not None:
                score_after = score_tuple_static(dem_sum, rep_sum, cfg.party)
                obj_after = objective(dem_sum, rep_sum, cfg.party, cfg)

                # also lock the newly flipped district if it's now a narrow win
                update_locks(step)

                accept_swap = (score_after > score_before) or (obj_after > obj_before + 1e-12)

                if accept_swap:
                    best_obj = obj_after
                    no_improve = 0
                    target_cooldown_until[int(d_target)] = step + TARGET_COOLDOWN
                    print(
                        f"[tiny] swap applied score {score_before}->{score_after} "
                        f"obj {obj_before:.6f}->{obj_after:.6f}",
                        flush=True,
                    )
                else:
                    # rollback swap
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
                    print("[tiny] swap rejected + rolled back", flush=True)

            else:
                # (B) best single-node move into target
                best = None
                best_gain = 0.0

                for node in bnodes:
                    src = int(labels[node])
                    if src == d_target:
                        continue
                    if is_locked(src, step) or is_locked(int(d_target), step):
                        continue
                    if not any(labels[nbr] == d_target for nbr in adj_idx[node]):
                        continue
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

                    dv = float(dem_votes[node]); rv = float(rep_votes[node])
                    gain = (dv - rv) if cfg.party == "dem" else (rv - dv)
                    if gain > best_gain:
                        best_gain = gain
                        best = (int(node), int(src), int(d_target))

                if best is not None and best_gain > 0:
                    node_best, src_best, dst_best = best
                    if is_locked(src_best, step) or is_locked(dst_best, step):
                        no_improve += 1
                        target_cooldown_until[int(d_target)] = step + TARGET_COOLDOWN
                    else:
                        score_before = score_tuple_static(dem_sum, rep_sum, cfg.party)
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

                        score_after = score_tuple_static(dem_sum, rep_sum, cfg.party)
                        obj_after = objective(dem_sum, rep_sum, cfg.party, cfg)
                        accept_move = (score_after > score_before) or (obj_after > obj_before + 1e-12)

                        if accept_move:
                            best_obj = obj_after
                            no_improve = 0
                            update_locks(step)
                            target_cooldown_until[int(d_target)] = step + TARGET_COOLDOWN
                            print(f"[tiny] single-node gain move applied gain={best_gain:.1f}", flush=True)
                        else:
                            # rollback
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
                else:
                    no_improve += 1
                    target_cooldown_until[int(d_target)] = step + TARGET_COOLDOWN
                    print("[tiny] no acceptable move found", flush=True)

            # --- Progress stats (compute once) ---
            m_now = party_margin(dem_sum, rep_sum, cfg.party)

            seats_now = int(np.sum(m_now > 0))
            losers_now = m_now[m_now <= 0]
            cl2 = float(losers_now.max()) if losers_now.size > 0 else float(m_now.min())

            ties2 = int(np.sum(m_now == 0))
            strict_losers2 = int(np.sum(m_now < 0))

            print(
                f"[hillclimb] step={step+1} seats={seats_now} closest_loss={cl2:.1f} "
                f"strict_losers={strict_losers2} ties={ties2} tested={tested} feasible={feasible} no_improve={no_improve}",
                flush=True,
            )

            # --- Flipbook frame (no recompute) ---
            if on_frame is not None and (step % frame_every == 0):
                locked_now = int(sum(1 for until in locked_until.values() if step < until))
                stats = {
                    "seats": seats_now,
                    "closest_loss": cl2,
                    "objective": float(best_obj),
                    "locked": locked_now,
                    # pass per-district party margins so FrameRecorder can highlight new wins
                    "margins": m_now.astype(float).tolist(),
                }
                on_frame(step, labels, stats)
            if on_frame is not None and (step % frame_every == 0):
            # Keep payload small + stable (no big arrays)
                stats = {
                    "seats": int(seats_won(dem_sum, rep_sum, cfg.party)),
                    "closest_loss": float(closest_loss_margin(dem_sum, rep_sum, cfg.party)),
                    "objective": float(best_obj),
                    "locked": int(sum(1 for d, until in locked_until.items() if step < until)) if "locked_until" in locals() else 0,
                }
                on_frame(step, labels, stats)

            if no_improve >= cfg.patience:
                print(f"[hillclimb] stopping: no improvement for {cfg.patience} steps", flush=True)
                break
            _emit_frame(step)
            continue

        # ---- NON-TINY MODE ----

        did_move = False

        # Phase 1: targeted move into d_target when near flip
        if d_target is not None and cl >= -cfg.near_flip_window:
            best = None
            best_delta = -1e18

            for node in bnodes:
                src = int(labels[node])
                if src == d_target:
                    continue
                if is_locked(src, step) or is_locked(int(d_target), step):
                    continue
                if not any(labels[nbr] == d_target for nbr in adj_idx[node]):
                    continue
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
                    if not (is_locked(src, step) or is_locked(dst, step)):
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
                        update_locks(step)
                        did_move = True

        # Phase 2: generic best-of-batch
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
                if is_locked(src, step):
                    continue

                nbr_districts = set(int(labels[nbr]) for nbr in adj_idx[node] if int(labels[nbr]) != src)
                if not nbr_districts:
                    continue

                for dst in nbr_districts:
                    if is_locked(dst, step):
                        continue
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
                if not (is_locked(src, step) or is_locked(dst, step)):
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
                    update_locks(step)
                else:
                    no_improve += 1
            else:
                no_improve += 1

        # Progress print
        # --- Progress stats (compute once) ---
        m_now = party_margin(dem_sum, rep_sum, cfg.party)

        seats_now = int(np.sum(m_now > 0))
        losers_now = m_now[m_now <= 0]
        cl2 = float(losers_now.max()) if losers_now.size > 0 else float(m_now.min())

        ties2 = int(np.sum(m_now == 0))
        strict_losers2 = int(np.sum(m_now < 0))

        print(
            f"[hillclimb] step={step+1} seats={seats_now} closest_loss={cl2:.1f} "
            f"strict_losers={strict_losers2} ties={ties2} tested={tested} feasible={feasible} no_improve={no_improve}",
            flush=True,
        )

        # --- Flipbook frame (no recompute) ---
        if on_frame is not None and (step % frame_every == 0):
            locked_now = int(sum(1 for until in locked_until.values() if step < until))
            stats = {
                "seats": seats_now,
                "closest_loss": cl2,
                "objective": float(best_obj),
                "locked": locked_now,
                # pass per-district party margins so FrameRecorder can highlight new wins
                "margins": m_now.astype(float).tolist(),
            }
            on_frame(step, labels, stats)

        if no_improve >= cfg.patience:
            print(f"[hillclimb] stopping: no improvement for {cfg.patience} steps", flush=True)
            break

    print("------------------------------------------------------")
    print(f"[hillclimb] Final seats: {seats_won(dem_sum, rep_sum, cfg.party)}")
    print(f"[hillclimb] Final objective: {best_obj:.6f}")
    return labels