# src/gerry/algos/pack_sinks.py
#
# Goal: Deterministic "opposition packing" algorithm.
# - Select top connected clusters of high opposition share as sink cores
# - Grow each sink district contiguously to near-equal population
# - Use short lookahead on the adjacency graph to “reach” nearby high-opposition nodes
# - Fill remaining districts with population-balanced contiguous growth
#
# Entry point:
#   run(pack, cfg, maximize="dem"|"rep") -> labels (np.ndarray of district ids)
#
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional

import numpy as np


# ----------------------------
# Config
# ----------------------------

@dataclass
class PackSinksParams:
    # Core redistricting constraint
    num_districts: int = 17
    pop_tolerance: float = 0.05

    # Packing behavior
    num_sinks: int = 3  # how many opposition "sink" districts to build first
    opp_share_threshold: float = 0.65  # node qualifies as "high-opposition" for clustering
    target_opp_share: float = 0.65     # sink goal at completion (tunable; may be unmet in some geographies)

    # Lookahead behavior (graph-distance)
    lookahead_depth: int = 4           # max BFS depth for "reach" paths to high-opposition nodes
    lookahead_min_gain: float = 0.0    # require strictly positive gain in sink score to accept a lookahead path (set >0 to be stricter)

    # Tie-breaking / determinism
    deterministic: bool = True


def _params_from_cfg(cfg: dict) -> PackSinksParams:
    p = PackSinksParams()
    run_cfg = cfg.get("run", {})
    algo_cfg = cfg.get("algo", {}).get("pack_sinks", {})

    p.num_districts = int(algo_cfg.get("num_districts", run_cfg.get("num_districts", p.num_districts)))
    p.pop_tolerance = float(algo_cfg.get("pop_tolerance", run_cfg.get("pop_tolerance", p.pop_tolerance)))

    p.num_sinks = int(algo_cfg.get("num_sinks", p.num_sinks))
    p.opp_share_threshold = float(algo_cfg.get("opp_share_threshold", p.opp_share_threshold))
    p.target_opp_share = float(algo_cfg.get("target_opp_share", p.target_opp_share))

    p.lookahead_depth = int(algo_cfg.get("lookahead_depth", p.lookahead_depth))
    p.lookahead_min_gain = float(algo_cfg.get("lookahead_min_gain", p.lookahead_min_gain))

    p.deterministic = bool(algo_cfg.get("deterministic", p.deterministic))
    return p


# ----------------------------
# Basic utilities
# ----------------------------

def _neighbors_sorted(adj: List[List[int]], u: int) -> List[int]:
    # keep deterministic expansion
    nbrs = adj[u]
    return sorted(nbrs)


def _compute_shares(dem: np.ndarray, rep: np.ndarray, maximize: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      opp_share[i]  = opponent_share at node i (two-party)
      opp_votes[i]  = opponent vote count at node i
    maximize = "dem" or "rep" (party we try to maximize seats for)
    Opponent is the other party.
    """
    maximize = maximize.lower().strip()
    if maximize == "rep":
        opp_votes = dem.astype(float)
        my_votes = rep.astype(float)
    else:
        opp_votes = rep.astype(float)
        my_votes = dem.astype(float)

    tot = (my_votes + opp_votes)
    opp_share = np.where(tot > 0, opp_votes / tot, 0.0)
    return opp_share, opp_votes


def _district_opp_share(opp_votes_sum: float, my_votes_sum: float) -> float:
    tot = opp_votes_sum + my_votes_sum
    return (opp_votes_sum / tot) if tot > 0 else 0.0


def _target_bounds(weight: np.ndarray, K: int, tol: float) -> Tuple[float, float, float]:
    total_pop = float(weight.sum())
    target = total_pop / K
    min_pop = target * (1 - tol)
    max_pop = target * (1 + tol)
    return target, min_pop, max_pop


# ----------------------------
# Clustering high-opposition nodes
# ----------------------------

def _high_opp_components(high_mask: np.ndarray, adj: List[List[int]], deterministic: bool) -> List[List[int]]:
    """
    Connected components on subgraph induced by high_mask==True.
    Deterministic ordering: components returned in descending size, then by min node id.
    """
    N = len(high_mask)
    seen = np.zeros(N, dtype=bool)
    comps: List[List[int]] = []

    for start in range(N):
        if not high_mask[start] or seen[start]:
            continue
        q = deque([start])
        seen[start] = True
        comp = [start]
        while q:
            u = q.popleft()
            for v in (_neighbors_sorted(adj, u) if deterministic else adj[u]):
                if high_mask[v] and not seen[v]:
                    seen[v] = True
                    q.append(v)
                    comp.append(v)
        comp.sort()
        comps.append(comp)

    # Sort for deterministic "top clusters"
    comps.sort(key=lambda c: (-len(c), c[0] if c else 10**18))
    return comps


def _score_cluster(comp: List[int], opp_votes: np.ndarray) -> float:
    # prioritize by total opponent votes (simple + effective for packing)
    return float(opp_votes[comp].sum()) if comp else -1.0


# ----------------------------
# Sink growth with short lookahead
# ----------------------------

def _frontier_of_set(nodes: Set[int], adj: List[List[int]], unassigned: Set[int], deterministic: bool) -> List[int]:
    frontier = set()
    for u in nodes:
        for v in (adj[u] if not deterministic else _neighbors_sorted(adj, u)):
            if v in unassigned:
                frontier.add(v)
    return sorted(frontier) if deterministic else list(frontier)


def _bfs_path_to_high(
    sources: List[int],
    adj: List[List[int]],
    unassigned: Set[int],
    high_mask: np.ndarray,
    max_depth: int,
    deterministic: bool,
) -> Optional[List[int]]:
    """
    Multi-source BFS starting from 'sources' (frontier nodes).
    Searches for a high_mask node within <= max_depth, traveling only through unassigned nodes.
    Returns a path [v0, v1, ..., vt] where v0 is a frontier node (in sources) and vt is high_mask.
    Deterministic: explores nodes and neighbors in sorted order.
    """
    if max_depth <= 0:
        return None

    # parent pointers
    parent: Dict[int, int] = {}
    depth: Dict[int, int] = {}

    # initialize queue
    q = deque()
    start_nodes = sorted(sources) if deterministic else sources
    for s in start_nodes:
        if s not in unassigned:
            continue
        parent[s] = -1
        depth[s] = 0
        q.append(s)

    while q:
        u = q.popleft()
        d = depth[u]
        if d > max_depth:
            continue

        if high_mask[u] and d > 0:
            # reconstruct path from u back to some source
            path = [u]
            while parent[path[-1]] != -1:
                path.append(parent[path[-1]])
            path.reverse()
            return path

        if d == max_depth:
            continue

        for v in (_neighbors_sorted(adj, u) if deterministic else adj[u]):
            if v not in unassigned:
                continue
            if v in parent:
                continue
            parent[v] = u
            depth[v] = d + 1
            q.append(v)

    return None


def _grow_sink_district(
    core: List[int],
    unassigned: Set[int],
    adj: List[List[int]],
    weight: np.ndarray,
    dem: np.ndarray,
    rep: np.ndarray,
    opp_votes: np.ndarray,
    maximize: str,
    min_pop: float,
    max_pop: float,
    params: PackSinksParams,
) -> List[int]:
    """
    Grow a sink district starting from a connected core (list of nodes).
    Maintains contiguity: only adds frontier nodes or frontier->path nodes (all connected).
    Tries to keep opponent share high, with lookahead to reach nearby high-opposition precincts.
    """
    maximize = maximize.lower().strip()
    deterministic = params.deterministic

    district: Set[int] = set()
    pop = 0.0
    dem_sum = 0.0
    rep_sum = 0.0

    # initialize from core (truncate if needed to respect max_pop)
    for u in core:
        if u not in unassigned:
            continue
        w = float(weight[u])
        if pop + w > max_pop and len(district) > 0:
            break
        district.add(u)
        unassigned.remove(u)
        pop += w
        dem_sum += float(dem[u])
        rep_sum += float(rep[u])

    # if core ended up empty (already assigned), return empty
    if not district:
        return []

    def current_opp_share() -> float:
        if maximize == "rep":
            return _district_opp_share(dem_sum, rep_sum)
        else:
            return _district_opp_share(rep_sum, dem_sum)

    # Growth loop: aim to reach min_pop, while keeping opp_share high
    # Strategy:
    #   - prefer adding high-opposition frontier nodes
    #   - if none, use lookahead path to a nearby high-opposition node
    #   - if still none, add the frontier node that best preserves opp_share
    safety = 0
    while pop < min_pop and unassigned and safety < 5_000_000:
        safety += 1

        frontier = _frontier_of_set(district, adj, unassigned, deterministic)
        if not frontier:
            break

        # Candidate 1: best high-opposition frontier node (by opponent share at node)
        # We'll approximate node opp_share using opp_votes / (dem+rep) at node.
        # That value exists implicitly via pack votes; recompute quickly:
        #   if maximize=dem => opp=rep; if maximize=rep => opp=dem.
        best_frontier = None
        best_frontier_score = -1e18

        for v in frontier:
            wv = float(weight[v])
            if pop + wv > max_pop:
                continue
            dv = float(dem[v])
            rv = float(rep[v])

            # resulting opp share if we add v
            if maximize == "rep":
                new_opp_share = _district_opp_share(dem_sum + dv, rep_sum + rv)
            else:
                new_opp_share = _district_opp_share(rep_sum + rv, dem_sum + dv)

            # score: prioritize higher new_opp_share, then higher opp_votes contribution
            ov = float(opp_votes[v])
            score = (1_000_000.0 * new_opp_share) + ov - 0.000001 * v
            if score > best_frontier_score:
                best_frontier_score = score
                best_frontier = v

        # Candidate 2: lookahead path to a high-opposition node within depth
        # Only attempt if it seems helpful: either we're below target_opp_share, or no good frontier fit.
        use_lookahead = (params.lookahead_depth > 0)

        # lookahead_path = None
        # if use_lookahead:
        #     path = _bfs_path_to_high(
        #         sources=frontier,
        #         adj=adj,
        #         unassigned=unassigned,
        #         high_mask=(lambda: None)(),  # placeholder, overwritten below
        #         max_depth=params.lookahead_depth,
        #         deterministic=deterministic,
        #     )

        # We need high_mask for opponent-heavy nodes, computed from current maximize.
        # Compute it once per loop cheaply (node-level opp share):
        # (We avoid caching here for simplicity; if desired, compute once at the top of run().)
        if maximize == "rep":
            node_tot = (dem + rep).astype(float)
            node_opp_share = np.where(node_tot > 0, dem.astype(float) / node_tot, 0.0)
        else:
            node_tot = (dem + rep).astype(float)
            node_opp_share = np.where(node_tot > 0, rep.astype(float) / node_tot, 0.0)
        high_mask = (node_opp_share >= params.opp_share_threshold)

        lookahead_path = None
        if use_lookahead:
            lookahead_path = _bfs_path_to_high(
                sources=frontier,
                adj=adj,
                unassigned=unassigned,
                high_mask=high_mask,
                max_depth=params.lookahead_depth,
                deterministic=deterministic,
            )

        # Decide what to add
        added_any = False

        # Try lookahead path if it fits and meaningfully improves our sink goal
        if lookahead_path is not None:
            # Check if adding entire path stays under max_pop
            path_pop = float(weight[lookahead_path].sum())
            if pop + path_pop <= max_pop:
                # Estimate new opp_share if we add the path
                path_dem = float(dem[lookahead_path].sum())
                path_rep = float(rep[lookahead_path].sum())
                if maximize == "rep":
                    new_opp_share = _district_opp_share(dem_sum + path_dem, rep_sum + path_rep)
                else:
                    new_opp_share = _district_opp_share(rep_sum + path_rep, dem_sum + path_dem)

                gain = new_opp_share - current_opp_share()
                # Accept if it improves opp share (or meets a minimum gain threshold), especially if we're below target.
                if (gain >= params.lookahead_min_gain) and (current_opp_share() < params.target_opp_share or best_frontier is None):
                    for v in lookahead_path:
                        if v in unassigned:
                            district.add(v)
                            unassigned.remove(v)
                            pop += float(weight[v])
                            dem_sum += float(dem[v])
                            rep_sum += float(rep[v])
                    added_any = True

        # Otherwise add the best frontier node
        if not added_any and best_frontier is not None:
            v = best_frontier
            district.add(v)
            unassigned.remove(v)
            pop += float(weight[v])
            dem_sum += float(dem[v])
            rep_sum += float(rep[v])
            added_any = True

        if not added_any:
            fallback_count += 1
            break

    return sorted(district) if deterministic else list(district)

"""HELPER 
will postprocess to enforce contiguity

"""
def _repair_contiguity_by_reassigning_islands(
    labels: np.ndarray,
    adj: List[List[int]],
    weight: np.ndarray,
    K: int,
    min_pop: float,
    max_pop: float,
    max_passes: int = 10,
    deterministic: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Post-pass contiguity repair:
      - For each district, keep largest connected component.
      - Reassign nodes in smaller components to a touching neighboring district.
      - Prefer neighbor with lowest population that stays within max_pop.
    Returns (labels, moved_count).
    """
    from collections import deque
    labels = labels.copy()

    # Track district pops so reassigning can be pop-aware
    district_pops = np.zeros(K, dtype=float)
    for i in range(len(labels)):
        district_pops[labels[i]] += float(weight[i])

    total_moved = 0

    for _pass in range(max_passes):
        moved_this_pass = 0

        for d in range(K):
            members = np.where(labels == d)[0]
            if members.size <= 1:
                continue

            member_set = set(members.tolist())

            # Find connected components within district d
            seen = set()
            comps = []

            for start in members:
                s = int(start)
                if s in seen:
                    continue
                q = deque([s])
                seen.add(s)
                comp = [s]
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

            # Keep largest component
            comps.sort(key=lambda c: (-len(c), min(c)))
            main = set(comps[0])

            # Reassign all other components
            for comp in comps[1:]:
                # deterministic ordering for stability
                comp_nodes = sorted(comp) if deterministic else comp

                for node in comp_nodes:
                    w = float(weight[node])

                    # candidate neighbor districts (must touch)
                    nbr_ds = {labels[n] for n in adj[node] if labels[n] != d}
                    if not nbr_ds:
                        continue

                    nbr_ds = sorted(nbr_ds) if deterministic else list(nbr_ds)

                    # prefer those that stay under max_pop
                    feasible = [dd for dd in nbr_ds if district_pops[dd] + w <= max_pop]

                    if feasible:
                        # pick lowest-pop feasible neighbor
                        dd_best = min(feasible, key=lambda dd: (district_pops[dd], dd))
                    else:
                        # no feasible neighbor; pick least overflow
                        dd_best = min(nbr_ds, key=lambda dd: ((district_pops[dd] + w) - max_pop, district_pops[dd], dd))

                    # apply reassignment
                    labels[node] = dd_best
                    district_pops[d] -= w
                    district_pops[dd_best] += w
                    moved_this_pass += 1

        total_moved += moved_this_pass
        if moved_this_pass == 0:
            break

    return labels, total_moved
def _count_components_per_district(labels, adj, K):
    from collections import deque
    import numpy as np

    results = {}

    for d in range(K):
        nodes = np.where(labels == d)[0]
        if len(nodes) == 0:
            continue

        node_set = set(nodes.tolist())
        seen = set()
        components = 0

        for start in nodes:
            if start in seen:
                continue
            components += 1
            q = deque([int(start)])
            seen.add(int(start))
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if v in node_set and v not in seen:
                        seen.add(v)
                        q.append(v)

        results[d] = components

    return results

# ----------------------------
# Fill remaining districts (population-balanced contiguous growth)
# ----------------------------

def _build_remaining_districts(
    labels: np.ndarray,
    start_district: int,
    unassigned: Set[int],
    adj: List[List[int]],
    weight: np.ndarray,
    K: int,
    min_pop: float,
    max_pop: float,
    deterministic: bool,
) -> None:
    """
    Deterministic contiguous fill for districts [start_district .. K-1].
    Strategy:
      - seed each remaining district with the smallest-id unassigned node
      - then repeatedly expand the district with smallest population by adding a frontier node
        that brings it closest to target, respecting max_pop.
    """
    fallback_count = 0 # checker
    target = float(weight.sum()) / K

    district_pops = np.zeros(K, dtype=float)

    # initialize pops for already-labeled nodes
    for i in range(len(labels)):
        d = labels[i]
        if d >= 0:
            district_pops[d] += float(weight[i])

    # seed remaining districts
    for d in range(start_district, K):
        if not unassigned:
            break
        seed = min(unassigned) if deterministic else next(iter(unassigned))
        labels[seed] = d
        unassigned.remove(seed)
        district_pops[d] += float(weight[seed])

    # maintain frontier sets
    frontiers: List[Set[int]] = [set() for _ in range(K)]
    for d in range(start_district, K):
        members = np.where(labels == d)[0]
        for u in members:
            for v in (_neighbors_sorted(adj, int(u)) if deterministic else adj[int(u)]):
                if v in unassigned:
                    frontiers[d].add(v)

    safety = 0
    while unassigned and safety < 5_000_000:
        safety += 1

        # pick district with smallest pop (among remaining districts only)
        d = min(range(start_district, K), key=lambda dd: district_pops[dd])

        # candidates from its frontier that fit max_pop
        cand = [v for v in (sorted(frontiers[d]) if deterministic else list(frontiers[d]))
                if v in unassigned and district_pops[d] + float(weight[v]) <= max_pop]

        if not cand:
            # try any remaining district
            found = False
            for dd in sorted(range(start_district, K), key=lambda x: district_pops[x]):
                cand2 = [v for v in (sorted(frontiers[dd]) if deterministic else list(frontiers[dd]))
                         if v in unassigned and district_pops[dd] + float(weight[v]) <= max_pop]
                if cand2:
                    d = dd
                    cand = cand2
                    found = True
                    break
            if not found:
                # last resort: assign the smallest unassigned node to the smallest-pop district (still contiguous may fail)
                v = min(unassigned) if deterministic else next(iter(unassigned))
                d = min(range(start_district, K), key=lambda dd: district_pops[dd])
                cand = [v]
                fallback_count += 1

        # choose node that makes district closest to target
        v_best = min(cand, key=lambda v: abs((district_pops[d] + float(weight[v])) - target))
        labels[v_best] = d
        unassigned.remove(v_best)
        district_pops[d] += float(weight[v_best])

       

        # update frontiers: remove v_best from all, add its neighbors to district d
        for dd in range(start_district, K):
            frontiers[dd].discard(v_best)
        for nbr in (_neighbors_sorted(adj, v_best) if deterministic else adj[v_best]):
            if nbr in unassigned:
                frontiers[d].add(nbr)
    print(f"[INFO] Remaining-district fallback assignments: {fallback_count}")

# ----------------------------
# Public entry point
# ----------------------------

def run(pack, cfg: dict, maximize: str = "dem") -> np.ndarray:
    """
    Deterministic opposition-packing algorithm.

    maximize: which party we are trying to maximize seats for ("dem" or "rep").
              The opponent is the other party, and we try to PACK opponent voters into sink districts.
    """
    maximize = maximize.lower().strip()
    if maximize not in {"dem", "rep"}:
        maximize = "dem"

    params = _params_from_cfg(cfg)

    dem = pack.dem.astype(float)
    rep = pack.rep.astype(float)
    weight = pack.weight.astype(float)
    adj = pack.adj
    N = len(weight)
    K = params.num_districts

    _, min_pop, max_pop = _target_bounds(weight, K, params.pop_tolerance)

    # compute opponent share at node and opponent votes
    opp_share, opp_votes = _compute_shares(dem, rep, maximize)

    # high-opposition mask for clustering
    high_mask = (opp_share >= params.opp_share_threshold)

    # find connected components among high-opposition nodes
    comps = _high_opp_components(high_mask, adj, params.deterministic)

    # rank clusters by total opponent votes (descending)
    scored = [(c, _score_cluster(c, opp_votes)) for c in comps]
    print(f"Found {len(comps)} high-opposition clusters.")
    for i, (comp, score) in enumerate(scored[:5]):
        print(f"  Cluster {i}: size={len(comp)}, opp_votes={score:.0f}")
    scored.sort(key=lambda x: (-x[1], x[0][0] if x[0] else 10**18))

    # initialize labels
    labels = np.full(N, -1, dtype=int)
    unassigned: Set[int] = set(range(N))

    # Build sink districts first: districts 0..S-1
    S = max(0, min(params.num_sinks, K))
    sink_idx = 0
    used_nodes: Set[int] = set()

    for comp, _score in scored:
        if sink_idx >= S:
            break

        # core must be fully unassigned to be usable; otherwise take the unassigned subset and ensure it stays connected-ish
        core = [u for u in comp if u in unassigned]
        if not core:
            continue

        print(f"\nBuilding sink district {sink_idx} from cluster size {len(core)}")
        # Grow sink from this core
        sink_nodes = _grow_sink_district(
            core=core,
            unassigned=unassigned,
            adj=adj,
            weight=weight,
            dem=dem,
            rep=rep,
            opp_votes=opp_votes,
            maximize=maximize,
            min_pop=min_pop,
            max_pop=max_pop,
            params=params,
        )
        if not sink_nodes:
            continue
            

        # assign sink labels
        for u in sink_nodes:
            labels[u] = sink_idx
            used_nodes.add(u)
        if sink_nodes:
            opp_sum = sum(opp_votes[u] for u in sink_nodes)
            tot_sum = sum(dem[u] + rep[u] for u in sink_nodes)
            share = opp_sum / tot_sum if tot_sum > 0 else 0
            print(f"  Sink {sink_idx} built: nodes={len(sink_nodes)}, opp_share={share:.3f}")

        # check achieved opponent share; if it’s too low, you might choose to not count it as a sink
        # For determinism we keep it, but you can change logic later if desired.
        sink_idx += 1

        ##CHECKPOINT
        print("\n[CHECK] After sink construction")
        comps = _count_components_per_district(labels, adj, K)
        for d, c in comps.items():
            if c > 1:
                print(f"  District {d} has {c} components")
        


    # If we couldn't build all sinks from clusters, seed remaining sinks from highest opposition nodes
    if sink_idx < S:
        # deterministic: iterate nodes in descending opp_share then by id
        order = list(range(N))
        order.sort(key=lambda i: (-float(opp_share[i]), -float(opp_votes[i]), i))
        for seed in order:
            if sink_idx >= S:
                break
            if seed not in unassigned:
                continue
            # core is just this seed; grow it
            sink_nodes = _grow_sink_district(
                core=[seed],
                unassigned=unassigned,
                adj=adj,
                weight=weight,
                dem=dem,
                rep=rep,
                opp_votes=opp_votes,
                maximize=maximize,
                min_pop=min_pop,
                max_pop=max_pop,
                params=params,
            )
            if not sink_nodes:
                continue
            for u in sink_nodes:
                labels[u] = sink_idx
            sink_idx += 1

    # Build remaining districts contiguously & population-balanced
    _build_remaining_districts(
        labels=labels,
        start_district=sink_idx,
        unassigned=unassigned,
        adj=adj,
        weight=weight,
        K=K,
        min_pop=min_pop,
        max_pop=max_pop,
        deterministic=params.deterministic,
    )
    print("\n[CHECK] After remaining district fill")
    fallback_count = 0 # checker
    comps = _count_components_per_district(labels, adj, K)
    for d, c in comps.items():
        if c > 1:
            print(f"  District {d} has {c} components")

       
    ''' Remove the following to get better pop contraints, but lsoe contitugity'''
    labels, moved = _repair_contiguity_by_reassigning_islands(
        labels=labels,
        adj=adj,
        weight=weight,
        K=K,
        min_pop=min_pop,
        max_pop=max_pop,
        max_passes=10,
        deterministic=params.deterministic,
    )
    print(f"[INFO] Contiguity repair moved {moved} precincts")
    ''' end removal here'''

    # # Final: any stragglers (should be rare) assign to smallest-index district with adjacency if possible
    # # This preserves contiguity as much as possible; but if it triggers, your inputs/tolerance are likely too tight.
    if unassigned:
        district_pops = np.zeros(K, dtype=float)
        for i in range(N):
            if labels[i] >= 0:
                district_pops[labels[i]] += float(weight[i])

        for v in sorted(unassigned):
            nbr_ds = sorted({int(labels[n]) for n in adj[v] if labels[n] != -1})
            if nbr_ds:
                # choose neighbor district with smallest pop that fits max_pop if possible
                candidates = [d for d in nbr_ds if district_pops[d] + float(weight[v]) <= max_pop]
                d = candidates[0] if candidates else nbr_ds[0]
            else:
                d = int(np.argmin(district_pops))
            labels[v] = d
            district_pops[d] += float(weight[v])

    labels = labels.astype(int)
    labels[labels < 0] = 0
    labels[labels >= K] = K - 1

    print("\nFinal district population stats:")
    district_pops = np.zeros(K)
    for i in range(N):
        district_pops[labels[i]] += weight[i]

    print("  min_pop:", district_pops.min())
    print("  max_pop:", district_pops.max())

    ## CHECKPOINT
    print("\n[CHECK] Final contiguity")
    comps = _count_components_per_district(labels, adj, K)
    for d, c in comps.items():
        if c > 1:
            print(f"  District {d} has {c} components")

    return labels

