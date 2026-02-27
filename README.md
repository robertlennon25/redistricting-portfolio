# Illinois Redistricting Algorithm Demo

---

# 1) Objective and Data Source

## Project Objective

This project explores algorithmic approaches to drawing U.S. congressional district maps under the following constraints:

- Equal (or near-equal) population per district  
- Contiguity (each district must form a connected region)  
- Optimization of partisan outcomes (maximize seats for a selected party)

The goal is to build a flexible experimental framework where different redistricting algorithms can be implemented, tested, visualized, and compared through a web interface.

This is a research/demo environment — not a production or legally compliant redistricting system.

---

## Data Sources

### Election Precinct Data

Illinois 2024 General Election precinct-level results and boundaries were sourced from:

Redistricting Data Hub (RDH):  
https://redistrictingdatahub.org

The dataset includes:

- Precinct geometries
- Congressional vote totals (Democrat, Republican)
- Unique precinct identifiers

### Census Population Data

Population data is derived from:

2020 Census PL 94-171 Redistricting Data (Block-level)

We spatially join Census blocks to precincts and aggregate total population (`P0010001`) to compute:

- `TOTPOP` (total population per precinct)

This ensures population equality constraints are based on Census population, not vote totals.

---

# 2) Algorithms Used and Why

We currently support two algorithm families:

---

## Greedy (Baseline)

### Purpose
A fast heuristic for building districts by expanding regions from seed nodes.

### Structure
1. Seed selection
2. Region growth under population constraints
3. Optional contiguity repair
4. Optional population balancing

### Strengths
- Simple
- Fast
- Easy to reason about

### Weaknesses
- Does not robustly enforce contiguity
- Can get stuck in poor local configurations
- Limited seat optimization

---

## smartA (Seeded Growth + Simulated Annealing)

### Purpose
A more advanced approach that:

1. Constructs a valid initial map (population-balanced + contiguous-ish)
2. Optimizes partisan seat outcomes via boundary swaps
3. Uses simulated annealing to escape local optima

### Key Concepts

- Seeded region growing
- Boundary precinct detection
- Connectivity checks before removal
- Population constraint enforcement
- Sigmoid-based seat objective
- Optional anti-waste penalty
- Optional paired swaps
- Annealing temperature schedule

### Why This Approach?

Seat maximization is a discontinuous objective (crossing 50% matters).  
Simulated annealing provides:

- Local improvement
- Escape from local minima
- Exploration early, refinement late

This structure is closer to real-world redistricting heuristics.

---

# 3) Chronological: How to Run the System

This section explains how a developer should use the pipeline from raw data to frontend.

---

## Step 0 — Folder Structure

Raw datasets should be stored in:

```
/raw-data/
```

Recommended structure:

```
raw-data/
  il_2024_gen_prec/
    il_2024_gen_cong_prec.shp
  il_pl2020_blocks/
    il_pl2020_p1_b.shp
```

After spatially joining population:

```
il_2024_gen_cong_prec_with_pop.gpkg
```

---

## Step 1 — Build Map Pack

Run:

```
python3 scripts/build_map_pack.py --config config.yaml
```

### What build_map_pack Does

- Reads precinct geometry file (.shp or .gpkg)
- Detects Dem/Rep vote columns dynamically
- Uses `weight_col` (e.g., `TOTPOP`) as population weight
- Computes centroids
- Builds adjacency graph (touch-based)
- Creates ID ↔ index mappings

### Output Location

```
assets/<pack_name>/
```

### Files Created

- `shapes.geojson`
- `attributes.csv`
- `adjacency.json`
- `id_to_idx.json`
- `idx_to_id.json`
- `meta.json`

This directory is called the **Map Pack**.

It is the static representation of the geographic problem.

---

## Step 2 — Run an Algorithm

### Greedy (example)

```
python3 scripts/run_greedy_pack_v2.py --config config.yaml --maximize dem
```

### Custom smartA

```
python3 scripts/run_custom_algo.py --config config.yaml --algo_key smartA --maximize dem
```

---

### What the Runner Does

The runner:

1. Loads the Map Pack
2. Calls the algorithm’s `run(pack, cfg, maximize)` function
3. Receives `labels` (district assignments)
4. Exports frontend-ready files
5. Updates `apps/web/public/outputs/latest.json`

---

## Step 3 — Output Files

Each run creates a timestamped directory:

```
apps/web/public/outputs/<algo_key>_<maximize>_<timestamp>/
```

Example:

```
apps/web/public/outputs/smartA_dem_20260226_214512/
```

Files written:

- `map_data.geojson`
- `districts.geojson`
- `district_stats.json`
- `district_stats.csv`
- `map.png`
- `unit_to_district.csv`

The `latest.json` manifest maps:

```
{
  "smartA_dem": "smartA_dem_20260226_214512",
  "smartA_rep": "smartA_rep_20260226_214530"
}
```

The frontend reads this file dynamically.

---

## Step 4 — Run Frontend

From:

```
apps/web/
```

Run:

```
npm install
npm run dev
```

The frontend:

1. Fetches `/outputs/latest.json`
2. Populates dropdown with algorithm keys
3. Fetches:
   - `map_data.geojson`
   - `district_stats.json`
   - `districts.geojson`
4. Renders interactive Leaflet map
5. Displays sidebar district stats

No backend server is required — all outputs are static files.

---

# 4) Current Known Bugs and Limitations

This is an experimental framework.

Current issues include:

## 1. Contiguity Not Fully Enforced

- Phase 1 fallback assignments may break contiguity.
- Connectivity checks are local (removal-based) and may miss global fragmentation.
- No final global contiguity verification pass.

## 2. Population Locking

- Tight tolerance can prevent legal swaps.
- Annealing may freeze if no feasible moves exist.

## 3. Objective Symmetry

- Dem and GOP runs can converge to identical maps if:
  - Annealing fails to explore
  - Phase 1 dominates solution
  - Vote signal is weak relative to constraints

## 4. No Compactness Constraint

Version A intentionally ignores shape quality.

Future versions may include:

- Perimeter penalties
- Polsby-Popper compactness
- Dispersion penalties

## 5. Performance

- Connectivity checks are O(V+E) per move.
- Large iteration counts may slow execution.
- Boundary sampling strategy may require optimization.

---

# 5) Credits

- Redistricting Data Hub (RDH)  
  https://redistrictingdatahub.org  
  Precinct boundaries and election data.

- U.S. Census Bureau  
  2020 PL 94-171 Redistricting Data (Block-level population)

- Python libraries:
  - GeoPandas
  - Shapely
  - NumPy
  - Matplotlib

---

# Final Notes

This project is an algorithmic sandbox for exploring redistricting under legal constraints.

The system is modular:

- Map Pack layer = static geographic representation
- Algorithm layer = experimental redistricting logic
- Runner layer = I/O + manifest handling
- Frontend layer = static visualization

New algorithms can be added by:

1. Creating a new file in `src/gerry/algos/`
2. Implementing `run(pack, cfg, maximize)`
3. Running via `scripts/run_custom_algo.py`
4. Viewing results immediately in the web UI

This structure enables rapid iteration and comparative experimentation between algorithm families.