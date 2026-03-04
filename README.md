=======
# Redistricting Portfolio

An interactive **algorithmic redistricting research project** that generates district maps, optimizes them using heuristic algorithms, evaluates their political properties, and visualizes the results through a **Next.js interactive web application**.

This repository contains:

- A **Python backend** for generating and optimizing district maps
- A **data pipeline** for converting precinct shapefiles into fast algorithm-ready map packs
- Multiple **redistricting algorithms** (K-Means, hillclimb optimization, etc.)
- Automated **export pipelines** for producing GeoJSON and statistics
- A **Next.js frontend** that allows users to explore maps and watch the algorithm evolve in a flipbook animation

The goal is to demonstrate how algorithmic redistricting works and provide a platform for experimenting with optimization approaches.

---

# Project Architecture

```
redistricting/
│
├── gerry/                  # Core Python library
│   ├── algos/              # Districting algorithms
│   ├── data/               # Map pack loader utilities
│   ├── viz/                # Frame recorder (flipbook generator)
│
├── scripts/                # CLI entrypoints for running algorithms
│   ├── build_map_pack.py
│   ├── run_kmeans_pack.py
│   ├── run_hillclimber.py
│
├── raw-data/               # Source shapefiles and precinct datasets
│
├── apps/web/               # Next.js frontend
│   ├── pages/
│   ├── components/
│   ├── public/outputs/     # Exported maps used by the frontend
│
└── config.yaml             # Configuration file controlling algorithms
```

The repository is divided into two major layers:

### Backend (Python)

Responsible for:

- ingesting shapefiles
- constructing adjacency graphs
- generating district assignments
- optimizing district configurations
- exporting data to files consumed by the frontend

### Frontend (Next.js)

Responsible for:

- loading precomputed runs
- rendering maps with Leaflet
- displaying district statistics
- visualizing algorithm progress

---

# Data Pipeline Overview

The redistricting workflow follows this pipeline:

```
Precinct shapefile
      │
      ▼
build_map_pack.py
      │
      ▼
Map Pack (algorithm-ready data)
      │
      ▼
Districting algorithm
      │
      ▼
Export run
      │
      ▼
GeoJSON + statistics
      │
      ▼
Next.js frontend
```

Each stage is described below.

---

# 1. Raw Input Data

The system begins with a **precinct-level shapefile** containing:

Required columns:

| Column | Description |
|------|-------------|
| `unit_id` | Unique precinct identifier |
| `dem_votes` | Democratic vote count |
| `rep_votes` | Republican vote count |
| `weight` | Population or voting population |
| `geometry` | Polygon or MultiPolygon |

Example:

```
unit_id: "BOONE-:-FLORA 1-(CONG-11)"
dem_votes: 263
rep_votes: 240
weight: 503
geometry: Polygon(...)
```

These datasets are stored under:

```
raw-data/<state>_precincts/
>>>>>>> clean-deploy
```

---

# 2. Map Pack Construction

The **map pack** converts raw shapefiles into a format optimized for algorithms.

Run:

```
python scripts/build_map_pack.py --config config.yaml --state il
```

This produces:

```
assets/<state>_precincts/
│
├── attributes.csv
├── adjacency.json
├── shapes.geojson
├── id_to_idx.json
```

## attributes.csv

Contains algorithm inputs:

| column | meaning |
|------|---------|
| unit_id | precinct identifier |
| weight | population |
| dem_votes | democratic votes |
| rep_votes | republican votes |
| centroid_x | geometry centroid |
| centroid_y | geometry centroid |

Example:

```
unit_id,weight,dem_votes,rep_votes,centroid_x,centroid_y
BOONE-:-FLORA 1-(CONG-11),503,263,240,-88.85,42.24
```

---

## adjacency.json

Precinct adjacency graph:

```
{
  "precinct_id": ["neighbor1","neighbor2","neighbor3"]
}
```

Used to enforce **contiguity constraints**.

---

## shapes.geojson

Contains the original precinct geometries used for visualization.

---

## id_to_idx.json

Maps precinct IDs to numeric indices for fast NumPy operations.

Example:

```
{
  "BOONE-:-FLORA 1-(CONG-11)": 0,
  "COOK-:-7700013-(CONG-11)": 1
}
```

---

# 3. Running Algorithms

Once the map pack exists, algorithms can be executed.

## K-Means Baseline

```
python scripts/run_kmeans_pack.py --config config.yaml --state il
```

Generates a population-balanced clustering using a soft capacity constraint.

---

## Hillclimb Optimization

```
python scripts/run_hillclimber.py --config config.yaml --state il --party dem
```

The hillclimber:

- starts from a K-means map
- performs boundary swaps
- maximizes the number of seats for a target party
- preserves:
  - population balance
  - district contiguity
  - seat stability constraints

Key features:

- tabu memory to prevent oscillations
- narrow-win district locking
- swap-assist moves for near ties
- cycle detection

---

# 4. Exporting Results

Each algorithm run creates an output folder:

```
apps/web/public/outputs/<state>/<run_name>/
```

Example:

```
outputs/il/hillclimb_dem_20260301_190700/
```

Contents:

```
map_data.geojson
districts.geojson
district_stats.json
unit_to_district.csv
flipbook/
```

---

## map_data.geojson

Precinct-level GeoJSON including district assignments.

Example properties:

```
{
 "unit_id": "...",
 "district": 10,
 "dem_votes": 263,
 "rep_votes": 240,
 "weight": 503,
 "district_dem": 143664,
 "district_rep": 179018,
 "district_weight": 322682,
 "district_winner": "GOP",
 "district_margin": -35354,
 "district_margin_pct": -10.95
}
```

This file drives the **frontend map rendering**.

---

## districts.geojson

District boundaries created by dissolving precincts.

Used to render bold district outlines.

---

## district_stats.json

Aggregated statistics per district:

```
{
  "district": 5,
  "dem_votes": 312000,
  "rep_votes": 280000,
  "weight": 592000,
  "winner": "Dem",
  "margin": 32000,
  "margin_pct": 5.4
}
```

---

## unit_to_district.csv

Mapping between precincts and districts:

```
unit_id,district
BOONE-:-FLORA 1-(CONG-11),10
...
```

Useful for downstream analysis or exporting to other tools.

---

# 5. Flipbook Generation

The hillclimber optionally records frames of the algorithm.

Frames are generated using:

```
FrameRecorder
```

Output:

```
flipbook/
│
├── frames/
│   ├── frame_000000.png
│   ├── frame_000005.png
│   ├── frame_000010.png
│
└── manifest.json
```

---

## manifest.json

Example:

```
{
  "state": "il",
  "title": "Hillclimb (DEM)",
  "fps": 12,
  "frame_every": 5,
  "frames": [...]
}
```

Used by the frontend flipbook viewer.

---

# Frontend

The frontend is a **Next.js application**.

Location:

```
apps/web
```

It uses:

- React
- Leaflet
- GeoJSON rendering

---

# Frontend Pages

### Map Page

Displays district maps with:

- rainbow or party coloring
- district hover tooltips
- vote totals
- district statistics

Users can:

- switch between algorithm runs
- toggle outlines
- inspect district-level results

---

### Redistricting In Action

Displays algorithm progression:

- frame slider
- play/pause animation
- step metadata
- explanatory text

Frames are loaded from:

```
/outputs/<state>/<run>/flipbook
```

---

### About Page

Explains:

- algorithm design
- redistricting concepts
- fairness metrics
- project goals

---

# Adding New Algorithms

To integrate a new algorithm:

1. Implement in:

```
gerry/algos/
```

2. Create a script:

```
scripts/run_<algorithm>.py
```

3. Export results using:

```
export_run()
```

which writes:

```
map_data.geojson
district_stats.json
districts.geojson
```

4. Update:

```
outputs/<state>/latest.json
```

so the frontend discovers the run.

---

# Example Workflow

```
# Build map pack
python scripts/build_map_pack.py --state il

# Run kmeans baseline
python scripts/run_kmeans_pack.py --state il

# Run hillclimb optimizer
python scripts/run_hillclimber.py --state il --party dem

# Start frontend
cd apps/web
npm run dev
```

---

# Key Concepts

### Contiguity

Ensured using adjacency graph checks during moves.

### Population Balance

Each district must remain within a tolerance of:

```
ideal_population = total_population / num_districts
```

### Seat Optimization

Objective function:

```
seat_weight * seats
+ flip_weight * closest_loss_margin
+ loss_weight * total_losing_margin
```

This encourages:

- maximizing seats
- flipping near losses
- improving margins

---

# Deployment

The frontend can be deployed to **Vercel**.

Important:

Only a subset of runs should be included in:

```
apps/web/public/outputs/
```

to avoid large repository sizes.

Typically include:

```
current_congress
kmeans
hillclimb runs
flipbooks
```

---

# Future Improvements

Potential extensions:

- fairness metrics
- compactness scores
- efficiency gap
- simulated annealing optimizer
- reinforcement learning district generator
- server-side map streaming
- WebGL rendering for large states

---

# License

MIT License.

---

# Author

Robert Lennon  
Brown University — Computer Science
