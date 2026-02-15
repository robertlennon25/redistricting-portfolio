## Overview

This project uses Illinois precinct-level voter data to generate districting figures using a variety of algorithms, including greedy packing and constrained k-means clustering. It demonstrates different approaches to partitioning the state into 17 congressional districts while respecting population balance, contiguity, and partisan objectives.

## Dependencies

* Python 3.7+
* geopandas
* networkx
* pandas
* numpy
* matplotlib
* shapely
* **k_means_constrained** (used in `kmeans_constrained_package.py`)



## Data

The scripts expect an Illinois 2022 General Election precinct-level shapefile. Set the `SHAPEFILE_PATH` variable at the top of each script to point to the local `.shp` file.

## Usage

1. Update `SHAPEFILE_PATH` in each script to your local precinct shapefile path.
2. Install dependencies, e.g.:

   ```bash
   pip install geopandas networkx pandas numpy matplotlib shapely k_means_constrained
   ```

3. Run the scripts:

   ```bash
   python greedy_packing.py --maximize dem
   python kmeans_constrained_package.py
   python kmeans_constrained_hard_pop.py
   python kmeans_constrained_soft.py
   python kmeans_constrained_hard_pop_contiguity_compact.py
   python linreg_voters.py
   python plot_precincts.py
   ```

4. Outputs (shapefiles, CSVs, figures) will be saved in each script’s designated output directory.

## File Descriptions

### `greedy_packing.py`

Implements a greedy, seed-and-grow districting algorithm:

* **compute_votes**: aggregates Democratic and Republican vote totals and precinct population
* **build_adjacency**: constructs a graph of precinct adjacencies
* **greedy_assign**: seeds districts with high vote share precincts, grows outward to fill
* **enforce_contiguity**: fixes disjointed district pieces
* **balance_population**: swaps precincts to meet ±5% ideal population
* **evaluate_districts**: reports vote totals and seat counts
* **Output**: saves shapefile `submission_figures/output_greedy_packing2_vX.shp` and corresponding map PNG/CSV

### `kmeans_constrained_package.py`

Uses the `k_means_constrained` library to enforce size constraints on clusters:

* **expand_by_population**: upscales representation of populous precincts
* **KMeansConstrained**: applies constrained k-means
* Maps clustered labels back to original precincts
* **Output**: saves to `figures_sectionv5/kmeans_constrained_districts.csv` and a PNG map

### `kmeans_constrained_hard_pop.py`

Custom constrained k-means with hard population cap:

* Penalizes overpopulated districts with a quadratic term
* Assigns each precinct to the lowest-score valid cluster
* Updates centers each iteration
* **Output**: PNG map and `kmeans_constrained_hard_pop/district_sizes_vX.csv`

### `kmeans_constrained_soft.py`

Custom constrained k-means with soft penalties and contiguity:

* Runs multiple restarts of randomized seeds
* **run_seed**: assigns based on distance, contiguity, and population reward
* Removes enclave fragments after assignment
* **Outputs**: 
  * `constrained_kmeans_districts.csv`
  * Text summary of seat outcomes and margins
  * PNG map with annotations

### `kmeans_constrained_hard_pop_contiguity_compact.py`

Adds compactness scoring to hard population/contiguity-constrained k-means:

* Scoring function = distance + contiguity bonus + population penalty
* Computes square-space and Polsby-Popper compactness scores
* **Output**: PNG map and CSV with compactness/district sizes in `kmeans_constrained_hard_pop/`

### `linreg_voters.py`

Performs linear regression to estimate presidential vote share based on local variables:

* **merge_vote_data**: joins 2020 presidential data with the precinct shapefile
* **run_regression**: computes linear fit for Dem % vs. covariates (e.g., area, population)
* **Output**: regression diagnostics and scatter plots saved to `submission_figures/output_linreg_voters_vX.png`/`.csv`

### `plot_precincts.py`

Creates visualizations of precinct-level and district-level vote outcomes:

* **plot_votes_by_district**: shows Democratic margin by district
* **plot_precincts_with_districts**: shades precincts by district assignment
* **Output**: saves PNG figures to `submission_figures/output_plot_precincts_vX.png`

## Outputs

* **Shapefiles**: e.g., `output_greedy_packing2_vX.shp`
* **CSV files**: district assignments and diagnostics (e.g., `district_sizes_vX.csv`)
* **Figures**: PNG maps in `submission_figures/`, `figures_sectionv5/`, or `kmeans_constrained_hard_pop/`
* **Text summaries**: seat counts and margins for soft-constrained k-means runs

### Additional Visualization Scripts (not part of K-Means)

#### `plot_non_house_votes.py`
- Creates visualizations for non-congressional races or voting patterns.
- **Output**: saves visualizations as PNG files to designated visualization directories.

#### `plot_precincts.py`
- Generates visualizations to show precinct-level voting distributions.
- **Output**: saves visualizations as PNG files for detailed precinct analysis.


This project utilizes the k-means-constrained Python package developed by Josh Levy:
@software{Levy-Kramer_k-means-constrained_2018,
  author = {Levy-Kramer, Josh},
  month = apr,
  title = {{k-means-constrained}},
  url = {https://github.com/joshlk/k-means-constrained},
  year = {2018}
}

#### NEW Stuff

Created virtual environment called gerryenv

need to change filepaths, have been changed in:
- kmeans_constrained_hard_pop_contiguity.py
  - still does not adhere to perfect contiguity constraints
  - has issue with ogr - "contains polygon(s) with rings with invalid winding order. Autocorrecting them, but that shapefile should be corrected using ogr2ogr for example.
  return ogr_read("
  - needs error messages or some sort of input shape for the files that are chosen 


## Project outline moving forward: ##
- Get data for other states
- Figure out ogr2ogr issue
- Reorganize state data into well-named folders
- delete unnecessary data files/folders
- run the currently existing code on the other state data
- Create a frontend with options for
  - state
  - data used (which election year)
  - option to introduce randomness to votes
- create model to introduce randomness to populations 
  - allow users to put their own data into the code? could create issues
- fix models on current data
  - ensure contiguity 
  - add compactness score/legality score
  - add some information about the courts in each state
  


