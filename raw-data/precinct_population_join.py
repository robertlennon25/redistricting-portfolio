from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--state", required=True, help="State key in config.yaml under states: (e.g., il, wi)")
    ap.add_argument(
        "--method",
        choices=["centroid_within", "intersects"],
        default="centroid_within",
        help="centroid_within is fast; intersects is more robust but may overcount without area-weighting.",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    states = cfg.get("states", {}) or {}
    if args.state not in states:
        raise KeyError(f"State '{args.state}' not found under config['states'].")

    scfg = states[args.state]

    precinct_path = Path(scfg["precinct_path"]).expanduser()
    blocks_path = Path(scfg["blocks_path"]).expanduser()
    out_gpkg = Path(scfg["out_gpkg"]).expanduser()
    out_layer = scfg.get("out_layer")
    if not out_layer:
        raise KeyError(f"states.{args.state}.out_layer is required (you said you want layers).")

    pop_col = scfg.get("pop_col", "P0010001")
    precinct_id_col = scfg.get("precinct_id_col", "UNIQUE_ID")
    target_epsg = int(scfg.get("target_epsg", 3857))

    print(f"[{args.state}] Loading precincts: {precinct_path}")
    precincts = gpd.read_file(precinct_path)

    print(f"[{args.state}] Loading blocks: {blocks_path}")
    blocks = gpd.read_file(blocks_path)

    if pop_col not in blocks.columns:
        raise ValueError(f"[{args.state}] pop_col '{pop_col}' not found in blocks. Columns: {list(blocks.columns)[:80]}")
    if precinct_id_col not in precincts.columns:
        raise ValueError(f"[{args.state}] precinct_id_col '{precinct_id_col}' not found in precincts. Columns: {list(precincts.columns)[:80]}")

    blocks[pop_col] = pd.to_numeric(blocks[pop_col], errors="coerce").fillna(0).astype(float)
    precincts[precinct_id_col] = precincts[precinct_id_col].astype(str)

    if precincts.crs is None:
        raise ValueError(f"[{args.state}] Precinct CRS is None. Check the source file (.prj).")
    if blocks.crs is None:
        raise ValueError(f"[{args.state}] Blocks CRS is None. Check the source file (.prj).")

    # Project both to planar CRS for stable spatial operations
    precincts = precincts.to_crs(epsg=target_epsg)
    blocks = blocks.to_crs(epsg=target_epsg)

    # Clean geometries
    precincts["geometry"] = precincts.geometry.buffer(0)
    blocks["geometry"] = blocks.geometry.buffer(0)

    if args.method == "centroid_within":
        blocks_use = blocks[[pop_col, "geometry"]].copy()
        blocks_use["geometry"] = blocks_use.geometry.centroid
        print(f"[{args.state}] Spatial join: block CENTROIDS within precincts ...")
        joined = gpd.sjoin(
            blocks_use,
            precincts[[precinct_id_col, "geometry"]],
            how="inner",
            predicate="within",
        )
    else:
        print(f"[{args.state}] Spatial join: blocks intersect precincts ...")
        joined = gpd.sjoin(
            blocks[[pop_col, "geometry"]],
            precincts[[precinct_id_col, "geometry"]],
            how="inner",
            predicate="intersects",
        )

    print(f"[{args.state}] Aggregating {pop_col} -> TOTPOP by precinct...")
    pop_by_precinct = (
        joined.groupby(precinct_id_col)[pop_col].sum().reset_index().rename(columns={pop_col: "TOTPOP"})
    )

    print(f"[{args.state}] Merging back onto precincts...")
    out = precincts.merge(pop_by_precinct, on=precinct_id_col, how="left")
    out["TOTPOP"] = out["TOTPOP"].fillna(0).astype(float)

    print(f"[{args.state}] Writing GPKG: {out_gpkg} | layer='{out_layer}'")
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: write to a specific layer name
    out.to_file(out_gpkg, layer=out_layer, driver="GPKG")

    print(f"[{args.state}] Done. TOTPOP describe():")
    print(out["TOTPOP"].describe())


if __name__ == "__main__":
    main()