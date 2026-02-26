from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime

import geopandas as gpd

import re

def compute_votes_dynamic(
    gdf: gpd.GeoDataFrame,
    contest_prefix: str = "GCON",   # change if your file uses different contest code
    dem_party: str = "D",
    rep_party: str = "R",
    weight_col: str = "TOTPOP"
) -> gpd.GeoDataFrame:
    """
    Auto-detect and sum vote columns by contest prefix and party letter.

    Expected column shapes:
      - GCON01Dxxxx, GCON01Rxxxx, ... (case-insensitive)

    If your 2024 file uses a different prefix (e.g., 'CONG', 'USCONG'), pass contest_prefix.
    """
    cols = list(gdf.columns)

    # Example: ^GCON\d{2}D  -> all Dem columns for 01..17 districts
    dem_re = re.compile(rf"^{re.escape(contest_prefix)}\d{{2}}{re.escape(dem_party)}", re.IGNORECASE)
    rep_re = re.compile(rf"^{re.escape(contest_prefix)}\d{{2}}{re.escape(rep_party)}", re.IGNORECASE)

    dem_cols = [c for c in cols if dem_re.match(c)]
    rep_cols = [c for c in cols if rep_re.match(c)]

    if not dem_cols or not rep_cols:
        preview = cols[:120]
        raise KeyError(
            f"Could not find vote columns for prefix='{contest_prefix}' with party letters "
            f"'{dem_party}'/'{rep_party}'.\n"
            f"Found dem_cols={dem_cols}\nFound rep_cols={rep_cols}\n"
            f"First columns: {preview}\n\n"
            "Fix: change contest_prefix in config (e.g., GCON/CONG/USCONG) or adjust regex."
        )

    gdf = gdf.copy()
    gdf["dem_votes"] = 0.0
    gdf["rep_votes"] = 0.0

    for c in dem_cols:
        gdf["dem_votes"] += gdf[c].fillna(0).astype(float)
    for c in rep_cols:
        gdf["rep_votes"] += gdf[c].fillna(0).astype(float)

    # edited 2/26/26 to change to total population from precinct_join_population
    # ----------------------------
    # NEW: weight from population
    # ----------------------------
    if weight_col not in gdf.columns:
        # Helpful error with suggestions
        candidates = [c for c in cols if "POP" in c.upper() or c.upper().startswith("P00")]
        raise KeyError(
            f"weight_col='{weight_col}' not found in precinct file.\n"
            f"Did you run the block→precinct population join and save it into the precinct layer?\n"
            f"Available columns with likely population signals: {candidates[:50]}\n"
            f"First columns: {cols[:80]}"
        )

    gdf["weight"] = gdf[weight_col].fillna(0).astype(float)

    # Helpful logging
    print(f"✅ Vote columns detected for prefix '{contest_prefix}':")
    print(f"  Dem cols ({len(dem_cols)}): {dem_cols[:8]}{' ...' if len(dem_cols) > 8 else ''}")
    print(f"  Rep cols ({len(rep_cols)}): {rep_cols[:8]}{' ...' if len(rep_cols) > 8 else ''}")

    return gdf


def build_adjacency_by_id(gdf: gpd.GeoDataFrame, unit_id_col: str) -> dict[str, list[str]]:
    gdf = gdf[[unit_id_col, "geometry"]].copy()
    gdf[unit_id_col] = gdf[unit_id_col].astype(str)
    gdf = gdf.reset_index(drop=True)

    sindex = gdf.sindex
    neighbors = {uid: [] for uid in gdf[unit_id_col].tolist()}

    for i, geom in enumerate(gdf.geometry):
        if i % 500 == 0:
            print(f"Adjacency: {i}/{len(gdf)}")
        uid_i = gdf.at[i, unit_id_col]
        for j in sindex.intersection(geom.bounds):
            if i >= j:
                continue
            if geom.touches(gdf.geometry[j]):
                uid_j = gdf.at[j, unit_id_col]
                neighbors[uid_i].append(uid_j)
                neighbors[uid_j].append(uid_i)

    return {k: sorted(set(v)) for k, v in neighbors.items()}


def main():
    import yaml, argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))

    shp = Path(cfg["data"]["precinct_shapefile_path"])
    unit_id_col = cfg["data"]["unit_id_col"]
    epsg = int(cfg["data"].get("crs_epsg", 3857))
    # --- Resolve assets_dir (support multiple config schemas) ---
    assets_dir_raw = (
        cfg.get("paths", {}).get("assets_dir")          # new schema
        or cfg.get("output", {}).get("assets_dir")      # old schema
        or cfg.get("assets_dir")                        # legacy flat key
    )
    if not assets_dir_raw:
        raise KeyError(
            "Config missing assets_dir. Provide one of:\n"
            "  paths.assets_dir: 'assets/il2024_precincts'\n"
            "  output.assets_dir: 'assets/il2024_precincts'\n"
            "  assets_dir: 'assets/il2024_precincts'"
        )

    out_dir = Path(assets_dir_raw).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_dir.mkdir(parents=True, exist_ok=True)

    # print("OUT_DIR raw:", cfg["output"]["assets_dir"])
    # print("OUT_DIR resolved:", out_dir.resolve())


    layer = cfg.get("data", {}).get("precinct_layer")
    if layer:
        print(f"Reading GPKG layer: {layer}")
        gdf = gpd.read_file(shp, layer=layer)
    else:
        gdf = gpd.read_file(shp)
    gdf = gdf.to_crs(epsg=epsg)
    gdf["geometry"] = gdf["geometry"].buffer(0)
    

    if unit_id_col not in gdf.columns:
        raise ValueError(f"unit_id_col='{unit_id_col}' not found. Available columns: {list(gdf.columns)[:50]} ...")

    gdf[unit_id_col] = gdf[unit_id_col].astype(str)

    # votes + weight
    gdf = compute_votes_dynamic(gdf)

    # centroids
    centroids = gdf.geometry.centroid
    gdf["centroid_x"] = centroids.x
    gdf["centroid_y"] = centroids.y

    # adjacency (static)
    adjacency = build_adjacency_by_id(gdf, unit_id_col=unit_id_col)

    # id <-> idx mapping (static)
    ids = gdf[unit_id_col].tolist()
    id_to_idx = {uid: i for i, uid in enumerate(ids)}
    idx_to_id = {i: uid for uid, i in id_to_idx.items()}

    # save shapes (static)
    shapes = gdf[[unit_id_col, "geometry"]].rename(columns={unit_id_col: "unit_id"})
    shapes.to_file(out_dir / "shapes.geojson", driver="GeoJSON")

    # save attributes (static)
    attrs = gdf[[unit_id_col, "dem_votes", "rep_votes", "weight", "centroid_x", "centroid_y"]].rename(columns={unit_id_col: "unit_id"})
    attrs.to_csv(out_dir / "attributes.csv", index=False)

    # save adjacency + mappings
    (out_dir / "adjacency.json").write_text(json.dumps(adjacency))
    (out_dir / "id_to_idx.json").write_text(json.dumps(id_to_idx))
    (out_dir / "idx_to_id.json").write_text(json.dumps(idx_to_id))

    meta = {
        "built_at": datetime.now().isoformat(),
        "source_shapefile": str(shp),
        "unit_id_col": unit_id_col,
        "epsg": epsg,
        "n_units": len(gdf),
        "note": "weight = dem_votes + rep_votes (two-party vote proxy, not census population)."
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"✅ Built map pack at: {out_dir}")
    print(f"Units: {len(gdf)} | adjacency keys: {len(adjacency)} | shapes: shapes.geojson")

if __name__ == "__main__":
    main()
