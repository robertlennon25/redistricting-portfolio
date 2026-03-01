from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import geopandas as gpd


def compute_votes_dynamic(
    gdf: gpd.GeoDataFrame,
    contest_prefix: str = "GCON",
    dem_party: str = "D",
    rep_party: str = "R",
    weight_col: str = "TOTPOP",
) -> gpd.GeoDataFrame:
    """
    Supports two common schemas:

    (A) IL-style with district numbers:
        GCON01D..., GCON01R..., GCON02D..., ...

    (B) Single contest code + trailing party letter:
        G24COND, G24CONR, (optionally G24CONO), etc.
    """
    cols = list(gdf.columns)

    # ---- Try schema A: prefix + 2 digits + party ----
    dem_re_A = re.compile(rf"^{re.escape(contest_prefix)}\d{{2}}{re.escape(dem_party)}", re.IGNORECASE)
    rep_re_A = re.compile(rf"^{re.escape(contest_prefix)}\d{{2}}{re.escape(rep_party)}", re.IGNORECASE)

    dem_cols = [c for c in cols if dem_re_A.match(c)]
    rep_cols = [c for c in cols if rep_re_A.match(c)]

    # ---- If none found, try schema B: exact contest column names ----
    if not dem_cols or not rep_cols:
        # Most common: exact names like "G24COND" and "G24CONR"
        dem_exact = [c for c in cols if c.upper() == f"{contest_prefix}{dem_party}".upper()]
        rep_exact = [c for c in cols if c.upper() == f"{contest_prefix}{rep_party}".upper()]

        # Slightly more flexible fallback: startswith contest_prefix and endswith party letter
        if not dem_exact:
            dem_re_B = re.compile(rf"^{re.escape(contest_prefix)}.*{re.escape(dem_party)}$", re.IGNORECASE)
            dem_exact = [c for c in cols if dem_re_B.match(c)]
        if not rep_exact:
            rep_re_B = re.compile(rf"^{re.escape(contest_prefix)}.*{re.escape(rep_party)}$", re.IGNORECASE)
            rep_exact = [c for c in cols if rep_re_B.match(c)]

        # If we found B-style columns, use them
        if dem_exact and rep_exact:
            dem_cols = dem_exact
            rep_cols = rep_exact

    if not dem_cols or not rep_cols:
        preview = cols[:120]
        raise KeyError(
            f"Could not find vote columns for prefix='{contest_prefix}' with party letters "
            f"'{dem_party}'/'{rep_party}'.\n"
            f"Found dem_cols={dem_cols}\nFound rep_cols={rep_cols}\n"
            f"First columns: {preview}\n\n"
            "Fix options:\n"
            "  1) Set votes.contest_prefix correctly for this state (e.g., 'G24CON').\n"
            "  2) Or adjust compute_votes_dynamic() patterns if your dataset uses a different schema."
        )

    gdf = gdf.copy()
    gdf["dem_votes"] = 0.0
    gdf["rep_votes"] = 0.0

    for c in dem_cols:
        gdf["dem_votes"] += gdf[c].fillna(0).astype(float)
    for c in rep_cols:
        gdf["rep_votes"] += gdf[c].fillna(0).astype(float)

    # Weight from population
    if weight_col not in gdf.columns:
        candidates = [c for c in cols if "POP" in c.upper() or c.upper().startswith("P00")]
        raise KeyError(
            f"weight_col='{weight_col}' not found in precinct file.\n"
            f"Did you run the block→precinct population join and save it into the precinct layer?\n"
            f"Available columns with likely population signals: {candidates[:50]}\n"
            f"First columns: {cols[:80]}"
        )

    gdf["weight"] = gdf[weight_col].fillna(0).astype(float)

    print(f"✅ Vote columns detected using contest_prefix='{contest_prefix}':")
    print(f"  Dem cols ({len(dem_cols)}): {dem_cols}")
    print(f"  Rep cols ({len(rep_cols)}): {rep_cols}")

    return gdf


def build_adjacency_by_id(
    gdf: gpd.GeoDataFrame,
    unit_id_col: str,
    eps: float = 1.0,
    use_boundary: bool = True,
) -> dict[str, list[str]]:
    """
    Build adjacency using a tolerant geometric test.
    """
    gdf = gdf[[unit_id_col, "geometry"]].copy()
    gdf[unit_id_col] = gdf[unit_id_col].astype(str)
    gdf = gdf.reset_index(drop=True)

    gdf["geometry"] = gdf["geometry"].buffer(0)

    sindex = gdf.sindex
    ids = gdf[unit_id_col].tolist()
    geoms = gdf.geometry.values

    neighbors: dict[str, set[str]] = {uid: set() for uid in ids}

    for i, geom_i in enumerate(geoms):
        if i % 500 == 0:
            print(f"Adjacency: {i}/{len(geoms)}")

        if geom_i is None or geom_i.is_empty:
            continue

        uid_i = ids[i]
        cand_idx = list(sindex.intersection(geom_i.bounds))

        for j in cand_idx:
            j = int(j)
            if i >= j:
                continue

            geom_j = geoms[j]
            if geom_j is None or geom_j.is_empty:
                continue

            try:
                if use_boundary:
                    ok = geom_i.boundary.buffer(eps).intersects(geom_j.boundary.buffer(eps))
                else:
                    ok = geom_i.buffer(eps).intersects(geom_j.buffer(eps))
            except Exception:
                ok = False

            if ok:
                uid_j = ids[j]
                neighbors[uid_i].add(uid_j)
                neighbors[uid_j].add(uid_i)

    return {k: sorted(v) for k, v in neighbors.items()}


def _apply_state_override(cfg: dict, state: str) -> dict:
    """
    If cfg has a states.<state> entry, override data/paths for this run.
    Does NOT write back to config.yaml; only modifies the in-memory cfg.
    """
    scfg = (cfg.get("states", {}) or {}).get(state)
    if not scfg:
        raise KeyError(f"State '{state}' not found under cfg['states'].")

    cfg = dict(cfg)  # shallow copy
    cfg.setdefault("data", {})
    cfg.setdefault("paths", {})

    # precinct input should be the "with_pop" output of your join script
    cfg["data"]["precinct_shapefile_path"] = scfg["out_gpkg"]
    if scfg.get("out_layer"):
        cfg["data"]["precinct_layer"] = scfg["out_layer"]

    cfg["data"]["unit_id_col"] = scfg.get("precinct_id_col", cfg["data"].get("unit_id_col", "UNIQUE_ID"))
    cfg["data"]["crs_epsg"] = int(scfg.get("target_epsg", cfg["data"].get("crs_epsg", 3857)))
    cfg["data"]["weight_col"] = scfg.get("weight_col", "TOTPOP")

    # where to write the pack
    if scfg.get("assets_dir"):
        cfg["paths"]["assets_dir"] = scfg["assets_dir"]

    # vote prefix / party letters can also be overridden per state if you want
    # vote prefix / party letters can be overridden per state
    svotes = (scfg.get("votes", {}) or {})

    cfg.setdefault("votes", {})

    if svotes.get("contest_prefix"):
        cfg["votes"]["contest_prefix"] = svotes["contest_prefix"]
    if svotes.get("dem_party_letter"):
        cfg["votes"]["dem_party_letter"] = svotes["dem_party_letter"]
    if svotes.get("rep_party_letter"):
        cfg["votes"]["rep_party_letter"] = svotes["rep_party_letter"]

        return cfg


def main():
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--state", default=None, help="Override to build map pack for a configured state key")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))

    # optional override (does not modify config.yaml)
    if args.state:
        cfg = _apply_state_override(cfg, args.state)
        print(f"✅ build_map_pack using state='{args.state}'")

    # Read config fields (either from base config or overridden)
    shp = Path(cfg["data"]["precinct_shapefile_path"]).expanduser()
    unit_id_col = cfg["data"]["unit_id_col"]
    epsg = int(cfg["data"].get("crs_epsg", 3857))
    weight_col = cfg["data"].get("weight_col", "TOTPOP")

    # votes config
    votes_cfg = cfg.get("votes", {}) or {}
    contest_prefix = votes_cfg.get("contest_prefix", cfg.get("data", {}).get("contest_prefix", "GCON"))
    dem_party = votes_cfg.get("dem_party_letter", "D")
    rep_party = votes_cfg.get("rep_party_letter", "R")

    # Resolve assets_dir (support multiple config schemas)
    assets_dir_raw = (
        cfg.get("paths", {}).get("assets_dir")
        or cfg.get("output", {}).get("assets_dir")
        or cfg.get("assets_dir")
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

    # Read precinct layer if provided (GPKG)
    layer = cfg.get("data", {}).get("precinct_layer")
    if layer:
        print(f"Reading GPKG layer: {layer}")
        gdf = gpd.read_file(shp, layer=layer)
    else:
        gdf = gpd.read_file(shp)

    gdf = gdf.to_crs(epsg=epsg)
    gdf["geometry"] = gdf["geometry"].buffer(0)

    if unit_id_col not in gdf.columns:
        raise ValueError(
            f"unit_id_col='{unit_id_col}' not found. Available columns: {list(gdf.columns)[:50]} ..."
        )

    gdf[unit_id_col] = gdf[unit_id_col].astype(str)

    # votes + weight
    gdf = compute_votes_dynamic(
        gdf,
        contest_prefix=contest_prefix,
        dem_party=dem_party,
        rep_party=rep_party,
        weight_col=weight_col,
    )

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

        # extra attributes to carry through into attributes.csv
    extra_attr_cols = cfg.get("data", {}).get("extra_attr_cols", []) or []

    # allow per-state override if you want it later
    # (only works if you add states.<state>.extra_attr_cols in config)
    if args.state:
        scfg = (cfg.get("states", {}) or {}).get(args.state, {}) or {}
        sdata = (scfg.get("data", {}) or {})
        extra_attr_cols = sdata.get("extra_attr_cols", extra_attr_cols) or []

    # keep only extras that actually exist
    extra_attr_cols = [c for c in extra_attr_cols if c in gdf.columns]

    # save attributes (static)
    base_cols = [unit_id_col, "dem_votes", "rep_votes", "weight", "centroid_x", "centroid_y"]
    attrs_cols = base_cols + extra_attr_cols

    attrs = gdf[attrs_cols].rename(columns={unit_id_col: "unit_id"})
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
        "weight_col": weight_col,
        "note": "weight is derived from precinct population join (e.g., TOTPOP from PL94-171 blocks).",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"✅ Built map pack at: {out_dir}")
    print(f"Units: {len(gdf)} | adjacency keys: {len(adjacency)} | shapes: shapes.geojson")


if __name__ == "__main__":
    main()