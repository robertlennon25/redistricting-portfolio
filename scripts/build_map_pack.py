from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime

import geopandas as gpd

def compute_votes_gcon(gdf):
    mapping = {
        '1st': ('GCON01DJAC','GCON01RCAR'),'2nd': ('GCON02DKEL','GCON02RLYN'),
        '3rd': ('GCON03DRAM','GCON03RBUR'),'4th': ('GCON04DGAR','GCON04RFAL'),
        '5th': ('GCON05DQUI','GCON05RHAN'),'6th': ('GCON06DCAS','GCON06RPEK'),
        '7th': ('GCON07DDAV','GCON07OWRI'),'8th': ('GCON08DKRI','GCON08RDAR'),
        '9th': ('GCON09DSCH','GCON09RRIC'),'10th': ('GCON10DSCH','GCON10RSEV'),
        '11th': ('GCON11DFOS','GCON11RLAU'),'12th': ('GCON12DMAR','GCON12RBOS'),
        '13th': ('GCON13DBUD','GCON13RDEE'),'14th': ('GCON14DUND','GCON14RGRY'),
        '15th': ('GCON15DLAN','GCON15RMIL'),'16th': ('GCON16DHAD','GCON16RLAH'),
        '17th': ('GCON17DSOR','GCON17RKIN')
    }
    gdf = gdf.copy()
    gdf["dem_votes"] = 0
    gdf["rep_votes"] = 0
    for dv, rv in mapping.values():
        gdf["dem_votes"] += gdf[dv].fillna(0).astype(float)
        gdf["rep_votes"] += gdf[rv].fillna(0).astype(float)
    gdf["weight"] = (gdf["dem_votes"] + gdf["rep_votes"]).astype(float)
   # turnout proxy
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
    out_dir = Path(cfg["output"]["assets_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(shp)
    gdf = gdf.to_crs(epsg=epsg)
    gdf["geometry"] = gdf["geometry"].buffer(0)
    

    if unit_id_col not in gdf.columns:
        raise ValueError(f"unit_id_col='{unit_id_col}' not found. Available columns: {list(gdf.columns)[:50]} ...")

    gdf[unit_id_col] = gdf[unit_id_col].astype(str)

    # votes + weight
    gdf = compute_votes_gcon(gdf)

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
