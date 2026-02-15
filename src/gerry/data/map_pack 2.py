from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
import geopandas as gpd

@dataclass
class MapPack:
    pack_dir: Path
    ids: list[str]
    id_to_idx: dict[str, int]
    idx_to_id: dict[int, str]
    dem: np.ndarray
    rep: np.ndarray
    weight: np.ndarray
    coords: np.ndarray  # (N,2)
    adj_ids: dict[str, list[str]]
    adj: list[list[int]]  # neighbors as indices
    shapes: gpd.GeoDataFrame  # unit_id + geometry

def load_map_pack(pack_dir: str | Path) -> MapPack:
    pack_dir = Path(pack_dir)

    attrs = pd.read_csv(pack_dir / "attributes.csv")
    attrs["unit_id"] = attrs["unit_id"].astype(str)

    id_to_idx = json.loads((pack_dir / "id_to_idx.json").read_text())
    idx_to_id_raw = json.loads((pack_dir / "idx_to_id.json").read_text())
    idx_to_id = {int(k): v for k, v in idx_to_id_raw.items()}

    adj_ids = json.loads((pack_dir / "adjacency.json").read_text())

    # Ensure attrs order matches id_to_idx order
    ids = [None] * len(id_to_idx)
    for uid, i in id_to_idx.items():
        ids[i] = uid

    attrs = attrs.set_index("unit_id").loc[ids].reset_index()

    dem = attrs["dem_votes"].to_numpy(dtype=float)
    rep = attrs["rep_votes"].to_numpy(dtype=float)
    weight = attrs["weight"].to_numpy(dtype=float)
    coords = attrs[["centroid_x","centroid_y"]].to_numpy(dtype=float)

    # Build adjacency list in idx space
    adj = [[] for _ in range(len(ids))]
    for uid, nbrs in adj_ids.items():
        i = id_to_idx[uid]
        adj[i] = [id_to_idx[n] for n in nbrs if n in id_to_idx]

    shapes = gpd.read_file(pack_dir / "shapes.geojson")
    shapes["unit_id"] = shapes["unit_id"].astype(str)
    shapes = shapes.set_index("unit_id").loc[ids].reset_index()

    return MapPack(
        pack_dir=pack_dir,
        ids=ids,
        id_to_idx=id_to_idx,
        idx_to_id=idx_to_id,
        dem=dem,
        rep=rep,
        weight=weight,
        coords=coords,
        adj_ids=adj_ids,
        adj=adj,
        shapes=shapes,
    )
