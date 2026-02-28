import argparse
import json
import yaml
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from gerry.data.map_pack import load_map_pack
from export_run import export_run

from gerry.algos.kmeans_softcap import kmeans_softcap_labels
from gerry.algos.hillclimber import (
    HillclimbConfig,
    build_adj_idx,
    hillclimb_max_seats,
)
from gerry.algos.kmeans_postprocess_contig import enforce_contiguity_by_reattach

''' First iteration of hillclimber
Starts from teh kmeans map that is generated 
example usage from repo root: 

python3 scripts/run_hillclimber.py --config config.yaml --state ny --party rep

 '''


# ---------------------------------------------------------------------
# Helpers copied from kmeans runner (state-aware)
# ---------------------------------------------------------------------

def _resolve_pack_dir(cfg: dict, state: str | None) -> Path:
    if state:
        scfg = (cfg.get("states", {}) or {}).get(state)
        if not scfg:
            raise KeyError(f"State '{state}' not found under cfg['states'].")
        return Path(scfg["assets_dir"]).expanduser().resolve()

    paths = cfg.get("paths", {}) or {}
    data = cfg.get("data", {}) or {}
    output = cfg.get("output", {}) or {}

    pack_dir_raw = data.get("map_pack_dir") or paths.get("assets_dir") or output.get("assets_dir")
    if not pack_dir_raw:
        raise KeyError("No map pack directory found.")
    return Path(pack_dir_raw).expanduser().resolve()


def _resolve_outputs_root(cfg: dict) -> Path:
    paths = cfg.get("paths", {}) or {}
    public_outputs_raw = paths.get("public_outputs_dir")
    if public_outputs_raw:
        return Path(public_outputs_raw).expanduser().resolve()

    return Path("apps/web/public/outputs").resolve()


def _update_latest_manifest(state_outputs_root: Path, key: str, folder_name: str):
    manifest_path = state_outputs_root / "latest.json"
    if manifest_path.exists():
        latest = json.loads(manifest_path.read_text())
    else:
        latest = {}
    latest[key] = folder_name
    manifest_path.write_text(json.dumps(latest, indent=2))
    print(f"Updated {manifest_path}")


def _load_pack_arrays(pack_dir: Path, pack):
    # coords
    if hasattr(pack, "coords") and pack.coords is not None:
        coords = np.asarray(pack.coords)
    else:
        attrs = pd.read_csv(pack_dir / "attributes.csv")
        coords = attrs[["centroid_x", "centroid_y"]].values.astype(float)

    # weight
    if hasattr(pack, "weight") and pack.weight is not None:
        weight = np.asarray(pack.weight).astype(float)
    else:
        attrs = pd.read_csv(pack_dir / "attributes.csv")
        weight = attrs["weight"].astype(float).values

    # votes
    attrs = pd.read_csv(pack_dir / "attributes.csv")
    dem_votes = attrs["dem_votes"].astype(float).values
    rep_votes = attrs["rep_votes"].astype(float).values

    # unit_ids in correct order
    id_to_idx = json.loads((pack_dir / "id_to_idx.json").read_text())
    idx_to_id = {int(v): str(k) for k, v in id_to_idx.items()}
    unit_ids = [idx_to_id[i] for i in range(len(idx_to_id))]

    return unit_ids, coords, weight, dem_votes, rep_votes


# ---------------------------------------------------------------------
# Load starting plan from latest kmeans if exists
# ---------------------------------------------------------------------

def _load_latest_kmeans_labels(state_outputs_root: Path, pack_dir: Path) -> np.ndarray | None:
    manifest_path = state_outputs_root / "latest.json"
    if not manifest_path.exists():
        return None

    latest = json.loads(manifest_path.read_text())
    folder = latest.get("kmeans_softcap")
    if not folder:
        return None

    run_dir = state_outputs_root / folder
    map_data_path = run_dir / "map_data.geojson"
    if not map_data_path.exists():
        return None

    print(f"Using latest kmeans plan from {run_dir}")

    # Load labels from exported GeoJSON
    import geopandas as gpd
    gdf = gpd.read_file(map_data_path)
    if "district" not in gdf.columns:
        return None

    # Must align with pack ordering
    id_to_idx = json.loads((pack_dir / "id_to_idx.json").read_text())
    labels = np.zeros(len(id_to_idx), dtype=int)

    for _, row in gdf.iterrows():
        uid = str(row["unit_id"])
        idx = id_to_idx.get(uid)
        if idx is not None:
            labels[int(idx)] = int(row["district"])

    return labels


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--state", required=True)
    ap.add_argument("--party", choices=["dem", "rep"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))

    pack_dir = _resolve_pack_dir(cfg, args.state)
    outputs_root = _resolve_outputs_root(cfg)
    state_outputs_root = outputs_root / args.state
    state_outputs_root.mkdir(parents=True, exist_ok=True)

    pack = load_map_pack(pack_dir)
    unit_ids, coords, weight, dem_votes, rep_votes = _load_pack_arrays(pack_dir, pack)

    num_districts = int(
        (cfg.get("states", {}) or {}).get(args.state, {}).get(
            "num_districts",
            (cfg.get("run", {}) or {}).get("num_districts", 17),
        )
    )

    # 1) Try loading latest kmeans
    labels_init = _load_latest_kmeans_labels(state_outputs_root, pack_dir)

    # 2) If not found, generate one
    if labels_init is None:
        print("No kmeans plan found — generating one now.")
        labels_init = kmeans_softcap_labels(
            coords=coords,
            weight=weight,
            num_districts=num_districts,
            pop_tolerance=0.05,
            max_iter=80,
            alpha=10000,
            seed=args.seed,
        )

    # 3) Build adjacency
    adj_json = json.loads((pack_dir / "adjacency.json").read_text())
    adj_idx = build_adj_idx(unit_ids, adj_json)
    # Phase A: fix kmeans plan so moves become feasible
    labels_init = enforce_contiguity_by_reattach(
        labels=labels_init,
        weight=weight,
        adj_idx=adj_idx,
        num_districts=num_districts,
        eps=0.25,   # <-- lax repair tolerance (try 0.20–0.30)
    )
    

    # 4) Hillclimb config
    hc_cfg = HillclimbConfig(
        party=args.party,
        pop_tolerance=0.15,     # <-- hillclimb tolerance (try 0.12–0.18)
        boundary_sample_k=4000,
        max_steps=200_000,
        patience=50_000,
        seed=args.seed,
        margin_weight=0.02,
    )

    # 5) Run hillclimber
    labels_final = hillclimb_max_seats(
        labels_init=labels_init,
        adj_idx=adj_idx,
        weight=weight,
        dem_votes=dem_votes,
        rep_votes=rep_votes,
        num_districts=num_districts,
        cfg=hc_cfg,
    )

    # 6) Export
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"hillclimb_{args.party}_{run_id}"
    run_dir = state_outputs_root / folder_name

    export_run(
        pack_dir=pack_dir,
        pack=pack,
        labels=labels_final,
        run_dir=run_dir,
        title=f"Hillclimb maximize {args.party.upper()} seats [{args.state}]",
    )

    _update_latest_manifest(state_outputs_root, f"hillclimb_{args.party}", folder_name)

    print("Saved:", run_dir)


if __name__ == "__main__":
    main()