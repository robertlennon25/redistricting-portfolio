import argparse
import json
import yaml
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd

from gerry.data.map_pack import load_map_pack
from export_run import export_run

from gerry.viz.frame_recorder import FrameRecorder, FrameMeta

from gerry.algos.kmeans_softcap import kmeans_softcap_labels
from gerry.algos.hillclimber import (
    HillclimbConfig,
    build_adj_idx,
    hillclimb_max_seats,
)
from gerry.algos.postprocess_contig import (
    enforce_contiguity_postprocess,
    rebalance_population_local,
    SeatGuard,
)

"""
First iteration of hillclimber
Starts from the kmeans map that is generated

example usage from repo root:
python3 scripts/run_hillclimber.py --config config.yaml --state ny --party rep
"""


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
    print(f"Using latest kmeans plan from {run_dir}")

    # Load labels from exported GeoJSON
    map_data_path = run_dir / "map_data.geojson"
    if not map_data_path.exists():
        return None

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
    ap.add_argument("--frame_every", type=int, default=5, help="Record a flipbook frame every N steps.")
    ap.add_argument("--fps", type=int, default=12, help="FPS metadata for flipbook playback.")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))

    pack_dir = _resolve_pack_dir(cfg, args.state)
    outputs_root = _resolve_outputs_root(cfg)
    state_outputs_root = outputs_root / args.state
    state_outputs_root.mkdir(parents=True, exist_ok=True)

    pack = load_map_pack(pack_dir)
    print("[4] pack loaded", flush=True)
    unit_ids, coords, weight, dem_votes, rep_votes = _load_pack_arrays(pack_dir, pack)

    num_districts = int(
        (cfg.get("states", {}) or {}).get(args.state, {}).get(
            "num_districts",
            (cfg.get("run", {}) or {}).get("num_districts", 17),
        )
    )

    # 1) Try loading latest kmeans
    print("[7] loading starting labels...", flush=True)
    labels_init = _load_latest_kmeans_labels(state_outputs_root, pack_dir)
    print("[8] latest labels:", "FOUND" if labels_init is not None else "NONE", flush=True)

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

    # 3) Build adjacency (must happen BEFORE repair)
    print("[11] reading adjacency.json...", flush=True)
    adj_json = json.loads((pack_dir / "adjacency.json").read_text())

    print("[12] building adj_idx...", flush=True)
    adj_idx = build_adj_idx(unit_ids, adj_json)
    print("[13] adj_idx built", flush=True)

    # Phase A: fix starting plan so moves become feasible (lax)
    print("[14] repairing contiguity...", flush=True)
    seat_guard_init = SeatGuard.from_arrays(labels_init, dem_votes, rep_votes, num_districts)
    labels_init = enforce_contiguity_postprocess(
        labels=labels_init,
        weight=weight,
        adj_idx=adj_idx,
        num_districts=num_districts,
        eps=0.25,
        max_passes=10,
        enable_bridge=True,
        max_bridge_len=30,
        seat_guard=seat_guard_init,
    )
    print("[15] contiguity repair done", flush=True)

    # 4) Hillclimb config (global + state override)
    hc_base = ((cfg.get("algos", {}) or {}).get("hillclimb", {}) or {}).copy()
    scfg = (cfg.get("states", {}) or {}).get(args.state, {}) or {}
    hc_override = (((scfg.get("algos", {}) or {}).get("hillclimb", {}) or {}))
    hc_base.update(hc_override)

    hc_cfg = HillclimbConfig(
        party=args.party,
        pop_tolerance=float(hc_base.get("pop_tolerance", 0.15)),
        boundary_sample_k=int(hc_base.get("boundary_sample_k", 4000)),
        max_steps=int(hc_base.get("max_steps", 300)),
        patience=int(hc_base.get("patience", 250)),
        seed=int(args.seed),
    )

    # 5) Define *final* run_dir BEFORE hillclimb so frames go into the right output folder
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"hillclimb_{args.party}_{run_id}"
    run_dir = state_outputs_root / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # 6) Flipbook recorder (must use final run_dir)
    rec = FrameRecorder(
        pack_dir=pack_dir,
        run_dir=run_dir,
        state=args.state,
        title=f"Hillclimb ({args.party.upper()}) [{args.state}]",
        dpi=140,
        figsize=(9.5, 9.5),
    )
    frame_no = 0
    FRAME_EVERY = int(args.frame_every)
    FPS = int(args.fps)

    def on_frame(step: int, labels: np.ndarray, stats: dict):
        nonlocal frame_no
        meta = FrameMeta(
            step=int(step),
            seats=int(stats["seats"]),
            closest_loss=float(stats["closest_loss"]),
            objective=float(stats["objective"]),
            locked=int(stats.get("locked", 0)),
        )
        rec.record(
            frame_no=frame_no,
            labels=labels,
            meta=meta,
            edgecolor=None,
            linewidth=0.10,
            margins=stats.get("margins")
        )
        frame_no += 1

    # 7) Run hillclimber (records frames)
    print("[16] starting hillclimb...", flush=True)
    labels_final = hillclimb_max_seats(
        labels_init=labels_init,
        adj_idx=adj_idx,
        weight=weight,
        dem_votes=dem_votes,
        rep_votes=rep_votes,
        num_districts=num_districts,
        cfg=hc_cfg,
        on_frame=on_frame,
        frame_every=FRAME_EVERY,
    )
    print("[17] hillclimb done", flush=True)

    # 8) FINAL SHIP POSTPROCESS
    print("[18] final contiguity + pop rebalance postprocess...", flush=True)

    seat_guard = SeatGuard.from_arrays(labels_final, dem_votes, rep_votes, num_districts)

    labels_final = enforce_contiguity_postprocess(
        labels=labels_final,
        weight=weight,
        adj_idx=adj_idx,
        num_districts=num_districts,
        eps=0.12,
        max_passes=10,
        enable_bridge=True,
        max_bridge_len=30,
        seat_guard=seat_guard,
    )

    labels_final = rebalance_population_local(
        labels=labels_final,
        weight=weight,
        adj_idx=adj_idx,
        num_districts=num_districts,
        max_moves=800,
        seat_guard=seat_guard,
    )

    print("[19] postprocess done", flush=True)

    # 9) Write flipbook manifest (after hillclimb frames recorded)
    manifest_path = rec.write_manifest(fps=FPS, frame_every=FRAME_EVERY)
    print("Flipbook manifest:", manifest_path, flush=True)

    # 10) Export (uses same run_dir)
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