import argparse
import json
import yaml
from pathlib import Path
from datetime import datetime

import numpy as np

from gerry.data.map_pack import load_map_pack
from gerry.algos.kmeans_softcap import kmeans_softcap_labels
from export_run import export_run

from gerry.algos.kmeans_postprocess_contig import (
    build_adj_idx,
    enforce_contiguity_by_reattach,
)


def _resolve_pack_dir(cfg: dict, state: str | None) -> Path:
    """
    Priority:
      1) --state with cfg.states.<state>.assets_dir
      2) data.map_pack_dir / paths.assets_dir / output.assets_dir (legacy)
    """
    if state:
        scfg = (cfg.get("states", {}) or {}).get(state)
        if not scfg:
            raise KeyError(f"State '{state}' not found under cfg['states'].")
        assets_dir = scfg.get("assets_dir")
        if not assets_dir:
            raise KeyError(f"cfg.states.{state}.assets_dir missing.")
        return Path(assets_dir).expanduser().resolve()

    paths = cfg.get("paths", {}) or {}
    data = cfg.get("data", {}) or {}
    output = cfg.get("output", {}) or {}

    pack_dir_raw = data.get("map_pack_dir") or paths.get("assets_dir") or output.get("assets_dir")
    if not pack_dir_raw:
        raise KeyError(
            "No map pack directory found. Provide one of:\n"
            "  --state <key> with states.<key>.assets_dir\n"
            "  data.map_pack_dir\n  paths.assets_dir\n  output.assets_dir"
        )

    return Path(pack_dir_raw).expanduser().resolve()


def _resolve_outputs_root(cfg: dict) -> Path:
    paths = cfg.get("paths", {}) or {}

    # preferred for web
    public_outputs_raw = paths.get("public_outputs_dir")
    if public_outputs_raw:
        return Path(public_outputs_raw).expanduser().resolve()

    # allow local outputs if present
    out_dir_raw = paths.get("out_dir") or (cfg.get("output", {}) or {}).get("out_dir")
    if out_dir_raw:
        return Path(out_dir_raw).expanduser().resolve()

    # fallback
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
    """
    Robustly get:
      - unit_ids (list[str]) in the SAME order as coords/weight
      - coords (np.ndarray Nx2)
      - weight (np.ndarray N,)
    Uses pack fields if available, otherwise reads pack_dir/attributes.csv and id_to_idx.json.
    """

    # 1) coords
    if hasattr(pack, "coords") and pack.coords is not None:
        coords = np.asarray(pack.coords)
    else:
        import pandas as pd
        attrs_path = pack_dir / "attributes.csv"
        if not attrs_path.exists():
            raise FileNotFoundError(f"Missing {attrs_path}; cannot derive coords.")
        attrs = pd.read_csv(attrs_path)
        if "centroid_x" not in attrs.columns or "centroid_y" not in attrs.columns:
            raise KeyError("attributes.csv missing centroid_x/centroid_y; rebuild pack.")
        coords = attrs[["centroid_x", "centroid_y"]].values.astype(float)

    # 2) weight
    if hasattr(pack, "weight") and pack.weight is not None:
        weight = np.asarray(pack.weight).astype(float)
    else:
        import pandas as pd
        attrs = pd.read_csv(pack_dir / "attributes.csv")
        if "weight" not in attrs.columns:
            raise KeyError("attributes.csv missing 'weight' column; rebuild pack.")
        weight = attrs["weight"].astype(float).values

    # 3) unit_ids in correct order
    if hasattr(pack, "unit_ids") and pack.unit_ids is not None:
        unit_ids = [str(x) for x in pack.unit_ids]
    else:
        id_to_idx_path = pack_dir / "id_to_idx.json"
        if id_to_idx_path.exists():
            id_to_idx = json.loads(id_to_idx_path.read_text())
            idx_to_id = {int(v): str(k) for k, v in id_to_idx.items()}
            unit_ids = [idx_to_id[i] for i in range(len(idx_to_id))]
        else:
            # fallback: attributes.csv order
            import pandas as pd
            attrs = pd.read_csv(pack_dir / "attributes.csv")
            if "unit_id" not in attrs.columns:
                raise KeyError("attributes.csv missing unit_id; rebuild pack.")
            unit_ids = [str(x) for x in attrs["unit_id"].tolist()]

    n = coords.shape[0]
    if weight.shape[0] != n:
        raise ValueError(f"coords N={n} but weight N={weight.shape[0]} (mismatch).")
    if len(unit_ids) != n:
        raise ValueError(f"coords N={n} but unit_ids N={len(unit_ids)} (mismatch).")

    return unit_ids, coords, weight


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--state", default=None, help="Use cfg.states.<state>.assets_dir and write outputs under /outputs/<state>/")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--contig_eps", type=float, default=0.10, help="Pop tolerance for contiguity repair postprocess.")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))

    # Resolve pack and outputs
    pack_dir = _resolve_pack_dir(cfg, args.state)
    outputs_root = _resolve_outputs_root(cfg)

    # State-scoped outputs directory (prevents clutter)
    state_key = args.state or (cfg.get("project", {}) or {}).get("default_state") or "default"
    state_outputs_root = outputs_root / state_key
    state_outputs_root.mkdir(parents=True, exist_ok=True)

    run_cfg = cfg.get("run", {}) or {}
    k_cfg = (cfg.get("algos", {}) or {}).get("kmeans_softcap", {}) or {}

    # allow per-state override of num_districts if present
    if args.state:
        scfg = (cfg.get("states", {}) or {}).get(args.state, {}) or {}
        num_districts = int(scfg.get("num_districts", run_cfg.get("num_districts", 17)))
    else:
        num_districts = int(run_cfg.get("num_districts", 17))

    pop_tolerance = float(run_cfg.get("pop_tolerance", 0.05))
    max_iter = int(k_cfg.get("max_iter", 80))
    alpha = float(k_cfg.get("alpha", 10000.0))

    # seed precedence: CLI > algos.kmeans_softcap.seed > run.seed > default
    seed = args.seed
    if seed is None:
        seed = k_cfg.get("seed", None)
    if seed is None:
        seed = run_cfg.get("seed", 42)
    seed = int(seed)

    pack = load_map_pack(pack_dir)
    unit_ids, coords, weight = _load_pack_arrays(pack_dir, pack)

    # --- KMeans ---
    labels = kmeans_softcap_labels(
        coords=coords,
        weight=weight,
        num_districts=num_districts,
        pop_tolerance=pop_tolerance,
        max_iter=max_iter,
        alpha=alpha,
        seed=seed,
    )

    # --- Contiguity postprocess ---
    adj_path = pack_dir / "adjacency.json"
    if adj_path.exists():
        adj_json = json.loads(adj_path.read_text())
        adj_idx = build_adj_idx(unit_ids, adj_json)
        labels = enforce_contiguity_by_reattach(
            labels=labels,
            weight=weight,
            adj_idx=adj_idx,
            num_districts=num_districts,
            eps=float(args.contig_eps),
        )
    else:
        print(f"⚠️ No adjacency.json found at {adj_path}; skipping contiguity postprocess.")

    # --- Export ---
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"kmeans_softcap_{run_id}"
    run_dir = state_outputs_root / folder_name

    export_run(
        pack_dir=pack_dir,
        pack=pack,
        labels=labels,
        run_dir=run_dir,
        title=f"K-Means (soft pop + contiguity) [{state_key}]",
    )

    _update_latest_manifest(state_outputs_root, "kmeans_softcap", folder_name)
    print("Saved:", run_dir)


if __name__ == "__main__":
    main()