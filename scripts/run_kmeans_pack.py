import argparse
import json
import yaml
from pathlib import Path
from datetime import datetime

from gerry.data.map_pack import load_map_pack
from gerry.algos.kmeans_softcap import kmeans_softcap_labels
from export_run import export_run


def _resolve_pack_dir(cfg: dict) -> Path:
    paths = cfg.get("paths", {})
    data = cfg.get("data", {})
    output = cfg.get("output", {})

    pack_dir_raw = (
        data.get("map_pack_dir")
        or paths.get("assets_dir")
        or output.get("assets_dir")
    )

    if not pack_dir_raw:
        raise KeyError("No map pack directory found in config.")

    return Path(pack_dir_raw).expanduser().resolve()


def _resolve_outputs_root(cfg: dict) -> Path:
    paths = cfg.get("paths", {})
    public_outputs_raw = paths.get("public_outputs_dir")

    if public_outputs_raw:
        return Path(public_outputs_raw).expanduser().resolve()

    # fallback
    return Path("apps/web/public/outputs").resolve()


def _update_latest_manifest(outputs_root: Path, key: str, folder_name: str):
    manifest_path = outputs_root / "latest.json"

    if manifest_path.exists():
        latest = json.loads(manifest_path.read_text())
    else:
        latest = {}

    latest[key] = folder_name
    manifest_path.write_text(json.dumps(latest, indent=2))
    print("Updated latest.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))

    pack_dir = _resolve_pack_dir(cfg)
    outputs_root = _resolve_outputs_root(cfg)
    outputs_root.mkdir(parents=True, exist_ok=True)

    run_cfg = cfg.get("run", {})

    num_districts = int(run_cfg.get("num_districts", 17))
    pop_tolerance = float(run_cfg.get("pop_tolerance", 0.05))
    max_iter = int(run_cfg.get("max_iter", 80))
    alpha = float(run_cfg.get("alpha", 10000))
    seed = args.seed if args.seed is not None else int(run_cfg.get("seed", 42))

    pack = load_map_pack(pack_dir)

    # Robust coords handling
    if hasattr(pack, "coords"):
        coords = pack.coords
    else:
        coords = pack.shapes[["centroid_x", "centroid_y"]].values

    labels = kmeans_softcap_labels(
        coords=coords,
        weight=pack.weight,
        num_districts=num_districts,
        pop_tolerance=pop_tolerance,
        max_iter=max_iter,
        alpha=alpha,
        seed=seed,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"k_means_new_{run_id}"
    run_dir = outputs_root / folder_name

    export_run(
        pack_dir=pack_dir,
        pack=pack,
        labels=labels,
        run_dir=run_dir,
        title="K-Means (soft pop constraint)"
    )

    _update_latest_manifest(outputs_root, "k_means_new", folder_name)

    print("Saved:", run_dir)


if __name__ == "__main__":
    main()
