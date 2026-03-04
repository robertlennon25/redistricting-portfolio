import argparse
import json
from pathlib import Path
from datetime import datetime

import yaml
import matplotlib.pyplot as plt
import pandas as pd
'''
This is a good runner for different algorithms
At the time, the issue is that they require you to change the run import
based on which algorithm you want to run. In version 3, we will fix this. 
'''
from gerry.data.map_pack import load_map_pack
# from gerry.algos.custom_algo import run as run_algorithm
from gerry.algos.pack_sinks_pop_no_con import run as run_algorithm


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_config_path(config_arg: str) -> Path:
    rr = _repo_root()
    p = Path(config_arg).expanduser()
    if not p.is_absolute():
        p = (rr / p).resolve()
    return p


def _resolve_pack_dir(cfg: dict) -> Path:
    rr = _repo_root()
    paths = cfg.get("paths", {})
    output = cfg.get("output", {})
    data = cfg.get("data", {})

    pack_dir_raw = (
        data.get("map_pack_dir")
        or cfg.get("map_pack_dir")
        or paths.get("assets_dir")
        or output.get("assets_dir")
    )
    if not pack_dir_raw:
        raise KeyError(
            "Missing map pack directory in config. Provide one of:\n"
            "  data.map_pack_dir: 'assets/il2024_precincts'\n"
            "  paths.assets_dir: 'assets/il2024_precincts'\n"
        )

    pack_dir = Path(pack_dir_raw).expanduser()
    if not pack_dir.is_absolute():
        pack_dir = (rr / pack_dir).resolve()
    return pack_dir


def _resolve_public_outputs_root(cfg: dict) -> Path:
    rr = _repo_root()
    paths = cfg.get("paths", {})
    public_outputs_raw = paths.get("public_outputs_dir")
    if public_outputs_raw:
        p = Path(public_outputs_raw).expanduser()
        if not p.is_absolute():
            p = (rr / p).resolve()
        return p
    return (rr / "apps/web/public/outputs").resolve()


def _update_latest_manifest(public_outputs_root: Path, key: str, run_folder_name: str):
    manifest_path = public_outputs_root / "latest.json"
    if manifest_path.exists():
        try:
            latest = json.loads(manifest_path.read_text())
            if not isinstance(latest, dict):
                latest = {}
        except Exception:
            latest = {}
    else:
        latest = {}

    latest[key] = run_folder_name
    manifest_path.write_text(json.dumps(latest, indent=2))
    print("✅ Updated manifest:", manifest_path)


def _export_frontend_files(pack_dir: Path, pack, labels, run_dir: Path, title: str):
    run_dir.mkdir(parents=True, exist_ok=True)

    gdf = pack.shapes.copy()
    labels_list = labels.tolist() if hasattr(labels, "tolist") else list(labels)
    if len(labels_list) != len(gdf):
        raise ValueError(f"labels length ({len(labels_list)}) != shapes length ({len(gdf)})")
    gdf["district"] = pd.Series(labels_list, index=gdf.index).astype(int)

    gdf[["unit_id", "district"]].to_csv(run_dir / "unit_to_district.csv", index=False)

    attrs = pd.read_csv(pack_dir / "attributes.csv")
    attrs["unit_id"] = attrs["unit_id"].astype(str)

    df = attrs.copy()
    df["district"] = gdf["district"].values

    district_stats = df.groupby("district")[["dem_votes", "rep_votes", "weight"]].sum().reset_index()
    district_stats["winner"] = district_stats.apply(
        lambda r: "Dem" if r["dem_votes"] > r["rep_votes"] else "GOP",
        axis=1,
    )
    district_stats["margin"] = district_stats["dem_votes"] - district_stats["rep_votes"]
    district_stats["margin_pct"] = (
        district_stats["margin"]
        / (district_stats["dem_votes"] + district_stats["rep_votes"]).replace(0, 1)
        * 100
    )

    (run_dir / "district_stats.json").write_text(json.dumps(district_stats.to_dict(orient="records"), indent=2))
    district_stats.to_csv(run_dir / "district_stats.csv", index=False)

    df = df.merge(
        district_stats.rename(
            columns={
                "dem_votes": "district_dem",
                "rep_votes": "district_rep",
                "weight": "district_weight",
                "winner": "district_winner",
                "margin": "district_margin",
                "margin_pct": "district_margin_pct",
            }
        ),
        on="district",
        how="left",
    )

    gdf = gdf.merge(
        df[
            [
                "unit_id",
                "dem_votes",
                "rep_votes",
                "weight",
                "district_dem",
                "district_rep",
                "district_weight",
                "district_winner",
                "district_margin",
                "district_margin_pct",
            ]
        ],
        on="unit_id",
        how="left",
    )

    gdf_web = gdf.to_crs(epsg=4326)
    gdf_web.to_file(run_dir / "map_data.geojson", driver="GeoJSON")

    districts = gdf.dissolve(by="district", as_index=False)
    districts_web = districts.to_crs(epsg=4326)
    districts_web.to_file(run_dir / "districts.geojson", driver="GeoJSON")

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(column="district", cmap="tab20", linewidth=0.1, edgecolor="white", ax=ax)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(run_dir / "map.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--maximize", choices=["dem", "rep"], required=True)
    ap.add_argument("--algo_key", default="", help="Key prefix in latest.json, e.g. greedy2, smartA, custom")
    args = ap.parse_args()

    cfg_path = _resolve_config_path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())

    pack_dir = _resolve_pack_dir(cfg)
    public_outputs_root = _resolve_public_outputs_root(cfg)
    public_outputs_root.mkdir(parents=True, exist_ok=True)

    pack = load_map_pack(pack_dir)

    labels = run_algorithm(pack=pack, cfg=cfg, maximize=args.maximize)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = f"{args.algo_key}_{args.maximize}_{run_id}"
    run_dir = public_outputs_root / run_folder

    _export_frontend_files(pack_dir=pack_dir, pack=pack, labels=labels, run_dir=run_dir,
                           title=f"{args.algo_key} ({args.maximize.upper()})")

    _update_latest_manifest(public_outputs_root, f"{args.algo_key}_{args.maximize}", run_folder)

    print("✅ Saved run:", run_dir)


if __name__ == "__main__":
    main()