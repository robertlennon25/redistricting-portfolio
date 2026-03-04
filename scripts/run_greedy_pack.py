import argparse
import json
from pathlib import Path
from datetime import datetime

import yaml
import matplotlib.pyplot as plt
import pandas as pd

from gerry.data.map_pack import load_map_pack
from gerry.algos.greedy_packing import greedy_packing_labels, fix_contiguity, post_balance_pop


def _resolve_config_path(config_arg: str) -> Path:
    repo_root = Path(__file__).resolve().parents[1]  # scripts/.. = repo root
    p = Path(config_arg).expanduser()
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    return p


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_pack_dir(cfg: dict, repo_root: Path) -> Path:
    paths = cfg.get("paths", {})
    output = cfg.get("output", {})
    data = cfg.get("data", {})

    pack_dir_raw = (
        data.get("map_pack_dir")               # old
        or cfg.get("map_pack_dir")             # legacy flat
        or paths.get("assets_dir")             # new
        or output.get("assets_dir")            # older
    )
    if not pack_dir_raw:
        raise KeyError(
            "Missing map pack directory in config.\n"
            "Provide one of:\n"
            "  data.map_pack_dir: 'assets/il2024_precincts'\n"
            "  paths.assets_dir: 'assets/il2024_precincts'\n"
        )

    pack_dir = Path(pack_dir_raw).expanduser()
    if not pack_dir.is_absolute():
        pack_dir = (repo_root / pack_dir).resolve()
    return pack_dir


def _resolve_public_outputs_root(cfg: dict, repo_root: Path) -> Path:
    """
    Where we write timestamped runs that the frontend can fetch.
    Default: apps/web/public/outputs
    """
    paths = cfg.get("paths", {})
    public_outputs_raw = paths.get("public_outputs_dir")  # optional config
    if public_outputs_raw:
        p = Path(public_outputs_raw).expanduser()
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        return p

    return (repo_root / "apps/web/public/outputs").resolve()


def _update_latest_manifest(public_outputs_root: Path, key: str, run_folder_name: str):
    """
    Writes apps/web/public/outputs/latest.json like:
    { "greedy_dem": "greedy_dem_YYYYMMDD_HHMMSS", "greedy_rep": "..." }
    """
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--maximize", choices=["dem", "rep"], default=None)
    args = ap.parse_args()

    repo_root = _resolve_repo_root()

    cfg_path = _resolve_config_path(args.config)
    print("1) loading config...")
    print("   config =", cfg_path)
    cfg = yaml.safe_load(cfg_path.read_text())

    pack_dir = _resolve_pack_dir(cfg, repo_root)
    public_outputs_root = _resolve_public_outputs_root(cfg, repo_root)
    public_outputs_root.mkdir(parents=True, exist_ok=True)

    print("   pack_dir =", pack_dir)
    print("   public_outputs_root =", public_outputs_root)

    run_cfg = cfg.get("run", {})
    maximize = args.maximize or run_cfg.get("maximize", "dem")
    num_districts = int(run_cfg.get("num_districts", 17))
    pop_tolerance = float(run_cfg.get("pop_tolerance", 0.05))

    print("2) loading map pack...")
    pack = load_map_pack(pack_dir)
    print("   loaded pack:", len(pack.ids), "units")

    print("3) running greedy_packing_labels...")
    labels = greedy_packing_labels(
        dem=pack.dem,
        rep=pack.rep,
        weight=pack.weight,
        adj=pack.adj,
        num_districts=num_districts,
        pop_tolerance=pop_tolerance,
        maximize=maximize,
    )
    print("   greedy done. unique labels:", len(set(labels.tolist())))

    print("4) fix_contiguity pass 1...")
    labels = fix_contiguity(labels, pack.adj, num_districts=num_districts)
    print("   fix_contiguity done")

    print("5) post_balance_pop...")
    labels = post_balance_pop(
        labels=labels,
        weight=pack.weight,
        adj=pack.adj,
        num_districts=num_districts,
        pop_tolerance=pop_tolerance,
    )
    print("   post_balance_pop done")

    print("6) fix_contiguity pass 2...")
    labels = fix_contiguity(labels, pack.adj, num_districts=num_districts)
    print("   fix_contiguity done (2)")

    # ---- output folder (timestamped, but inside public outputs) ----
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = f"greedy_{maximize}_{run_id}"
    run_dir = public_outputs_root / run_folder
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build precinct-level GeoDataFrame with district labels ----
    gdf = pack.shapes.copy()
    gdf["district"] = labels

    # save simple mapping
    gdf[["unit_id", "district"]].to_csv(run_dir / "unit_to_district.csv", index=False)

    # ---- Compute district totals for sidebar + hover ----
    attrs_path = Path(pack_dir) / "attributes.csv"
    attrs = pd.read_csv(attrs_path)
    attrs["unit_id"] = attrs["unit_id"].astype(str)

    df = attrs.copy()
    df["district"] = labels

    district_stats = (
        df.groupby("district")[["dem_votes", "rep_votes", "weight"]]
        .sum()
        .reset_index()
    )
    district_stats["winner"] = district_stats.apply(
        lambda r: "Dem" if r["dem_votes"] > r["rep_votes"] else "GOP",
        axis=1,
    )
    district_stats["margin"] = (district_stats["dem_votes"] - district_stats["rep_votes"])
    district_stats["margin_pct"] = (
        district_stats["margin"] / (district_stats["dem_votes"] + district_stats["rep_votes"]).replace(0, 1)
    ) * 100

    # save sidebar table data
    (run_dir / "district_stats.json").write_text(json.dumps(district_stats.to_dict(orient="records"), indent=2))
    district_stats.to_csv(run_dir / "district_stats.csv", index=False)

    # Merge district totals back to each precinct so hover can show district totals
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

    # ---- Create frontend-ready precinct GeoJSON ----
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

    # IMPORTANT: Leaflet expects EPSG:4326 (lon/lat)
    gdf_web = gdf.to_crs(epsg=4326)
    gdf_web.to_file(run_dir / "map_data.geojson", driver="GeoJSON")
    print("Saved GeoJSON:", run_dir / "map_data.geojson")

    # ---- District boundary overlay (bold borders) ----
    districts = gdf.dissolve(by="district", as_index=False)
    districts_web = districts.to_crs(epsg=4326)
    districts_web.to_file(run_dir / "districts.geojson", driver="GeoJSON")
    print("Saved district outlines:", run_dir / "districts.geojson")

    # ---- Save a static PNG too (optional) ----
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(column="district", cmap="tab20", linewidth=0.1, edgecolor="white", ax=ax)
    ax.set_title(f"Greedy packing ({maximize.upper()})")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(run_dir / "map.png", dpi=200)
    plt.close(fig)

    # ---- Update latest manifest for frontend ----
    _update_latest_manifest(public_outputs_root, f"greedy_{maximize}", run_folder)

    print("✅ Saved run:", run_dir)


if __name__ == "__main__":
    main()
