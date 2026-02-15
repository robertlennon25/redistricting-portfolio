import yaml
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

from gerry.data.map_pack import load_map_pack
from gerry.algos.greedy_packing import greedy_packing_labels, fix_contiguity, post_balance_pop


def main():
    print("1) loading config...")
    cfg = yaml.safe_load(open("config.yaml"))
    pack_dir = cfg["data"]["map_pack_dir"]
    print("   pack_dir =", pack_dir)

    out_root = Path(cfg["output"]["out_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    maximize = cfg["run"].get("maximize", "dem")
    num_districts = int(cfg["run"].get("num_districts", 17))
    pop_tolerance = float(cfg["run"].get("pop_tolerance", 0.05))

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

    # output folder
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"greedy_{maximize}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build precinct-level GeoDataFrame with district labels ----
    gdf = pack.shapes.copy()
    gdf["district"] = labels

    # save simple mapping
    gdf[["unit_id", "district"]].to_csv(run_dir / "unit_to_district.csv", index=False)

    # ---- Compute district totals for sidebar + hover ----
    # Load attributes (votes/weight) from the map pack
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
    (run_dir / "district_stats.json").write_text(district_stats.to_json(orient="records"))
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

    # ---- Create frontend-ready GeoJSON ----
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

    geojson_path = run_dir / "map_data.geojson"
    # # gdf.to_file(geojson_path, driver="GeoJSON")
    # # Leaflet expects GeoJSON in EPSG:4326 (lat/lon)
    # gdf_web = gdf.to_crs(epsg=4326)
    # gdf_web.to_file(geojson_path, driver="GeoJSON")
    geojson_path = run_dir / "map_data.geojson"

    # IMPORTANT: Leaflet expects GeoJSON in EPSG:4326 (lon/lat)
    gdf_web = gdf.to_crs(epsg=4326)
    gdf_web.to_file(geojson_path, driver="GeoJSON")


    print("Saved GeoJSON:", geojson_path)

    # ---- Save a static PNG too ----
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(column="district", cmap="tab20", linewidth=0.1, edgecolor="white", ax=ax)
    ax.set_title(f"Greedy packing ({maximize.upper()})")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(run_dir / "map.png", dpi=200)
    plt.close(fig)

    print("Saved:", run_dir)


if __name__ == "__main__":
    main()
