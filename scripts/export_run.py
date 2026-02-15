from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def export_run(pack_dir: str, pack, labels, run_dir: Path, title: str):
    run_dir.mkdir(parents=True, exist_ok=True)

    # join labels
    gdf = pack.shapes.copy()
    gdf["district"] = labels

    # mapping
    gdf[["unit_id", "district"]].to_csv(run_dir / "unit_to_district.csv", index=False)

    # district stats
    attrs = pd.read_csv(Path(pack_dir) / "attributes.csv")
    attrs["unit_id"] = attrs["unit_id"].astype(str)
    df = attrs.copy()
    df["district"] = labels

    district_stats = (
        df.groupby("district")[["dem_votes", "rep_votes", "weight"]].sum().reset_index()
    )
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

    (run_dir / "district_stats.json").write_text(district_stats.to_json(orient="records"))

    # merge totals for hover
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

    # Export for Leaflet
    gdf_web = gdf.to_crs(epsg=4326)
    gdf_web["geometry"] = gdf_web["geometry"].simplify(0.0002, preserve_topology=True)
    gdf_web.to_file(run_dir / "map_data.geojson", driver="GeoJSON")

    # PNG preview
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(column="district", cmap="tab20", linewidth=0.1, edgecolor="white", ax=ax)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(run_dir / "map.png", dpi=200)
    plt.close(fig)
