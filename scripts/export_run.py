from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

def export_run(
    pack_dir: str,
    pack,
    labels,
    run_dir: Path,
    title: str,
    simplify_tol_precincts: float = 0.0002,
    simplify_tol_districts: float = 0.0003,
):
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- Join labels to shapes ----
    gdf = pack.shapes.copy()

    # Ensure labels is a plain list aligned to gdf rows
    if hasattr(labels, "tolist"):
        labels_list = labels.tolist()
    else:
        labels_list = list(labels)

    if len(labels_list) != len(gdf):
        raise ValueError(f"labels length ({len(labels_list)}) != shapes length ({len(gdf)})")

    gdf["district"] = pd.Series(labels_list, index=gdf.index).astype(int)

    # Mapping file (debug/useful)
    gdf[["unit_id", "district"]].to_csv(run_dir / "unit_to_district.csv", index=False)

    # ---- District stats (sidebar) ----
    attrs = pd.read_csv(Path(pack_dir) / "attributes.csv")
    attrs["unit_id"] = attrs["unit_id"].astype(str)

    df = attrs.copy()
    df["district"] = gdf["district"].values  # guaranteed aligned

    district_stats = (
        df.groupby("district")[["dem_votes", "rep_votes", "weight"]]
        .sum()
        .reset_index()
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

    # Write JSON (pretty)
    (run_dir / "district_stats.json").write_text(
        json.dumps(district_stats.to_dict(orient="records"), indent=2)
    )
    district_stats.to_csv(run_dir / "district_stats.csv", index=False)

    # ---- Merge district totals back for hover ----
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

    # ---- Export precinct GeoJSON for Leaflet ----
    gdf_web = gdf.to_crs(epsg=4326)

    if simplify_tol_precincts and simplify_tol_precincts > 0:
        gdf_web["geometry"] = gdf_web["geometry"].simplify(
            simplify_tol_precincts, preserve_topology=True
        )

    gdf_web.to_file(run_dir / "map_data.geojson", driver="GeoJSON")

    # ---- Export district boundary overlay (for bold borders) ----
    # Dissolve precincts into district polygons
    districts = gdf.dissolve(by="district", as_index=False)
    districts_web = districts.to_crs(epsg=4326)

    if simplify_tol_districts and simplify_tol_districts > 0:
        districts_web["geometry"] = districts_web["geometry"].simplify(
            simplify_tol_districts, preserve_topology=True
        )

    districts_web.to_file(run_dir / "districts.geojson", driver="GeoJSON")

    # ---- PNG preview ----
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(column="district", cmap="tab20", linewidth=0.1, edgecolor="white", ax=ax)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(run_dir / "map.png", dpi=200)
    plt.close(fig)

    print(f"✅ Exported run to: {run_dir}")
    print(f"   - map_data.geojson (precincts)")
    print(f"   - districts.geojson (district borders)")
    print(f"   - district_stats.json")
