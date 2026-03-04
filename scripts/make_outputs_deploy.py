#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable, Optional

import geopandas as gpd


# ---------- helpers ----------

def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copytree(src: Path, dst: Path, *, ignore: Optional[callable] = None) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=ignore)


def _simplify_geometries_wgs84(gdf: gpd.GeoDataFrame, simplify_m: float) -> gpd.GeoDataFrame:
    """
    Simplify geometry in meters by projecting to EPSG:3857 then back to EPSG:4326.
    Your exported GeoJSON is CRS84/WGS84-ish; this keeps output compatible with Leaflet.
    """
    if simplify_m <= 0:
        return gdf

    gdf = gdf.copy()
    # make geometry valid-ish
    try:
        gdf["geometry"] = gdf["geometry"].buffer(0)
    except Exception:
        pass

    # if CRS missing, assume WGS84 for exported geojson
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)

    gdf_3857 = gdf.to_crs(epsg=3857)
    gdf_3857["geometry"] = gdf_3857["geometry"].simplify(simplify_m, preserve_topology=True)
    gdf_out = gdf_3857.to_crs(epsg=4326)

    # final validity pass
    try:
        gdf_out["geometry"] = gdf_out["geometry"].buffer(0)
    except Exception:
        pass

    return gdf_out


def _slim_map_data(
    in_path: Path,
    out_path: Path,
    simplify_m: float,
    keep_props: list[str],
) -> None:
    gdf = gpd.read_file(in_path)

    # keep only the props we need + geometry
    cols = [c for c in keep_props if c in gdf.columns]
    if "geometry" not in cols:
        cols = cols + ["geometry"]
    gdf = gdf[cols].copy()

    # enforce types that your frontend expects
    if "district" in gdf.columns:
        gdf["district"] = gdf["district"].astype(int)

    gdf = _simplify_geometries_wgs84(gdf, simplify_m=simplify_m)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GeoJSON")


def _slim_districts_geojson(in_path: Path, out_path: Path, simplify_m: float) -> None:
    gdf = gpd.read_file(in_path)
    cols = [c for c in ["district", "geometry"] if c in gdf.columns]
    gdf = gdf[cols].copy()

    if "district" in gdf.columns:
        gdf["district"] = gdf["district"].astype(int)

    gdf = _simplify_geometries_wgs84(gdf, simplify_m=simplify_m)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GeoJSON")


def _ignore_everything_except_manifest(_dir: str, files: list[str]) -> set[str]:
    """
    Copy flipbook folder but optionally ignore frames if needed.
    (Not used by default; see --drop_flipbook_frames.)
    """
    keep = {"manifest.json"}
    return {f for f in files if f not in keep}


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="apps/web/public/outputs", help="Source outputs folder")
    ap.add_argument("--out_dir", default="apps/web/public/outputs_deploy", help="Destination deploy outputs folder")
    ap.add_argument(
        "--states",
        default=None,
        help="Comma-separated state keys to include (e.g. il,ny). If omitted, uses states.json if present.",
    )
    ap.add_argument(
        "--keep_keys",
        default=None,
        help="Comma-separated keys to keep from latest.json (e.g. current_congress,hillclimb_dem,hillclimb_rep,kmeans_softcap). "
             "If omitted, keeps ALL keys in latest.json.",
    )
    ap.add_argument("--simplify_m", type=float, default=75.0, help="Geometry simplify tolerance in meters (EPSG:3857). 0 disables.")
    ap.add_argument("--copy_flipbook", action="store_true", help="Copy flipbook folder (frames + manifest) for kept runs.")
    ap.add_argument("--drop_flipbook_frames", action="store_true", help="If copying flipbooks, keep manifest.json only (no frames).")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # decide which states
    states: list[str] = []
    if args.states:
        states = [s.strip() for s in args.states.split(",") if s.strip()]
    else:
        states_path = in_dir / "states.json"
        if states_path.exists():
            j = json.loads(states_path.read_text())
            if isinstance(j, list):
                states = [str(x) for x in j]
        if not states:
            # fallback: infer states by directory names
            states = sorted([p.name for p in in_dir.iterdir() if p.is_dir()])

    keep_keys = None
    if args.keep_keys:
        keep_keys = {k.strip() for k in args.keep_keys.split(",") if k.strip()}

    # copy states.json if it exists
    if (in_dir / "states.json").exists():
        _copy_file(in_dir / "states.json", out_dir / "states.json")

    # minimal set of properties to keep for your current MapView + tooltips
    # (MapView reads: unit_id, district, dem_votes, rep_votes, weight, district_winner)
    keep_props = ["unit_id", "district", "dem_votes", "rep_votes", "weight", "district_winner"]

    for st in states:
        st_in = in_dir / st
        st_out = out_dir / st
        st_out.mkdir(parents=True, exist_ok=True)

        latest_path = st_in / "latest.json"
        if not latest_path.exists():
            print(f"[skip] {st}: no latest.json at {latest_path}")
            continue

        latest = _read_json(latest_path)
        _write_json(st_out / "latest.json", latest)  # keep same latest pointers

        # determine which run folders to copy
        items = []
        for key, folder in latest.items():
            if keep_keys is not None and key not in keep_keys:
                continue
            if not folder:
                continue
            items.append((key, str(folder)))

        if not items:
            print(f"[warn] {st}: latest.json has no matching folders to copy for keep_keys={keep_keys}")
            continue

        print(f"[state] {st}: copying {len(items)} run(s)")

        for key, folder in items:
            src_run = st_in / folder
            dst_run = st_out / folder

            if not src_run.exists():
                print(f"  [missing] {st}/{folder} (skipping)")
                continue

            # Copy everything first (small runs only). Then overwrite slimmed files.
            # If you want to copy only specific files, we can tighten this.
            _copytree(src_run, dst_run)

            # Slim map_data.geojson (required)
            in_map = dst_run / "map_data.geojson"
            if in_map.exists():
                print(f"  [slim] {st}/{folder}/map_data.geojson (simplify {args.simplify_m}m)")
                _slim_map_data(
                    in_path=in_map,
                    out_path=in_map,             # overwrite in place
                    simplify_m=float(args.simplify_m),
                    keep_props=keep_props,
                )
            else:
                print(f"  [warn] missing map_data.geojson in {st}/{folder}")

            # Slim districts.geojson (optional)
            in_districts = dst_run / "districts.geojson"
            if in_districts.exists():
                print(f"  [slim] {st}/{folder}/districts.geojson (simplify {args.simplify_m}m)")
                _slim_districts_geojson(
                    in_path=in_districts,
                    out_path=in_districts,
                    simplify_m=float(args.simplify_m),
                )

            # Ensure district_stats.json exists (copytree already handled)
            # If you ever want to recompute it, do it elsewhere.

            # Flipbook handling
            if args.copy_flipbook:
                fb_src = src_run / "flipbook"
                fb_dst = dst_run / "flipbook"
                if fb_src.exists():
                    if args.drop_flipbook_frames:
                        print(f"  [flipbook] copying manifest only (no frames) for {st}/{folder}")
                        _copytree(fb_src, fb_dst, ignore=_ignore_everything_except_manifest)
                    else:
                        # already copied by copytree; nothing to do
                        pass
                else:
                    # If folder doesn’t have flipbook, it’s fine.
                    pass
            else:
                # If not copying flipbook, remove it from deploy outputs to save space
                fb_dst = dst_run / "flipbook"
                if fb_dst.exists():
                    shutil.rmtree(fb_dst)

    print("\n✅ Done.")
    print(f"Deploy outputs written to: {out_dir}")
    print("Next step: swap outputs_deploy -> outputs (or point frontend to /outputs_deploy).")


if __name__ == "__main__":
    main()