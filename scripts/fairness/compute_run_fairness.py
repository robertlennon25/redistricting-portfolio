#!/usr/bin/env python3
"""
Compute fairness metrics for a single exported run folder.

Reads (from --run-dir):
  - district_stats.json  (district-level dem/rep totals)
  - districts.geojson    (district boundary polygons)

Writes (to --run-dir):
  - fairness.json

Metrics:
  - Compactness (Polsby–Popper) per district
  - Mean/median/statewide vote share diagnostics
  - Efficiency gap (wasted votes) + per-district wasted votes

Usage example:
  python scripts/fairness/compute_run_fairness.py \
    --run-dir apps/web/public/outputs/il/hillclimb_dem_20260301_183125 \
    --party dem

Notes:
  - Uses EPSG:5070 by default for area/perimeter (CONUS Albers).
  - If your state is non-CONUS, pass a better projected EPSG via --area-epsg.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    import geopandas as gpd
except Exception as e:
    raise RuntimeError(
        "geopandas is required. Install deps from your project environment."
    ) from e

try:
    # Shapely 2.x
    from shapely.validation import make_valid
except Exception:
    make_valid = None  # Shapely < 2 fallback


# -----------------------------
# Helpers: IO + normalization
# -----------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def _infer_state_and_run_name(run_dir: Path) -> Tuple[Optional[str], str]:
    """
    Given .../outputs/<state>/<run_name>, infer state + run_name.
    If structure doesn't match, state=None.
    """
    parts = run_dir.resolve().parts
    # try to find ".../outputs/<state>/<run>"
    # e.g., apps/web/public/outputs/il/hillclimb_dem_...
    try:
        idx = parts.index("outputs")
        state = parts[idx + 1] if idx + 1 < len(parts) else None
        run_name = parts[idx + 2] if idx + 2 < len(parts) else run_dir.name
        return state, run_name
    except ValueError:
        return None, run_dir.name

def _load_district_stats_json(path: Path) -> pd.DataFrame:
    """
    Accepts either:
      - a list of district objects
      - or a dict keyed by district with values as objects
      - or {"districts": [...]} wrappers
    Returns DataFrame with at least: district, dem_votes, rep_votes
    """
    raw = json.loads(path.read_text())

    if isinstance(raw, dict) and "districts" in raw and isinstance(raw["districts"], list):
        rows = raw["districts"]
    elif isinstance(raw, list):
        rows = raw
    elif isinstance(raw, dict):
        # dict keyed by district?
        # e.g. {"0": {"dem_votes":..., "rep_votes":...}, ...}
        maybe_rows = []
        all_keys_numeric = True
        for k, v in raw.items():
            try:
                int(k)
            except Exception:
                all_keys_numeric = False
                break
            if isinstance(v, dict):
                v2 = dict(v)
                v2["district"] = int(k)
                maybe_rows.append(v2)
        if all_keys_numeric and maybe_rows:
            rows = maybe_rows
        else:
            raise ValueError(f"Unsupported district_stats.json structure: keys={list(raw.keys())[:10]}")
    else:
        raise ValueError("Unsupported district_stats.json structure")

    df = pd.DataFrame(rows)

    # normalize district column
    if "district" not in df.columns:
        # sometimes might be "District" or similar
        for cand in ["District", "district_id", "districtIndex"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "district"})
                break
    if "district" not in df.columns:
        raise ValueError("district_stats.json missing a 'district' column")

    # normalize vote columns
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ("dem_votes", "dem", "demvote", "dem_votes_total"):
            col_map[c] = "dem_votes"
        if lc in ("rep_votes", "rep", "repvote", "rep_votes_total"):
            col_map[c] = "rep_votes"
    df = df.rename(columns=col_map)

    if "dem_votes" not in df.columns or "rep_votes" not in df.columns:
        raise ValueError(f"district_stats.json missing dem/rep vote columns. columns={list(df.columns)}")

    df["district"] = df["district"].astype(int)
    df["dem_votes"] = pd.to_numeric(df["dem_votes"], errors="coerce").fillna(0).astype(float)
    df["rep_votes"] = pd.to_numeric(df["rep_votes"], errors="coerce").fillna(0).astype(float)

    # drop duplicates if any (keep first)
    df = df.sort_values("district").drop_duplicates(subset=["district"], keep="first").reset_index(drop=True)
    return df

def _load_districts_geojson(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)

    # normalize district column
    if "district" not in gdf.columns:
        for cand in ["District", "district_id", "DISTRICT", "districtIndex"]:
            if cand in gdf.columns:
                gdf = gdf.rename(columns={cand: "district"})
                break
    if "district" not in gdf.columns:
        raise ValueError(f"{path.name} missing 'district' property. columns={list(gdf.columns)[:50]}")

    gdf["district"] = gdf["district"].astype(int)

    # fix invalid geometries if possible
    if make_valid is not None:
        gdf["geometry"] = gdf["geometry"].apply(lambda geom: make_valid(geom) if geom is not None else geom)
    else:
        # shapely <2 fallback: buffer(0) can fix some invalid polys
        gdf["geometry"] = gdf["geometry"].buffer(0)

    return gdf


# -----------------------------
# Metrics
# -----------------------------

def compute_polsby_popper(
    districts_gdf: gpd.GeoDataFrame,
    area_epsg: int = 5070,
) -> pd.DataFrame:
    """
    Returns DataFrame with:
      district, area_km2, perimeter_km, polsby_popper
    """
    gdf = districts_gdf.copy()

    if gdf.crs is None:
        # Most exports for web are EPSG:4326; assume if missing.
        gdf = gdf.set_crs(epsg=4326)

    gdf_proj = gdf.to_crs(epsg=area_epsg)

    # meters-based (typical projected CRS)
    area_m2 = gdf_proj.geometry.area
    perim_m = gdf_proj.geometry.length

    area_km2 = area_m2 / 1e6
    perimeter_km = perim_m / 1e3

    # Polsby–Popper: 4πA / P^2
    pp = []
    for A, P in zip(area_m2.tolist(), perim_m.tolist()):
        if P <= 0 or A <= 0:
            pp.append(None)
        else:
            pp.append(float(4.0 * math.pi * A / (P * P)))

    out = pd.DataFrame({
        "district": gdf_proj["district"].astype(int).tolist(),
        "area_km2": area_km2.astype(float).tolist(),
        "perimeter_km": perimeter_km.astype(float).tolist(),
        "polsby_popper": pp,
    })

    return out.sort_values("district").reset_index(drop=True)

def compute_vote_share_diagnostics(
    stats_df: pd.DataFrame,
    party: str,
) -> Dict[str, Any]:
    """
    Computes district shares and statewide/mean/median summaries for the target party.
    """
    party = party.lower()
    if party not in ("dem", "rep"):
        raise ValueError("--party must be 'dem' or 'rep'")

    total = stats_df["dem_votes"] + stats_df["rep_votes"]
    total = total.replace(0, float("nan"))

    if party == "dem":
        share = stats_df["dem_votes"] / total
    else:
        share = stats_df["rep_votes"] / total

    share = share.fillna(0.0).clip(0.0, 1.0)

    # unweighted across districts
    mean_share = float(share.mean()) if len(share) else 0.0
    median_share = float(share.median()) if len(share) else 0.0

    # vote-weighted statewide share
    total_votes = float((stats_df["dem_votes"] + stats_df["rep_votes"]).sum())
    if total_votes <= 0:
        statewide_share = 0.0
    else:
        statewide_share = float(
            (stats_df["dem_votes"].sum() / total_votes) if party == "dem" else (stats_df["rep_votes"].sum() / total_votes)
        )

    return {
        "party": party,
        "statewide_share": statewide_share,
        "mean_share": mean_share,
        "median_share": median_share,
        "mean_median_diff": mean_share - median_share,
        "mean_vs_state_diff": mean_share - statewide_share,
        "median_vs_state_diff": median_share - statewide_share,
        "district_shares": pd.DataFrame({"district": stats_df["district"].astype(int), "party_share": share.astype(float)}),
        "two_party_votes_total": total_votes,
    }

def compute_efficiency_gap(
    stats_df: pd.DataFrame,
    votes_to_win_mode: str = "majority_plus_one",
) -> Dict[str, Any]:
    """
    Efficiency gap based on wasted votes.

    votes_to_win_mode:
      - "majority_plus_one": floor(total/2)+1 (typical discrete)
      - "half": total/2 (continuous variant)
    """
    wasted_dem = []
    wasted_rep = []

    for _, row in stats_df.iterrows():
        dem = float(row["dem_votes"])
        rep = float(row["rep_votes"])
        total = dem + rep

        if total <= 0:
            wasted_dem.append(0.0)
            wasted_rep.append(0.0)
            continue

        if votes_to_win_mode == "majority_plus_one":
            votes_to_win = math.floor(total / 2.0) + 1.0
        elif votes_to_win_mode == "half":
            votes_to_win = total / 2.0
        else:
            raise ValueError("votes_to_win_mode must be 'majority_plus_one' or 'half'")

        if dem > rep:
            wasted_dem.append(max(0.0, dem - votes_to_win))
            wasted_rep.append(rep)
        else:
            wasted_rep.append(max(0.0, rep - votes_to_win))
            wasted_dem.append(dem)

    W_dem = float(sum(wasted_dem))
    W_rep = float(sum(wasted_rep))
    V = float((stats_df["dem_votes"] + stats_df["rep_votes"]).sum())

    eg = 0.0
    if V > 0:
        # Common sign convention: (W_rep - W_dem)/V
        # Positive => advantage for Democrats (Republicans waste more)
        eg = float((W_rep - W_dem) / V)

    per = pd.DataFrame({
        "district": stats_df["district"].astype(int),
        "wasted_dem": wasted_dem,
        "wasted_rep": wasted_rep,
    })

    return {
        "efficiency_gap": eg,
        "total_wasted_dem": W_dem,
        "total_wasted_rep": W_rep,
        "per_district": per.sort_values("district").reset_index(drop=True),
        "two_party_votes_total": V,
        "votes_to_win_mode": votes_to_win_mode,
    }

def label_packed_cracked(
    shares_df: pd.DataFrame,
    party: str,
    statewide_share: float,
    competitive_band: float = 0.05,
    packed_margin: float = 0.10,
    cracked_margin: float = 0.05,
) -> pd.DataFrame:
    """
    Simple first-pass labeling.

    Balanced: within +/- competitive_band of 50%
    Packed: won by party and party_share - statewide_share >= packed_margin
    Cracked: lost by party, but party_share >= statewide_share - cracked_margin (i.e. meaningful support diluted)
    Else: other
    """
    party = party.lower()
    df = shares_df.copy()
    df["share_minus_state"] = df["party_share"] - statewide_share

    def _label(p_share: float) -> str:
        if abs(p_share - 0.5) <= competitive_band:
            return "balanced"
        won = p_share > 0.5
        if won and (p_share - statewide_share) >= packed_margin:
            return "packed"
        if (not won) and (p_share >= (statewide_share - cracked_margin)):
            return "cracked"
        return "other"

    df["label"] = df["party_share"].apply(_label)
    return df


# -----------------------------
# Main script
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Path to a single run folder under apps/web/public/outputs/<state>/<run>")
    ap.add_argument("--party", default="dem", choices=["dem", "rep"], help="Party to compute vote-share diagnostics for")
    ap.add_argument("--area-epsg", type=int, default=5070, help="Projected CRS EPSG for area/perimeter (default: 5070)")
    ap.add_argument("--votes-to-win-mode", default="majority_plus_one", choices=["majority_plus_one", "half"])
    ap.add_argument("--competitive-band", type=float, default=0.05)
    ap.add_argument("--packed-margin", type=float, default=0.10)
    ap.add_argument("--cracked-margin", type=float, default=0.05)
    ap.add_argument("--out-name", default="fairness.json", help="Output filename (default: fairness.json)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"--run-dir not found: {run_dir}")

    stats_path = run_dir / "district_stats.json"
    districts_path = run_dir / "districts.geojson"

    if not stats_path.exists():
        raise FileNotFoundError(f"Missing file: {stats_path}")
    if not districts_path.exists():
        raise FileNotFoundError(f"Missing file: {districts_path}")

    stats_df = _load_district_stats_json(stats_path)
    districts_gdf = _load_districts_geojson(districts_path)

    # Compactness
    compact_df = compute_polsby_popper(districts_gdf, area_epsg=args.area_epsg)

    # Vote shares + diagnostics
    vote_diag = compute_vote_share_diagnostics(stats_df, party=args.party)
    shares_df = vote_diag["district_shares"]

    # Packed/cracked labeling
    labels_df = label_packed_cracked(
        shares_df=shares_df,
        party=args.party,
        statewide_share=vote_diag["statewide_share"],
        competitive_band=args.competitive_band,
        packed_margin=args.packed_margin,
        cracked_margin=args.cracked_margin,
    )

    # Efficiency gap
    eg = compute_efficiency_gap(stats_df, votes_to_win_mode=args.votes_to_win_mode)

    # Merge per-district outputs
    out_df = stats_df[["district", "dem_votes", "rep_votes"]].copy()
    out_df["two_party_total"] = out_df["dem_votes"] + out_df["rep_votes"]

    out_df = out_df.merge(compact_df, on="district", how="left")
    out_df = out_df.merge(shares_df, on="district", how="left")
    out_df = out_df.merge(labels_df[["district", "label", "share_minus_state"]], on="district", how="left")
    out_df = out_df.merge(eg["per_district"], on="district", how="left")

    # Summary stats
    pp_series = out_df["polsby_popper"].dropna()
    compact_summary = {
        "mean_polsby_popper": float(pp_series.mean()) if len(pp_series) else None,
        "min_polsby_popper": float(pp_series.min()) if len(pp_series) else None,
        "max_polsby_popper": float(pp_series.max()) if len(pp_series) else None,
    }

    state, run_name = _infer_state_and_run_name(run_dir)

    fairness: Dict[str, Any] = {
        "meta": {
            "state": state,
            "run": run_name,
            "generated_at": _utc_now_iso(),
            "inputs": {
                "district_stats": "district_stats.json",
                "districts_geojson": "districts.geojson",
            },
            "params": {
                "party": args.party,
                "area_epsg": args.area_epsg,
                "votes_to_win_mode": args.votes_to_win_mode,
                "competitive_band": args.competitive_band,
                "packed_margin": args.packed_margin,
                "cracked_margin": args.cracked_margin,
            },
        },
        "summary": {
            "two_party_votes_total": float(vote_diag["two_party_votes_total"]),
            "party": args.party,
            "statewide_share": float(vote_diag["statewide_share"]),
            "mean_share": float(vote_diag["mean_share"]),
            "median_share": float(vote_diag["median_share"]),
            "mean_median_diff": float(vote_diag["mean_median_diff"]),
            "mean_vs_state_diff": float(vote_diag["mean_vs_state_diff"]),
            "median_vs_state_diff": float(vote_diag["median_vs_state_diff"]),
            "efficiency_gap": float(eg["efficiency_gap"]),
            "total_wasted_dem": float(eg["total_wasted_dem"]),
            "total_wasted_rep": float(eg["total_wasted_rep"]),
            "compactness": compact_summary,
        },
        "districts": [],
    }

    # Per-district records (JSON-friendly)
    for _, r in out_df.sort_values("district").iterrows():
        fairness["districts"].append({
            "district": int(r["district"]),
            "dem_votes": float(r["dem_votes"]),
            "rep_votes": float(r["rep_votes"]),
            "two_party_total": float(r["two_party_total"]),
            "party_share": float(r["party_share"]) if pd.notnull(r["party_share"]) else None,
            "packed_cracked": {
                "label": str(r["label"]) if pd.notnull(r["label"]) else None,
                "share_minus_state": float(r["share_minus_state"]) if pd.notnull(r["share_minus_state"]) else None,
            },
            "compactness": {
                "polsby_popper": float(r["polsby_popper"]) if pd.notnull(r["polsby_popper"]) else None,
                "area_km2": float(r["area_km2"]) if pd.notnull(r["area_km2"]) else None,
                "perimeter_km": float(r["perimeter_km"]) if pd.notnull(r["perimeter_km"]) else None,
            },
            "efficiency": {
                "wasted_dem": float(r["wasted_dem"]) if pd.notnull(r["wasted_dem"]) else None,
                "wasted_rep": float(r["wasted_rep"]) if pd.notnull(r["wasted_rep"]) else None,
            },
        })

    out_path = run_dir / args.out_name
    out_path.write_text(json.dumps(fairness, indent=2))
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()