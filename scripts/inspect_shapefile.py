from __future__ import annotations
from pathlib import Path
import argparse
import geopandas as gpd
import pandas as pd
import re

VOTE_PATTERNS = [
    r"\bDEM\b", r"\bREP\b", r"DEMOCRAT", r"REPUBLIC",
    r"\bPRES\b", r"PRESIDENT", r"\bGOV\b", r"\bSEN\b", r"\bUSS\b",
    r"\bCON\b", r"\bGCON\b", r"TRUMP", r"BIDEN", r"CLINTON"
]
POP_PATTERNS = [r"\bPOP\b", r"POPULATION", r"VAP", r"TOTAL", r"TOTPOP", r"PRECINCT_POP"]

def find_columns(cols, patterns):
    hits = []
    for c in cols:
        for p in patterns:
            if re.search(p, c, flags=re.IGNORECASE):
                hits.append(c)
                break
    return sorted(set(hits))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Path to .shp or a directory containing shapefile pieces")
    ap.add_argument("--head", type=int, default=3)
    args = ap.parse_args()

    path = Path(args.path)

    # Allow passing either the .shp or the folder that contains it
    shp = path
    if path.is_dir():
        candidates = list(path.glob("*.shp"))
        if not candidates:
            raise FileNotFoundError(f"No .shp found in {path}")
        shp = candidates[0]
        print(f"[info] Using shapefile: {shp}")

    gdf = gpd.read_file(shp)

    print("\n=== BASIC INFO ===")
    print("Rows:", len(gdf))
    print("Columns:", len(gdf.columns))
    print("CRS:", gdf.crs)
    print("Geometry types:", gdf.geometry.geom_type.value_counts().to_dict())

    print("\n=== COLUMN PREVIEW (first 60) ===")
    cols = list(map(str, gdf.columns))
    print(cols[:60])

    vote_cols = find_columns(cols, VOTE_PATTERNS)
    pop_cols = find_columns(cols, POP_PATTERNS)

    print("\n=== POSSIBLE VOTE COLUMNS (pattern match) ===")
    print(vote_cols[:200] if vote_cols else "None found")

    print("\n=== POSSIBLE POPULATION COLUMNS (pattern match) ===")
    print(pop_cols[:200] if pop_cols else "None found")

    # Show numeric summary for likely vote/pop columns
    likely_numeric = [c for c in (vote_cols + pop_cols) if pd.api.types.is_numeric_dtype(gdf[c])]
    likely_numeric = likely_numeric[:25]

    if likely_numeric:
        print("\n=== QUICK NUMERIC SUMMARY (first 25 likely columns) ===")
        desc = gdf[likely_numeric].describe().T[["min", "mean", "max"]]
        print(desc.to_string())
    else:
        print("\n=== QUICK NUMERIC SUMMARY ===")
        print("No likely numeric vote/pop columns detected (or columns are non-numeric).")

    print("\n=== SAMPLE ROWS ===")
    show_cols = (vote_cols + pop_cols)[:12]
    show_cols = show_cols if show_cols else cols[:12]
    print(gdf[show_cols].head(args.head).to_string(index=False))

if __name__ == "__main__":
    main()
