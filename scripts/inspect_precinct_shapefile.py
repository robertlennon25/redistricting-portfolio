import geopandas as gpd
import sys

path = sys.argv[1]
gdf = gpd.read_file(path)

print("Rows:", len(gdf))
print("CRS:", gdf.crs)
print("First 50 columns:\n", list(gdf.columns)[:50])

# helpful: show likely ID columns
id_candidates = [c for c in gdf.columns if "GEOID" in str(c).upper() or "ID" in str(c).upper() or "PREC" in str(c).upper()]
print("\nID candidates:", id_candidates[:30])

# show vote-ish columns
vote_candidates = [c for c in gdf.columns if str(c).upper().startswith("GCON") or "PRES" in str(c).upper() or "SEN" in str(c).upper()]
print("\nVote-ish candidates (first 80):", vote_candidates[:80])
