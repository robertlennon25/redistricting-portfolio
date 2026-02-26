from pathlib import Path
import geopandas as gpd
import pandas as pd
"""
Run this code to join the population file (Census 2020, PL4171)
 to the voting district voting results from 2024 congresisonal elections
 
 This is a temporary fix to join them for Illinois, will be refactored to autoimatically 
 run when we add new states 

 """

PRECINCT_SHP = Path("/Users/robertlennon/Desktop/redistricting/raw-data/il_2024_gen_prec/il_2024_gen_cong_prec/il_2024_gen_cong_prec.shp")
BLOCKS_SHP   = Path("/Users/robertlennon/Desktop/redistricting/raw-data/il_pl2020_b (1)/il_pl2020_b.shp")  # adjust if needed
OUT_PATH     = Path("/Users/robertlennon/Desktop/redistricting/raw-data/il_2024_gen_prec/il_2024_gen_cong_prec/il_2024_gen_cong_prec_with_pop.gpkg")

POP_COL = "P0010001"   # total population (use P0030001 for VAP)
PRECINCT_ID_COL = "UNIQUE_ID"

def main():
    print("Loading precincts:", PRECINCT_SHP)
    precincts = gpd.read_file(PRECINCT_SHP)

    print("Loading blocks:", BLOCKS_SHP)
    blocks = gpd.read_file(BLOCKS_SHP)

    if POP_COL not in blocks.columns:
        raise ValueError(f"{POP_COL} not found in blocks. Available: {list(blocks.columns)[:50]} ...")

    # Ensure numeric
    blocks[POP_COL] = pd.to_numeric(blocks[POP_COL], errors="coerce").fillna(0).astype(float)

    # Align CRS (project both to a planar CRS for spatial ops)
    # If precincts CRS missing, you need to set it (rare with RDH, but possible)
    if precincts.crs is None:
        raise ValueError("Precincts CRS is None. Set it before joining (check .prj file).")
    if blocks.crs is None:
        raise ValueError("Blocks CRS is None. Set it before joining (check .prj file).")

    # Use the precinct CRS
    blocks = blocks.to_crs(precincts.crs)

    # Centroid join is much faster than polygon intersects and usually good enough for block->precinct
    # (blocks are small; centroid almost always falls inside the correct precinct)
    blocks_cent = blocks.copy()
    blocks_cent["geometry"] = blocks_cent.geometry.centroid

    print("Spatial join (centroid within precinct)...")
    joined = gpd.sjoin(
        blocks_cent[[POP_COL, "geometry"]],
        precincts[[PRECINCT_ID_COL, "geometry"]],
        how="inner",
        predicate="within",
    )

    print("Aggregating population by precinct...")
    pop_by_precinct = joined.groupby(PRECINCT_ID_COL)[POP_COL].sum().reset_index()
    pop_by_precinct = pop_by_precinct.rename(columns={POP_COL: "TOTPOP"})

    print("Merging back onto precincts...")
    out = precincts.merge(pop_by_precinct, on=PRECINCT_ID_COL, how="left")
    out["TOTPOP"] = out["TOTPOP"].fillna(0).astype(float)

    print("Writing:", OUT_PATH)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_file(OUT_PATH, driver="GPKG")

    print("Done. TOTPOP stats:")
    print(out["TOTPOP"].describe())

if __name__ == "__main__":
    main()