import geopandas as gpd
from pathlib import Path

"""
OBJECTIVE: prints the column titles of .shp, .gpkg files
-  used to figure out how to join populations and voting districts initially
NOT PART OF END PIPELINE, USED DURING CODING
"""

# Path to your .shp file (not the .dbf directly)

# shapefile_path = Path("/Users/robertlennon/Desktop/redistricting/raw-data/il_2024_gen_prec/il_2024_gen_cong_prec/il_2024_gen_cong_prec.shp")
# shapefile_path = Path("/Users/robertlennon/Desktop/redistricting/raw-data/il_2024_gen_prec/il_2024_gen_cong_prec/il_2024_gen_cong_prec_with_pop.gpkg")
shapefile_path = Path("/Users/robertlennon/Desktop/redistricting/raw-data/ny_2024_gen_prec/ny_2024_gen_all_prec_with_pop.gpkg")

gdf = gpd.read_file(shapefile_path)

print("Columns in shapefile:", shapefile_path)
for col in gdf.columns:
    print(col)