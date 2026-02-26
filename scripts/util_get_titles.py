import geopandas as gpd
from pathlib import Path

"""
OBJECTIVE: prints the column titles of .shp files
-  used to figure out how to join populations and voting districts initially
NOT PART OF END PIPELINE, USED DURING CODING
"""

# Path to your .shp file (not the .dbf directly)
shapefile_path = Path("/Users/robertlennon/Desktop/redistricting/raw-data/il_pl2020_b (1)/il_pl2020_b.shp")
gdf = gpd.read_file(shapefile_path)

print("Columns in shapefile:", shapefile_path)
for col in gdf.columns:
    print(col)