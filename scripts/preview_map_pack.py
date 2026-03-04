import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
import sys

pack = Path(sys.argv[1])

gdf = gpd.read_file(pack / "shapes.geojson")
print("rows:", len(gdf), "crs:", gdf.crs, "geom:", gdf.geometry.geom_type.value_counts().to_dict())

fig, ax = plt.subplots(figsize=(10,10))
gdf.plot(ax=ax, linewidth=0.1, edgecolor="black", facecolor="none")
ax.set_title("Map Pack Preview: shapes.geojson")
ax.axis("off")
plt.show()
