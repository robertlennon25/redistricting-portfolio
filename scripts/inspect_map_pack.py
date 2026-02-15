import json
from pathlib import Path
import pandas as pd

pack = Path("assets/il2022_precincts")  # change if needed

attrs = pd.read_csv(pack / "attributes.csv")
adj = json.loads((pack / "adjacency.json").read_text())

print("Units in attributes:", len(attrs))
print("Adjacency keys:", len(adj))
print("Sample unit_id:", attrs["unit_id"].iloc[0])
print("Sample neighbors count:", len(adj[str(attrs['unit_id'].iloc[0])]))

print("\nVotes summary:")
print(attrs[["dem_votes","rep_votes","weight"]])

print("\nDisconnected units (0 neighbors):", sum(len(v)==0 for v in adj.values()))
