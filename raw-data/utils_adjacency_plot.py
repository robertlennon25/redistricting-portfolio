import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Path to your adjacency.json
ADJ_PATH = Path("/Users/robertlennon/Desktop/redistricting/assets/il2024_precincts/adjacency.json")  # <-- adjust if needed

with open(ADJ_PATH, "r") as f:
    adj = json.load(f)

# Degree = number of neighbors
degrees = np.array([len(neighbors) for neighbors in adj.values()])

print("Adjacency Graph Stats")
print("---------------------")
print(f"Number of nodes: {len(degrees)}")
print(f"Mean degree:   {degrees.mean():.2f}")
print(f"Median degree: {np.median(degrees):.2f}")
print(f"Min degree:    {degrees.min()}")
print(f"Max degree:    {degrees.max()}")
zero_degree = sum(degrees == 0)
one_degree = sum(degrees == 1)

print(f"Zero-degree nodes: {zero_degree}")
print(f"One-degree nodes:  {one_degree}")

# Histogram
plt.figure(figsize=(8, 5))
plt.hist(degrees, bins=30, edgecolor="black")
plt.title("Adjacency Degree Distribution")
plt.xlabel("Number of Neighbors (Degree)")
plt.ylabel("Number of Precincts")
plt.tight_layout()
plt.show()