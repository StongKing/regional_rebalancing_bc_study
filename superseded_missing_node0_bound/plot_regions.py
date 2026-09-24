"""Plot the predeclared coordinate-only partition; no outcomes are read."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
index = json.loads((HERE/"inputs/index.json").read_text(encoding="utf-8"))
case = Path(index["cases"][0]["directory"])
metadata = json.loads((case/"metadata.json").read_text(encoding="utf-8"))
with np.load(case/"instance.npz", allow_pickle=False) as archive:
    coordinates = archive["coords"]
    labels = archive["region_labels"]
    identifiers = archive["station_ids"].astype(str)

fig, ax = plt.subplots(figsize=(7.2, 7.2), layout="constrained")
palette = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
for region, color in enumerate(palette):
    selected = labels == region
    ax.scatter(coordinates[selected, 1], coordinates[selected, 0], s=18, c=color,
               alpha=.85, linewidths=.3, edgecolors="white", label=f"Region {region+1} ({selected.sum()} stations)")
depot = np.flatnonzero(identifiers == str(metadata["depot_station_id"]))
if len(depot) == 1:
    point = coordinates[depot[0]]
    ax.scatter(point[1], point[0], marker="X", s=95, color="#222222", zorder=5, label="Physical depot")
ax.set(xlabel="Longitude", ylabel="Latitude", title="Fixed geographic partition of the 567-station network")
ax.set_aspect(1/np.cos(np.deg2rad(coordinates[:, 0].mean())))
ax.grid(alpha=.15)
ax.legend(loc="lower left", fontsize=9, framealpha=.96)
fig.supxlabel("Coordinates only; all methods share this partition. Cross-region truck routes remain feasible.", fontsize=9)
out = HERE/"results/figures"
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out/"fixed_four_regions.png", dpi=180)
fig.savefig(out/"fixed_four_regions.pdf")
plt.close(fig)
print(str(out/"fixed_four_regions.png"))
