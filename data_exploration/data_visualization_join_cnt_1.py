# -*- coding: utf-8 -*-
"""
To run:
JOIN COUNT SYNTHETIC DATA (TRAIN-DEV-TEST) VS BIRD DATA FOR SPECIFIC DATABASE

PYTHONPATH=src python -m data_exploration.data_visualization_join_cnt_1


Grouped bar chart: join_cnt_per_sql_per_level across datasets and levels.
- Uses matplotlib (no seaborn, no custom colors).
- Tries to read the JSON from the given path; falls back to the embedded data below if missing.
- Saves figure to /mnt/data/join_cnt_per_sql_per_level.png and displays it.
"""
from pathlib import Path
import json
import matplotlib.pyplot as plt

# --- Config ---
DB_NAME = "california_schools"
levels = ["simple", "moderate", "challenging", "window", "overall"]
# (key, label)
datasets = [
    ("bird_dev", "Bird-Dev"),
    ("syn_train", "Synthetic Train"),
    ("syn_dev", "Synthetic Dev"),
    ("syn_test", "Synthetic Test"),  
]

# --- Attempt to load from the provided path ---
data_exploration_dir_path = Path(f"./data_exploration/data_exploration_details/{DB_NAME}")
report_file_path = data_exploration_dir_path / f"data_exploration_{DB_NAME}.json"

data = None
if report_file_path.exists():
    with open(report_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

# --- Extract the values for plotting ---
values_by_dataset = []
for key, label in datasets:
    ds = data.get(key, {})
    join_stats = (ds or {}).get("join_cnt_per_sql_per_level", {})
    # Ensure order matches `levels`
    vals = [float(join_stats.get(level, 0.0)) for level in levels]
    values_by_dataset.append((label, vals))

# --- Plotting (single figure, grouped bars) ---
num_groups = len(levels)
num_datasets = len(values_by_dataset)
bar_width = 0.18           # width of each bar
group_gap = 0.35           # extra space between groups

# Calculate x positions
group_width = num_datasets * bar_width + group_gap
x_positions = []
for i in range(num_groups):
    group_start = i * group_width
    x_positions.append([group_start + j * bar_width for j in range(num_datasets)])

plt.figure(figsize=(12, 6))
for j, (label, vals) in enumerate(values_by_dataset):
    xs = [pos[j] for pos in x_positions]
    # bars = plt.bar(xs, vals, width=bar_width, label=label)
    cmap = plt.cm.get_cmap("Spectral", num_datasets)
    dataset_colors = [cmap(i) for i in range(num_datasets)]
    bars = plt.bar(xs, vals, width=bar_width, label=label,
                color=dataset_colors[j], alpha=0.85,
                edgecolor="gray", linewidth=0.8, zorder=3)
    plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.3, zorder=0)
    # Add value labels
    for rect in bars:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, f"{height:.2f}",
                 ha="center", va="bottom", fontsize=9, rotation=0)

# Set x-ticks at the center of each group
group_centers = [positions[0] + (num_datasets * bar_width) / 2.0 - (bar_width / 2.0) for positions in x_positions]
plt.xticks(group_centers, [lvl.capitalize() for lvl in levels])

plt.ylabel("Joins per SQL")
plt.title("Join Count per SQL per Level across Datasets")
plt.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
plt.tight_layout()

out_path = data_exploration_dir_path / "join_cnt_per_sql_per_level.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"Saved figure to: {out_path}")
