# -*- coding: utf-8 -*-
""""
To run:
PYTHONPATH=src python -m data_exploration.data_visualization_3
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_column_usage_log_barh(
    db_name: str = "california_schools",
    base_dir: Path = Path("./data_exploration/data_exploration_details"),
    datasets: List[str] = None,
    outfile_name: str = "column_coverage_comparison_california_schools.png",
) -> Path:
    """
    Plot a grouped horizontal bar chart (one row per column across all tables)
    of the column usage counts for each dataset, on a logarithmic (symlog) x-axis.

    The figure is tailored for ~89 columns:
      - Horizontal bars for label readability.
      - Sorted by total usage across datasets (descending).
      - xscale('symlog', linthresh=1) allows true zeros on the axis.
      - xlim(-1, max*1.15) keeps 0 away from the left edge so zero markers are visible.
      - Each zero-valued bar has a small marker at x=0 for visibility.

    Args:
        db_name: Database name (used to locate the JSON report file).
        base_dir: Base directory containing the data_exploration JSONs.
        datasets: Ordered list of dataset keys to include. If None, uses
                  ["bird_dev", "syn_train", "syn_dev", "syn_test"].
        outfile_name: Name of the output PNG saved alongside the JSON.

    Returns:
        Path: Full path to the saved PNG file.

    Raises:
        FileNotFoundError: If the JSON report file is not found.
        ValueError: If required blocks are missing or malformed.
    """
    if datasets is None:
        datasets = ["bird_dev", "syn_train", "syn_dev", "syn_test"]

    # ---- Load JSON (no manual numbers, no fallbacks) ----
    data_dir: Path = base_dir / db_name
    report_path: Path = data_dir / f"data_exploration_{db_name}.json"
    if not report_path.exists():
        raise FileNotFoundError(
            f"Report not found: {report_path}\n"
            f"Generate or place the JSON before plotting."
        )

    with open(report_path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    # ---- Flatten columns as "table.column" and collect counts per dataset ----
    all_cols: List[str] = []
    col_counts: Dict[str, Dict[str, float]] = {}  # "table.col" -> dataset -> count

    for ds in datasets:
        ds_block = data.get(ds)
        if not isinstance(ds_block, dict):
            raise ValueError(f"Dataset block missing or malformed: '{ds}'")

        tables = ds_block.get("db_columns_count_in_data")
        if not isinstance(tables, dict):
            raise ValueError(f"'db_columns_count_in_data' missing for dataset '{ds}'")

        for table_name, cols in tables.items():
            if not isinstance(cols, dict):
                continue
            for col_name, cnt in cols.items():
                key = f"{table_name}.{col_name}"
                if key not in col_counts:
                    col_counts[key] = {k: 0.0 for k in datasets}
                    all_cols.append(key)
                try:
                    col_counts[key][ds] = float(cnt)
                except Exception:
                    col_counts[key][ds] = 0.0

    # Order by total usage across datasets (descending)
    order: List[str] = sorted(
        all_cols,
        key=lambda k: sum(col_counts[k][ds] for ds in datasets),
        reverse=True,
    )

    n: int = len(order)
    num_datasets: int = len(datasets)

    # ---- Geometry for grouped horizontal bars ----
    bar_height: float = 0.18
    offsets: np.ndarray = (np.arange(num_datasets) - (num_datasets - 1) / 2.0) * bar_height
    y_base: np.ndarray = np.arange(n)

    # ---- Figure sizing for ~89 rows ----
    fig_height: float = max(12.0, n * 0.22)  # ~0.22 inch/row
    plt.figure(figsize=(16, fig_height))

    # ---- Colors: soft but colorful (no gray/white) ----
    dataset_colors: List[str] = ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3"]  # Set2 family
    if num_datasets > len(dataset_colors):
        cmap = plt.cm.get_cmap("tab20", num_datasets)
        dataset_colors = [cmap(i) for i in range(num_datasets)]

    # ---- Draw bars & remember zero positions for markers ----
    max_val: float = 0.0
    zero_points: List[Tuple[float, float, str]] = []  # (x=0, y, color)

    for j, ds in enumerate(datasets):
        xs: List[float] = []
        ys: List[float] = []
        for i, key in enumerate(order):
            val = float(col_counts[key][ds])
            xs.append(val)                  # true value (may be 0)
            ys.append(y_base[i] + offsets[j])
            if val == 0.0:
                zero_points.append((0.0, ys[-1], dataset_colors[j]))
            else:
                max_val = max(max_val, val)

        plt.barh(
            ys, xs, height=bar_height, label=ds,
            color=dataset_colors[j], alpha=0.9,
            edgecolor="gray", linewidth=0.6, zorder=3,
        )

    # ---- Axes: symlog allows zero; push left bound < 0 so 0 isn't at the edge ----
    plt.xscale("symlog", linthresh=1, linscale=1, base=10)
    if max_val <= 0:
        max_val = 1.0
    plt.xlim(-1, max_val * 1.15)

    plt.xlabel("Count", fontsize=16, fontweight="bold", labelpad=6)
    plt.ylabel("Columns (table.column)", fontsize=16, fontweight="bold", labelpad=6)
    plt.yticks(y_base, order, fontsize=7)
    plt.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.3, zorder=0)
    plt.title("Column Coverage Comparison for California Schools Database (Bird vs. Synthetic Dataset)", fontsize=18, fontweight="bold", pad=10)

    # ---- Zero markers at x=0 so zero-use columns are visible ----
    if zero_points:
        for x0, y0, c in zero_points:
            plt.scatter([x0], [y0], s=10, marker="o", color=c, edgecolor="black", linewidths=0.3, zorder=4)

        zero_proxy = Line2D([0], [0], marker='o', color='w',
                            markerfacecolor='none', markeredgecolor='black',
                            markeredgewidth=0.6, markersize=6, linestyle='None',
                            label='0 (marker at x=0)')
        # Extend legend with zero marker
        handles, labels = plt.gca().get_legend_handles_labels()
        handles.append(zero_proxy)
        labels.append('0 (marker at x=0)')
        plt.legend(handles, labels, ncol=min(5, num_datasets + 1), loc="lower right")
    else:
        plt.legend(ncol=min(5, num_datasets), loc="lower right")

    plt.tight_layout()

    out_path: Path = data_dir / outfile_name
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


if __name__ == "__main__":
    db_name = "california_schools"
    base_dir = Path("./data_exploration/data_exploration_details")
    plot_column_usage_log_barh(db_name=db_name, base_dir=base_dir, datasets=None)
