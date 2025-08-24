# -*- coding: utf-8 -*-
"""
JOIN COUNT SYNTHETIC DATA (AVG) VS BIRD DATA FOR SPECIFIC DATABASE
To run:
PYTHONPATH=src python -m data_exploration.data_visualization_join_cnt_2
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import matplotlib.pyplot as plt

def plot_syn_avg_vs_bird_dev(
    db_name: str = "california_schools",
    base_dir: Path = Path("./data_exploration/data_exploration_details"),
    outfile_name: str = "join_cnt_syn_avg_vs_bird_per_level.png",
) -> Path:
    """
    Plot a grouped bar chart comparing bird_dev vs the average of syn_* datasets
    (syn_train, syn_dev, syn_test) for `join_cnt_per_sql_per_level`.

    Args:
        db_name: Database name used to resolve the JSON file location.
        base_dir: Base directory containing the data_exploration JSONs.
        outfile_name: Filename for the saved figure (PNG).

    Returns:
        Path: Full path to the saved figure.

    Raises:
        FileNotFoundError: If the expected JSON report file does not exist.
        ValueError: If required keys are missing from the report data.
    """
    levels: List[str] = ["simple", "moderate", "challenging", "window", "overall"]
    syn_keys: List[str] = ["syn_train", "syn_dev", "syn_test"]
    bird_key: str = "bird_dev"

    # --- Load data ---
    data_dir: Path = base_dir / db_name
    report_path: Path = data_dir / f"data_exploration_{db_name}.json"
    if not report_path.exists():
        raise FileNotFoundError(
            f"Report not found at {report_path}. Ensure you generated it first."
        )

    with open(report_path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    # Validate presence
    if bird_key not in data:
        raise ValueError(f"'{bird_key}' section is missing in the report JSON.")
    for k in syn_keys:
        if k not in data:
            raise ValueError(f"'{k}' section is missing in the report JSON.")

    def get_join_cnt_per_sql_per_level(section: Dict[str, Any]) -> Dict[str, float]:
        stats = section.get("join_cnt_per_sql_per_level")
        if not isinstance(stats, dict):
            raise ValueError("Missing 'join_cnt_per_sql_per_level' in a dataset section.")
        # Cast to float; ignore missing keys by returning 0.0
        return {lvl: float(stats.get(lvl, 0.0)) for lvl in levels}

    def get_total_join_cnt_per_level(section: Dict[str, Any]) -> Dict[str, float]:
        stats = section.get("total_join_cnt_per_level")
        if not isinstance(stats, dict):
            raise ValueError("Missing 'total_join_cnt_per_level' in a dataset section.")
        # Cast to float; ignore missing keys by returning 0.0
        return {lvl: float(stats.get(lvl, 0.0)) for lvl in levels}

    def get_question_counts_per_level(section: Dict[str, Any]) -> Dict[str, float]:
        stats = section.get("question_counts_per_level")
        if not isinstance(stats, dict):
            raise ValueError("Missing 'question_counts_per_level' in a dataset section.")
        # Cast to float; ignore missing keys by returning 0.0
        return {lvl: float(stats.get(lvl, 0.0)) for lvl in levels}

    bird_jcpspl_map: Dict[str, float] = get_join_cnt_per_sql_per_level(data[bird_key])
    syn_jcpspl_maps: List[Dict[str, float]] = [get_join_cnt_per_sql_per_level(data[k]) for k in syn_keys]
    syn_tjcpl_maps: List[Dict[str, float]] = [get_total_join_cnt_per_level(data[k]) for k in syn_keys]
    syn_qcpl_maps: List[Dict[str, float]] = [get_question_counts_per_level(data[k]) for k in syn_keys]

    # --- Compute syn average per level (ignore missing by averaging over present syn_* values) ---
    syn_avg: List[float] = []
    for lvl_idx, lvl in enumerate(levels):
        # vals = [m.get(lvl, 0.0) for m in syn_jcpspl_maps if lvl in m]
        # syn_avg.append(sum(vals) / len(vals) if vals else 0.0)
        total_vals = [m.get(lvl, 0.0) for m in syn_tjcpl_maps if lvl in m]
        total_cnts = [m.get(lvl, 0.0) for m in syn_qcpl_maps if lvl in m]

        syn_avg.append(sum(total_vals) / sum(total_cnts) if total_cnts else 0.0)

    bird_vals: List[float] = [bird_jcpspl_map[lvl] for lvl in levels]

    # --- Plot (two bars per group: bird_dev vs syn_avg) ---
    num_groups: int = len(levels)
    num_series: int = 2  # bird_dev, syn_avg
    bar_width: float = 0.28
    group_gap: float = 0.40
    alpha_val: float = 0.85

    cmap = plt.cm.get_cmap("Spectral", num_series)
    colors = [cmap(i) for i in range(num_series)]

    group_width: float = num_series * bar_width + group_gap
    x_positions: List[List[float]] = []
    for i in range(num_groups):
        start = i * group_width
        x_positions.append([start + j * bar_width for j in range(num_series)])

    plt.figure(figsize=(10, 6))

    # Series 1: bird_dev
    xs_bird = [pos[0] for pos in x_positions]
    bars_bird = plt.bar(
        xs_bird, bird_vals, width=bar_width, label="Bird-Dev",
        color=colors[0], alpha=alpha_val, edgecolor="gray", linewidth=0.8, zorder=3,
    )

    # Series 2: syn_avg
    xs_syn = [pos[1] for pos in x_positions]
    bars_syn = plt.bar(
        xs_syn, syn_avg, width=bar_width, label="Synthetic",
        color=colors[1], alpha=alpha_val, edgecolor="gray", linewidth=0.8, zorder=3,
    )

    # Value labels
    def _annotate(bars):
        for rect in bars:
            h = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2.0, h, f"{h:.2f}",
                     ha="center", va="bottom", fontsize=9, zorder=4)

    _annotate(bars_bird)
    _annotate(bars_syn)

    # X tick centers
    centers = [positions[0] + (num_series * bar_width) / 2.0 - (bar_width / 2.0)
               for positions in x_positions]
    plt.xticks(centers, [lvl.capitalize() for lvl in levels])

    plt.ylabel("Joins per SQL")
    plt.title("Join Count per SQL per Level Across Datasets")
    plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.3, zorder=0)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2)
    plt.tight_layout()

    out_path: Path = data_dir / outfile_name
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return out_path

if __name__ == "__main__":
    print(plot_syn_avg_vs_bird_dev())
