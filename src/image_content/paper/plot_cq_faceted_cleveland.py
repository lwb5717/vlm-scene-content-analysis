# -*- coding: utf-8 -*-
"""
Plot merged Cleveland dot plots for CQ complexity metrics:
1) Scene x (Clutter+Quantity 9-bin)
2) Distance x (Clutter+Quantity 9-bin)

Each y-category shows two metrics:
- normalized expected CQ score
- normalized entropy of CQ distribution
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

ROOT_DIR = Path(__file__).resolve().parents[3]

DATASET_CONFIG = {
    "SPAQ": {
        "complexity": ROOT_DIR / "result" / "complexity_results_spaq.csv",
        "scene": ROOT_DIR / "result" / "scene_analysis_results_spaq.csv",
    },
    "KonIQ-10K": {
        "complexity": ROOT_DIR / "result" / "complexity_results_koniq10k.csv",
        "scene": ROOT_DIR / "result" / "scene_analysis_results_koniq10k.csv",
    },
    "LIVE Wild": {
        "complexity": ROOT_DIR / "result" / "complexity_results_live.csv",
        "scene": ROOT_DIR / "result" / "scene_analysis_results_livewild.csv",
    },
    "CID2013": {
        "complexity": ROOT_DIR / "result" / "complexity_results_cid2013.csv",
        "scene": ROOT_DIR / "result" / "scene_analysis_results_cid2013.csv",
    },
}

DEFAULT_OUTPUT_DIR = ROOT_DIR / "result" / "paper_figures" / "cq_cleveland"

SCENE_ORDER = ["natural_landscape", "portrait", "indoor", "urban", "other"]
DISTANCE_ORDER = ["close", "medium", "far", "unknown"]
LEVEL_MAP = {"low": 0, "medium": 1, "high": 2}
N_CQ_BINS = 9

TITLE_FONTSIZE = 23
LABEL_FONTSIZE = 18
TICK_FONTSIZE = 16
MARKER_SIZE = 108
LEGEND_FONTSIZE = 11
LEGEND_TITLE_FONTSIZE = 13

MEAN_COLOR = "#2C7FB8"
ENT_COLOR = "#D95F0E"
CONNECTOR_COLOR = "#AFAFAF"
SHOW_CONNECTOR = False
MISSING_COLOR = "#7F7F7F"
POINT_ALPHA = 0.82
POINT_EDGE_WIDTH = 1.8
POINT_HALO_WIDTH = 4.0

GROUP_TITLE_MAP = {
    "scene_type": "Scene Type",
    "camera_to_object_distance": "Camera-to-Object Distance",
}
DATASET_COLORS = {
    "SPAQ": "#4E79A7",
    "KonIQ-10K": "#F28E2B",
    "LIVE Wild": "#59A14F",
    "CID2013": "#B07AA1",
}
MEAN_LABEL = r"$\mathbb{E}[CQ]$ (norm)"
ENTROPY_LABEL = "H(CQ) (norm)"


def _apply_ieee_font() -> None:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"]


def _save_figure(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"), format="pdf")


def _normalize_image_name(value: str) -> str:
    text = str(value).replace("\\", "/").strip().lower()
    return text.split("/")[-1]


def _load_and_merge(dataset_name: str) -> pd.DataFrame:
    cfg = DATASET_CONFIG[dataset_name]
    c_path = cfg["complexity"]
    s_path = cfg["scene"]

    if not c_path.exists():
        raise FileNotFoundError(f"Complexity CSV not found: {c_path}")
    if not s_path.exists():
        raise FileNotFoundError(f"Scene CSV not found: {s_path}")

    c_df = pd.read_csv(c_path)
    s_df = pd.read_csv(s_path)

    required_c = {"image_name", "quantity_level", "clutter_level"}
    required_s = {"image_name"}
    if not required_c.issubset(c_df.columns):
        raise ValueError(f"Complexity CSV missing columns: {required_c - set(c_df.columns)}")
    if not required_s.issubset(s_df.columns):
        raise ValueError(f"Scene CSV missing columns: {required_s - set(s_df.columns)}")

    # Keep schema stable even when some fields are absent in source CSV.
    if "scene_type" not in s_df.columns:
        s_df["scene_type"] = "other"
    if "camera_to_object_distance" not in s_df.columns:
        s_df["camera_to_object_distance"] = "unknown"

    c_df = c_df.copy()
    s_df = s_df.copy()
    c_df["image_key"] = c_df["image_name"].map(_normalize_image_name)
    s_df["image_key"] = s_df["image_name"].map(_normalize_image_name)

    merged = c_df.merge(
        s_df[["image_key", "scene_type", "camera_to_object_distance"]],
        on="image_key",
        how="inner",
    )
    if merged.empty:
        raise RuntimeError(f"No matched samples after merge for dataset: {dataset_name}")

    q = merged["quantity_level"].astype(str).str.lower().map(LEVEL_MAP)
    c = merged["clutter_level"].astype(str).str.lower().map(LEVEL_MAP)
    valid = q.notna() & c.notna()
    merged = merged.loc[valid].copy()

    # CQ bin index in [1, 9]: quantity-major then clutter-minor.
    merged["cq_bin"] = (q[valid].astype(int) * 3 + c[valid].astype(int) + 1).astype(int)
    merged["scene_type"] = merged["scene_type"].fillna("other").astype(str).str.lower().str.strip()
    merged["camera_to_object_distance"] = (
        merged["camera_to_object_distance"].fillna("unknown").astype(str).str.lower().str.strip()
    )

    # Fold unexpected labels to fallback classes so facet rows stay consistent.
    merged.loc[~merged["scene_type"].isin(SCENE_ORDER), "scene_type"] = "other"
    merged.loc[~merged["camera_to_object_distance"].isin(DISTANCE_ORDER), "camera_to_object_distance"] = "unknown"
    return merged


def _group_metrics(df: pd.DataFrame, group_col: str, group_order: List[str]) -> pd.DataFrame:
    rows = []
    max_entropy = np.log2(N_CQ_BINS)

    for group in group_order:
        bins = df.loc[df[group_col] == group, "cq_bin"]
        if bins.empty:
            rows.append({"group": group, "mean_norm": np.nan, "entropy_norm": np.nan, "n": 0})
            continue

        pmf = bins.value_counts(normalize=True).reindex(range(1, N_CQ_BINS + 1), fill_value=0.0)
        mean_score = float((pmf.index.to_numpy(dtype=float) * pmf.to_numpy()).sum())
        entropy = float(-(pmf[pmf > 0] * np.log2(pmf[pmf > 0])).sum())

        mean_norm = (mean_score - 1.0) / (N_CQ_BINS - 1.0)  # [0, 1]
        entropy_norm = entropy / max_entropy if max_entropy > 0 else 0.0  # [0, 1]

        rows.append(
            {
                "group": group,
                "mean_norm": mean_norm,
                "entropy_norm": entropy_norm,
                "n": int(len(bins)),
            }
        )

    return pd.DataFrame(rows)


def _group_bin_distribution(df: pd.DataFrame, group_col: str, group_order: List[str]) -> pd.DataFrame:
    rows = []
    for group in group_order:
        bins = df.loc[df[group_col] == group, "cq_bin"]
        pmf = bins.value_counts(normalize=True).reindex(range(1, N_CQ_BINS + 1), fill_value=0.0)
        counts = bins.value_counts().reindex(range(1, N_CQ_BINS + 1), fill_value=0)
        for cq_bin in range(1, N_CQ_BINS + 1):
            rows.append(
                {
                    "group": group,
                    "cq_bin": cq_bin,
                    "count": int(counts.loc[cq_bin]),
                    "percentage": float(pmf.loc[cq_bin] * 100.0),
                }
            )
    return pd.DataFrame(rows)


def _pretty_label(value: str) -> str:
    text = str(value).replace("_", " ")
    if text == "natural landscape":
        return "nature"
    return text


def _pretty_group_title(value: str) -> str:
    key = str(value).strip().lower()
    return GROUP_TITLE_MAP.get(key, _pretty_label(value).title())


def _shade_dataset_color(base_color: str, kind: str) -> str:
    rgb = np.array(mcolors.to_rgb(base_color), dtype=float)
    if kind == "mean":
        # Slightly darker for E[CQ]
        out = rgb * 0.82
    elif kind == "ent_edge":
        # Darker edge in the same hue family for H(CQ) markers.
        out = rgb * 0.62
    else:
        # Slightly lighter for H(CQ)
        out = rgb + (1.0 - rgb) * 0.35
    return mcolors.to_hex(np.clip(out, 0.0, 1.0))


def _plot_faceted(
    metric_tables: Dict[str, pd.DataFrame],
    group_col_title: str,
    out_path: Path,
) -> None:
    pretty_group = _pretty_group_title(group_col_title)
    dataset_names = list(metric_tables.keys())
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.8), sharex=True)
    axes = axes.flatten()

    for i, name in enumerate(dataset_names):
        ax = axes[i]
        table = metric_tables[name].copy()
        y = np.arange(len(table))
        has_data = table["n"] > 0
        missing = ~has_data

        if SHOW_CONNECTOR:
            # connector between two metrics for each category
            ax.hlines(
                y[has_data],
                table.loc[has_data, "mean_norm"],
                table.loc[has_data, "entropy_norm"],
                color=CONNECTOR_COLOR,
                linewidth=1.4,
                zorder=1,
            )
        ax.scatter(
            table.loc[has_data, "mean_norm"],
            y[has_data],
            color=MEAN_COLOR,
            s=MARKER_SIZE,
            zorder=3,
            label=MEAN_LABEL,
        )
        ax.scatter(
            table.loc[has_data, "entropy_norm"],
            y[has_data],
            color=ENT_COLOR,
            s=MARKER_SIZE,
            marker="^",
            zorder=3,
            label=ENTROPY_LABEL,
        )

        # Keep visual row consistency for groups with n=0.
        if missing.any():
            ax.scatter(
                np.zeros(int(missing.sum())),
                y[missing],
                facecolors="none",
                edgecolors=MISSING_COLOR,
                s=MARKER_SIZE,
                linewidths=1.2,
                zorder=2,
            )
            ax.scatter(
                np.zeros(int(missing.sum())),
                y[missing],
                facecolors="none",
                edgecolors=MISSING_COLOR,
                s=MARKER_SIZE,
                linewidths=1.2,
                marker="^",
                zorder=2,
            )

        ax.set_yticks(y)
        ax.set_yticklabels([_pretty_label(v) for v in table["group"]], fontsize=TICK_FONTSIZE, fontweight="bold")
        for lbl in ax.get_yticklabels():
            lbl.set_rotation(45)
            lbl.set_ha("right")
            lbl.set_va("center")
            lbl.set_rotation_mode("anchor")
        ax.invert_yaxis()
        ax.set_xlim(-0.02, 1.02)
        ax.grid(axis="x", color="#E6E6E6", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
        for lbl in ax.get_xticklabels():
            lbl.set_fontweight("bold")

        if i % 2 == 0:
            ax.set_ylabel(pretty_group, fontsize=LABEL_FONTSIZE, fontweight="bold")
        else:
            ax.set_ylabel("")

        if i >= 2:
            ax.set_xlabel("normalized score (0-1)", fontsize=LABEL_FONTSIZE, fontweight="bold")
        else:
            ax.set_xlabel("")

    for i in range(len(dataset_names), 4):
        axes[i].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles[:2],
        labels[:2],
        loc="upper left",
        bbox_to_anchor=(0.36, 0.965),
        ncol=2,
        frameon=True,
        fontsize=LEGEND_FONTSIZE,
        title="metric",
    )
    plt.setp(legend.get_title(), fontsize=LEGEND_TITLE_FONTSIZE, fontweight="bold")
    for txt in legend.get_texts():
        txt.set_fontweight("bold")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save_figure(fig, out_path)
    plt.close(fig)


def _plot_overlaid_single(
    metric_tables: Dict[str, pd.DataFrame],
    group_title: str,
    out_path: Path,
) -> None:
    dataset_names = list(metric_tables.keys())
    fig, ax = plt.subplots(figsize=(9.0, 5.2))

    # Spread datasets a bit more within each category row.
    if len(dataset_names) <= 1:
        ds_offsets = np.array([0.0])
    else:
        ds_offsets = np.linspace(-0.12, 0.12, len(dataset_names))

    first_table = next(iter(metric_tables.values()))
    categories = first_table["group"].tolist()
    # Tighten inter-category spacing to reduce overall figure height.
    y_base = np.arange(len(categories), dtype=float) * 0.65
    # Soft 0.2-bin background bands to improve x-interval readability.
    band_color = "#DCEAF7"
    band_alphas = [0.5, 0.4, 0.3, 0.2, 0.1]
    for i in range(5):
        x0 = 0.2 * i
        x1 = 0.2 * (i + 1)
        ax.axvspan(x0, x1, facecolor=band_color, alpha=band_alphas[i], zorder=0)

    for i, ds in enumerate(dataset_names):
        t = metric_tables[ds].copy()
        y = y_base + ds_offsets[i]
        has_data = t["n"] > 0
        missing = ~has_data
        ds_color = DATASET_COLORS.get(ds, "#4E79A7")
        mean_color = _shade_dataset_color(ds_color, "mean")
        ent_color = _shade_dataset_color(ds_color, "ent")
        ent_edge_color = _shade_dataset_color(ds_color, "ent_edge")

        # dataset hue is primary; metric encoded by point vs segment.
        e_x = t.loc[has_data, "mean_norm"].to_numpy(dtype=float)
        e_y = y[has_data]
        # E[CQ]: segment from x=0 to its value.
        ax.hlines(
            e_y,
            0.0,
            e_x,
            colors=mean_color,
            linewidth=5.6,
            alpha=POINT_ALPHA,
            zorder=2,
        )
        ax.vlines(
            e_x,
            e_y - 0.045,
            e_y + 0.045,
            colors=mean_color,
            linewidth=3.2,
            alpha=POINT_ALPHA,
            zorder=3,
        )
        # H[CQ]: point marker.
        entropy_points = ax.scatter(
            t.loc[has_data, "entropy_norm"],
            y[has_data],
            facecolors=ent_color,
            edgecolors=ent_edge_color,
            marker="o",
            s=74,
            linewidths=POINT_EDGE_WIDTH,
            alpha=POINT_ALPHA,
            zorder=3,
        )
        entropy_points.set_path_effects(
            [pe.Stroke(linewidth=POINT_HALO_WIDTH, foreground="white"), pe.Normal()]
        )

        if missing.any():
            missing_points = ax.scatter(
                np.zeros(int(missing.sum())),
                y[missing],
                facecolors="none",
                edgecolors=ent_edge_color,
                marker="o",
                s=74,
                linewidths=POINT_EDGE_WIDTH,
                alpha=POINT_ALPHA,
                zorder=2,
            )
            missing_points.set_path_effects(
                [pe.Stroke(linewidth=POINT_HALO_WIDTH, foreground="white"), pe.Normal()]
            )
            # no E segment for missing rows to avoid implying magnitude without data

    ax.set_yticks(y_base)
    ax.set_yticklabels([_pretty_label(v) for v in categories], fontsize=TICK_FONTSIZE, fontweight="bold")
    for lbl in ax.get_yticklabels():
        lbl.set_rotation(60)
        lbl.set_ha("right")
        lbl.set_va("center")
        lbl.set_rotation_mode("anchor")
    ax.invert_yaxis()
    ax.set_xlim(-0.005, 1.05)
    # Fill side margins so the banded background covers the full plotting width.
    x_left, x_right = ax.get_xlim()
    if x_left < 0:
        ax.axvspan(x_left, 0.0, facecolor=band_color, alpha=band_alphas[0], zorder=0)
    if x_right > 1:
        ax.axvspan(1.0, x_right, facecolor=band_color, alpha=band_alphas[-1], zorder=0)
    # Very light dashed guides for each dataset offset band in every category row.
    x0, x1 = ax.get_xlim()
    for off in ds_offsets:
        ax.hlines(
            y_base + off,
            xmin=x0,
            xmax=x1,
            colors="#D8D8D8",
            linestyles=(0, (2.0, 2.0)),
            linewidth=0.6,
            alpha=0.45,
            zorder=0,
        )
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
    for lbl in ax.get_xticklabels():
        lbl.set_fontweight("bold")
    ax.set_xlabel("")
    ax.set_ylabel(group_title, fontsize=LABEL_FONTSIZE, fontweight="bold", labelpad=2)

    ds_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor="white",
            markeredgecolor=DATASET_COLORS.get(ds, "#4E79A7"),
            markeredgewidth=1.5,
            color="none",
            linestyle="None",
            markersize=7,
            label=ds,
        )
        for ds in dataset_names
    ]
    metric_handles = [
        Line2D(
            [0, 1],
            [0, 0],
            color="#777777",
            linewidth=5.6,
            linestyle="-",
            label=MEAN_LABEL,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor="#555555",
            markeredgecolor="#444444",
            markeredgewidth=0.8,
            color="none",
            linestyle="None",
            markersize=7,
            label=ENTROPY_LABEL,
        ),
    ]
    legend1 = fig.legend(
        handles=ds_handles,
        title="Dataset",
        loc="upper center",
        bbox_to_anchor=(0.908, 0.5),
        ncol=1,
        frameon=True,
        fontsize=LEGEND_FONTSIZE,
        handletextpad=0.55,
        labelspacing=0.45,
        columnspacing=1.1,
        borderpad=0.45,
        borderaxespad=0.25,
        handlelength=1.4,
    )
    plt.setp(legend1.get_title(), fontsize=LEGEND_TITLE_FONTSIZE, fontweight="bold")
    for txt in legend1.get_texts():
        txt.set_fontweight("bold")

    legend2 = fig.legend(
        handles=metric_handles,
        title="Metric",
        loc="upper center",
        bbox_to_anchor=(0.9, 0.27),
        ncol=1,
        frameon=True,
        fontsize=LEGEND_FONTSIZE,
        handletextpad=0.55,
        labelspacing=0.45,
        columnspacing=1.1,
        borderpad=0.45,
        borderaxespad=0.25,
        handlelength=1.4,
    )
    plt.setp(legend2.get_title(), fontsize=LEGEND_TITLE_FONTSIZE, fontweight="bold")
    for txt in legend2.get_texts():
        txt.set_fontweight("bold")

    fig.tight_layout(rect=(0.002, 0.005, 0.998, 0.992))
    _save_figure(fig, out_path)
    plt.close(fig)


def main() -> int:
    _apply_ieee_font()

    parser = argparse.ArgumentParser(
        description="Plot faceted and merged Cleveland dots for CQ complexity metrics."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write plots.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_by_dataset: Dict[str, pd.DataFrame] = {}
    for dataset_name in DATASET_CONFIG:
        merged_by_dataset[dataset_name] = _load_and_merge(dataset_name)

    scene_metrics: Dict[str, pd.DataFrame] = {}
    dist_metrics: Dict[str, pd.DataFrame] = {}
    scene_distributions: List[pd.DataFrame] = []
    dist_distributions: List[pd.DataFrame] = []
    metric_rows: List[pd.DataFrame] = []
    for dataset_name, df in merged_by_dataset.items():
        scene_metrics[dataset_name] = _group_metrics(df, "scene_type", SCENE_ORDER)
        dist_metrics[dataset_name] = _group_metrics(df, "camera_to_object_distance", DISTANCE_ORDER)
        scene_metric_df = scene_metrics[dataset_name].copy()
        scene_metric_df.insert(0, "dataset", dataset_name)
        scene_metric_df.insert(1, "group_type", "scene_type")
        metric_rows.append(scene_metric_df)

        dist_metric_df = dist_metrics[dataset_name].copy()
        dist_metric_df.insert(0, "dataset", dataset_name)
        dist_metric_df.insert(1, "group_type", "camera_to_object_distance")
        metric_rows.append(dist_metric_df)

        scene_dist_df = _group_bin_distribution(df, "scene_type", SCENE_ORDER)
        scene_dist_df.insert(0, "dataset", dataset_name)
        scene_dist_df.insert(1, "group_type", "scene_type")
        scene_distributions.append(scene_dist_df)

        dist_dist_df = _group_bin_distribution(df, "camera_to_object_distance", DISTANCE_ORDER)
        dist_dist_df.insert(0, "dataset", dataset_name)
        dist_dist_df.insert(1, "group_type", "camera_to_object_distance")
        dist_distributions.append(dist_dist_df)

        sample_export = df[
            [
                "image_name",
                "scene_type",
                "camera_to_object_distance",
                "quantity_level",
                "clutter_level",
                "cq_bin",
            ]
        ].copy()
        sample_export.insert(0, "dataset", dataset_name)
        sample_export.to_csv(out_dir / f"cq_sample_assignments_{dataset_name.lower().replace(' ', '_').replace('-', '_')}.csv", index=False, encoding="utf-8-sig")

    pd.concat(metric_rows, ignore_index=True).to_csv(
        out_dir / "cq_group_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(scene_distributions + dist_distributions, ignore_index=True).to_csv(
        out_dir / "cq_group_distributions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    _plot_faceted(
        scene_metrics,
        "scene_type",
        out_dir / "faceted_cleveland_scene_cq_metrics.png",
    )
    _plot_faceted(
        dist_metrics,
        "camera_to_object_distance",
        out_dir / "faceted_cleveland_distance_cq_metrics.png",
    )
    _plot_overlaid_single(
        scene_metrics,
        "Scene Type",
        out_dir / "merged_cleveland_scene_cq_metrics.png",
    )
    _plot_overlaid_single(
        dist_metrics,
        "Camera-to-Object Distance",
        out_dir / "merged_cleveland_distance_cq_metrics.png",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
