# -*- coding: utf-8 -*-
"""
Visualize quantity/clutter level distributions across multiple datasets.
"""

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors as mcolors
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

ROOT_DIR = Path(__file__).resolve().parents[3]

DEFAULT_INPUTS = {
    "spaq": ROOT_DIR / "result" / "complexity_results_spaq.csv",
    "koniq10k": ROOT_DIR / "result" / "complexity_results_koniq10k.csv",
    "live": ROOT_DIR / "result" / "complexity_results_live.csv",
    "cid2013": ROOT_DIR / "result" / "complexity_results_cid2013.csv",
}
OUTPUT_DIR = ROOT_DIR / "result" / "paper_figures" / "complexity"

LEVEL_ORDER = ["low", "medium", "high"]
Y_AXIS_LEVEL_ORDER = ["high", "medium", "low"]
BAR_COLORS = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759"]

# Font configuration: bar charts
BAR_TITLE_FONTSIZE = 22
BAR_LABEL_FONTSIZE = 20
BAR_TICK_FONTSIZE = 18
BAR_LEGEND_FONTSIZE = 13
BAR_LEGEND_TITLE_FONTSIZE = 14
BAR_VALUE_LABEL_FONTSIZE = 14
BAR_MIRRORED_VALUE_LABEL_FONTSIZE = 14
BAR_SIDE_LABEL_FONTSIZE = 19

# Font configuration: heatmaps
HEATMAP_TITLE_FONTSIZE = 20
HEATMAP_SUPTITLE_FONTSIZE = 22
HEATMAP_LABEL_FONTSIZE = 18
HEATMAP_TICK_FONTSIZE = 17
HEATMAP_COLORBAR_LABEL_FONTSIZE = 17
HEATMAP_COLORBAR_TICK_FONTSIZE = 16
HEATMAP_ANNOT_FONTSIZE = 14

DATASET_DISPLAY_MAP = {
    "spaq": "SPAQ",
    "koniq10k": "KonIQ-10K",
    "koniq-10k": "KonIQ-10K",
    "live": "LIVE Wild",
    "livewild": "LIVE Wild",
    "cid2013": "CID2013",
}


class _HandlerBiColorPatch(HandlerBase):
    """Legend handler that draws a left-right bi-color patch."""

    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        left_color, right_color = orig_handle
        left = Rectangle(
            (xdescent, ydescent),
            width / 2.0,
            height,
            facecolor=left_color,
            edgecolor="#2A2A2A",
            linewidth=0.4,
            transform=trans,
        )
        right = Rectangle(
            (xdescent + width / 2.0, ydescent),
            width / 2.0,
            height,
            facecolor=right_color,
            edgecolor="#2A2A2A",
            linewidth=0.4,
            transform=trans,
        )
        return [left, right]


def _pretty_label(value: str) -> str:
    return str(value).replace("_", " ")


def _dataset_display_name(value: str) -> str:
    key = str(value).strip().lower()
    return DATASET_DISPLAY_MAP.get(key, value)


def _apply_ieee_font() -> None:
    # IEEE Transactions manuscripts conventionally use Times-style serif fonts.
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"]


def _save_figure(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight", pad_inches=0.02)


def _load_frames(inputs: List[Path], dataset_names: List[str]) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for name, path in zip(dataset_names, inputs):
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        df = pd.read_csv(path)
        if "quantity_level" not in df.columns or "clutter_level" not in df.columns:
            raise ValueError(f"CSV missing required columns in {path}: quantity_level, clutter_level")
        frames[name] = df
    return frames


def _grouped_percent_bar(
    value_tables: Dict[str, pd.Series],
    x_order: List[str],
    title: str,
    xlabel: str,
    out_path: Path,
    label_fontsize: int = BAR_VALUE_LABEL_FONTSIZE,
    show_zero_labels: bool = False,
) -> None:
    dataset_names = list(value_tables.keys())

    count_table = pd.DataFrame(index=dataset_names, columns=x_order, dtype=float).fillna(0)
    for name in dataset_names:
        count_table.loc[name] = value_tables[name].reindex(x_order, fill_value=0)

    pct_table = count_table.div(count_table.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100.0

    x = np.arange(len(x_order))
    width = 0.8 / max(len(dataset_names), 1)
    colors = [BAR_COLORS[i % len(BAR_COLORS)] for i in range(len(dataset_names))]

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    for i, name in enumerate(dataset_names):
        offsets = x - 0.4 + (i + 0.5) * width
        bars = ax.bar(
            offsets,
            pct_table.loc[name],
            width=width,
            color=colors[i],
            label=_dataset_display_name(name),
        )
        for bar, count in zip(bars, count_table.loc[name]):
            if count <= 0 and not show_zero_labels:
                continue
            y_base = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_base + 0.35 + i * 0.12,
                f"{int(count)}",
                ha="center",
                va="bottom",
                fontsize=label_fontsize,
                fontweight="bold",
            )

    ax.set_xlabel(_pretty_label(xlabel), fontweight="bold", fontsize=BAR_LABEL_FONTSIZE)
    ax.set_ylabel("percentage (%)", fontweight="bold", fontsize=BAR_LABEL_FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [ _pretty_label(v) for v in x_order ],
        rotation=30,
        ha="center",
        fontsize=BAR_TICK_FONTSIZE,
        fontweight="bold",
    )
    ax.tick_params(axis="y", labelsize=BAR_TICK_FONTSIZE)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ymax = float(pct_table.to_numpy().max())
    headroom = 14.0 if len(x_order) > 3 else 22.0
    ax.set_ylim(0, min(105.0, ymax + headroom))
    legend = ax.legend(title="dataset", loc="upper right", frameon=True, fontsize=BAR_LEGEND_FONTSIZE)
    plt.setp(legend.get_title(), fontsize=BAR_LEGEND_TITLE_FONTSIZE, fontweight="bold")
    for txt in legend.get_texts():
        txt.set_fontweight("bold")
    fig.tight_layout(pad=0.2)
    _save_figure(fig, out_path)
    plt.close(fig)


def _mirrored_quantity_clutter_bar(
    quantity_tables: Dict[str, pd.Series],
    clutter_tables: Dict[str, pd.Series],
    x_order: List[str],
    out_path: Path,
    label_fontsize: int = BAR_MIRRORED_VALUE_LABEL_FONTSIZE,
    show_zero_labels: bool = True,
) -> None:
    dataset_names = list(quantity_tables.keys())

    q_count = pd.DataFrame(index=dataset_names, columns=x_order, dtype=float).fillna(0)
    c_count = pd.DataFrame(index=dataset_names, columns=x_order, dtype=float).fillna(0)
    for name in dataset_names:
        q_count.loc[name] = quantity_tables[name].reindex(x_order, fill_value=0)
        c_count.loc[name] = clutter_tables[name].reindex(x_order, fill_value=0)

    q_pct = q_count.div(q_count.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100.0
    c_pct = c_count.div(c_count.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100.0

    rows = []
    q_vals = []
    c_vals = []
    q_counts = []
    c_counts = []
    for level in x_order:
        for name in dataset_names:
            rows.append(f"{_pretty_label(level)} | {_dataset_display_name(name)}")
            q_vals.append(float(q_pct.loc[name, level]))
            c_vals.append(float(c_pct.loc[name, level]))
            q_counts.append(int(q_count.loc[name, level]))
            c_counts.append(int(c_count.loc[name, level]))

    y = np.arange(len(rows), dtype=float)
    q_vals = np.array(q_vals, dtype=float)
    c_vals = np.array(c_vals, dtype=float)
    q_counts = np.array(q_counts, dtype=int)
    c_counts = np.array(c_counts, dtype=int)

    # Keep dataset identity with distinct colors; left/right use gentle hue-shift variants.
    base_colors = [BAR_COLORS[i % len(BAR_COLORS)] for i in range(len(dataset_names))]
    cool_tint = np.array(mcolors.to_rgb("#8EC5FF"), dtype=float)  # slight cyan-blue shift for quantity
    warm_tint = np.array(mcolors.to_rgb("#FFB37A"), dtype=float)  # slight orange shift for clutter

    q_color_map = {}
    c_color_map = {}
    for i, name in enumerate(dataset_names):
        base = np.array(mcolors.to_rgb(base_colors[i]), dtype=float)
        # quantity: mildly cooler + slightly lighter
        q_rgb = base * 0.80 + cool_tint * 0.20
        q_rgb = q_rgb + (1.0 - q_rgb) * 0.06
        # clutter: mildly warmer + slightly darker
        c_rgb = base * 0.84 + warm_tint * 0.16
        c_rgb = c_rgb * 0.90
        q_color_map[name] = mcolors.to_hex(np.clip(q_rgb, 0.0, 1.0))
        c_color_map[name] = mcolors.to_hex(np.clip(c_rgb, 0.0, 1.0))

    fig, ax = plt.subplots(figsize=(9.6, 6.5))

    q_colors = [q_color_map[name] for _lvl in x_order for name in dataset_names]
    c_colors = [c_color_map[name] for _lvl in x_order for name in dataset_names]
    q_bars = ax.barh(y, -q_vals, height=0.72, color=q_colors, edgecolor="#2A2A2A", linewidth=0.4)
    c_bars = ax.barh(y, c_vals, height=0.72, color=c_colors, edgecolor="#2A2A2A", linewidth=0.4)

    # Center line.
    ax.axvline(0, color="#C33", linewidth=1.2, zorder=3)

    # Level block separators and labels (inside left side, vertical).
    n_ds = max(1, len(dataset_names))
    for i, level in enumerate(x_order):
        start = i * n_ds
        end = start + n_ds - 1
        center = (start + end) / 2.0
        ax.text(
            0.012,
            center,
            _pretty_label(level),
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            rotation=90,
            fontsize=BAR_TICK_FONTSIZE,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.3, pad=0.8),
        )
        if i < len(x_order) - 1:
            ax.axhline(end + 0.5, color="#E2E2E2", linewidth=0.9, zorder=1)

    # Count labels at bar ends.
    for bar, cnt in zip(q_bars, q_counts):
        if cnt <= 0 and not show_zero_labels:
            continue
        x_end = bar.get_width()  # negative
        ax.text(x_end - 0.9, bar.get_y() + bar.get_height() / 2, f"{cnt}", ha="right", va="center",
                fontsize=label_fontsize, fontweight="bold")
    for bar, cnt in zip(c_bars, c_counts):
        if cnt <= 0 and not show_zero_labels:
            continue
        x_end = bar.get_width()  # positive
        ax.text(x_end + 0.9, bar.get_y() + bar.get_height() / 2, f"{cnt}", ha="left", va="center",
                fontsize=label_fontsize, fontweight="bold")

    xmax = max(float(np.max(q_vals)), float(np.max(c_vals)))
    xlim = min(100.0, max(15.0, xmax + 8.0))
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{abs(v):.0f}"))
    ax.grid(axis="x", color="#ECECEC", linewidth=0.8)
    ax.set_axisbelow(True)

    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.tick_params(axis="y", labelleft=False, left=False)
    ax.tick_params(axis="x", labelsize=BAR_TICK_FONTSIZE)
    for lbl in ax.get_xticklabels():
        lbl.set_fontweight("bold")

    ax.set_xlabel("percentage (%)", fontweight="bold", fontsize=BAR_LABEL_FONTSIZE)
    ax.set_ylabel("dataset (grouped by level)", fontweight="bold", fontsize=BAR_LABEL_FONTSIZE)

    legend_handles = [(q_color_map[name], c_color_map[name]) for name in dataset_names]
    legend_labels = [_dataset_display_name(name) for name in dataset_names]
    legend = ax.legend(
        legend_handles,
        legend_labels,
        title="dataset",
        loc="lower left",
        bbox_to_anchor=(0.8, 0.45),
        frameon=True,
        fontsize=BAR_LEGEND_FONTSIZE,
        handler_map={tuple: _HandlerBiColorPatch()},
    )
    plt.setp(legend.get_title(), fontsize=BAR_LEGEND_TITLE_FONTSIZE, fontweight="bold")
    for txt in legend.get_texts():
        txt.set_fontweight("bold")

    ax.text(
        0.12,
        0.965,
        "Quantity",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=BAR_SIDE_LABEL_FONTSIZE,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.35, pad=0.6),
    )
    ax.text(
        0.88,
        0.965,
        "Clutter",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=BAR_SIDE_LABEL_FONTSIZE,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.35, pad=0.6),
    )

    fig.tight_layout(rect=(0, 0, 1, 0.995), pad=0.2)
    _save_figure(fig, out_path)
    plt.close(fig)


def _joint_9level_heatmap_2x2(
    combo_tables: Dict[str, pd.Series],
    combo_order: List[str],
    out_path: Path,
) -> None:
    dataset_names = list(combo_tables.keys())
    y_row_indices = [LEVEL_ORDER.index(level) for level in Y_AXIS_LEVEL_ORDER]
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 9.6 * 0.75), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.03, wspace=0.035, hspace=0.07)

    pct_mats = {}
    count_mats = {}
    vmax = 0.0
    for name in dataset_names:
        counts = combo_tables[name].reindex(combo_order, fill_value=0).to_numpy(dtype=float).reshape(3, 3)
        counts = counts[y_row_indices, :]
        total = counts.sum()
        pct = (counts / total * 100.0) if total > 0 else counts
        pct_mats[name] = pct
        count_mats[name] = counts
        vmax = max(vmax, float(np.max(pct)))
    vmax = max(vmax, 1.0)

    cmap = "YlGnBu"
    last_mappable = None
    flat_axes = axes.flatten()

    for idx, ax in enumerate(flat_axes):
        if idx >= len(dataset_names):
            ax.axis("off")
            continue
        name = dataset_names[idx]
        pct = pct_mats[name]
        counts = count_mats[name]
        annot = np.array(
            [[f"{pct[r, c]:.1f}%\n({int(counts[r, c])})" for c in range(3)] for r in range(3)],
            dtype=object,
        )
        hm = sns.heatmap(
            pct,
            ax=ax,
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            cbar=False,
            square=False,
            linewidths=0.8,
            linecolor="white",
            annot=annot,
            fmt="",
            annot_kws={"fontsize": HEATMAP_ANNOT_FONTSIZE, "fontweight": "bold"},
            xticklabels=[_pretty_label(v) for v in LEVEL_ORDER],
            yticklabels=[_pretty_label(v) for v in Y_AXIS_LEVEL_ORDER],
        )
        last_mappable = hm.collections[0]
        ax.set_title(_dataset_display_name(name), fontweight="bold", fontsize=HEATMAP_TITLE_FONTSIZE)
        ax.set_aspect("auto")
        row, col = divmod(idx, 2)
        if row == 1:
            ax.set_xlabel("clutter level", fontweight="bold", fontsize=HEATMAP_LABEL_FONTSIZE)
            ax.tick_params(axis="x", rotation=0, labelsize=HEATMAP_TICK_FONTSIZE)
        else:
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=False, bottom=False)
        if col == 0:
            ax.set_ylabel("quantity level", fontweight="bold", fontsize=HEATMAP_LABEL_FONTSIZE)
            ax.tick_params(axis="y", labelsize=HEATMAP_TICK_FONTSIZE, pad=16)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False, left=False)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")
        for lbl in ax.get_yticklabels():
            lbl.set_rotation(90)
            lbl.set_ha("center")
            lbl.set_va("center")
            lbl.set_rotation_mode("default")

    cbar = fig.colorbar(last_mappable, ax=flat_axes.tolist(), fraction=0.03, pad=0.012)
    cbar.set_label("percentage (%)", fontweight="bold", fontsize=HEATMAP_COLORBAR_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=HEATMAP_COLORBAR_TICK_FONTSIZE)
    _save_figure(fig, out_path)
    plt.close(fig)


def main() -> None:
    _apply_ieee_font()

    parser = argparse.ArgumentParser(description="Plot multi-dataset complexity distributions.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[str(v) for v in DEFAULT_INPUTS.values()],
        help="Paths to complexity CSVs.",
    )
    parser.add_argument(
        "--dataset-names",
        nargs="*",
        default=list(DEFAULT_INPUTS.keys()),
        help="Display names for datasets. Must match number of --inputs.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory to write plots.",
    )
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.inputs]
    dataset_names = args.dataset_names
    if len(input_paths) != len(dataset_names):
        raise ValueError("--dataset-names length must equal --inputs length.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = _load_frames(input_paths, dataset_names)

    quantity_tables: Dict[str, pd.Series] = {}
    clutter_tables: Dict[str, pd.Series] = {}
    combo_tables: Dict[str, pd.Series] = {}

    combo_order = [f"{q}_{c}" for q in LEVEL_ORDER for c in LEVEL_ORDER]

    for name, df in frames.items():
        quantity = df["quantity_level"].astype(str).str.lower().value_counts().reindex(LEVEL_ORDER, fill_value=0)
        clutter = df["clutter_level"].astype(str).str.lower().value_counts().reindex(LEVEL_ORDER, fill_value=0)

        q = df["quantity_level"].astype(str).str.lower()
        c = df["clutter_level"].astype(str).str.lower()
        combo = (q + "_" + c).value_counts().reindex(combo_order, fill_value=0)

        quantity_tables[name] = quantity
        clutter_tables[name] = clutter
        combo_tables[name] = combo

    _mirrored_quantity_clutter_bar(
        quantity_tables,
        clutter_tables,
        Y_AXIS_LEVEL_ORDER,
        out_dir / "quantity_clutter_mirrored_distribution_all_datasets.png",
        show_zero_labels=True,
    )
    _joint_9level_heatmap_2x2(
        combo_tables,
        combo_order,
        out_dir / "quantity_clutter_9_levels_all_datasets.png",
    )

    summary_rows = []
    for name in dataset_names:
        for level in LEVEL_ORDER:
            summary_rows.append(
                {
                    "dataset": name,
                    "metric": "quantity_level",
                    "level": level,
                    "count": int(quantity_tables[name][level]),
                }
            )
            summary_rows.append(
                {
                    "dataset": name,
                    "metric": "clutter_level",
                    "level": level,
                    "count": int(clutter_tables[name][level]),
                }
            )
        for combo in combo_order:
            summary_rows.append(
                {
                    "dataset": name,
                    "metric": "quantity_clutter",
                    "level": combo,
                    "count": int(combo_tables[name][combo]),
                }
            )

    pd.DataFrame(summary_rows).to_csv(out_dir / "complexity_counts_all_datasets.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
