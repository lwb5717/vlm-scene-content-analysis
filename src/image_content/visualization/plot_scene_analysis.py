import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors

SCENE_TYPE_ORDER = ["natural_landscape", "urban", "indoor", "portrait", "other"]
LIGHTING_TYPE_ORDER = ["daylight", "artificial", "mixed", "unknown"]
LUMINANCE_LEVEL_ORDER = ["high", "medium", "low"]
CAMERA_DISTANCE_ORDER = ["close", "medium", "far", "unknown"]
MEMORY_COLOR_ORDER = [
    "sky_blue",
    "grass_foliage_green",
    "human_skin_tone",
    "water_blue",
    "snow_cloud_white",
    "none",
]

DEFAULT_DATASET_INPUTS = {
    "SPAQ": "result/scene_analysis_results_spaq.csv",
    "Koniq-10K": "result/scene_analysis_results_koniq10k.csv",
    "LIVE Wild": "result/scene_analysis_results_livewild.csv",
    "CID2013": "result/scene_analysis_results_cid2013.csv",
}

BAR_TITLE_FONTSIZE = 22
BAR_LABEL_FONTSIZE = 21
BAR_TICK_FONTSIZE = 20
BAR_LEGEND_FONTSIZE = 14
BAR_LEGEND_TITLE_FONTSIZE = 16
BAR_VALUE_FONTSIZE = 9

HEATMAP_TITLE_FONTSIZE = 22
HEATMAP_LABEL_FONTSIZE = 21
HEATMAP_TICK_FONTSIZE = 20
HEATMAP_COLORBAR_TICK_FONTSIZE = 16
HEATMAP_COLORBAR_LABEL_FONTSIZE = 18

BAR_DATASET_COLORS = ["#5B7DB1", "#D97A5D", "#6DAA9F", "#C17BAA"]
OVERVIEW_BAR_COLORS = ["#4E79A7", "#F28E2B", "#59A14F", "#B07AA1"]
STACKED_CATEGORY_COLORS = ["#1F6F8B", "#3BA99C", "#F2B134", "#E07A5F", "#81B29A", "#6A4C93"]
HEATMAP_CMAP = "YlGnBu"
HEATMAP_X_TICK_ROTATION = 45
HEATMAP_Y_TICK_ROTATION = 45
HEATMAP_ANNOT_FONTSIZE_MIN = 8
HEATMAP_ANNOT_FONTSIZE_MAX = 14
HEATMAP_TAIL_LABELS = {"other", "none", "unknown"}
HEATMAP_MEMORY_SHORT_LABELS = {
    "sky_blue": "sky blue",
    "grass_foliage_green": "green",
    "human_skin_tone": "skin tone",
    "water_blue": "water blue",
    "snow_cloud_white": "white",
    "none": "none",
}
BAR_SHORT_LABELS = {
    "natural_landscape": "nature",
    "daylight": "day",
    "sky_blue": "sky",
    "grass_foliage_green": "green",
    "human_skin_tone": "skin",
    "water_blue": "water",
    "snow_cloud_white": "white",
}


def _pretty_label(value: object) -> str:
    return str(value).replace("_", " ")


def _heatmap_label(value: object) -> str:
    key = str(value).strip().lower()
    if key in BAR_SHORT_LABELS:
        return BAR_SHORT_LABELS[key]
    if key in HEATMAP_MEMORY_SHORT_LABELS:
        return HEATMAP_MEMORY_SHORT_LABELS[key]
    return _pretty_label(value)


def _bar_label(value: object) -> str:
    key = str(value).strip().lower()
    if key in BAR_SHORT_LABELS:
        return BAR_SHORT_LABELS[key]
    return _pretty_label(value)


def _sorted_for_heatmap(labels: List[object]) -> List[object]:
    head = [v for v in labels if str(v).strip().lower() not in HEATMAP_TAIL_LABELS]
    tail = [v for v in labels if str(v).strip().lower() in HEATMAP_TAIL_LABELS]
    head_sorted = sorted(head, key=lambda x: (len(_heatmap_label(x)), _heatmap_label(x)))
    return head_sorted + tail


def _apply_ieee_font() -> None:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"]


def _auto_heatmap_annot_fontsize(n_rows: int, n_cols: int) -> int:
    # More cells -> smaller annotations; fewer cells -> larger annotations.
    n_cells = max(1, n_rows * n_cols)
    size = int(round(20 - 0.35 * n_cells))
    return max(HEATMAP_ANNOT_FONTSIZE_MIN, min(HEATMAP_ANNOT_FONTSIZE_MAX, size))


def _save_figure(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight", pad_inches=0.02)


def _explode_memory_color(df: pd.DataFrame) -> pd.DataFrame:
    mem = df["memory_color"].fillna("").astype(str)
    mem = mem.str.split(";")
    exploded = df.copy()
    exploded["memory_color"] = mem
    exploded = exploded.explode("memory_color")
    exploded["memory_color"] = exploded["memory_color"].str.strip()
    exploded = exploded[exploded["memory_color"] != ""]
    return exploded


def _fixed_order(counts: pd.Series, order: List[str], tail_labels: Tuple[str, ...] = ()) -> pd.Series:
    tail_set = set(tail_labels)
    preferred_head = [label for label in order if label not in tail_set]
    extras = [label for label in counts.index if label not in order and label not in tail_set]
    tail = [label for label in tail_labels if label in order or label in counts.index]
    final_order = preferred_head + extras + tail
    return counts.reindex(final_order, fill_value=0)


def _grouped_bar_percent_with_count_labels(
    datasets: Dict[str, pd.DataFrame],
    column: str,
    title: str,
    xlabel: str,
    out_path: Path,
    order: Optional[List[str]] = None,
    tail_labels: Tuple[str, ...] = (),
) -> None:
    dataset_names = list(datasets.keys())
    counts_by_dataset: Dict[str, pd.Series] = {}
    all_counts = pd.Series(dtype=float)

    for name, df in datasets.items():
        values = df[column].fillna("unknown").astype(str).str.strip()
        values = values.replace("", "unknown")
        counts = values.value_counts()
        counts_by_dataset[name] = counts
        all_counts = all_counts.add(counts, fill_value=0)

    if order:
        all_counts = _fixed_order(all_counts, order, tail_labels=tail_labels)
        categories = list(all_counts.index)
    else:
        categories = list(all_counts.sort_values(ascending=False).index)

    count_table = pd.DataFrame(index=dataset_names, columns=categories, dtype=float).fillna(0)
    for name in dataset_names:
        count_table.loc[name] = counts_by_dataset[name].reindex(categories, fill_value=0)

    totals = count_table.sum(axis=1).replace(0, np.nan)
    pct_table = count_table.div(totals, axis=0).fillna(0.0) * 100.0

    x = np.arange(len(categories))
    width = 0.8 / max(len(dataset_names), 1)
    palette = [BAR_DATASET_COLORS[i % len(BAR_DATASET_COLORS)] for i in range(len(dataset_names))]

    fig, ax = plt.subplots(figsize=(max(10, 1.3 * len(categories)), 3.6))
    for i, name in enumerate(dataset_names):
        offsets = x - 0.4 + (i + 0.5) * width
        bars = ax.bar(offsets, pct_table.loc[name], width=width, color=palette[i], label=name)
        for bar, count in zip(bars, count_table.loc[name]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{int(count)}",
                ha="center",
                va="bottom",
                fontsize=BAR_VALUE_FONTSIZE,
                fontweight="bold",
                rotation=0,
            )

    ax.set_xlabel("")
    ax.set_ylabel("percentage (%)", fontsize=BAR_LABEL_FONTSIZE, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_bar_label(v) for v in categories],
        rotation=0,
        ha="center",
        fontsize=BAR_TICK_FONTSIZE,
        fontweight="bold",
    )
    ax.tick_params(axis="y", labelsize=BAR_TICK_FONTSIZE)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.set_ylim(0, 80.0)
    legend = ax.legend(title="dataset", loc="upper right", frameon=True, fontsize=BAR_LEGEND_FONTSIZE)
    plt.setp(legend.get_title(), fontsize=BAR_LEGEND_TITLE_FONTSIZE, fontweight="bold")
    for txt in legend.get_texts():
        txt.set_fontweight("bold")
    fig.tight_layout(pad=0.2)
    _save_figure(fig, out_path)
    plt.close(fig)


def _build_pct_table(
    datasets: Dict[str, pd.DataFrame],
    column: str,
    order: Optional[List[str]] = None,
    tail_labels: Tuple[str, ...] = (),
) -> pd.DataFrame:
    dataset_names = list(datasets.keys())
    counts_by_dataset: Dict[str, pd.Series] = {}
    all_counts = pd.Series(dtype=float)

    for name, df in datasets.items():
        values = df[column].fillna("unknown").astype(str).str.strip()
        values = values.replace("", "unknown")
        counts = values.value_counts()
        counts_by_dataset[name] = counts
        all_counts = all_counts.add(counts, fill_value=0)

    if order:
        all_counts = _fixed_order(all_counts, order, tail_labels=tail_labels)
        categories = list(all_counts.index)
    else:
        categories = list(all_counts.sort_values(ascending=False).index)

    count_table = pd.DataFrame(index=dataset_names, columns=categories, dtype=float).fillna(0)
    for name in dataset_names:
        count_table.loc[name] = counts_by_dataset[name].reindex(categories, fill_value=0)

    pct_table = count_table.div(count_table.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) * 100.0
    return pct_table


def _cleveland_dot_overview(
    datasets: Dict[str, pd.DataFrame],
    exploded_datasets: Dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    groups = [
        ("Scene Type", datasets, "scene_type", SCENE_TYPE_ORDER, ("other",)),
        ("Lighting Type", datasets, "lighting_type", LIGHTING_TYPE_ORDER, ("unknown",)),
        ("Luminance Level", datasets, "luminance_level", LUMINANCE_LEVEL_ORDER, ()),
        ("Camera Distance", datasets, "camera_to_object_distance", CAMERA_DISTANCE_ORDER, ("unknown",)),
        ("Memory Color", exploded_datasets, "memory_color", MEMORY_COLOR_ORDER, ("none",)),
    ]

    dataset_names = list(datasets.keys())
    dataset_colors = [OVERVIEW_BAR_COLORS[i % len(OVERVIEW_BAR_COLORS)] for i in range(len(dataset_names))]

    row_keys: List[Tuple[str, str]] = []
    row_values: Dict[Tuple[str, str], pd.Series] = {}
    group_centers: List[Tuple[str, float]] = []
    group_bounds: List[float] = []

    y_cursor = 0.0
    gap = 1.2
    for group_name, source, col, order, tail in groups:
        pct_table = _build_pct_table(source, col, order=order, tail_labels=tail)
        cats = list(pct_table.columns)
        ys = []
        for cat in cats:
            key = (group_name, cat)
            row_keys.append(key)
            row_values[key] = pct_table[cat]
            ys.append(y_cursor)
            y_cursor += 1.0
        group_centers.append((group_name, float(np.mean(ys))))
        group_bounds.append(y_cursor - 0.5)
        y_cursor += gap

    y_positions = np.arange(len(row_keys), dtype=float)
    key_to_y = {k: y_positions[i] for i, k in enumerate(row_keys)}

    fig, ax = plt.subplots(figsize=(14.5, 8.8))

    # faint separators between attribute groups
    cursor_idx = 0
    for group_name, source, col, order, tail in groups[:-1]:
        cursor_idx += len(order)
        sep_y = cursor_idx - 0.5
        ax.axhline(sep_y, color="#DDDDDD", linewidth=1.0, zorder=0)

    # connector lines (min-max per category) + dataset points
    for key in row_keys:
        y = key_to_y[key]
        vals = row_values[key].reindex(dataset_names).to_numpy(dtype=float)
        ax.hlines(y, np.min(vals), np.max(vals), color="#C8C8C8", linewidth=1.6, zorder=1)
        for j, name in enumerate(dataset_names):
            ax.scatter(
                vals[j],
                y,
                s=62,
                color=dataset_colors[j],
                edgecolor="white",
                linewidth=0.6,
                label=name if key == row_keys[0] else None,
                zorder=3,
            )

    # y tick labels are all categories; group labels shown on the left margin.
    ax.set_yticks(y_positions)
    ax.set_yticklabels([_bar_label(cat) for _, cat in row_keys], fontsize=BAR_TICK_FONTSIZE, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("percentage (%)", fontsize=BAR_LABEL_FONTSIZE, fontweight="bold")
    ax.set_ylabel("category (grouped)", fontsize=BAR_LABEL_FONTSIZE, fontweight="bold")
    ax.tick_params(axis="x", labelsize=BAR_TICK_FONTSIZE)
    for lbl in ax.get_xticklabels():
        lbl.set_fontweight("bold")
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.9)
    ax.set_axisbelow(True)
    ax.set_xlim(left=0)

    # group headers in left margin
    group_start = 0
    for group_name, source, col, order, tail in groups:
        group_len = len(order)
        group_y = np.mean(np.arange(group_start, group_start + group_len))
        ax.text(
            -0.06,
            group_y,
            group_name,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=BAR_LABEL_FONTSIZE - 1,
            fontweight="bold",
        )
        group_start += group_len

    legend = ax.legend(title="dataset", loc="upper right", frameon=True, fontsize=BAR_LEGEND_FONTSIZE)
    plt.setp(legend.get_title(), fontsize=BAR_LEGEND_TITLE_FONTSIZE, fontweight="bold")
    for txt in legend.get_texts():
        txt.set_fontweight("bold")

    fig.tight_layout(rect=(0.08, 0.03, 0.98, 0.995))
    _save_figure(fig, out_path)
    plt.close(fig)


def _bar_overview_all_attributes(
    datasets: Dict[str, pd.DataFrame],
    exploded_datasets: Dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    groups = [
        ("Scene Type", datasets, "scene_type", SCENE_TYPE_ORDER, ("other",)),
        ("Lighting Type", datasets, "lighting_type", LIGHTING_TYPE_ORDER, ("unknown",)),
        ("Luminance Level", datasets, "luminance_level", LUMINANCE_LEVEL_ORDER, ()),
        ("Camera Distance", datasets, "camera_to_object_distance", CAMERA_DISTANCE_ORDER, ("unknown",)),
        ("Memory Color", exploded_datasets, "memory_color", MEMORY_COLOR_ORDER, ("none",)),
    ]

    dataset_names = list(datasets.keys())
    dataset_colors = [BAR_DATASET_COLORS[i % len(BAR_DATASET_COLORS)] for i in range(len(dataset_names))]

    # Build stacked row plan with blank gaps between attribute groups.
    row_labels: List[str] = []
    row_values: List[np.ndarray] = []
    row_y: List[float] = []
    row_group_bounds: List[Tuple[float, float]] = []
    row_group_centers: List[Tuple[str, float]] = []
    y_cursor = 0.0
    gap = 1.0

    for group_name, source, col, order, tail in groups:
        pct_table = _build_pct_table(source, col, order=order, tail_labels=tail)
        group_start = y_cursor
        group_row_ys: List[float] = []
        for cat in pct_table.columns:
            row_labels.append(_bar_label(cat))
            row_values.append(pct_table[cat].reindex(dataset_names).to_numpy(dtype=float))
            row_y.append(y_cursor)
            group_row_ys.append(y_cursor)
            y_cursor += 1.0
        group_end = y_cursor - 1.0
        row_group_centers.append((group_name, (group_start + group_end) / 2.0))
        row_group_bounds.append((group_row_ys[0] - 0.5, group_row_ys[-1] + 0.5))
        y_cursor += gap

    n_rows = len(row_labels)
    y_base = np.array(row_y, dtype=float)
    bar_h = 0.8 / max(len(dataset_names), 1)

    fig_h = max(10.6, 0.52 * n_rows + 3.1)
    fig, ax = plt.subplots(figsize=(12.5, fig_h))

    vals_matrix = np.vstack(row_values) if row_values else np.zeros((0, len(dataset_names)))
    for i, ds in enumerate(dataset_names):
        y_offsets = y_base - 0.4 + (i + 0.5) * bar_h
        widths = vals_matrix[:, i] if len(vals_matrix) else np.array([])
        ax.barh(
            y_offsets,
            widths,
            height=bar_h,
            color=dataset_colors[i],
            alpha=0.88,
            edgecolor="#222222",
            linewidth=0.45,
            label=ds,
            zorder=2,
        )

    # Group separators/titles.
    for idx, (y0, y1) in enumerate(row_group_bounds):
        if idx < len(row_group_bounds) - 1:
            boundary = (y1 + row_group_bounds[idx + 1][0]) / 2.0
            ax.axhline(boundary, color="#BCBCBC", linewidth=1.2, zorder=1)
    for group_name, center_y in row_group_centers:
        ax.text(
            0.985,
            center_y,
            group_name,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=BAR_LABEL_FONTSIZE - 1,
            fontweight="bold",
            rotation=90,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.35, pad=0.8),
        )

    ax.set_yticks(y_base)
    ax.set_yticklabels(row_labels, fontsize=BAR_TICK_FONTSIZE, fontweight="bold")
    ax.tick_params(axis="y", labelrotation=45, pad=6)
    for lbl in ax.get_yticklabels():
        lbl.set_ha("right")
        lbl.set_va("center")
        lbl.set_rotation_mode("anchor")
    ax.yaxis.tick_left()
    ax.tick_params(axis="y", labelleft=True, left=False, labelright=False, right=False)
    ax.invert_yaxis()
    ax.set_xlabel("percentage (%)", fontsize=BAR_LABEL_FONTSIZE, fontweight="bold")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=BAR_TICK_FONTSIZE)
    for lbl in ax.get_xticklabels():
        lbl.set_fontweight("bold")
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.9)
    ax.set_axisbelow(True)
    max_val = float(np.nanmax(vals_matrix)) if len(vals_matrix) else 100.0
    ax.set_xlim(0, min(100.0, max(10.0, max_val + 6.0)))

    legend = ax.legend(title="dataset", loc="upper right", frameon=True, fontsize=BAR_LEGEND_FONTSIZE)
    plt.setp(legend.get_title(), fontsize=BAR_LEGEND_TITLE_FONTSIZE, fontweight="bold")
    for txt in legend.get_texts():
        txt.set_fontweight("bold")

    fig.tight_layout(rect=(0.11, 0.03, 0.98, 0.995))
    _save_figure(fig, out_path)
    plt.close(fig)


def _stacked_percent_facet(
    crosstabs: Dict[str, pd.DataFrame],
    title: str,
    xlabel: str,
    out_path: Path,
) -> None:
    dataset_names = list(crosstabs.keys())
    cols = 2
    rows = math.ceil(len(dataset_names) / cols)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(11.8, 3.0 * rows),
        squeeze=False,
        sharey=True,
        constrained_layout=True,
    )

    first_table = next(iter(crosstabs.values()))
    colors = [STACKED_CATEGORY_COLORS[i % len(STACKED_CATEGORY_COLORS)] for i in range(len(first_table.columns))]

    for idx, name in enumerate(dataset_names):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        table = crosstabs[name]
        total = float(table.to_numpy().sum())
        if total > 0:
            pct = table / total * 100.0
        else:
            pct = table.astype(float)
        pct_display = pct.copy()
        pct_display.index = [_bar_label(v) for v in pct_display.index]
        pct_display.columns = [_bar_label(v) for v in pct_display.columns]
        pct_display.plot(kind="bar", stacked=True, color=colors, ax=ax, legend=False)
        ax.set_xlabel("")
        ax.set_ylabel("percentage (%)", fontsize=BAR_LABEL_FONTSIZE, fontweight="bold")
        ax.tick_params(axis="x", rotation=0, labelsize=BAR_TICK_FONTSIZE)
        ax.tick_params(axis="y", labelsize=BAR_TICK_FONTSIZE)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")

    for idx in range(len(dataset_names), rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        [_pretty_label(v) for v in labels],
        title=_bar_label(first_table.columns.name or ""),
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        frameon=True,
        fontsize=BAR_LEGEND_FONTSIZE,
    )
    fig.supxlabel(_pretty_label(xlabel), y=-0.02, fontsize=BAR_LABEL_FONTSIZE, fontweight="bold")
    fig.set_constrained_layout_pads(hspace=0.01, wspace=0.03, h_pad=0.03, w_pad=0.04)
    _save_figure(fig, out_path)
    plt.close(fig)


def _heatmap_percent_facet(
    crosstabs: Dict[str, pd.DataFrame],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
) -> None:
    dataset_names = list(crosstabs.keys())
    cols = 2
    rows = math.ceil(len(dataset_names) / cols)
    fig_side = 4.8 * max(rows, cols)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(fig_side, fig_side * 0.7),
        squeeze=False,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.08, top=0.96, wspace=0.02, hspace=0.12)

    pct_tables: Dict[str, pd.DataFrame] = {}
    vmax = 100.0
    first_name = dataset_names[0]
    base_table = crosstabs[first_name]
    row_order = _sorted_for_heatmap(list(base_table.index))
    col_order = _sorted_for_heatmap(list(base_table.columns))

    for name, table in crosstabs.items():
        pct = table.div(table.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) * 100.0
        pct = pct.reindex(index=row_order, columns=col_order, fill_value=0.0)
        pct_display = pct.copy()
        pct_display.index = [_heatmap_label(v) for v in pct_display.index]
        pct_display.columns = [_heatmap_label(v) for v in pct_display.columns]
        pct_tables[name] = pct_display
    norm = colors.Normalize(vmin=0, vmax=vmax)

    for idx, name in enumerate(dataset_names):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        annot_vals = pct_tables[name].apply(lambda col: col.map(lambda v: f"{v:.1f}%"))
        annot_fs = _auto_heatmap_annot_fontsize(
            pct_tables[name].shape[0],
            pct_tables[name].shape[1],
        )
        sns.heatmap(
            pct_tables[name],
            cmap=HEATMAP_CMAP,
            vmin=0,
            vmax=vmax,
            cbar=False,
            ax=ax,
            annot=annot_vals,
            fmt="",
            annot_kws={"fontsize": annot_fs, "fontweight": "bold"},
        )
        ax.set_title(_pretty_label(name), fontsize=HEATMAP_TITLE_FONTSIZE - 2, fontweight="bold", pad=1)
        if r == rows - 1:
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=HEATMAP_X_TICK_ROTATION, labelsize=HEATMAP_TICK_FONTSIZE, pad=6)
        else:
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=False, bottom=False)
        if c == 0:
            ax.set_ylabel("")
            ax.tick_params(axis="y", rotation=HEATMAP_Y_TICK_ROTATION, labelsize=HEATMAP_TICK_FONTSIZE)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False, left=False)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
            lbl.set_va("top")
            lbl.set_rotation_mode("anchor")
        for lbl in ax.get_yticklabels():
            lbl.set_ha("right")
            lbl.set_va("center")
            lbl.set_rotation_mode("anchor")

    for idx in range(len(dataset_names), rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=HEATMAP_CMAP)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("percentage (%)", fontsize=HEATMAP_COLORBAR_LABEL_FONTSIZE, fontweight="bold")
    cbar.ax.tick_params(labelsize=HEATMAP_COLORBAR_TICK_FONTSIZE)

    _save_figure(fig, out_path)
    plt.close(fig)


def main() -> int:
    _apply_ieee_font()

    parser = argparse.ArgumentParser(description="Plot multi-dataset scene analysis bars and stacks.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=list(DEFAULT_DATASET_INPUTS.values()),
        help="Paths to scene analysis CSVs.",
    )
    parser.add_argument(
        "--dataset-names",
        nargs="*",
        default=list(DEFAULT_DATASET_INPUTS.keys()),
        help="Display names for datasets. Must match number of --inputs.",
    )
    parser.add_argument(
        "--output-dir",
        default="result/paper_figures/stats",
        help="Directory to write plots.",
    )
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.inputs]
    dataset_names = args.dataset_names
    if len(dataset_names) != len(input_paths):
        raise ValueError("--dataset-names length must equal --inputs length.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_frames = {
        name: pd.read_csv(path) for name, path in zip(dataset_names, input_paths)
    }
    exploded_frames = {name: _explode_memory_color(df) for name, df in dataset_frames.items()}

    _grouped_bar_percent_with_count_labels(
        dataset_frames,
        "scene_type",
        "Scene Type Distribution",
        "scene_type",
        out_dir / "bar_scene_type_all_datasets.png",
        order=SCENE_TYPE_ORDER,
        tail_labels=("other",),
    )
    _grouped_bar_percent_with_count_labels(
        dataset_frames,
        "lighting_type",
        "Lighting Type Distribution",
        "lighting_type",
        out_dir / "bar_lighting_type_all_datasets.png",
        order=LIGHTING_TYPE_ORDER,
        tail_labels=("unknown",),
    )
    _grouped_bar_percent_with_count_labels(
        dataset_frames,
        "luminance_level",
        "Luminance Level Distribution",
        "luminance_level",
        out_dir / "bar_luminance_level_all_datasets.png",
        order=LUMINANCE_LEVEL_ORDER,
    )
    _grouped_bar_percent_with_count_labels(
        dataset_frames,
        "camera_to_object_distance",
        "Camera to Object Distance Distribution",
        "camera_to_object_distance",
        out_dir / "bar_camera_to_object_distance_all_datasets.png",
        order=CAMERA_DISTANCE_ORDER,
        tail_labels=("unknown",),
    )
    _grouped_bar_percent_with_count_labels(
        exploded_frames,
        "memory_color",
        "Memory Color Frequency",
        "memory_color",
        out_dir / "bar_memory_color_all_datasets.png",
        order=MEMORY_COLOR_ORDER,
        tail_labels=("none",),
    )
    _bar_overview_all_attributes(
        dataset_frames,
        exploded_frames,
        out_dir / "bar_overview_all_attributes_all_datasets.png",
    )
    _cleveland_dot_overview(
        dataset_frames,
        exploded_frames,
        out_dir / "cleveland_overview_all_datasets.png",
    )

    lighting_luminance_tabs: Dict[str, pd.DataFrame] = {}
    scene_memory_tabs: Dict[str, pd.DataFrame] = {}
    scene_distance_tabs: Dict[str, pd.DataFrame] = {}
    memory_luminance_tabs: Dict[str, pd.DataFrame] = {}
    for name in dataset_names:
        df = dataset_frames[name]
        exploded = exploded_frames[name]

        lighting_luminance = pd.crosstab(df["lighting_type"], df["luminance_level"])
        lighting_luminance = lighting_luminance.reindex(index=LIGHTING_TYPE_ORDER, fill_value=0)
        lighting_luminance = lighting_luminance.reindex(columns=LUMINANCE_LEVEL_ORDER, fill_value=0)
        lighting_luminance.columns.name = "luminance_level"
        lighting_luminance_tabs[name] = lighting_luminance

        scene_memory = pd.crosstab(exploded["scene_type"], exploded["memory_color"])
        scene_memory = scene_memory.reindex(index=SCENE_TYPE_ORDER, fill_value=0)
        scene_memory = scene_memory.reindex(columns=MEMORY_COLOR_ORDER, fill_value=0)
        scene_memory.columns.name = "memory_color"
        scene_memory_tabs[name] = scene_memory

        scene_distance = pd.crosstab(df["scene_type"], df["camera_to_object_distance"])
        scene_distance = scene_distance.reindex(index=SCENE_TYPE_ORDER, fill_value=0)
        scene_distance = scene_distance.reindex(columns=CAMERA_DISTANCE_ORDER, fill_value=0)
        scene_distance.columns.name = "camera_to_object_distance"
        scene_distance_tabs[name] = scene_distance

        memory_luminance = pd.crosstab(exploded["memory_color"], exploded["luminance_level"])
        memory_luminance = memory_luminance.reindex(index=MEMORY_COLOR_ORDER, fill_value=0)
        memory_luminance = memory_luminance.reindex(columns=LUMINANCE_LEVEL_ORDER, fill_value=0)
        memory_luminance.columns.name = "luminance_level"
        memory_luminance_tabs[name] = memory_luminance

    _heatmap_percent_facet(
        lighting_luminance_tabs,
        "Lighting Type x Luminance Level (Conditional Percentage by Dataset)",
        "luminance_level",
        "lighting_type",
        out_dir / "heatmap_lighting_luminance_conditional_all_datasets.png",
    )
    _heatmap_percent_facet(
        scene_memory_tabs,
        "Scene Type x Memory Color (Conditional Percentage by Dataset)",
        "memory_color",
        "scene_type",
        out_dir / "heatmap_scene_memory_color_conditional_all_datasets.png",
    )
    _heatmap_percent_facet(
        scene_distance_tabs,
        "Scene Type x Camera to Object Distance (Conditional Percentage by Dataset)",
        "camera_to_object_distance",
        "scene_type",
        out_dir / "heatmap_scene_distance_conditional_all_datasets.png",
    )
    _heatmap_percent_facet(
        memory_luminance_tabs,
        "Memory Color x Luminance Level (Conditional Percentage by Dataset)",
        "luminance_level",
        "memory_color",
        out_dir / "heatmap_memory_luminance_conditional_all_datasets.png",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
