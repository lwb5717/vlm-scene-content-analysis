import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    "KonIQ-10K": "result/scene_analysis_results_koniq10k.csv",
    "LIVE Wild": "result/scene_analysis_results_livewild.csv",
    "CID2013": "result/scene_analysis_results_cid2013.csv",
}

DATASET_COLORS = ["#4E79A7", "#F28E2B", "#59A14F", "#B07AA1", "#76B7B2", "#E15759"]
TITLE_FONTSIZE = 18
LABEL_FONTSIZE = 14
TICK_FONTSIZE = 12
LEGEND_FONTSIZE = 11
COUNT_FONTSIZE = 9


def _apply_plot_style() -> None:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"]


def _save_figure(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight", pad_inches=0.03)


def _pretty_label(value: object) -> str:
    label = str(value).replace("_", " ")
    if label == "natural landscape":
        return "nature"
    return label


def _load_frames(inputs: List[Path], dataset_names: List[str]) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    required = {
        "scene_type",
        "lighting_type",
        "luminance_level",
        "memory_color",
        "camera_to_object_distance",
    }
    for name, path in zip(dataset_names, inputs):
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        df = pd.read_csv(path)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns in {path}: {sorted(missing)}")
        frames[name] = df
    return frames


def _normalize_counts(series: pd.Series, order: Iterable[str]) -> pd.Series:
    counts = series.value_counts()
    order_list = list(order)
    extras = [label for label in counts.index if label not in order_list]
    final_order = order_list + extras
    return counts.reindex(final_order, fill_value=0)


def _explode_memory_colors(df: pd.DataFrame) -> pd.Series:
    values = df["memory_color"].fillna("").astype(str).str.split(";").explode().str.strip()
    values = values[values != ""]
    if values.empty:
        return pd.Series(dtype=str)
    return values


def _build_distribution_table(
    frames: Dict[str, pd.DataFrame],
    column: str,
    order: Iterable[str],
    explode_memory: bool = False,
) -> pd.DataFrame:
    counts_by_dataset: Dict[str, pd.Series] = {}
    categories: List[str] = list(order)

    for name, df in frames.items():
        if explode_memory:
            values = _explode_memory_colors(df)
        else:
            values = df[column].fillna("unknown").astype(str).str.strip().replace("", "unknown")
        counts = _normalize_counts(values, order)
        counts_by_dataset[name] = counts
        for label in counts.index:
            if label not in categories:
                categories.append(label)

    table = pd.DataFrame(index=list(frames.keys()), columns=categories, dtype=float).fillna(0)
    for name, counts in counts_by_dataset.items():
        table.loc[name] = counts.reindex(categories, fill_value=0)
    return table


def _save_count_and_pct_tables(count_table: pd.DataFrame, stem: str, csv_dir: Path) -> None:
    pct_table = count_table.div(count_table.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) * 100.0
    csv_dir.mkdir(parents=True, exist_ok=True)
    count_out = count_table.copy()
    count_out.index.name = "dataset"
    count_out.reset_index().to_csv(csv_dir / f"{stem}_counts.csv", index=False, encoding="utf-8-sig")

    pct_out = pct_table.round(4)
    pct_out.index.name = "dataset"
    pct_out.reset_index().to_csv(csv_dir / f"{stem}_percentages.csv", index=False, encoding="utf-8-sig")


def _build_crosstab(
    left: pd.Series,
    right: pd.Series,
    left_order: Iterable[str],
    right_order: Iterable[str],
) -> pd.DataFrame:
    table = pd.crosstab(left, right)
    left_labels = list(left_order)
    right_labels = list(right_order)
    left_extras = [label for label in table.index if label not in left_labels]
    right_extras = [label for label in table.columns if label not in right_labels]
    table = table.reindex(index=left_labels + left_extras, fill_value=0)
    table = table.reindex(columns=right_labels + right_extras, fill_value=0)
    return table


def _save_crosstab_bundle(
    frames: Dict[str, pd.DataFrame],
    stem: str,
    left_col: str,
    right_col: str,
    left_order: Iterable[str],
    right_order: Iterable[str],
    csv_dir: Path,
    explode_left: bool = False,
    explode_right: bool = False,
) -> None:
    rows: List[Dict[str, object]] = []
    for dataset_name, df in frames.items():
        left_values = _explode_memory_colors(df) if explode_left else df[left_col].fillna("unknown").astype(str).str.strip().replace("", "unknown")
        right_values = _explode_memory_colors(df) if explode_right else df[right_col].fillna("unknown").astype(str).str.strip().replace("", "unknown")

        if explode_left or explode_right:
            work_df = df.copy()
            if explode_left:
                work_df[left_col] = work_df[left_col].fillna("").astype(str).str.split(";")
                work_df = work_df.explode(left_col)
                work_df[left_col] = work_df[left_col].astype(str).str.strip()
                work_df = work_df[work_df[left_col] != ""]
            if explode_right:
                work_df[right_col] = work_df[right_col].fillna("").astype(str).str.split(";")
                work_df = work_df.explode(right_col)
                work_df[right_col] = work_df[right_col].astype(str).str.strip()
                work_df = work_df[work_df[right_col] != ""]
            left_values = work_df[left_col]
            right_values = work_df[right_col]

        table = _build_crosstab(left_values, right_values, left_order, right_order)
        row_pct = table.div(table.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) * 100.0
        for row_label in table.index:
            for col_label in table.columns:
                rows.append(
                    {
                        "dataset": dataset_name,
                        "row_category": row_label,
                        "column_category": col_label,
                        "count": int(table.loc[row_label, col_label]),
                        "row_percentage": float(row_pct.loc[row_label, col_label]),
                    }
                )

    csv_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_dir / f"{stem}.csv", index=False, encoding="utf-8-sig")


def _plot_grouped_distribution(
    count_table: pd.DataFrame,
    title: str,
    out_path: Path,
) -> None:
    pct_table = count_table.div(count_table.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) * 100.0
    dataset_names = list(count_table.index)
    categories = list(count_table.columns)
    x = np.arange(len(categories), dtype=float)
    width = 0.8 / max(1, len(dataset_names))

    fig_w = max(7.0, 1.2 * len(categories))
    fig, ax = plt.subplots(figsize=(fig_w, 4.2))

    for idx, name in enumerate(dataset_names):
        offsets = x - 0.4 + (idx + 0.5) * width
        bars = ax.bar(
            offsets,
            pct_table.loc[name],
            width=width,
            color=DATASET_COLORS[idx % len(DATASET_COLORS)],
            label=name,
        )
        for bar, count in zip(bars, count_table.loc[name]):
            if count <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.7,
                str(int(count)),
                ha="center",
                va="bottom",
                fontsize=COUNT_FONTSIZE,
                fontweight="bold",
            )

    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.set_ylabel("percentage (%)", fontsize=LABEL_FONTSIZE, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_pretty_label(label) for label in categories],
        rotation=0,
        ha="center",
        fontsize=TICK_FONTSIZE,
        fontweight="bold",
    )
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    ymax = float(pct_table.to_numpy().max()) if not pct_table.empty else 100.0
    ax.set_ylim(0, min(100.0, ymax + 18.0))
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)
    legend = ax.legend(title="dataset", frameon=True, fontsize=LEGEND_FONTSIZE)
    plt.setp(legend.get_title(), fontsize=LEGEND_FONTSIZE, fontweight="bold")
    for text in legend.get_texts():
        text.set_fontweight("bold")

    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


def main() -> int:
    _apply_plot_style()

    parser = argparse.ArgumentParser(description="Plot basic scene-attribute distributions.")
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
        default="result/plots/scene",
        help="Directory to write plots.",
    )
    args = parser.parse_args()

    input_paths = [Path(path) for path in args.inputs]
    dataset_names = args.dataset_names
    if len(input_paths) != len(dataset_names):
        raise ValueError("--dataset-names length must equal --inputs length.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = out_dir / "csv"
    frames = _load_frames(input_paths, dataset_names)

    plot_specs = [
        ("scene_type", SCENE_TYPE_ORDER, "Scene Type Distribution", "scene_type_distribution.png", False),
        ("lighting_type", LIGHTING_TYPE_ORDER, "Lighting Type Distribution", "lighting_type_distribution.png", False),
        ("luminance_level", LUMINANCE_LEVEL_ORDER, "Luminance Level Distribution", "luminance_level_distribution.png", False),
        (
            "camera_to_object_distance",
            CAMERA_DISTANCE_ORDER,
            "Camera Distance Distribution",
            "camera_distance_distribution.png",
            False,
        ),
        ("memory_color", MEMORY_COLOR_ORDER, "Memory Color Distribution", "memory_color_distribution.png", True),
    ]

    for column, order, title, filename, explode_memory in plot_specs:
        table = _build_distribution_table(frames, column, order, explode_memory=explode_memory)
        _plot_grouped_distribution(table, title, out_dir / filename)
        _save_count_and_pct_tables(table, filename.replace(".png", ""), csv_dir)

    _save_crosstab_bundle(
        frames,
        "lighting_luminance",
        "lighting_type",
        "luminance_level",
        LIGHTING_TYPE_ORDER,
        LUMINANCE_LEVEL_ORDER,
        csv_dir,
    )
    _save_crosstab_bundle(
        frames,
        "scene_camera_distance",
        "scene_type",
        "camera_to_object_distance",
        SCENE_TYPE_ORDER,
        CAMERA_DISTANCE_ORDER,
        csv_dir,
    )
    _save_crosstab_bundle(
        frames,
        "scene_memory_color",
        "scene_type",
        "memory_color",
        SCENE_TYPE_ORDER,
        MEMORY_COLOR_ORDER,
        csv_dir,
        explode_right=True,
    )
    _save_crosstab_bundle(
        frames,
        "memory_color_luminance",
        "memory_color",
        "luminance_level",
        MEMORY_COLOR_ORDER,
        LUMINANCE_LEVEL_ORDER,
        csv_dir,
        explode_left=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
