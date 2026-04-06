import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from image_content.common.plot_style import apply_serif_plot_style, get_palette, save_png, slugify_label

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

TITLE_FONTSIZE = 18
LABEL_FONTSIZE = 14
TICK_FONTSIZE = 12
LEGEND_FONTSIZE = 11
COUNT_FONTSIZE = 9


def _apply_plot_style() -> None:
    apply_serif_plot_style()


def _save_figure(fig: plt.Figure, out_path: Path) -> None:
    save_png(fig, out_path, pad_inches=0.03)


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


def _build_distribution_counts(
    df: pd.DataFrame,
    column: str,
    order: Iterable[str],
    explode_memory: bool = False,
) -> pd.Series:
    if explode_memory:
        values = _explode_memory_colors(df)
    else:
        values = df[column].fillna("unknown").astype(str).str.strip().replace("", "unknown")
    return _normalize_counts(values, order)


def _save_count_and_pct_tables(counts: pd.Series, stem: str, csv_dir: Path, dataset_name: str) -> None:
    pct = counts / max(float(counts.sum()), 1.0) * 100.0
    csv_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "dataset": dataset_name,
            "category": counts.index,
            "count": counts.astype(int).values,
        }
    ).to_csv(csv_dir / f"{stem}_counts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "dataset": dataset_name,
            "category": pct.index,
            "percentage": pct.round(4).values,
        }
    ).to_csv(csv_dir / f"{stem}_percentages.csv", index=False, encoding="utf-8-sig")


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
    df: pd.DataFrame,
    dataset_name: str,
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


def _plot_single_distribution(
    counts: pd.Series,
    dataset_name: str,
    title: str,
    out_path: Path,
) -> None:
    total = max(float(counts.sum()), 1.0)
    pct = counts / total * 100.0
    categories = list(counts.index)
    x = np.arange(len(categories), dtype=float)
    category_colors = get_palette(len(categories))

    fig_w = max(7.0, 1.2 * len(categories))
    fig, ax = plt.subplots(figsize=(fig_w, 4.2))
    bars = ax.bar(x, pct.values, width=0.72, color=category_colors)
    for bar, count in zip(bars, counts.values):
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

    ax.set_title(f"{dataset_name}: {title}", fontsize=TITLE_FONTSIZE, fontweight="bold")
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
    ymax = float(pct.max()) if not pct.empty else 100.0
    ax.set_ylim(0, min(100.0, ymax + 18.0))
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)

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

    for dataset_name, df in frames.items():
        dataset_slug = slugify_label(dataset_name)
        dataset_dir = out_dir / dataset_slug
        dataset_dir.mkdir(parents=True, exist_ok=True)
        csv_dir = dataset_dir / "csv"

        for column, order, title, filename, explode_memory in plot_specs:
            counts = _build_distribution_counts(df, column, order, explode_memory=explode_memory)
            _plot_single_distribution(counts, dataset_name, title, dataset_dir / filename)
            _save_count_and_pct_tables(counts, filename.replace(".png", ""), csv_dir, dataset_name)

        _save_crosstab_bundle(
            df,
            dataset_name,
            "lighting_luminance",
            "lighting_type",
            "luminance_level",
            LIGHTING_TYPE_ORDER,
            LUMINANCE_LEVEL_ORDER,
            csv_dir,
        )
        _save_crosstab_bundle(
            df,
            dataset_name,
            "scene_camera_distance",
            "scene_type",
            "camera_to_object_distance",
            SCENE_TYPE_ORDER,
            CAMERA_DISTANCE_ORDER,
            csv_dir,
        )
        _save_crosstab_bundle(
            df,
            dataset_name,
            "scene_memory_color",
            "scene_type",
            "memory_color",
            SCENE_TYPE_ORDER,
            MEMORY_COLOR_ORDER,
            csv_dir,
            explode_right=True,
        )
        _save_crosstab_bundle(
            df,
            dataset_name,
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
