import argparse
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from image_content.common.plot_style import apply_serif_plot_style, get_palette, save_png, slugify_label

ROOT_DIR = Path(__file__).resolve().parents[3]

DEFAULT_INPUTS = {
    "SPAQ": ROOT_DIR / "result" / "complexity_results_spaq.csv",
    "KonIQ-10K": ROOT_DIR / "result" / "complexity_results_koniq10k.csv",
    "LIVE Wild": ROOT_DIR / "result" / "complexity_results_live.csv",
    "CID2013": ROOT_DIR / "result" / "complexity_results_cid2013.csv",
}

LEVEL_ORDER = ["low", "medium", "high"]
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
    return str(value).replace("_", " ")


def _load_frames(inputs: List[Path], dataset_names: List[str]) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    required = {"quantity_level", "clutter_level"}
    for name, path in zip(dataset_names, inputs):
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        df = pd.read_csv(path)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns in {path}: {sorted(missing)}")
        frames[name] = df
    return frames


def _build_level_counts(df: pd.DataFrame, column: str) -> pd.Series:
    counts = df[column].fillna("").astype(str).str.lower().value_counts()
    return counts.reindex(LEVEL_ORDER, fill_value=0)


def _save_table_bundle(counts: pd.Series, stem: str, output_dir: Path, dataset_name: str) -> None:
    pct = counts / max(float(counts.sum()), 1.0) * 100.0
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "dataset": dataset_name,
            "level": counts.index,
            "count": counts.astype(int).values,
        }
    ).to_csv(output_dir / f"{stem}_counts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "dataset": dataset_name,
            "level": pct.index,
            "percentage": pct.round(4).values,
        }
    ).to_csv(output_dir / f"{stem}_percentages.csv", index=False, encoding="utf-8-sig")


def _plot_single_distribution(counts: pd.Series, dataset_name: str, title: str, out_path: Path) -> None:
    pct = counts / max(float(counts.sum()), 1.0) * 100.0
    x = np.arange(len(LEVEL_ORDER), dtype=float)
    level_colors = get_palette(len(LEVEL_ORDER))

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    bars = ax.bar(x, pct.values, width=0.72, color=level_colors)
    for bar, count in zip(bars, counts.values):
        if count <= 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            str(int(count)),
            ha="center",
            va="bottom",
            fontsize=COUNT_FONTSIZE,
            fontweight="bold",
        )

    ax.set_title(f"{dataset_name}: {title}", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.set_ylabel("percentage (%)", fontsize=LABEL_FONTSIZE, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(LEVEL_ORDER, fontsize=TICK_FONTSIZE, fontweight="bold")
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


def _build_joint_matrix(df: pd.DataFrame) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=int)
    q_map = {label: idx for idx, label in enumerate(LEVEL_ORDER)}
    c_map = {label: idx for idx, label in enumerate(LEVEL_ORDER)}
    quantity = df["quantity_level"].fillna("").astype(str).str.lower()
    clutter = df["clutter_level"].fillna("").astype(str).str.lower()
    for q, c in zip(quantity, clutter):
        if q in q_map and c in c_map:
            matrix[q_map[q], c_map[c]] += 1
    return matrix


def main() -> int:
    _apply_plot_style()

    parser = argparse.ArgumentParser(description="Plot basic visual-complexity distributions.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[str(path) for path in DEFAULT_INPUTS.values()],
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
        default=str(ROOT_DIR / "result" / "plots" / "complexity"),
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

    for dataset_name, df in frames.items():
        dataset_slug = slugify_label(dataset_name)
        dataset_dir = out_dir / dataset_slug
        dataset_dir.mkdir(parents=True, exist_ok=True)

        quantity_counts = _build_level_counts(df, "quantity_level")
        clutter_counts = _build_level_counts(df, "clutter_level")
        _plot_single_distribution(quantity_counts, dataset_name, "Quantity Level Distribution", dataset_dir / "quantity_level_distribution.png")
        _plot_single_distribution(clutter_counts, dataset_name, "Clutter Level Distribution", dataset_dir / "clutter_level_distribution.png")
        _save_table_bundle(quantity_counts, "quantity_level_distribution", dataset_dir, dataset_name)
        _save_table_bundle(clutter_counts, "clutter_level_distribution", dataset_dir, dataset_name)

        matrix = _build_joint_matrix(df)
        total = matrix.sum()
        joint_rows = []
        for row_idx, quantity_level in enumerate(LEVEL_ORDER):
            for col_idx, clutter_level in enumerate(LEVEL_ORDER):
                count = int(matrix[row_idx, col_idx])
                joint_rows.append(
                    {
                        "dataset": dataset_name,
                        "quantity_level": quantity_level,
                        "clutter_level": clutter_level,
                        "count": count,
                        "percentage": (count / total * 100.0) if total > 0 else 0.0,
                    }
                )
        pd.DataFrame(joint_rows).to_csv(dataset_dir / "complexity_joint_distribution.csv", index=False, encoding="utf-8-sig")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
