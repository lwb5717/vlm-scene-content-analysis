import argparse
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]

DEFAULT_INPUTS = {
    "SPAQ": ROOT_DIR / "result" / "complexity_results_spaq.csv",
    "KonIQ-10K": ROOT_DIR / "result" / "complexity_results_koniq10k.csv",
    "LIVE Wild": ROOT_DIR / "result" / "complexity_results_live.csv",
    "CID2013": ROOT_DIR / "result" / "complexity_results_cid2013.csv",
}

LEVEL_ORDER = ["low", "medium", "high"]
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


def _build_level_table(frames: Dict[str, pd.DataFrame], column: str) -> pd.DataFrame:
    table = pd.DataFrame(index=list(frames.keys()), columns=LEVEL_ORDER, dtype=float).fillna(0)
    for name, df in frames.items():
        counts = df[column].fillna("").astype(str).str.lower().value_counts()
        table.loc[name] = counts.reindex(LEVEL_ORDER, fill_value=0)
    return table


def _save_table_bundle(table: pd.DataFrame, stem: str, csv_dir: Path) -> None:
    pct_table = table.div(table.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) * 100.0
    csv_dir.mkdir(parents=True, exist_ok=True)

    count_out = table.copy()
    count_out.index.name = "dataset"
    count_out.reset_index().to_csv(csv_dir / f"{stem}_counts.csv", index=False, encoding="utf-8-sig")

    pct_out = pct_table.round(4)
    pct_out.index.name = "dataset"
    pct_out.reset_index().to_csv(csv_dir / f"{stem}_percentages.csv", index=False, encoding="utf-8-sig")


def _plot_grouped_distribution(count_table: pd.DataFrame, title: str, out_path: Path) -> None:
    pct_table = count_table.div(count_table.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) * 100.0
    dataset_names = list(count_table.index)
    x = np.arange(len(LEVEL_ORDER), dtype=float)
    width = 0.8 / max(1, len(dataset_names))

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
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
                bar.get_height() + 0.8,
                str(int(count)),
                ha="center",
                va="bottom",
                fontsize=COUNT_FONTSIZE,
                fontweight="bold",
            )

    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.set_ylabel("percentage (%)", fontsize=LABEL_FONTSIZE, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(LEVEL_ORDER, fontsize=TICK_FONTSIZE, fontweight="bold")
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


def _plot_joint_heatmaps(frames: Dict[str, pd.DataFrame], out_path: Path) -> None:
    dataset_names = list(frames.keys())
    cols = 2
    rows = math.ceil(len(dataset_names) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(8.5, 3.8 * rows), squeeze=False)

    matrices = {name: _build_joint_matrix(df) for name, df in frames.items()}
    pct_matrices = {}
    pct_max = 0.0
    for name, matrix in matrices.items():
        total = matrix.sum()
        pct = matrix / total * 100.0 if total > 0 else matrix.astype(float)
        pct_matrices[name] = pct
        pct_max = max(pct_max, float(pct.max()))
    pct_max = max(pct_max, 1.0)

    for idx, name in enumerate(dataset_names):
        row, col = divmod(idx, cols)
        ax = axes[row][col]
        matrix = matrices[name]
        pct = pct_matrices[name]
        image = ax.imshow(pct, cmap="YlGnBu", vmin=0, vmax=pct_max)

        ax.set_title(name, fontsize=TITLE_FONTSIZE, fontweight="bold")
        ax.set_xticks(range(3))
        ax.set_xticklabels(LEVEL_ORDER, fontsize=TICK_FONTSIZE, fontweight="bold")
        ax.set_yticks(range(3))
        ax.set_yticklabels(LEVEL_ORDER, fontsize=TICK_FONTSIZE, fontweight="bold")
        ax.set_xlabel("clutter level", fontsize=LABEL_FONTSIZE, fontweight="bold")
        ax.set_ylabel("quantity level", fontsize=LABEL_FONTSIZE, fontweight="bold")

        for r in range(3):
            for c in range(3):
                text_color = "white" if pct[r, c] >= max(pct.max() * 0.55, 1.0) else "black"
                ax.text(
                    c,
                    r,
                    f"{pct[r, c]:.1f}%\n({int(matrix[r, c])})",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color=text_color,
                )

    for idx in range(len(dataset_names), rows * cols):
        row, col = divmod(idx, cols)
        axes[row][col].axis("off")

    colorbar = fig.colorbar(image, ax=axes, fraction=0.03, pad=0.03)
    colorbar.set_label("percentage (%)", fontsize=LABEL_FONTSIZE, fontweight="bold")
    colorbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


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
    csv_dir = out_dir / "csv"
    frames = _load_frames(input_paths, dataset_names)

    quantity_table = _build_level_table(frames, "quantity_level")
    clutter_table = _build_level_table(frames, "clutter_level")
    _plot_grouped_distribution(quantity_table, "Quantity Level Distribution", out_dir / "quantity_level_distribution.png")
    _plot_grouped_distribution(clutter_table, "Clutter Level Distribution", out_dir / "clutter_level_distribution.png")
    _plot_joint_heatmaps(frames, out_dir / "complexity_9_levels.png")
    _save_table_bundle(quantity_table, "quantity_level_distribution", csv_dir)
    _save_table_bundle(clutter_table, "clutter_level_distribution", csv_dir)

    summary_rows = []
    joint_pct_rows = []
    for name, df in frames.items():
        for level in LEVEL_ORDER:
            summary_rows.append(
                {
                    "dataset": name,
                    "metric": "quantity_level",
                    "level": level,
                    "count": int(quantity_table.loc[name, level]),
                }
            )
            summary_rows.append(
                {
                    "dataset": name,
                    "metric": "clutter_level",
                    "level": level,
                    "count": int(clutter_table.loc[name, level]),
                }
            )

        matrix = _build_joint_matrix(df)
        total = matrix.sum()
        for row_idx, quantity_level in enumerate(LEVEL_ORDER):
            for col_idx, clutter_level in enumerate(LEVEL_ORDER):
                count = int(matrix[row_idx, col_idx])
                summary_rows.append(
                    {
                        "dataset": name,
                        "metric": "quantity_clutter",
                        "level": f"{quantity_level}_{clutter_level}",
                        "count": count,
                    }
                )
                joint_pct_rows.append(
                    {
                        "dataset": name,
                        "quantity_level": quantity_level,
                        "clutter_level": clutter_level,
                        "count": count,
                        "percentage": (count / total * 100.0) if total > 0 else 0.0,
                    }
                )

    pd.DataFrame(summary_rows).to_csv(out_dir / "complexity_counts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(joint_pct_rows).to_csv(out_dir / "complexity_joint_distribution.csv", index=False, encoding="utf-8-sig")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
