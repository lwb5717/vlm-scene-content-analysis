# -*- coding: utf-8 -*-
"""
Plot multi-dataset skin_type distributions in one grouped bar chart.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from image_content.common.plot_style import apply_serif_plot_style, get_palette, save_png

ROOT_DIR = Path(__file__).resolve().parents[3]

DEFAULT_INPUTS = {
    "spaq": ROOT_DIR / "result" / "skin_type" / "spaq" / "skin_type_results_spaq.csv",
    "koniq10k": ROOT_DIR / "result" / "skin_type" / "Koniq10K" / "skin_type_results_koniq10k.csv",
    "livewild": ROOT_DIR / "result" / "skin_type" / "livewild" / "skin_type_results_livewild.csv",
    "cid2013": ROOT_DIR / "result" / "skin_type" / "CID2013" / "skin_type_results_cid2013.csv",
}
DEFAULT_OUTPUT_DIR = ROOT_DIR / "result" / "plots" / "skin_type"

ORDER = ["light", "medium", "dark", "uncertain"]
TITLE_FONTSIZE = 22
LABEL_FONTSIZE = 20
TICK_FONTSIZE = 19
LEGEND_FONTSIZE = 14
LEGEND_TITLE_FONTSIZE = 15
LABEL_VALUE_FONTSIZE = 14

DATASET_DISPLAY_MAP = {
    "spaq": "SPAQ",
    "koniq10k": "KonIQ-10K",
    "koniq-10k": "KonIQ-10K",
    "live": "LIVE Wild",
    "livewild": "LIVE Wild",
    "cid2013": "CID2013",
}

def _pretty_label(value: str) -> str:
    return str(value).replace("_", " ")


def _dataset_display_name(value: str) -> str:
    key = str(value).strip().lower()
    return DATASET_DISPLAY_MAP.get(key, value)


def _apply_ieee_font() -> None:
    apply_serif_plot_style()


def _save_figure(fig: plt.Figure, out_path: Path) -> None:
    save_png(fig, out_path, pad_inches=0.02)


def _parse_skin_types(value: str) -> List[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if text.lower() == "uncertain":
        return ["uncertain"]
    try:
        payload = json.loads(text)
        if isinstance(payload, list):
            return [str(item).strip().lower() for item in payload if str(item).strip()]
    except json.JSONDecodeError:
        pass
    return [v.strip().lower() for v in text.split(";") if v.strip()]


def _load_counts(csv_path: Path) -> pd.Series:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "skin_types" not in df.columns:
        raise ValueError(f"CSV missing required column 'skin_types': {csv_path}")

    labels: List[str] = []
    for value in df["skin_types"]:
        labels.extend(_parse_skin_types(value))

    return pd.Series(labels).value_counts().reindex(ORDER, fill_value=0)


def _plot_grouped_bar(
    counts_by_dataset: Dict[str, pd.Series],
    out_png: Path,
    show_zero_labels: bool = True,
) -> None:
    dataset_names = list(counts_by_dataset.keys())
    dataset_colors = get_palette(len(dataset_names))

    count_table = pd.DataFrame(index=dataset_names, columns=ORDER, dtype=float).fillna(0)
    for name in dataset_names:
        count_table.loc[name] = counts_by_dataset[name].reindex(ORDER, fill_value=0)

    pct_table = count_table.div(count_table.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) * 100.0

    x = np.arange(len(ORDER))
    width = 0.8 / max(len(dataset_names), 1)

    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    for i, name in enumerate(dataset_names):
        offsets = x - 0.4 + (i + 0.5) * width
        bars = ax.bar(
            offsets,
            pct_table.loc[name],
            width=width,
            color=dataset_colors[i],
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
                fontsize=LABEL_VALUE_FONTSIZE,
                fontweight="bold",
            )

    ax.set_xlabel("skin types", fontweight="bold", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("percentage (%)", fontweight="bold", fontsize=LABEL_FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_pretty_label(v) for v in ORDER],
        rotation=0,
        ha="center",
        fontsize=TICK_FONTSIZE,
        fontweight="bold",
    )
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ymax = float(pct_table.to_numpy().max())
    ax.set_ylim(0, min(105.0, ymax + 18.0))
    legend = ax.legend(title="dataset", loc="upper right", frameon=True, fontsize=LEGEND_FONTSIZE)
    plt.setp(legend.get_title(), fontsize=LEGEND_TITLE_FONTSIZE, fontweight="bold")
    for txt in legend.get_texts():
        txt.set_fontweight("bold")
    fig.tight_layout(pad=0.2)
    _save_figure(fig, out_png)
    plt.close(fig)


def main() -> int:
    _apply_ieee_font()

    parser = argparse.ArgumentParser(description="Plot multi-dataset skin_type distributions.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[str(v) for v in DEFAULT_INPUTS.values()],
        help="Paths to skin_type result CSVs.",
    )
    parser.add_argument(
        "--dataset-names",
        nargs="*",
        default=list(DEFAULT_INPUTS.keys()),
        help="Display names for datasets. Must match number of --inputs.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write plot and counts CSV.",
    )
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.inputs]
    dataset_names = args.dataset_names
    if len(input_paths) != len(dataset_names):
        raise ValueError("--dataset-names length must equal --inputs length.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts_by_dataset: Dict[str, pd.Series] = {}
    for name, path in zip(dataset_names, input_paths):
        counts_by_dataset[name] = _load_counts(path)

    out_png = out_dir / "skin_type_distribution.png"
    _plot_grouped_bar(counts_by_dataset, out_png=out_png, show_zero_labels=True)

    summary = pd.DataFrame(counts_by_dataset).T.reindex(columns=ORDER)
    summary.index.name = "dataset"
    summary.reset_index().to_csv(
        out_dir / "skin_type_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )
    percentages = summary.div(summary.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) * 100.0
    percentages.index.name = "dataset"
    percentages.reset_index().to_csv(
        out_dir / "skin_type_percentages.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Saved plot to {out_png}")
    print(f"Saved counts to {out_dir / 'skin_type_counts.csv'}")
    print(f"Saved percentages to {out_dir / 'skin_type_percentages.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
