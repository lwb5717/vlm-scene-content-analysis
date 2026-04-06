import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt

# Tableau-style qualitative palette with enough separation for small-to-medium category counts.
QUALITATIVE_PALETTE = [
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#E15759",
    "#B07AA1",
    "#76B7B2",
    "#EDC948",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
]

NEUTRAL_GREY = "#8C8C8C"


def apply_serif_plot_style() -> None:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"]


def save_png(fig: plt.Figure, out_path: Path, pad_inches: float = 0.03, dpi: int = 300) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)


def get_palette(n: int) -> List[str]:
    if n <= 0:
        return []
    return [QUALITATIVE_PALETTE[idx % len(QUALITATIVE_PALETTE)] for idx in range(n)]


def build_color_map(
    labels: Iterable[str],
    preferred_order: Optional[Iterable[str]] = None,
    overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    overrides = overrides or {}
    ordered_labels: List[str] = []

    if preferred_order is not None:
        for label in preferred_order:
            text = str(label)
            if text not in ordered_labels:
                ordered_labels.append(text)

    for label in labels:
        text = str(label)
        if text not in ordered_labels:
            ordered_labels.append(text)

    dynamic_labels = [label for label in ordered_labels if label not in overrides]
    palette = get_palette(len(dynamic_labels))
    color_map = {label: color for label, color in zip(dynamic_labels, palette)}
    color_map.update(overrides)
    return color_map


def slugify_label(value: str) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "dataset"
