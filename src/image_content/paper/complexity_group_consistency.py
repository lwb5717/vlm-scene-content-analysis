# -*- coding: utf-8 -*-
"""
Check group-wise consistency of quantity/clutter levels for repeated-scene groups.
"""

import argparse
import math
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_CSV = ROOT_DIR / "result" / "complexity_results_cid2013.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "result" / "paper_figures" / "complexity"

LEVEL_ORDER = ["low", "medium", "high"]
LEVEL_TO_INT = {name: idx for idx, name in enumerate(LEVEL_ORDER)}


def _normalize_level(value: str) -> str:
    return str(value).strip().lower()


def _group_key(image_name: str) -> str:
    name = Path(str(image_name)).name
    stem = Path(name).stem
    if "_D" in stem:
        return stem.split("_D")[0]
    return stem


def _entropy(values: List[int]) -> float:
    if not values:
        return 0.0
    counts: Dict[int, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    total = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        prob = count / total
        entropy -= prob * math.log(prob, 2)
    return entropy


def _mode(values: List[int]) -> int:
    counts: Dict[int, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(counts.items(), key=lambda item: item[1])[0]


def _adjacent_only(values: List[int]) -> bool:
    levels = set(values)
    return not ({0, 2}.issubset(levels))


def _within_adjacent(values: List[int], mode: int) -> float:
    if not values:
        return 0.0
    ok = sum(1 for value in values if abs(value - mode) <= 1)
    return ok / len(values)


def _collect_group_stats(values: List[int]) -> Dict[str, float]:
    mode = _mode(values)
    mode_count = sum(1 for value in values if value == mode)
    return {
        "size": len(values),
        "mode": mode,
        "mode_rate": mode_count / len(values) if values else 0.0,
        "entropy": _entropy(values),
        "adjacent_only": _adjacent_only(values),
        "adjacent_rate_to_mode": _within_adjacent(values, mode),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export complexity group-consistency statistics.")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_CSV),
        help="Input complexity CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write group-consistency outputs.",
    )
    args = parser.parse_args()

    input_csv = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "complexity_group_consistency.csv"
    output_summary_csv = output_dir / "complexity_group_consistency_summary.csv"
    output_summary = output_dir / "complexity_group_consistency_summary.txt"

    if not input_csv.exists():
        raise FileNotFoundError(f"CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    required = {"image_name", "quantity_level", "clutter_level"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    df = df.copy()
    df["group"] = df["image_name"].map(_group_key)
    df["quantity_level"] = df["quantity_level"].map(_normalize_level)
    df["clutter_level"] = df["clutter_level"].map(_normalize_level)
    df = df[df["quantity_level"].isin(LEVEL_TO_INT) & df["clutter_level"].isin(LEVEL_TO_INT)]
    if df.empty:
        raise RuntimeError("No valid rows after filtering levels.")

    df["quantity_int"] = df["quantity_level"].map(LEVEL_TO_INT)
    df["clutter_int"] = df["clutter_level"].map(LEVEL_TO_INT)

    rows: List[Dict[str, object]] = []
    for group, group_df in df.groupby("group"):
        quantity_vals = group_df["quantity_int"].tolist()
        clutter_vals = group_df["clutter_int"].tolist()
        quantity_stats = _collect_group_stats(quantity_vals)
        clutter_stats = _collect_group_stats(clutter_vals)
        rows.append(
            {
                "group": group,
                "count": quantity_stats["size"],
                "quantity_mode": LEVEL_ORDER[quantity_stats["mode"]],
                "quantity_mode_rate": quantity_stats["mode_rate"],
                "quantity_entropy": quantity_stats["entropy"],
                "quantity_adjacent_only": quantity_stats["adjacent_only"],
                "quantity_adjacent_rate": quantity_stats["adjacent_rate_to_mode"],
                "clutter_mode": LEVEL_ORDER[clutter_stats["mode"]],
                "clutter_mode_rate": clutter_stats["mode_rate"],
                "clutter_entropy": clutter_stats["entropy"],
                "clutter_adjacent_only": clutter_stats["adjacent_only"],
                "clutter_adjacent_rate": clutter_stats["adjacent_rate_to_mode"],
            }
        )

    out_df = pd.DataFrame(rows).sort_values("group")
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    total_groups = len(out_df)
    quantity_adjacent_only_rate = out_df["quantity_adjacent_only"].mean() if total_groups else 0.0
    clutter_adjacent_only_rate = out_df["clutter_adjacent_only"].mean() if total_groups else 0.0
    quantity_entropy_mean = out_df["quantity_entropy"].mean() if total_groups else 0.0
    clutter_entropy_mean = out_df["clutter_entropy"].mean() if total_groups else 0.0

    pd.DataFrame(
        [
            {
                "total_groups": total_groups,
                "quantity_adjacent_only_rate": quantity_adjacent_only_rate,
                "clutter_adjacent_only_rate": clutter_adjacent_only_rate,
                "quantity_entropy_mean": quantity_entropy_mean,
                "clutter_entropy_mean": clutter_entropy_mean,
            }
        ]
    ).to_csv(output_summary_csv, index=False, encoding="utf-8-sig")

    output_summary.write_text(
        "\n".join(
            [
                f"total_groups={total_groups}",
                f"quantity_adjacent_only_rate={quantity_adjacent_only_rate}",
                f"clutter_adjacent_only_rate={clutter_adjacent_only_rate}",
                f"quantity_entropy_mean={quantity_entropy_mean}",
                f"clutter_entropy_mean={clutter_entropy_mean}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Saved group stats to {output_csv}")
    print(f"Saved summary CSV to {output_summary_csv}")
    print(f"Saved summary to {output_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
