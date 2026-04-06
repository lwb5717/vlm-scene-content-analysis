# -*- coding: utf-8 -*-
"""
Validate scene analysis results against SPAQ scene category labels.
"""

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from image_content.common.plot_style import save_png

ROOT_DIR = Path(__file__).resolve().parents[3]
RESULTS_CSV = ROOT_DIR / "result" / "scene_analysis_results_spaq.csv"
LABELS_XLSX = ROOT_DIR / "Scene category labels.xlsx"
OUTPUT_DIR = ROOT_DIR / "result" / "result_gallery" / "spaq_scene_correlation"

# Heatmap font configuration
HEATMAP_TITLE_FONTSIZE = 26
HEATMAP_TICK_FONTSIZE = 13
HEATMAP_COLORBAR_LABEL_FONTSIZE = 15
HEATMAP_COLORBAR_TICK_FONTSIZE = 12
HEATMAP_CMAP = "RdBu_r"
HEATMAP_TICK_ROTATION = 35

CORE_MAP = {
    "Cityscape": "cityscape",
    "Human": "human",
    "Indoor scene": "indoor",
    "Landscape": "landscape",
    "Plant": "plant",
}
EXTRA_MAP = {
    "Night scene": "night",
    "Still-life": "still_life_strict",
}


def _normalize_image_name(value: str) -> str:
    text = str(value).replace("\\", "/").strip()
    return text.split("/")[-1].lower()


def _split_memory_colors(value: str) -> List[str]:
    if pd.isna(value):
        return []
    parts = [item.strip().lower() for item in str(value).split(";") if item.strip()]
    return parts


def _build_predictions(df: pd.DataFrame) -> pd.DataFrame:
    mem = df["memory_color"].fillna("").astype(str)
    mem_list = mem.apply(_split_memory_colors)
    df = df.copy()
    scene = df["scene_type"].str.lower()
    lighting = df["lighting_type"].str.lower()
    luminance = df["luminance_level"].str.lower()
    distance = df["camera_to_object_distance"].str.lower()

    has_human_skin = mem_list.apply(lambda items: "human_skin_tone" in items)
    has_sky = mem_list.apply(lambda items: "sky_blue" in items)
    has_water = mem_list.apply(lambda items: "water_blue" in items)
    has_snow = mem_list.apply(lambda items: "snow_cloud_white" in items)
    has_plant = mem_list.apply(lambda items: "grass_foliage_green" in items)

    df["pred_still_life_strict"] = (
        distance.eq("close")
        & ~scene.eq("portrait")
        & ~has_human_skin
        & ~has_sky
        & ~has_water
    ).astype(int)

    df["pred_cityscape"] = scene.eq("urban").astype(int)
    df["pred_human"] = (scene.eq("portrait") | has_human_skin).astype(int)
    df["pred_indoor"] = (scene.eq("indoor") & ~df["pred_still_life_strict"].astype(bool)).astype(int)
    df["pred_landscape"] = (
        scene.eq("natural_landscape")
        & distance.isin(["far", "medium"])
        & (has_sky | has_water | has_snow)
    ).astype(int)
    df["pred_plant"] = (
        has_plant
        & distance.isin(["close", "medium"])
        & ~scene.eq("urban")
    ).astype(int)
    df["pred_night"] = (~luminance.eq("high") & ~lighting.eq("daylight")).astype(int)
    return df


def _score_binary(pred: pd.Series, label: pd.Series) -> Dict[str, float]:
    pred = pred.astype(int)
    label = label.astype(int)
    acc = (pred == label).mean()
    tp = ((pred == 1) & (label == 1)).sum()
    fp = ((pred == 1) & (label == 0)).sum()
    fn = ((pred == 0) & (label == 1)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _save_figure(fig, out_path: Path) -> None:
    save_png(fig, out_path, pad_inches=0.02)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate SPAQ labels and export CSV summaries.")
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory to write outputs.",
    )
    parser.add_argument(
        "--with-plots",
        action="store_true",
        help="Also render validation plots. By default this script exports CSV/TXT only.",
    )
    args = parser.parse_args()

    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"Results CSV not found: {RESULTS_CSV}")
    if not LABELS_XLSX.exists():
        raise FileNotFoundError(f"Labels Excel not found: {LABELS_XLSX}")

    results = pd.read_csv(RESULTS_CSV)
    labels = pd.read_excel(LABELS_XLSX)

    if "image_name" not in results.columns:
        raise ValueError("Results CSV missing required column: image_name")
    if "Image name" not in labels.columns:
        raise ValueError("Labels Excel missing required column: Image name")

    results = results.copy()
    results["image_key"] = results["image_name"].map(_normalize_image_name)
    labels = labels.copy()
    labels["image_key"] = labels["Image name"].map(_normalize_image_name)

    merged = results.merge(labels, on="image_key", how="inner")
    if merged.empty:
        raise RuntimeError("No matching images between results and labels.")

    label_cols = [col for col in labels.columns if col not in ("Image name", "image_key")]
    label_weights = labels[label_cols].fillna(0).sum().sort_values(ascending=False)

    merged = _build_predictions(merged)

    rows = []
    combined_map = {**CORE_MAP, **EXTRA_MAP}
    for label_col, pred_key in combined_map.items():
        if label_col not in merged.columns:
            raise ValueError(f"Labels Excel missing column: {label_col}")
        label_bin = (merged[label_col].fillna(0) > 0).astype(int)
        pred_col = f"pred_{pred_key}"
        if pred_col not in merged.columns:
            raise ValueError(f"Prediction column missing: {pred_col}")
        scores = _score_binary(merged[pred_col], label_bin)
        scores["label"] = label_col
        scores["pred_rule"] = pred_key
        rows.append(scores)

    score_df = pd.DataFrame(rows).set_index("label")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    score_df.to_csv(out_dir / "spaq_validation_scores.csv", encoding="utf-8-sig")
    label_weights.rename_axis("label").reset_index(name="sum_of_label_weights").to_csv(
        out_dir / "spaq_label_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    core_labels = list(CORE_MAP.keys())
    overall = score_df.loc[core_labels]["accuracy"].mean()

    with (out_dir / "spaq_validation_summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"overall_core_accuracy={overall}\n")
        f.write("per_label_accuracy:\n")
        f.write(score_df["accuracy"].to_string())
        f.write("\n")

    corr_labels = {}
    corr_preds = {}
    for label_col, pred_key in combined_map.items():
        corr_labels[label_col] = (merged[label_col].fillna(0) > 0).astype(int)
        corr_preds[f"pred_{label_col}"] = merged[f"pred_{pred_key}"]
    corr_df = pd.DataFrame({**corr_preds, **corr_labels})
    corr_matrix = corr_df.corr()
    corr_matrix.to_csv(out_dir / "spaq_validation_correlation.csv", encoding="utf-8-sig")

    if args.with_plots:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        score_df["accuracy"].plot(kind="bar", color="#4C72B0")
        plt.axhline(overall, color="#DD8452", linestyle="--", linewidth=1.5, label="core_avg")
        plt.title("SPAQ Validation Accuracy by Label")
        plt.xlabel("label")
        plt.ylabel("accuracy")
        plt.xticks(rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        _save_figure(plt.gcf(), out_dir / "spaq_validation_accuracy.png")
        plt.close()

        plt.figure(figsize=(8, 4))
        label_weights.plot(kind="bar", color="#4C72B0")
        plt.title("SPAQ Official Label Distribution")
        plt.xlabel("label")
        plt.ylabel("sum of label weights")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        _save_figure(plt.gcf(), out_dir / "spaq_label_distribution.png")
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 8))
        hm = ax.imshow(corr_matrix, cmap=HEATMAP_CMAP, vmin=-1, vmax=1)
        cbar = fig.colorbar(hm, ax=ax)
        cbar.set_label("correlation", fontsize=HEATMAP_COLORBAR_LABEL_FONTSIZE, fontweight="bold")
        cbar.ax.tick_params(labelsize=HEATMAP_COLORBAR_TICK_FONTSIZE)
        for lbl in cbar.ax.get_yticklabels():
            lbl.set_fontweight("bold")

        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_xticklabels(
            corr_matrix.columns,
            rotation=HEATMAP_TICK_ROTATION,
            ha="right",
            fontsize=HEATMAP_TICK_FONTSIZE,
            fontweight="bold",
        )
        ax.set_yticks(range(len(corr_matrix.index)))
        ax.set_yticklabels(
            corr_matrix.index,
            rotation=HEATMAP_TICK_ROTATION,
            ha="right",
            va="center",
            rotation_mode="anchor",
            fontsize=HEATMAP_TICK_FONTSIZE,
            fontweight="bold",
        )
        fig.tight_layout()
        _save_figure(fig, out_dir / "spaq_validation_correlation.png")
        plt.close(fig)

    print(f"Saved scores to {out_dir / 'spaq_validation_scores.csv'}")
    print(f"Saved label distribution CSV to {out_dir / 'spaq_label_distribution.csv'}")
    print(f"Saved correlation CSV to {out_dir / 'spaq_validation_correlation.csv'}")


if __name__ == "__main__":
    main()
