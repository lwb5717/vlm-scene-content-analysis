# -*- coding: utf-8 -*-
"""
Build multi-hot semantic vectors from scene analysis CSV, run PCA + KMeans,
and export vector matrix and plots.
"""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from image_content.common.plot_style import (
    NEUTRAL_GREY,
    build_color_map,
    get_palette,
    save_png,
)

ROOT_DIR = Path(__file__).resolve().parents[3]
DATASET_NAME = "koniq10k"
INPUT_CSV = ROOT_DIR / "result" / f"scene_analysis_results_{DATASET_NAME}.csv"
OUTPUT_DIR = ROOT_DIR / "result" / "paper_figures" / "pca"
JITTER_STD = 0.02

# Scatter plot font configuration
PLOT_TITLE_FONTSIZE = 19
PLOT_LABEL_FONTSIZE = 16
PLOT_TICK_FONTSIZE = 15
PLOT_LEGEND_FONTSIZE = 9
PLOT_LEGEND_TITLE_FONTSIZE = 10
PLOT_LEGEND_MARKERSIZE = 8
PLOT_LEGEND_HANDLETEXT_PAD = 0.35
PLOT_COLORBAR_LABEL_FONTSIZE = 11
PLOT_COLORBAR_TICK_FONTSIZE = 10

SCENE_TYPES = ["natural_landscape", "urban", "indoor", "portrait", "other"]
LIGHTING_TYPES = ["daylight", "artificial", "mixed"]
LUMINANCE_LEVELS = ["high", "medium", "low"]
CAMERA_DISTANCES = ["close", "medium", "far"]
MEMORY_COLORS = [
    "sky_blue",
    "grass_foliage_green",
    "human_skin_tone",
    "water_blue",
    "snow_cloud_white",
    "none",
]

def _apply_plot_style() -> None:
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"


def _split_memory_colors(value: str) -> List[str]:
    if pd.isna(value):
        return []
    parts = [v.strip().lower() for v in str(value).split(";")]
    return [v for v in parts if v and v != "unknown"]


def _normalize_label(value: str) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _semantic_complexity(row: pd.Series) -> int:
    memory_count = len([v for v in _split_memory_colors(row["memory_color"]) if v != "none"])
    fields = ["scene_type", "lighting_type", "luminance_level", "camera_to_object_distance"]
    non_unknown = sum(1 for field in fields if _normalize_label(row[field]) not in ("", "unknown"))
    return memory_count + non_unknown


def _memory_color_label(value: str) -> str:
    colors = _split_memory_colors(value)
    if not colors:
        return "none"
    unique = sorted(set(colors))
    if len(unique) == 1:
        return unique[0]
    return "multi"


def _build_vocab(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    base_cols = [
        ("scene_type", SCENE_TYPES),
        ("lighting_type", LIGHTING_TYPES),
        ("luminance_level", LUMINANCE_LEVELS),
        ("camera_to_object_distance", CAMERA_DISTANCES),
    ]
    vocab = []
    for prefix, labels in base_cols:
        vocab.extend([f"{prefix}={label}" for label in labels])
    vocab.extend([f"memory_color={label}" for label in MEMORY_COLORS])
    return vocab, [prefix for prefix, _ in base_cols]


def _encode_multi_hot(df: pd.DataFrame, vocab: List[str]) -> pd.DataFrame:
    vectors = pd.DataFrame(0, index=df.index, columns=vocab, dtype=int)

    for idx, row in df.iterrows():
        scene = _normalize_label(row["scene_type"])
        if scene and scene != "unknown":
            key = f"scene_type={scene}"
            if key in vectors.columns:
                vectors.at[idx, key] = 1

        lighting = _normalize_label(row["lighting_type"])
        if lighting and lighting != "unknown":
            key = f"lighting_type={lighting}"
            if key in vectors.columns:
                vectors.at[idx, key] = 1

        luminance = _normalize_label(row["luminance_level"])
        if luminance and luminance != "unknown":
            key = f"luminance_level={luminance}"
            if key in vectors.columns:
                vectors.at[idx, key] = 1

        distance = _normalize_label(row["camera_to_object_distance"])
        if distance and distance != "unknown":
            key = f"camera_to_object_distance={distance}"
            if key in vectors.columns:
                vectors.at[idx, key] = 1

        for mem in _split_memory_colors(row["memory_color"]):
            key = f"memory_color={mem}"
            if key in vectors.columns:
                vectors.at[idx, key] = 1

    return vectors


def _plot_scatter(
    points: np.ndarray,
    labels: Iterable,
    title: str,
    out_path: Path,
    color_map: Optional[Dict[str, str]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.3))
    rng = np.random.default_rng(42)
    jitter = rng.normal(loc=0.0, scale=JITTER_STD, size=points.shape)
    points = points + jitter
    labels_array = np.array(labels)
    if np.issubdtype(labels_array.dtype, np.number):
        color_values = labels_array
        scatter = ax.scatter(points[:, 0], points[:, 1], c=color_values, cmap="tab10", s=18, alpha=0.85)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("value", fontsize=PLOT_COLORBAR_LABEL_FONTSIZE, fontweight="bold")
        cbar.ax.tick_params(labelsize=PLOT_COLORBAR_TICK_FONTSIZE)
        for lbl in cbar.ax.get_yticklabels():
            lbl.set_fontweight("bold")
    else:
        labels_str = pd.Series(labels_array).fillna("unknown").astype(str)
        if color_map:
            colors = [color_map.get(label, "#8C8C8C") for label in labels_str]
            scatter = ax.scatter(points[:, 0], points[:, 1], c=colors, s=18, alpha=0.85)
            legend_labels = list(dict.fromkeys(labels_str.tolist()))
            handles = [
                Line2D([0], [0], marker="o", color="w", label=label,
                       markerfacecolor=color_map.get(label, NEUTRAL_GREY), markersize=PLOT_LEGEND_MARKERSIZE)
                for label in legend_labels
            ]
            legend = ax.legend(
                handles,
                legend_labels,
                title="label",
                loc="best",
                fontsize=PLOT_LEGEND_FONTSIZE,
                handletextpad=PLOT_LEGEND_HANDLETEXT_PAD,
            )
            plt.setp(legend.get_title(), fontsize=PLOT_LEGEND_TITLE_FONTSIZE, fontweight="bold")
            for txt in legend.get_texts():
                txt.set_fontweight("bold")
        else:
            codes, uniques = pd.factorize(labels_str, sort=True)
            scatter = ax.scatter(points[:, 0], points[:, 1], c=codes, cmap="tab10", s=18, alpha=0.85)
            handles, _ = scatter.legend_elements()
            legend = ax.legend(
                handles,
                uniques,
                title="label",
                loc="best",
                fontsize=PLOT_LEGEND_FONTSIZE,
                handletextpad=PLOT_LEGEND_HANDLETEXT_PAD,
            )
            plt.setp(legend.get_title(), fontsize=PLOT_LEGEND_TITLE_FONTSIZE, fontweight="bold")
            for txt in legend.get_texts():
                txt.set_fontweight("bold")
    ax.set_xlabel("PC1", fontsize=PLOT_LABEL_FONTSIZE, fontweight="bold")
    ax.set_ylabel("PC2", fontsize=PLOT_LABEL_FONTSIZE, fontweight="bold")
    ax.tick_params(axis="both", labelsize=PLOT_TICK_FONTSIZE)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    fig.tight_layout(pad=0.2)
    save_png(fig, out_path, pad_inches=0.02)
    plt.close(fig)


def _top_contributors(loadings: pd.DataFrame, component: str, top_k: int = 5) -> pd.DataFrame:
    positive = loadings.nlargest(top_k, component)[["semantic_feature", component]].copy()
    positive["direction"] = "positive"
    negative = loadings.nsmallest(top_k, component)[["semantic_feature", component]].copy()
    negative["direction"] = "negative"
    result = pd.concat([positive, negative], ignore_index=True)
    result.insert(0, "component", component)
    return result


def main() -> int:
    _apply_plot_style()

    parser = argparse.ArgumentParser(description="PCA + KMeans on semantic vectors.")
    parser.add_argument(
        "--input",
        default=str(INPUT_CSV),
        help="Input scene analysis CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Output directory for vectors and plots.",
    )
    parser.add_argument(
        "--color-by",
        choices=["scene_type", "semantic_complexity"],
        default="scene_type",
        help="Color scatter by scene_type or semantic_complexity.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="KMeans k (suggested 3-6).",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df["semantic_complexity"] = df.apply(_semantic_complexity, axis=1)
    df["memory_color_label"] = df["memory_color"].map(_memory_color_label)

    vocab, _ = _build_vocab(df)
    vectors = _encode_multi_hot(df, vocab)

    id_col = "image_name" if "image_name" in df.columns else "image_path"
    if id_col in df.columns:
        vectors_with_id = pd.concat([df[id_col], vectors], axis=1)
    else:
        vectors_with_id = vectors

    vectors_with_id.to_csv(out_dir / "semantic_vectors.csv", index=False, encoding="utf-8-sig")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(vectors.values)
    pca = PCA(n_components=2, random_state=42)
    points = pca.fit_transform(scaled)
    pca_coordinates = pd.DataFrame(
        {
            id_col: df[id_col] if id_col in df.columns else np.arange(len(df)),
            "PC1": points[:, 0],
            "PC2": points[:, 1],
            "scene_type": df["scene_type"].fillna("unknown"),
            "lighting_type": df["lighting_type"].fillna("unknown"),
            "luminance_level": df["luminance_level"].fillna("unknown"),
            "camera_to_object_distance": df["camera_to_object_distance"].fillna("unknown"),
            "memory_color_label": df["memory_color_label"].fillna("none"),
            "semantic_complexity": df["semantic_complexity"],
        }
    )
    explained_variance = pd.DataFrame(
        {
            "component": ["PC1", "PC2"],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "eigenvalue": pca.explained_variance_,
        }
    )
    explained_variance.to_csv(out_dir / "pca_explained_variance.csv", index=False, encoding="utf-8-sig")
    loadings = pd.DataFrame(
        {
            "semantic_feature": vectors.columns,
            "PC1_loading": pca.components_[0],
            "PC2_loading": pca.components_[1],
        }
    )
    loadings["PC1_abs"] = loadings["PC1_loading"].abs()
    loadings["PC2_abs"] = loadings["PC2_loading"].abs()
    loadings_sorted = loadings.sort_values(["PC1_abs", "PC2_abs"], ascending=False)
    loadings_sorted.to_csv(out_dir / "pca_loadings.csv", index=False, encoding="utf-8-sig")
    top_contributors = pd.concat(
        [
            _top_contributors(loadings, "PC1_loading"),
            _top_contributors(loadings, "PC2_loading"),
        ],
        ignore_index=True,
    )
    top_contributors.to_csv(out_dir / "pca_top_contributors.csv", index=False, encoding="utf-8-sig")

    scene_color_map = build_color_map(
        df["scene_type"].fillna("unknown").astype(str),
        preferred_order=SCENE_TYPES + ["unknown"],
        overrides={"unknown": NEUTRAL_GREY},
    )
    lighting_color_map = build_color_map(
        df["lighting_type"].fillna("unknown").astype(str),
        preferred_order=LIGHTING_TYPES + ["unknown"],
        overrides={"unknown": NEUTRAL_GREY},
    )
    distance_color_map = build_color_map(
        df["camera_to_object_distance"].fillna("unknown").astype(str),
        preferred_order=CAMERA_DISTANCES + ["unknown"],
        overrides={"unknown": NEUTRAL_GREY},
    )
    memory_color_map = build_color_map(
        df["memory_color_label"].fillna("none").astype(str),
        preferred_order=MEMORY_COLORS + ["multi"],
        overrides={"none": NEUTRAL_GREY, "unknown": NEUTRAL_GREY},
    )

    if args.color_by == "scene_type":
        labels = df["scene_type"].fillna("unknown")
        _plot_scatter(
            points,
            labels,
            "Semantic PCA (colored by scene_type)",
            out_dir / "pca_scene_type.png",
            color_map=scene_color_map,
        )
    else:
        labels = df["semantic_complexity"]
        _plot_scatter(
            points,
            labels,
            "Semantic PCA (colored by semantic_complexity)",
            out_dir / "pca_semantic_complexity.png",
        )

    _plot_scatter(
        points,
        df["lighting_type"].fillna("unknown"),
        "Semantic PCA (colored by lighting_type)",
        out_dir / "pca_lighting_type.png",
        color_map=lighting_color_map,
    )
    _plot_scatter(
        points,
        df["camera_to_object_distance"].fillna("unknown"),
        "Semantic PCA (colored by camera_to_object_distance)",
        out_dir / "pca_camera_distance.png",
        color_map=distance_color_map,
    )
    _plot_scatter(
        points,
        df["memory_color_label"].fillna("none"),
        "Semantic PCA (colored by memory_color)",
        out_dir / "pca_memory_color.png",
        color_map=memory_color_map,
    )

    kmeans = KMeans(n_clusters=args.k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled)
    pca_coordinates["kmeans_cluster"] = cluster_labels
    pca_coordinates.to_csv(out_dir / "pca_coordinates.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        kmeans.cluster_centers_,
        columns=vectors.columns,
    ).assign(kmeans_cluster=lambda frame: range(len(frame))).to_csv(
        out_dir / "kmeans_cluster_centroids.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _plot_scatter(points, cluster_labels, f"PCA with KMeans (k={args.k})", out_dir / "pca_kmeans.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
