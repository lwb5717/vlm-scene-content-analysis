# Result Gallery

This directory is the curated result gallery for the project.

It is not limited to figures that were finally used in the manuscript. It also keeps representative outputs that help explain what the full pipeline produces.

## Included Sections

- `stats/`
  Cross-dataset scene-attribute summaries and relationship heatmaps.
- `complexity/`
  Cross-dataset complexity distributions and exported statistics.
- `skin_type_analysis/`
  Cross-dataset skin-type summaries.
- `pca/`
  Semantic PCA and clustering exports.
- `spaq_scene_correlation/`
  SPAQ validation outputs.
- `cq_cleveland/`
  CQ-based grouped complexity analysis.

## How To Read It

- If you want the fastest high-level overview, start with `stats/` and `complexity/`.
- If you want manuscript-style secondary analyses, then open `pca/`, `spaq_scene_correlation/`, and `cq_cleveland/`.
- If you only need the minimal runnable example, use `examples/sample_outputs/cid2013/` instead.

## Scope

This folder is a curated showcase of project results:

- some files were used directly in the paper
- some files were generated during the same analysis pipeline but not finally placed in the paper
- CSV files are kept together with representative PNG outputs when they help explain the results
