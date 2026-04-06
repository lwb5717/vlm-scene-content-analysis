# CID2013 Sample Outputs

This directory contains a small curated example of repository outputs for the `CID2013` dataset.

It is included so that GitHub visitors can immediately see:

- what the generated CSV files look like
- what the plotting outputs look like
- what kinds of figures are produced from the analysis pipeline

This folder is intentionally selective. It is not a full dump of all local experimental outputs.

## Included Files

- `scene_analysis_results.csv`: scene attribute extraction output
- `complexity_results.csv`: visual complexity output
- `skin_type_results.csv`: skin type analysis output
- `figures/scene_type_distribution.png`: example scene distribution figure
- `figures/complexity_9_levels.png`: example complexity figure
- `figures/skin_type_distribution.png`: example skin-type figure
- `figures/semantic_summary.txt`: example text summary generated from scene analysis

If you run the scripts on your own image directory, you should obtain outputs with the same overall structure, although the exact labels and distributions will depend on the dataset, model version, and prompt version.
