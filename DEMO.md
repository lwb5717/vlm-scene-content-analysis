# Demo

This repository supports two demo modes.

## Project Figure

Use the project illustration below as the main repository figure:

- [figure_scene_content_info.pdf](./figure_scene_content_info.pdf)

## 1. Zero-Setup Demo

If you just want to see what the project produces, open:

- [examples/sample_outputs/cid2013](./examples/sample_outputs/cid2013)

This folder contains:

- `scene_analysis_results.csv`
- `complexity_results.csv`
- `skin_type_results.csv`
- complexity CSV outputs for downstream statistics or custom plotting

You can also inspect the text summary:

- [semantic_summary.txt](./examples/sample_outputs/cid2013/figures/semantic_summary.txt)

This is the fastest way to understand the output format without running the model.

## 2. Run-It-Yourself Demo

After setting your API key in `.env`, run the pipeline on your own small image folder:

```bash
python scripts/run_scene_inference.py --input-dir "path/to/images" --output-csv "outputs/scene_results.csv"
python scripts/run_complexity_inference.py --input-dir "path/to/images" --output-csv "outputs/complexity_results.csv"
python scripts/run_skin_type_inference.py --input-dir "path/to/images" --metadata-csv "outputs/scene_results.csv" --output-csv "outputs/skin_type_results.csv"
```

Then generate figures:

```bash
python scripts/plot_scene_results.py --inputs "outputs/scene_results.csv" --dataset-names "MyDataset" --output-dir "outputs/scene_figures"
python scripts/plot_complexity_results.py --inputs "outputs/complexity_results.csv" --dataset-names "MyDataset" --output-dir "outputs/complexity_figures"
python scripts/plot_skin_type_results.py --inputs "outputs/skin_type_results.csv" --dataset-names "MyDataset" --output-dir "outputs/skin_type_figures"
```

## Recommended GitHub Positioning

For GitHub visitors, the best reading order is:

1. `README.md`
2. `DEMO.md`
3. `examples/sample_outputs/cid2013/`
