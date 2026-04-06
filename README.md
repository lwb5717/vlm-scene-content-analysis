# Image Content Analysis

Run a vision-language model on your own image dataset, export structured CSV results, and optionally generate basic figures from those results.

This repository is designed as a simple research artifact:

- input: an image directory
- inference: scene attributes, visual complexity, and skin-type analysis
- output: CSV files, basic figures, and optional paper-analysis CSV exports

## Demo

Start here if you want the fastest overview:

- [DEMO.md](./DEMO.md)
- [examples/sample_outputs/cid2013](./examples/sample_outputs/cid2013)

### Project Figure

- [figure_scene_content_info.pdf](./figure_scene_content_info.pdf)

## Features

- configurable dataset path from the command line
- prompt files separated from code
- `.env`-based API key setup
- reusable inference modules under `src/`
- ready-to-run CLI scripts under `scripts/`
- curated sample outputs under `examples/sample_outputs/`

## Repository Structure

- `configs/prompts/`
  prompt files for scene, complexity, and skin-type tasks
- `src/image_content/common/`
  shared utilities for image loading, API calls, and CSV writing
- `src/image_content/inference/`
  core task implementations for scene, complexity, and skin-type inference
- `src/image_content/visualization/`
  plotting implementations
- `src/image_content/advanced_analysis/`
  optional advanced analyses such as SPAQ validation, PCA, CQ, and complexity-consistency validation
- `scripts/`
  user-facing entry points
- `examples/sample_outputs/`
  small example outputs kept for GitHub display
- `result/result_gallery/`
  curated result gallery for cross-dataset and manuscript-style outputs

## Setup

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Create a local `.env` file:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Then edit `.env`:

```env
DASHSCOPE_API_KEY=your_api_key_here
```

The scripts automatically read `.env` from the project root. This is the recommended setup.

## Quick Start

### 1. Run scene inference

```bash
python scripts/run_scene_inference.py --input-dir "path/to/images" --output-csv "outputs/scene_results.csv"
```

### 2. Run complexity inference

```bash
python scripts/run_complexity_inference.py --input-dir "path/to/images" --output-csv "outputs/complexity_results.csv"
```

### 3. Run skin-type inference

If you already generated scene results, you can use `--metadata-csv` to prefilter images whose `memory_color` contains `human_skin_tone`.

```bash
python scripts/run_skin_type_inference.py --input-dir "path/to/images" --metadata-csv "outputs/scene_results.csv" --output-csv "outputs/skin_type_results.csv"
```

### 4. Generate figures

```bash
python scripts/plot_scene_results.py --inputs "outputs/scene_results.csv" --dataset-names "MyDataset" --output-dir "outputs/scene_figures"
python scripts/plot_complexity_results.py --inputs "outputs/complexity_results.csv" --dataset-names "MyDataset" --output-dir "outputs/complexity_figures"
python scripts/plot_skin_type_results.py --inputs "outputs/skin_type_results.csv" --dataset-names "MyDataset" --output-dir "outputs/skin_type_figures"
```

### 5. Optional Advanced Analyses

After you have generated scene or complexity CSV results, you can optionally run the more advanced analyses used in our study, such as:

- semantic PCA
- CQ-based grouped complexity analysis
- complexity consistency validation
- SPAQ label validation

These scripts are provided as reusable examples of downstream analysis. They are not required for the basic pipeline, and users do not need to reproduce our exact study setup.

Most users can stop after the inference and basic plotting steps above.

## Main Tasks

- scene attribute extraction
- visual complexity analysis
- skin-type analysis

## Key Runtime Options

All inference scripts support the same main configuration pattern:

- `--model`
- `--base-url`
- `--api-env-var`
- `--max-workers`
- `--checkpoint-csv`
- `--checkpoint-interval`
- `--timeout`
- `--temperature`
- `--prompt-file`

## Example Outputs

See the curated example output directory:

- [examples/sample_outputs/cid2013](./examples/sample_outputs/cid2013)

This is intentionally small. The repository keeps representative outputs, not every local experimental artifact.

## Result Gallery

For a broader showcase of project outputs beyond the minimal example, see:

- [result/result_gallery](./result/result_gallery)

This directory now acts as a curated result gallery. It includes both manuscript-facing outputs and additional representative results that help show the full scope of the analysis pipeline.

## Paper Coverage

This repository now includes the executable code and retained figures for:

- closed-set scene attribute annotation
- visual complexity annotation
- skin-type analysis
- cross-dataset attribute plots
- SPAQ validation
- CID2013 complexity group-consistency analysis
- semantic PCA exports
- CQ complexity exports

The low-level descriptor analysis discussed in the manuscript is not yet included as an executable pipeline in this repository snapshot.

## Notes

- Input images are discovered recursively under `--input-dir`.
- Output identifiers are stored as paths relative to the input root when possible.
- If needed, you can still set `DASHSCOPE_API_KEY` manually in your shell instead of using `.env`.
