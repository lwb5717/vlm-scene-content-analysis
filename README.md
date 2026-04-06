# Image Content Analysis

Run a vision-language model on your own image dataset, export structured CSV results, and generate paper-style figures from those results.

This repository is designed as a simple research artifact:

- input: an image directory
- inference: scene attributes, visual complexity, and skin-type analysis
- output: CSV files and summary figures

## Demo

Start here if you want the fastest overview:

- [DEMO.md](./DEMO.md)
- [examples/sample_outputs/cid2013](./examples/sample_outputs/cid2013)

### Visual Preview

#### Scene Distribution

![Scene distribution](examples/sample_outputs/cid2013/figures/scene_type_distribution.png)

#### Complexity Distribution

![Complexity distribution](examples/sample_outputs/cid2013/figures/complexity_9_levels.png)

#### Skin-Type Distribution

![Skin-type distribution](examples/sample_outputs/cid2013/figures/skin_type_distribution.png)

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
- `scripts/`
  user-facing entry points
- `examples/sample_outputs/`
  small example outputs kept for GitHub display
- `result/paper_figures/`
  selected paper-facing figures

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

## Notes

- Input images are discovered recursively under `--input-dir`.
- Output identifiers are stored as paths relative to the input root when possible.
- If needed, you can still set `DASHSCOPE_API_KEY` manually in your shell instead of using `.env`.
