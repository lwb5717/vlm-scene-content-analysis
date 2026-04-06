import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

def main() -> int:
    parser = argparse.ArgumentParser(description="Plot scene statistics from scene-analysis CSV files.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Paths to scene analysis CSVs.")
    parser.add_argument(
        "--dataset-names",
        nargs="*",
        required=True,
        help="Display names for datasets. Must match number of --inputs.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to write plots.")
    parser.parse_args()

    from image_content.visualization.plot_scene_analysis import main as plot_main
    return plot_main()


if __name__ == "__main__":
    raise SystemExit(main())
