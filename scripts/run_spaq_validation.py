import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main() -> int:
    from image_content.paper.validate_spaq_labels import main as run_main
    run_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

