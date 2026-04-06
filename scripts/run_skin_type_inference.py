import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run skin-type inference on an image directory.")
    parser.add_argument("--input-dir", required=True, help="Root directory of images.")
    parser.add_argument("--output-csv", required=True, help="Output CSV path.")
    parser.add_argument("--metadata-csv", help="Optional scene-analysis CSV used to prefilter human-skin images.")
    parser.add_argument("--checkpoint-csv", help="Optional checkpoint CSV path.")
    parser.add_argument("--prompt-file", default=str(ROOT_DIR / "configs" / "prompts" / "skin_type.txt"))
    parser.add_argument("--model", default="qwen3-vl-plus")
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api-env-var", default="DASHSCOPE_API_KEY")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    import pandas as pd

    from image_content.common.images import enumerate_images
    from image_content.common.vlm import create_client
    from image_content.inference.base import run_inference_task
    from image_content.inference.skin import build_processor, load_prompt, resolve_csv_images

    input_dir = Path(args.input_dir).resolve()
    output_csv = Path(args.output_csv).resolve()
    checkpoint_csv = Path(args.checkpoint_csv).resolve() if args.checkpoint_csv else None

    image_paths = enumerate_images(input_dir)
    if args.metadata_csv:
        metadata_df = pd.read_csv(Path(args.metadata_csv).resolve())
        image_paths = resolve_csv_images(metadata_df, input_dir, image_paths)

    client = create_client(api_env_var=args.api_env_var, base_url=args.base_url)
    prompt_text = load_prompt(Path(args.prompt_file))
    process_image = build_processor(
        client=client,
        prompt_text=prompt_text,
        model_name=args.model,
        timeout=args.timeout,
        temperature=args.temperature,
        root_dir=input_dir,
    )

    written_rows = run_inference_task(
        image_paths=image_paths,
        process_image=process_image,
        output_csv=output_csv,
        checkpoint_csv=checkpoint_csv,
        checkpoint_interval=args.checkpoint_interval,
        max_workers=args.max_workers,
        progress_label="Skin analysis",
    )
    print(f"Processed {len(image_paths)} images and wrote {written_rows} rows to {output_csv}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
