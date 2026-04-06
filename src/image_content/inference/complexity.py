import json
from pathlib import Path
from typing import Dict

from tqdm import tqdm

from image_content.common.images import format_image_identifier
from image_content.common.vlm import load_text_file, run_image_prompt

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_PROMPT_PATH = ROOT_DIR / "configs" / "prompts" / "visual_complexity.txt"

LEVELS = {"low", "medium", "high"}
ERROR_VALUE = "error"


def load_prompt(prompt_path: Path = DEFAULT_PROMPT_PATH) -> str:
    return load_text_file(prompt_path)


def strip_code_fence(text: str) -> str:
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        return "\n".join(lines).strip()
    return text.strip()


def parse_response(raw_text: str) -> Dict[str, str]:
    if not raw_text:
        raise ValueError("Empty response from model.")

    cleaned = strip_code_fence(raw_text)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON decode error: {exc}")

    required = [
        "quantity_level",
        "quantity_evidence",
        "clutter_level",
        "clutter_evidence",
    ]
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}")

    quantity_level = str(payload["quantity_level"]).strip().lower()
    clutter_level = str(payload["clutter_level"]).strip().lower()
    if quantity_level not in LEVELS:
        raise ValueError(f"Invalid quantity_level: {quantity_level}")
    if clutter_level not in LEVELS:
        raise ValueError(f"Invalid clutter_level: {clutter_level}")

    return {
        "quantity_level": quantity_level,
        "quantity_evidence": str(payload["quantity_evidence"]).strip(),
        "clutter_level": clutter_level,
        "clutter_evidence": str(payload["clutter_evidence"]).strip(),
    }


def error_row(image_path: Path, root_dir: Path, message: str) -> Dict[str, str]:
    tqdm.write(f"[ERROR] {image_path.name}: {message}")
    return {
        "image_name": format_image_identifier(image_path, root_dir),
        "quantity_level": ERROR_VALUE,
        "quantity_evidence": ERROR_VALUE,
        "clutter_level": ERROR_VALUE,
        "clutter_evidence": ERROR_VALUE,
    }


def build_processor(
    client,
    prompt_text: str,
    model_name: str,
    timeout: int,
    temperature: float,
    root_dir: Path,
):
    def process_image(image_path: Path) -> Dict[str, str]:
        try:
            text = run_image_prompt(
                client=client,
                image_path=image_path,
                prompt_text=prompt_text,
                model_name=model_name,
                timeout=timeout,
                temperature=temperature,
            )
            parsed = parse_response(text)
        except Exception as exc:
            return error_row(image_path, root_dir, str(exc))

        parsed["image_name"] = format_image_identifier(image_path, root_dir)
        return parsed

    return process_image
