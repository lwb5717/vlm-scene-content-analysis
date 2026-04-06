import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

from tqdm import tqdm

from image_content.common.images import format_image_identifier
from image_content.common.vlm import load_text_file, run_image_prompt

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_PROMPT_PATH = ROOT_DIR / "configs" / "prompts" / "scene_attributes.txt"

SCENE_TYPES = {"natural_landscape", "urban", "indoor", "portrait", "other"}
LIGHTING_TYPES = {"daylight", "artificial", "mixed", "unknown"}
LUMINANCE_LEVELS = {"high", "medium", "low"}
CAMERA_DISTANCES = {"close", "medium", "far", "unknown"}
MEMORY_COLORS = {
    "sky_blue",
    "grass_foliage_green",
    "human_skin_tone",
    "water_blue",
    "snow_cloud_white",
    "none",
}
ERROR_VALUE = "error"


def load_prompt(prompt_path: Path = DEFAULT_PROMPT_PATH) -> str:
    return load_text_file(prompt_path)


def strip_code_fence(text: str) -> str:
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        return "\n".join(lines).strip()
    return text.strip()


def normalize_label(payload: Dict, field: str, allowed: Iterable[str]) -> str:
    value = str(payload[field]).strip().lower()
    if value not in allowed:
        raise ValueError(f"Invalid {field}: {value}")
    return value


def parse_scene_attributes(raw_text: str) -> Dict[str, str]:
    if not raw_text:
        raise ValueError("Empty response from model.")

    cleaned = strip_code_fence(raw_text)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON decode error: {exc}")

    required_fields = [
        "scene_type",
        "lighting_type",
        "luminance_level",
        "memory_color",
        "camera_to_object_distance",
    ]
    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}")

    memory_raw = payload["memory_color"]
    memory_list: List[str] = []
    if isinstance(memory_raw, str):
        memory_list = [item.strip().lower() for item in re.split(r"[;,]", memory_raw) if item.strip()]
    elif isinstance(memory_raw, Iterable) and not isinstance(memory_raw, (str, bytes)):
        memory_list = [str(item).strip().lower() for item in memory_raw if str(item).strip()]
    else:
        value = str(memory_raw).strip().lower()
        if value:
            memory_list = [value]

    if not memory_list:
        memory_list = ["none"]

    invalid = [label for label in memory_list if label not in MEMORY_COLORS]
    if invalid:
        raise ValueError(f"Invalid memory_color: {', '.join(invalid)}")
    if "none" in memory_list and len(memory_list) > 1:
        raise ValueError("memory_color cannot include 'none' with other labels")

    return {
        "scene_type": normalize_label(payload, "scene_type", SCENE_TYPES),
        "lighting_type": normalize_label(payload, "lighting_type", LIGHTING_TYPES),
        "luminance_level": normalize_label(payload, "luminance_level", LUMINANCE_LEVELS),
        "memory_color": ";".join(memory_list),
        "camera_to_object_distance": normalize_label(
            payload,
            "camera_to_object_distance",
            CAMERA_DISTANCES,
        ),
    }


def error_row(image_path: Path, root_dir: Path, message: str) -> Dict[str, str]:
    tqdm.write(f"[ERROR] {image_path.name}: {message}")
    return {
        "image_name": format_image_identifier(image_path, root_dir),
        "scene_type": ERROR_VALUE,
        "lighting_type": ERROR_VALUE,
        "luminance_level": ERROR_VALUE,
        "memory_color": ERROR_VALUE,
        "camera_to_object_distance": ERROR_VALUE,
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
            parsed = parse_scene_attributes(text)
        except Exception as exc:
            return error_row(image_path, root_dir, str(exc))

        parsed["image_name"] = format_image_identifier(image_path, root_dir)
        return parsed

    return process_image
