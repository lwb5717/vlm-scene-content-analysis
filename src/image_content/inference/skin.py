import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from image_content.common.images import build_image_index, format_image_identifier
from image_content.common.vlm import load_text_file, run_image_prompt

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_PROMPT_PATH = ROOT_DIR / "configs" / "prompts" / "skin_type.txt"

FITZ_GROUPS = {"light", "medium", "dark"}
FITZ_GROUP_ORDER = ["light", "medium", "dark"]


def load_prompt(prompt_path: Path = DEFAULT_PROMPT_PATH) -> str:
    return load_text_file(prompt_path)


def strip_code_fence(text: str) -> str:
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        return "\n".join(lines).strip()
    return text.strip()


def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower_map = {col.lower(): col for col in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def parse_tag_value(value: str) -> List[str]:
    return [item.strip().lower() for item in re.split(r"[;,]", str(value)) if item.strip()]


def row_has_human_skin_tag(value: str) -> Optional[bool]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    tags = parse_tag_value(value)
    if not tags:
        return None
    return "human_skin_tone" in tags


def resolve_csv_images(
    df: pd.DataFrame,
    dataset_root: Path,
    image_paths: List[Path],
) -> List[Path]:
    identifier_map, filename_map = build_image_index(image_paths, dataset_root)
    path_column = find_column(df, ["image_path", "path"])
    name_column = find_column(df, ["image_name", "image"])
    tag_column = find_column(df, ["memory_color"])
    if not tag_column:
        raise ValueError("Metadata CSV is missing required column: memory_color")

    resolved: Dict[Path, Path] = {}
    for _, row in df.iterrows():
        if row_has_human_skin_tag(row[tag_column]) is not True:
            continue

        image_path: Optional[Path] = None
        if path_column:
            raw_path = str(row[path_column]).strip()
            if raw_path:
                candidate = Path(raw_path)
                if not candidate.is_absolute():
                    candidate = (dataset_root / candidate).resolve()
                if candidate.exists():
                    image_path = candidate

        if image_path is None and name_column:
            name_value = str(row[name_column]).strip()
            if name_value in identifier_map:
                image_path = identifier_map[name_value]
            else:
                candidates = filename_map.get(Path(name_value).name, [])
                if len(candidates) == 1:
                    image_path = candidates[0]

        if image_path is not None:
            resolved[image_path] = image_path

    return sorted(resolved.keys())


def parse_skin_response(raw_text: str) -> Tuple[bool, Optional[List[str]]]:
    if not raw_text:
        raise ValueError("Empty response from model.")

    cleaned = strip_code_fence(raw_text)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON decode error: {exc}")

    if "has_human_skin" not in payload or "skin_types" not in payload:
        raise ValueError("Missing fields in response.")

    has_skin = bool(payload["has_human_skin"])
    if not has_skin:
        return False, None

    def split_labels(value: str) -> List[str]:
        return [item.strip().lower() for item in re.split(r"[;,/]", value) if item.strip()]

    raw_types = payload["skin_types"]
    if isinstance(raw_types, str):
        if raw_types.strip().lower() == "uncertain":
            return True, None
        raw_types = split_labels(raw_types)

    if isinstance(raw_types, Iterable) and not isinstance(raw_types, (str, bytes)):
        normalized: List[str] = []
        for item in raw_types:
            normalized.extend(split_labels(str(item)))
        if "uncertain" in normalized:
            return True, None
        normalized = [item for item in normalized if item in FITZ_GROUPS]
        if not normalized:
            return True, None
        return True, sorted(set(normalized), key=FITZ_GROUP_ORDER.index)

    return True, None


def build_processor(
    client,
    prompt_text: str,
    model_name: str,
    timeout: int,
    temperature: float,
    root_dir: Path,
):
    def process_image(image_path: Path) -> Optional[Dict[str, str]]:
        try:
            text = run_image_prompt(
                client=client,
                image_path=image_path,
                prompt_text=prompt_text,
                model_name=model_name,
                timeout=timeout,
                temperature=temperature,
            )
            has_skin, skin_types = parse_skin_response(text)
        except Exception as exc:
            tqdm.write(f"[ERROR] {image_path.name}: {exc}")
            return {
                "image_path": format_image_identifier(image_path, root_dir),
                "skin_types": "uncertain",
            }

        if not has_skin:
            return None

        return {
            "image_path": format_image_identifier(image_path, root_dir),
            "skin_types": "uncertain" if not skin_types else json.dumps(skin_types),
        }

    return process_image
