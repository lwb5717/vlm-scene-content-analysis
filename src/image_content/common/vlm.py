import os
from pathlib import Path
from typing import Optional

from openai import OpenAI

from image_content.common.images import image_to_data_url

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_ENV_FILE = ROOT_DIR / ".env"


def load_text_file(path: Path) -> str:
    return path.resolve().read_text(encoding="utf-8").strip()


def load_env_file(env_path: Path = DEFAULT_ENV_FILE) -> None:
    env_path = env_path.resolve()
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def create_client(
    api_env_var: str = "DASHSCOPE_API_KEY",
    base_url: str = DEFAULT_BASE_URL,
    api_key: Optional[str] = None,
) -> OpenAI:
    load_env_file()
    key = (api_key or os.environ.get(api_env_var, "")).strip()
    if not key:
        raise RuntimeError(
            f"API key missing. Add {api_env_var}=... to the project .env file "
            f"(copy .env.example to .env), or set the {api_env_var} environment variable."
        )
    return OpenAI(api_key=key, base_url=base_url)


def run_image_prompt(
    client: OpenAI,
    image_path: Path,
    prompt_text: str,
    model_name: str,
    timeout: int,
    temperature: float = 0.0,
) -> str:
    img_data_url = image_to_data_url(image_path)
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": img_data_url}},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ],
        stream=False,
        temperature=temperature,
        timeout=timeout,
    )
    if not completion.choices:
        raise ValueError("No choices in response.")
    return completion.choices[0].message.content
