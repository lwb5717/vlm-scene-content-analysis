import base64
from io import BytesIO
from mimetypes import guess_type
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageOps

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
MAX_DATA_URL_BYTES = 10_200_000
MAX_IMAGE_DIM = 1536
JPEG_QUALITY = 85


def enumerate_images(root: Path) -> List[Path]:
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Image directory not found: {root}")
    files: List[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(path.resolve())
    return sorted(files)


def format_image_identifier(image_path: Path, root_dir: Optional[Path] = None) -> str:
    image_path = image_path.resolve()
    if root_dir is not None:
        try:
            return image_path.relative_to(root_dir.resolve()).as_posix()
        except ValueError:
            pass
    return image_path.name


def build_image_index(
    image_paths: List[Path],
    root_dir: Optional[Path] = None,
) -> Tuple[Dict[str, Path], Dict[str, List[Path]]]:
    identifier_map: Dict[str, Path] = {}
    filename_map: Dict[str, List[Path]] = {}
    for path in image_paths:
        identifier_map[format_image_identifier(path, root_dir)] = path
        filename_map.setdefault(path.name, []).append(path)
    return identifier_map, filename_map


def _data_url_size(byte_len: int, mime_type: str) -> int:
    header = f"data:{mime_type};base64,"
    base64_len = 4 * ((byte_len + 2) // 3)
    return len(header) + base64_len


def _guess_mime_and_format(image_path: Path, raw_bytes: bytes) -> Tuple[str, str]:
    try:
        with Image.open(BytesIO(raw_bytes)) as img:
            image_format = (img.format or "").upper()
    except Exception:
        image_format = ""

    if image_format == "JPEG":
        return "image/jpeg", "JPEG"
    if image_format == "PNG":
        return "image/png", "PNG"
    if image_format == "WEBP":
        return "image/webp", "WEBP"
    if image_format == "BMP":
        return "image/bmp", "BMP"
    if image_format in ("TIFF", "TIF"):
        return "image/tiff", "TIFF"

    mime_type, _ = guess_type(image_path.name)
    suffix = image_path.suffix.lstrip(".").upper() or "JPEG"
    return mime_type or "application/octet-stream", suffix


def encode_image_bytes(
    image_path: Path,
    max_data_url_bytes: int = MAX_DATA_URL_BYTES,
    max_image_dim: int = MAX_IMAGE_DIM,
    jpeg_quality: int = JPEG_QUALITY,
) -> Tuple[bytes, str]:
    image_path = image_path.resolve()
    data = image_path.read_bytes()
    mime_type, image_format = _guess_mime_and_format(image_path, data)
    if _data_url_size(len(data), mime_type) <= max_data_url_bytes:
        return data, mime_type

    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img)
        if not image_format:
            image_format = image_path.suffix.lstrip(".").upper() or "JPEG"

        keep_alpha = image_format == "PNG" and img.mode in ("RGBA", "LA")
        if keep_alpha:
            img = img.copy()
        elif img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        else:
            img = img.copy()

        scale = 1.0
        quality = jpeg_quality
        current_mime = mime_type

        while True:
            resized = img.copy()
            resized.thumbnail((int(max_image_dim * scale), int(max_image_dim * scale)), Image.LANCZOS)
            buffer = BytesIO()
            save_kwargs = {}

            if image_format in ("JPG", "JPEG"):
                image_format = "JPEG"
                current_mime = "image/jpeg"
                save_kwargs = {"quality": quality, "optimize": True}
            elif image_format == "WEBP":
                current_mime = "image/webp"
                save_kwargs = {"quality": quality, "method": 6}
            elif image_format == "PNG":
                current_mime = "image/png"
                save_kwargs = {"optimize": True, "compress_level": 9}
            else:
                image_format = "JPEG"
                current_mime = "image/jpeg"
                save_kwargs = {"quality": quality, "optimize": True}

            resized.save(buffer, format=image_format, **save_kwargs)
            data = buffer.getvalue()

            if _data_url_size(len(data), current_mime) <= max_data_url_bytes:
                break

            if quality > 30 and image_format in ("JPEG", "WEBP"):
                quality -= 10
            else:
                scale *= 0.85
                if scale < 0.4:
                    break

        if _data_url_size(len(data), current_mime) > max_data_url_bytes:
            if img.mode in ("RGBA", "LA"):
                background = Image.new("RGBA", img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(background, img.convert("RGBA")).convert("RGB")
            elif img.mode != "RGB":
                img = img.convert("RGB")

            image_format = "JPEG"
            current_mime = "image/jpeg"
            scale = 0.85
            quality = 70

            while _data_url_size(len(data), current_mime) > max_data_url_bytes and scale >= 0.2:
                resized = img.copy()
                resized.thumbnail((int(max_image_dim * scale), int(max_image_dim * scale)), Image.LANCZOS)
                buffer = BytesIO()
                resized.save(buffer, format="JPEG", quality=quality, optimize=True)
                data = buffer.getvalue()
                if _data_url_size(len(data), current_mime) <= max_data_url_bytes:
                    break
                if quality > 30:
                    quality -= 10
                else:
                    scale *= 0.85

    return data, current_mime


def image_to_data_url(image_path: Path) -> str:
    data, mime_type = encode_image_bytes(image_path)
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
