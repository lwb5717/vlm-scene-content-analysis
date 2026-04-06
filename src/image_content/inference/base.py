from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional

from tqdm import tqdm

from image_content.common.io import save_rows


def run_inference_task(
    image_paths: List[Path],
    process_image: Callable[[Path], Optional[Dict[str, str]]],
    output_csv: Path,
    checkpoint_csv: Optional[Path] = None,
    checkpoint_interval: int = 50,
    max_workers: int = 8,
    progress_label: str = "Inference",
) -> int:
    if not image_paths:
        raise RuntimeError("No image files were selected for processing.")

    results: List[Dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_image, path): path for path in image_paths}
        for index, future in enumerate(
            tqdm(as_completed(futures), total=len(futures), desc=progress_label),
            start=1,
        ):
            row = future.result()
            if row is not None:
                results.append(row)
            if checkpoint_csv is not None and checkpoint_interval > 0 and index % checkpoint_interval == 0:
                save_rows(results, checkpoint_csv)

    save_rows(results, output_csv)
    return len(results)
