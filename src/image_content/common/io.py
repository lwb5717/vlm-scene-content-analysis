from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def sort_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not rows:
        return rows
    for key in ("image_name", "image_path"):
        if key in rows[0]:
            return sorted(rows, key=lambda row: str(row.get(key, "")))
    return rows


def save_rows(rows: List[Dict[str, str]], csv_path: Optional[Path]) -> None:
    if not rows or csv_path is None:
        return
    csv_path = csv_path.resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sort_rows(rows)).to_csv(csv_path, index=False, encoding="utf-8-sig")
