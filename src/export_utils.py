from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def export_dataset(rows: List[Dict[str, Any]], filename: str) -> str:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    return str(path)


def export_csv(rows: List[Dict[str, Any]], filename: str) -> str:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return str(path)
    fieldnames = list(rows[0].keys())
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return str(path)
