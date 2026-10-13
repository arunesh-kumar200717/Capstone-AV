from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.export_utils import export_dataset


report = [
    {'section': 'architecture', 'status': 'implemented', 'notes': 'Core components created.'},
    {'section': 'retrieval', 'status': 'implemented', 'notes': 'Hybrid and BM25 retrieval available.'},
    {'section': 'llm', 'status': 'configurable', 'notes': 'OpenAI-compatible and Ollama supported.'},
    {'section': 'evaluation', 'status': 'draft', 'notes': 'Uses real evaluation data when available.'},
]

if __name__ == '__main__':
    output_file = Path('artifacts/submission_report.json')
    export_dataset(report, str(output_file))
    print(f'Submission report generated: {output_file}')
