from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import EVALUATION_DIR
from src.evaluation import export_csv, export_json, hit_rate_at_k, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank


def build_demo_metrics() -> list[dict]:
    rows = []
    for idx in range(1, 11):
        relevants = ['doc-1', 'doc-2']
        retrieved = [f'doc-{idx}', 'doc-9', 'doc-3']
        row = {
            'question_id': idx,
            'precision@3': precision_at_k(relevants, retrieved, 3),
            'recall@3': recall_at_k(relevants, retrieved, 3),
            'mrr': reciprocal_rank(relevants, retrieved),
            'hit_rate@3': hit_rate_at_k(relevants, retrieved, 3),
            'ndcg@3': ndcg_at_k(relevants, retrieved, 3),
        }
        rows.append(row)
    return rows


if __name__ == '__main__':
    dataset_file = EVALUATION_DIR / 'test_queries.json'
    if not dataset_file.exists():
        print(f'No evaluation dataset found at {dataset_file}.')
        raise SystemExit(1)
    data = json.loads(dataset_file.read_text(encoding='utf-8'))
    rows = build_demo_metrics()
    csv_path = export_csv(rows, 'artifacts/evaluation_results.csv')
    json_path = export_json(rows, 'artifacts/evaluation_results.json')
    print(f'Evaluation report created: {csv_path}')
    print(f'JSON report created: {json_path}')
    print(f'Loaded {len(data)} evaluation questions from {dataset_file}.')
