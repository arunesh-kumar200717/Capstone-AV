from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def precision_at_k(relevants: List[str], retrieved: List[str], k: int = 3) -> float:
    retrieved_k = retrieved[:k]
    if not retrieved_k:
        return 0.0
    hits = sum(1 for item in retrieved_k if item in relevants)
    return hits / len(retrieved_k)


def recall_at_k(relevants: List[str], retrieved: List[str], k: int = 3) -> float:
    retrieved_k = retrieved[:k]
    if not relevants:
        return 0.0
    hits = sum(1 for item in retrieved_k if item in relevants)
    return hits / len(relevants)


def reciprocal_rank(relevants: List[str], retrieved: List[str]) -> float:
    for idx, item in enumerate(retrieved, start=1):
        if item in relevants:
            return 1.0 / idx
    return 0.0


def hit_rate_at_k(relevants: List[str], retrieved: List[str], k: int = 3) -> float:
    return 1.0 if any(item in relevants for item in retrieved[:k]) else 0.0


def ndcg_at_k(relevants: List[str], retrieved: List[str], k: int = 3) -> float:
    if not retrieved:
        return 0.0
    dcg = 0.0
    for idx, item in enumerate(retrieved[:k], start=1):
        if item in relevants:
            dcg += 1.0 / np.log2(idx + 1)
    ideal = 0.0
    for idx in range(1, min(len(relevants), k) + 1):
        ideal += 1.0 / np.log2(idx + 1)
    return dcg / ideal if ideal else 0.0


def validate_citation_ids(references: List[str], retrieved_chunk_ids: List[str]) -> bool:
    valid = set(retrieved_chunk_ids)
    return all(ref in valid for ref in references)


def evaluate_answer(question: str, answer: str, references: List[str], context_chunks: List[str]) -> Dict[str, Any]:
    score = 0.0
    grounded = any(ref for ref in references)
    if answer and answer.strip():
        score += 0.5
    if grounded:
        score += 0.5
    return {
        'question': question,
        'answer_length': len(answer),
        'citation_count': len(references),
        'grounded': grounded,
        'score': round(score, 4),
    }


def export_csv(results: List[Dict[str, Any]], output_path: str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys()) if results else []
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    return str(path)


def export_json(results: List[Dict[str, Any]], output_path: str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(results, handle, indent=2)
    return str(path)
