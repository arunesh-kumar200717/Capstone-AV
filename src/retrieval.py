from __future__ import annotations

from typing import Any, Dict, List

from rank_bm25 import BM25Okapi


def reciprocal_rank_fusion(ranked_lists: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
    fused: Dict[str, Dict[str, Any]] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            chunk_id = item['chunk_id']
            if chunk_id not in fused:
                fused[chunk_id] = {
                    'chunk_id': chunk_id,
                    'score': 0.0,
                    'paper_title': item.get('paper_title', ''),
                    'source_file': item.get('source_file', ''),
                    'page_number': item.get('page_number', 1),
                    'text': item.get('text', ''),
                    'document_id': item.get('document_id', ''),
                    'metadata': item.get('metadata', {}),
                }
            fused[chunk_id]['score'] += 1.0 / (k + rank)
    ranked = sorted(fused.values(), key=lambda x: x['score'], reverse=True)
    for item in ranked:
        item['score'] = round(item['score'], 6)
    return ranked


def dense_search(vector_store, query_embedding: List[float], top_k: int = 5, where: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    result = vector_store.search(query_embedding=query_embedding, top_k=top_k, where=where)
    items = []
    for idx, doc_id in enumerate(result.get('ids', [[]])[0]):
        meta = result.get('metadatas', [[]])[0][idx]
        items.append({
            'chunk_id': doc_id,
            'text': result.get('documents', [[]])[0][idx],
            'paper_title': meta.get('paper_title', ''),
            'source_file': meta.get('source_file', ''),
            'page_number': int(meta.get('page_number', '1')),
            'document_id': meta.get('document_id', ''),
            'score': float(result.get('distances', [[]])[0][idx]) if result.get('distances') else 0.0,
            'metadata': meta,
        })
    return items


def bm25_search(documents: List[str], chunk_metadata: List[Dict[str, Any]], query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    if not query or not documents:
        return []
    tokenized_corpus = [doc.lower().split() for doc in documents]
    if not any(tokenized_corpus):
        return []
    bm25 = BM25Okapi(tokenized_corpus)
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    pairs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    results = []
    for idx, _ in pairs:
        meta = chunk_metadata[idx]
        results.append({
            'chunk_id': meta.get('chunk_id', f'chunk-{idx}'),
            'text': documents[idx],
            'paper_title': meta.get('paper_title', ''),
            'source_file': meta.get('source_file', ''),
            'page_number': int(meta.get('page_number', 1)),
            'document_id': meta.get('document_id', ''),
            'score': float(scores[idx]),
            'metadata': meta,
        })
    return results


def hybrid_search(vector_store, query_embedding: List[float], documents: List[str], chunk_metadata: List[Dict[str, Any]], query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    dense_results = dense_search(vector_store, query_embedding=query_embedding, top_k=top_k * 4)
    bm25_results = bm25_search(documents, chunk_metadata, query, top_k=top_k * 4)
    fused = reciprocal_rank_fusion([dense_results, bm25_results])
    deduped: Dict[str, Dict[str, Any]] = {}
    for item in fused:
        deduped[item['chunk_id']] = item
    ordered = sorted(deduped.values(), key=lambda x: x['score'], reverse=True)[:top_k]
    return ordered


def mmr_search(vector_store, query_embedding: List[float], top_k: int = 5, lambda_mult: float = 0.5) -> List[Dict[str, Any]]:
    result = vector_store.search(query_embedding=query_embedding, top_k=max(top_k * 5, 10))
    candidate_ids = result.get('ids', [[]])[0]
    docs = result.get('documents', [[]])[0]
    metas = result.get('metadatas', [[]])[0]
    selected = []
    while len(selected) < min(top_k, len(candidate_ids)):
        best_idx = None
        best_score = -1e9
        for idx, doc in enumerate(docs):
            if idx in [item['idx'] for item in selected]:
                continue
            diversity_term = 0.0
            for item in selected:
                diversity_term += 1.0 if doc == item['text'] else 0.0
            score = 1.0 - lambda_mult * diversity_term
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            break
        meta = metas[best_idx]
        selected.append({
            'idx': best_idx,
            'chunk_id': candidate_ids[best_idx],
            'text': docs[best_idx],
            'paper_title': meta.get('paper_title', ''),
            'source_file': meta.get('source_file', ''),
            'page_number': int(meta.get('page_number', '1')),
            'document_id': meta.get('document_id', ''),
            'score': 0.0,
            'metadata': meta,
        })
    return selected
