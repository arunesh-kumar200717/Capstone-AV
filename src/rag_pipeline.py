from __future__ import annotations

import time
import logging
from typing import Any, Dict, List

from config.prompts import SYSTEM_PROMPT, build_retrieval_prompt
from src.embeddings import EmbeddingManager
from src.llm_provider import LLMProvider, parse_json_response
from src.retrieval import bm25_search, dense_search, hybrid_search, mmr_search

LOGGER = logging.getLogger(__name__)


class ResearchRAG:
    def __init__(self, vector_store, document_index=None):
        self.vector_store = vector_store
        self.embedding_manager = EmbeddingManager()
        self.llm = LLMProvider()
        self.document_index = document_index or {}

    def build_context(self, chunks: List[Dict[str, Any]]) -> str:
        sections = []
        for item in chunks:
            sections.append(f"[Chunk ID: {item['chunk_id']}] Paper: {item['paper_title']} | Page {item['page_number']} | File: {item['source_file']}\n{item['text']}")
        return '\n\n'.join(sections)

    def answer_question(self, question: str, model_name: str | None = None, strategy: str = 'hybrid', top_k: int = 3, history: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
        if not question or not question.strip():
            raise ValueError('Question cannot be empty.')
        start = time.time()
        history = history or []
        query_embedding = self.embedding_manager.embed_query(question, model_name=model_name)
        all_docs = self.vector_store.collection.get(include=['documents', 'metadatas'])
        docs = all_docs.get('documents', []) or []
        metas = all_docs.get('metadatas', []) or []
        ids = all_docs.get('ids', []) or []
        if not docs:
            return {
                'answer': 'No indexed documents are available. Upload and index at least one research paper before asking questions.',
                'references': [],
                'sources': [],
                'strategy': strategy,
                'model_name': model_name,
                'response_time_ms': round((time.time() - start) * 1000, 1),
                'insufficient_evidence': True,
                'warning': 'The document library is empty.',
            }

        flattened_docs = []
        flattened_metas = []
        flattened_ids = []
        for doc in docs:
            if isinstance(doc, list):
                flattened_docs.extend(doc)
            else:
                flattened_docs.append(doc)
        for meta in metas:
            if isinstance(meta, list):
                flattened_metas.extend(meta)
            else:
                flattened_metas.append(meta)
        for chunk_ids in ids:
            if isinstance(chunk_ids, list):
                flattened_ids.extend(chunk_ids)
            else:
                flattened_ids.append(chunk_ids)

        chunk_records = []
        for idx, text in enumerate(flattened_docs):
            item = flattened_metas[idx] if idx < len(flattened_metas) else {}
            chunk_records.append({
                'chunk_id': flattened_ids[idx] if idx < len(flattened_ids) else item.get('chunk_id') or f'chunk-{idx}',
                'document_id': item.get('document_id', ''),
                'paper_title': item.get('paper_title', ''),
                'source_file': item.get('source_file', ''),
                'page_number': int(item.get('page_number', 1)),
                'text': text,
            })

        if strategy == 'dense':
            results = dense_search(self.vector_store, query_embedding=query_embedding, top_k=top_k)
        elif strategy == 'mmr':
            results = mmr_search(self.vector_store, query_embedding=query_embedding, top_k=top_k)
        elif strategy == 'bm25':
            results = bm25_search(flattened_docs, chunk_records, question, top_k=top_k)
        else:
            results = hybrid_search(self.vector_store, query_embedding=query_embedding, documents=flattened_docs, chunk_metadata=chunk_records, query=question, top_k=top_k)
        results = self._deduplicate(results)

        passages = []
        for item in results:
            passages.append({
                'chunk_id': item['chunk_id'],
                'paper_title': item.get('paper_title', ''),
                'page_number': item.get('page_number', 1),
                'source_file': item.get('source_file', ''),
                'text': item.get('text', ''),
                'score': item.get('score', 0.0),
            })
        context = self.build_context(passages)
        prompt = build_retrieval_prompt(question, context)

        try:
            if not self.llm.is_configured():
                structured = {
                    'answer': 'LLM is not configured. The system can show retrieved evidence, but no grounded model-generated answer is available yet.',
                    'references': [item['chunk_id'] for item in passages],
                    'insufficient_evidence': True,
                    'warning': 'LLM provider not configured. Retrieval-only mode is active.',
                }
            else:
                raw = self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT)
                structured = parse_json_response(raw.get('text', ''))
                structured['references'] = self._validate_references(structured.get('references', []), passages)
                if not structured.get('references'):
                    structured['insufficient_evidence'] = True
        except Exception as exc:
            LOGGER.exception('LLM generation failed for provider=%s model=%s', self.llm.provider, self.llm.model_name)
            structured = {
                'answer': f'LLM request failed: {exc}',
                'references': [item['chunk_id'] for item in passages],
                'insufficient_evidence': True,
                'warning': f'LLM error: {exc}',
            }

        response_time = round((time.time() - start) * 1000, 1)
        structured.update({
            'sources': passages,
            'strategy': strategy,
            'model_name': model_name,
            'response_time_ms': response_time,
            'top_supporting_sources': passages[:3]
        })
        return structured

    def _deduplicate(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        deduped = []
        for item in items:
            chunk_id = item.get('chunk_id')
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            deduped.append(item)
        return deduped

    def _validate_references(self, references: List[str], passages: List[Dict[str, Any]]) -> List[str]:
        valid_ids = {item['chunk_id'] for item in passages}
        cleaned = []
        for ref in references:
            if ref in valid_ids:
                cleaned.append(ref)
        return list(dict.fromkeys(cleaned))
