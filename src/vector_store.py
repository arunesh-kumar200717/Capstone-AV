from __future__ import annotations

from typing import Any, Dict, List, Optional

import chromadb

from config.settings import CHROMA_DIR


class ResearchVectorStore:
    def __init__(self, collection_name: str = 'research_chunks'):
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(name=collection_name, metadata={'hnsw:space': 'cosine'})

    def reset_collection(self):
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name=self.collection_name, metadata={'hnsw:space': 'cosine'})

    def add_documents(self, records: List[Dict[str, Any]], embeddings: List[List[float]], model_name: str) -> None:
        if not records:
            return
        ids = [record['chunk_id'] for record in records]
        metadata = [{
            'chunk_id': record.get('chunk_id'),
            'document_id': record.get('document_id'),
            'paper_title': record.get('paper_title', ''),
            'source_file': record.get('source_file', ''),
            'page_number': str(record.get('page_number', 0)),
            'chunk_index': str(record.get('chunk_index', 0)),
            'model_name': model_name,
        } for record in records]
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadata,
            documents=[record.get('text', '') for record in records],
        )

    def search(self, query_embedding: List[float], top_k: int = 5, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )

    def count(self) -> int:
        return self.collection.count()

    def delete_document(self, document_id: str) -> None:
        ids = self.collection.get(where={'document_id': document_id})['ids']
        if ids:
            self.collection.delete(ids=ids)

    def get_by_document(self, document_id: str) -> Dict[str, Any]:
        return self.collection.get(where={'document_id': document_id})

    def persist(self):
        # PersistentClient writes immediately; keep a no-op hook for call sites.
        return None
