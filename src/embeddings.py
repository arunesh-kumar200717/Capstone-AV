from __future__ import annotations

from typing import List

from sklearn.feature_extraction.text import HashingVectorizer

from config.settings import DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODELS


class EmbeddingManager:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or DEFAULT_EMBEDDING_MODEL
        self._models: dict[str, object] = {}
        self._fallback_vectorizers: dict[str, HashingVectorizer] = {}

    def get_model(self, model_name: str | None = None):
        name = (model_name or self.model_name).strip()
        if name not in self._models:
            try:
                from sentence_transformers import SentenceTransformer

                self._models[name] = SentenceTransformer(name)
            except Exception:
                self._models[name] = None
        return self._models[name]

    def embed_documents(self, texts: List[str], model_name: str | None = None) -> List[List[float]]:
        model = self.get_model(model_name)
        if model is not None:
            embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            return embeddings.tolist()
        vectorizer = self._fallback_vectorizers.setdefault((model_name or self.model_name).strip(), HashingVectorizer(n_features=768, alternate_sign=False, norm='l2'))
        matrix = vectorizer.transform(texts)
        return matrix.toarray().tolist()

    def embed_query(self, query: str, model_name: str | None = None) -> List[float]:
        model = self.get_model(model_name)
        if model is not None:
            embedding = model.encode([query], show_progress_bar=False, normalize_embeddings=True)[0]
            return embedding.tolist()
        vectorizer = self._fallback_vectorizers.setdefault((model_name or self.model_name).strip(), HashingVectorizer(n_features=768, alternate_sign=False, norm='l2'))
        array = vectorizer.transform([query]).toarray()[0]
        return array.tolist()

    def available_models(self) -> List[str]:
        return list(EMBEDDING_MODELS)
