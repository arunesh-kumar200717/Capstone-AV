from src.rag_pipeline import ResearchRAG


class FakeVectorStore:
    def __init__(self):
        self.collection = self

    def get(self, include=None):
        return {
            'ids': [['c1']],
            'documents': [['Retrieval-augmented generation uses external evidence.']],
            'metadatas': [[{'chunk_id': 'c1', 'document_id': 'd1', 'paper_title': 'Paper', 'source_file': 'paper.pdf', 'page_number': 2}]],
        }

    def search(self, query_embedding, top_k=5, where=None):
        return {
            'ids': [['c1']],
            'documents': [['Retrieval-augmented generation uses external evidence.']],
            'metadatas': [[{'id': 'c1', 'document_id': 'd1', 'paper_title': 'Paper', 'source_file': 'paper.pdf', 'page_number': 2}]],
            'distances': [[0.1]],
        }


def test_rag_without_llm_returns_insufficient_evidence_message(monkeypatch):
    rag = ResearchRAG(FakeVectorStore())
    monkeypatch.setattr(rag.llm, 'is_configured', lambda: False)
    result = rag.answer_question('What is RAG?', model_name='sentence-transformers/all-MiniLM-L6-v2', strategy='dense', top_k=3)
    assert 'LLM is not configured' in result['answer']
    assert result['insufficient_evidence'] is True
