from src.retrieval import bm25_search


def test_bm25_search_returns_ranked_results():
    documents = ['Retrieval augmented generation uses external memory.', 'LLMs can hallucinate without evidence.']
    metadata = [
        {'chunk_id': 'c1', 'paper_title': 'Paper One', 'source_file': 'one.pdf', 'page_number': 3, 'document_id': 'd1'},
        {'chunk_id': 'c2', 'paper_title': 'Paper Two', 'source_file': 'two.pdf', 'page_number': 5, 'document_id': 'd2'},
    ]
    results = bm25_search(documents, metadata, 'retrieval augmented generation', 3)
    assert results
    assert results[0]['chunk_id'] == 'c1'
