from src.text_splitter import split_chunks_for_pages


def test_split_chunks_preserves_page_metadata():
    document_id = 'doc-1'
    chunks = split_chunks_for_pages(
        document_id=document_id,
        title='Example Paper',
        source_file='example.pdf',
        pages=[{'page_number': 1, 'text': 'Alpha beta gamma. ' * 20}],
        chunk_size=50,
        chunk_overlap=10,
    )
    assert chunks
    assert chunks[0]['document_id'] == document_id
    assert chunks[0]['page_number'] == 1
    assert chunks[0]['source_file'] == 'example.pdf'
