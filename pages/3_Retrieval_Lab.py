from __future__ import annotations

import streamlit as st

from src.embeddings import EmbeddingManager
from src.retrieval import bm25_search, dense_search, hybrid_search, mmr_search
from src.vector_store import ResearchVectorStore

st.set_page_config(page_title='Retrieval Lab', layout='wide')

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = ResearchVectorStore('research_chunks')
if 'embedding_manager' not in st.session_state:
    st.session_state.embedding_manager = EmbeddingManager('sentence-transformers/all-MiniLM-L6-v2')

st.title('Retrieval Laboratory')
question = st.text_input('Enter a retrieval query', value='What is retrieval-augmented generation?')
strategy = st.selectbox('Retrieval strategy', ['hybrid', 'dense', 'mmr', 'bm25'])
model_name = st.selectbox('Embedding model', ['sentence-transformers/all-MiniLM-L6-v2', 'BAAI/bge-small-en-v1.5'], index=0)
ks = st.slider('Top-k', min_value=1, max_value=10, value=3)

if st.button('Run retrieval'):
    storage = st.session_state.vector_store
    all_items = storage.collection.get(include=['documents', 'metadatas'])
    docs = all_items.get('documents', []) or []
    metas = all_items.get('metadatas', []) or []
    if not docs:
        st.info('No indexed documents are available for retrieval.')
        st.stop()

    flattened_docs = []
    for doc in docs:
        if isinstance(doc, list):
            flattened_docs.extend(doc)
        else:
            flattened_docs.append(doc)

    flattened_metas = []
    for meta in metas:
        if isinstance(meta, list):
            flattened_metas.extend(meta)
        else:
            flattened_metas.append(meta)

    chunk_metadata = [
        {
            'chunk_id': item.get('id') or f'chunk-{idx}',
            'paper_title': item.get('paper_title', ''),
            'source_file': item.get('source_file', ''),
            'page_number': int(item.get('page_number', 1)),
            'document_id': item.get('document_id', ''),
        }
        for idx, item in enumerate(flattened_metas)
    ]

    query_embedding = st.session_state.embedding_manager.embed_query(question, model_name=model_name)
    if strategy == 'dense':
        results = dense_search(storage, query_embedding=query_embedding, top_k=ks)
    elif strategy == 'mmr':
        results = mmr_search(storage, query_embedding=query_embedding, top_k=ks)
    elif strategy == 'bm25':
        results = bm25_search(flattened_docs, chunk_metadata, question, top_k=ks)
    else:
        results = hybrid_search(storage, query_embedding=query_embedding, documents=flattened_docs, chunk_metadata=chunk_metadata, query=question, top_k=ks)

    if not results:
        st.warning('No results for the current query. Try another question or index more PDFs.')
    else:
        st.subheader('Top ranked passages')
        for idx, result in enumerate(results[:ks], start=1):
            st.markdown(f'### Rank {idx}: **{result.get("paper_title", "Unknown")}** — Page {result.get("page_number", 1)}')
            st.caption(f'Chunk ID: {result.get("chunk_id", "n/a")} | Source file: {result.get("source_file", "n/a")} | Score: {result.get("score", 0)}')
            st.write(result.get('text', ''))
