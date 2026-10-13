from __future__ import annotations

from pathlib import Path

import streamlit as st

from config.settings import CHUNK_OVERLAP, CHUNK_SIZE, DEFAULT_EMBEDDING_MODEL, PAPERS_DIR
from src.document_loader import load_pdf_text
from src.embeddings import EmbeddingManager
from src.text_splitter import split_chunks_for_pages
from src.vector_store import ResearchVectorStore

st.set_page_config(page_title='Document Library', layout='wide')

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = ResearchVectorStore('research_chunks')
if 'embedding_manager' not in st.session_state:
    st.session_state.embedding_manager = EmbeddingManager(DEFAULT_EMBEDDING_MODEL)

PAPERS_DIR.mkdir(parents=True, exist_ok=True)
st.title('Document Library')

uploaded = st.file_uploader('Upload research PDFs', type=['pdf'], accept_multiple_files=True)
if uploaded:
    for file in uploaded:
        target = PAPERS_DIR / file.name
        if not target.exists():
            target.write_bytes(file.read())
            st.success(f'Saved {file.name} to the document library.')
        else:
            st.warning(f'{file.name} already exists in the library and was not duplicated.')

paper_files = sorted(PAPERS_DIR.glob('*.pdf'))
if not paper_files:
    st.info('No PDFs are currently available. Upload a paper to begin retrieval.')
    st.stop()

for paper in paper_files:
    document, errors = load_pdf_text(str(paper))
    with st.container():
        st.markdown('---')
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.markdown(f'**{document.title if document else paper.stem}**')
        with col2:
            st.write(paper.name)
        with col3:
            if st.button('Remove', key=f'remove_{paper.name}'):
                paper.unlink(missing_ok=True)
                st.session_state.vector_store.delete_document(document.document_id if document else paper.stem)
                st.rerun()
        if document:
            st.write(f'Pages: {document.page_count} | Extracted text pages: {len(document.pages)}')
            if errors:
                st.warning('; '.join(errors[:3]))
        else:
            st.error('Unable to load this PDF.')

        if st.button('Re-index', key=f'reindex_{paper.name}'):
            if document is None:
                st.error('This PDF could not be read and cannot be indexed.')
            else:
                chunks = split_chunks_for_pages(document.document_id, document.title, document.filename, document.pages, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
                if not chunks:
                    st.warning('This PDF produced no extractable text chunks.')
                else:
                    embeddings = st.session_state.embedding_manager.embed_documents([chunk['text'] for chunk in chunks], model_name=DEFAULT_EMBEDDING_MODEL)
                    st.session_state.vector_store.add_documents(chunks, embeddings, model_name=DEFAULT_EMBEDDING_MODEL)
                    st.session_state.vector_store.persist()
                    st.success(f'Indexed {len(chunks)} chunks from {document.title}.')

        if document and document.pages:
            with st.expander('Preview extracted text'):
                preview = '\n\n'.join(f"Page {p['page_number']}: {p['text'][:500]}" for p in document.pages[:3])
                st.text(preview)
