from __future__ import annotations

from pathlib import Path

import streamlit as st

from config.settings import CHROMA_DIR, DEFAULT_EMBEDDING_MODEL, PAPERS_DIR, TOP_K
from src.embeddings import EmbeddingManager
from src.llm_provider import LLMProvider
from src.vector_store import ResearchVectorStore

st.set_page_config(page_title='Research Paper Answer Bot', layout='wide')

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = ResearchVectorStore('research_chunks')
if 'embedding_manager' not in st.session_state:
    st.session_state.embedding_manager = EmbeddingManager(DEFAULT_EMBEDDING_MODEL)
if 'llm' not in st.session_state:
    st.session_state.llm = LLMProvider()

st.title('Research Paper Answer Bot')
st.caption('Grounded research assistant for PDF-based literature exploration.')

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric('Indexed papers', len(list(PAPERS_DIR.glob('*.pdf'))))
with col2:
    st.metric('Indexed chunks', st.session_state.vector_store.count())
with col3:
    st.metric('Embedding model', DEFAULT_EMBEDDING_MODEL)
with col4:
    st.metric('LLM configured', 'Yes' if st.session_state.llm.is_configured() else 'No')

st.write(f'Chroma persistence directory: {CHROMA_DIR}')
st.write(f'Top-k retrieval: {TOP_K}')

if not st.session_state.llm.is_configured():
    st.info('No LLM provider is configured. Ingestion and retrieval are available, but answer generation remains in retrieval-only mode until a valid provider is configured.')

st.markdown('---')
st.subheader('Upload PDFs to the library')
files = st.file_uploader('Upload research PDFs', type=['pdf'], accept_multiple_files=True)
if files:
    saved = 0
    for uploaded in files:
        target = PAPERS_DIR / uploaded.name
        if not target.exists():
            target.write_bytes(uploaded.getvalue())
            saved += 1
    if saved:
        st.success(f'Saved {saved} new PDF(s) to the document library.')

st.markdown('---')
col_a, col_b = st.columns(2)
with col_a:
    st.page_link('pages/1_Document_Library.py', label='Open document library')
with col_b:
    st.page_link('pages/2_Research_Chat.py', label='Open research chat')

st.page_link('pages/3_Retrieval_Lab.py', label='Open retrieval lab')
st.page_link('pages/4_Evaluation_Dashboard.py', label='Open evaluation dashboard')
