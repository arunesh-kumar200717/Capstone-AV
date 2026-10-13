import streamlit as st

st.set_page_config(page_title='Project Information', layout='wide')
st.title('Project Information')
st.markdown('''
This capstone project implements a research-paper answer bot using:
- PDF ingestion with metadata preservation.
- ChromaDB vector storage.
- Sentence-transformers embeddings.
- Hybrid retrieval and RAG.
- Streamlit-based interface and evaluation workflows.

Key requirement areas covered:
- Document management and indexing.
- Research chat with citations.
- Retrieval experiment dashboard.
- Evaluation and report exports.
- Windows-friendly setup instructions.
''')
