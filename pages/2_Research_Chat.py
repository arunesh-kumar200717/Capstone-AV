from __future__ import annotations

import time

import streamlit as st

from config.settings import DEFAULT_EMBEDDING_MODEL, TOP_K
from src.conversation import ConversationManager
from src.rag_pipeline import ResearchRAG
from src.vector_store import ResearchVectorStore

st.set_page_config(page_title='Research Chat', layout='wide')

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = ResearchVectorStore('research_chunks')
if 'conversation_manager' not in st.session_state:
    st.session_state.conversation_manager = ConversationManager()
if 'rag' not in st.session_state:
    st.session_state.rag = ResearchRAG(st.session_state.vector_store)

st.title('Research Chat')

if st.button('Clear chat'):
    st.session_state.chat_history = []
    st.session_state.conversation_manager = ConversationManager()

for msg in st.session_state.chat_history:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])
        if 'sources' in msg:
            with st.expander('Supporting sources'):
                for source in msg['sources'][:3]:
                    st.markdown(f"- **{source['paper_title']}** (Page {source['page_number']}) — {source['source_file']}")
                    st.caption(source['text'])

prompt = st.chat_input('Ask a question about the research corpus...')
if prompt:
    st.session_state.chat_history.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)
    prior_question = None
    if len(st.session_state.chat_history) >= 2:
        prior_question = st.session_state.chat_history[-2].get('content', '') if st.session_state.chat_history[-2]['role'] == 'user' else None
    rewritten = st.session_state.conversation_manager.rewrite_follow_up(prompt, prior_question)
    start = time.time()
    result = st.session_state.rag.answer_question(rewritten, model_name=DEFAULT_EMBEDDING_MODEL, strategy='hybrid', top_k=TOP_K)
    response = result.get('answer', 'No answer available.')
    sources = result.get('sources', [])
    st.session_state.chat_history.append({'role': 'assistant', 'content': response, 'sources': sources, 'references': result.get('references', [])})
    with st.chat_message('assistant'):
        st.markdown(response)
        st.caption(f'Retrieval strategy: hybrid | Response time: {result.get("response_time_ms", 0)} ms')
        if result.get('references'):
            st.caption(f'Validated references: {result["references"]}')
        else:
            st.caption('No validated citations available.')
        with st.expander('Show supporting evidence'):
            for source in sources[:3]:
                st.markdown(f"**{source['paper_title']}** (Page {source['page_number']})\n{source['text']}")
