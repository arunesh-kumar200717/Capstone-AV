from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from config.settings import EVALUATION_DIR

st.set_page_config(page_title='Evaluation Dashboard', layout='wide')
st.title('Evaluation Dashboard')

query_file = EVALUATION_DIR / 'test_queries.json'
if query_file.exists():
    data = json.loads(query_file.read_text(encoding='utf-8'))
    st.write(f'{len(data)} evaluation questions loaded.')
    df = pd.DataFrame(data)
    st.dataframe(df[['question', 'domain', 'expected_answer']].head(10), use_container_width=True)

    fig = px.histogram(df, x='domain', title='Question domains')
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info('No evaluation dataset is present yet. Add test_queries.json or run the evaluation script.')

st.subheader('Evaluation notes')
st.write('Use the exported CSV or JSON evaluation reports after running the evaluation script to compare retrieval metrics and answer quality.')
