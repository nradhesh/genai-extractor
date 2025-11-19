import streamlit as st
from query_agent import run_streamlit_app

# st.set_page_config() MUST be the first Streamlit command
st.set_page_config(page_title="Scholarly RAG Assistant", page_icon="📚", layout="wide")

run_streamlit_app()
