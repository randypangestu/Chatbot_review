import streamlit as st
import os
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    get_response_synthesizer,
    Settings
    )
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from lib.run_query import load_query_engine
import time
#from openai import OpenAI



st.title("Spotify Review Chatbot")
if not os.path.exists("./embedding_storage"):
    st.warning("""Embeddings not found. Please follow the instructions in the README to generate embeddings.
               and make sure the embedding_storage is in the './embedding_storage' directory.""")
    st.stop()

with st.spinner("Calling Chatbot..."):
    query_engine = load_query_engine(embedding_dir="./embedding_storage")
    time.sleep(2)  # Simulating a long-running process
st.success("Chatbot is here!")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "User", "content": prompt})

    response = query_engine.query(prompt)
    st.session_state.messages.append({"role": "bot", "content": response})
    with st.chat_message("Assistant"):
        st.markdown(response)

