import streamlit as st
import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from lib.run_query import load_query_engine
from llama_index.core.response.notebook_utils import display_response
import time

from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    get_response_synthesizer,
    Settings
)


def check_embeddings():
    if not os.path.exists("./embedding_storage"):
        st.warning("""Embeddings not found. Please follow the instructions in the README to generate embeddings.
                   and make sure the embedding_storage is in the './embedding_storage' directory.""")
        st.stop()


def load_chatbot():

    with st.spinner("Calling Chatbot..., this make take a while..."):
        query_engine = load_query_engine(embedding_dir="./embedding_storage", load_custom_llm=False)
    time.sleep(2)  # Simulating a long-running process
    st.success("Chatbot is here!")
    return query_engine


def initialize_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def process_user_input(query_engine):
    if prompt := st.chat_input("What is up?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "User", "content": prompt})

        response = query_engine.query(prompt)
        st.session_state.messages.append({"role": "bot", "content": response})
        if response.response == "Empty Response":
            response = "I'm sorry, I don't have an answer for that. Please ask me something else."
        with st.chat_message("Assistant"):
            st.markdown(response)


def main():
    st.title("Spotify Review Chatbot")
    check_embeddings()
    
    query_engine = load_chatbot()
    initialize_chat_history()
    process_user_input(query_engine)


if __name__ == "__main__":
    main()
