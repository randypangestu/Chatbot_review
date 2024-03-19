import os.path
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
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dir", default="./embedding_storage", help="Directory to store the embeddings")
    parser.add_argument("--prompt","-p", type=str, default="what is the common reasons people give low ratings, answer in list from the most common", help="Query to search")
    return parser.parse_args()


def load_query_engine(embedding_dir):
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )
    storage_context = StorageContext.from_defaults(persist_dir=embedding_dir)
    index = load_index_from_storage(storage_context)
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=20,
    )
    response_synthesizer = get_response_synthesizer()

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
    )
    return query_engine

if __name__ == "__main__":
    args = get_args()
    query_engine = load_query_engine(embedding_dir=args.embedding_dir) 
    response = query_engine.query(args.prompt)
    print(response)