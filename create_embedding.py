import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import argparse
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
)
    

def generate_embedding(document_folder, embedding_dir, local_embedding=True):
    if local_embedding:
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )
    documents = SimpleDirectoryReader(document_folder).load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=embedding_dir)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dir", default="./embedding_storage", help="Directory to store the embeddings")
    parser.add_argument("--document_folder", default="./dataset_review", help="Folder containing the documents")
    parser.add_argument("--use_gpt", action="store_false", help="Use local LLM")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    print('Generating embeddings using local LLM: {}'.format(args.use_gpt))
    generate_embedding(args.document_folder, args.embedding_dir, local_embedding=args.use_gpt)
    print(f"Embeddings generated and stored in {args.embedding_dir}")
