import os.path
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    get_response_synthesizer,
    Settings
    )
import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core import PromptTemplate
from llama_index.llms.openai import OpenAI

from transformers import BitsAndBytesConfig
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM



import argparse

QA_PROMPT_TEMPLATE = (
    """Context information is below.\n
    You are a chatbot for our company spotify, your purpose is to answer based on context given\n
    ---------------------\n
    {context_str}\n"
    ---------------------\n
    only answer the query based on the context given\n
    answer in detail based on the context given\n
    if the query is not relevant to context or context is empty, answer as 'query is not relevan' \n
    Query: {query_str}\n
    Answer: """
)



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dir", default="./embedding_storage", help="Directory to store the embeddings")
    parser.add_argument("--prompt","-p", type=str, default="what is the common reasons people give low ratings, answer in list from the most common", help="Query to search")
    return parser.parse_args()

def load_mistral():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


    llm = HuggingFaceLLM(
        model_name="mistralai/Mistral-7B-Instruct-v0.1",
        tokenizer_name="mistralai/Mistral-7B-Instruct-v0.1",
        query_wrapper_prompt=PromptTemplate("<s>[INST] {query_str} [/INST] </s>\n"),
        context_window=3900,
        max_new_tokens=256,
        model_kwargs={"quantization_config": quantization_config},
        # tokenizer_kwargs={},
        generate_kwargs={"temperature": 0.2, "top_k": 5, "do_sample":True},
        device_map="cuda:0",
    )
    return llm


def load_query_engine(embedding_dir, load_custom_llm=False):
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )
    storage_context = StorageContext.from_defaults(persist_dir=embedding_dir)
    index = load_index_from_storage(storage_context)
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=20,
    )
    if load_custom_llm: # still has bug
        llm = load_mistral()
        Settings.llm = llm
        query_engine = index.as_query_engine(response_mode="compact")

    else:
        llm = OpenAI(model="gpt-3.5-turbo-0613")
        Settings.llm = llm
        qa_prompt = PromptTemplate(QA_PROMPT_TEMPLATE)
        summarizer = get_response_synthesizer(
            response_mode="compact",  
            verbose=True, 
            llm=llm, 
            structured_answer_filtering=True,
            text_qa_template=qa_prompt,

            )
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=summarizer,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.1)],
        )

    
    return query_engine

if __name__ == "__main__":
    args = get_args()
    query_engine = load_query_engine(embedding_dir=args.embedding_dir) 
    response = query_engine.query(args.prompt)
    print(response)