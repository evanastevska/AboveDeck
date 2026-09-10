"""
ablation.py, run one config through the full pipeline.

Takes a config dict and a query, builds the pipeline according
to those settings, returns the result.

Expensive steps (only needed when chunk_size, chunking_strategy,
or embedding_model change):
    - chunking
    - embedding + store

Cheap steps (change freely between runs):
    - retrieval method
    - reranker on/off
    - top-k
    - query transform
"""

import chunk
import embed
import retrieve
import generate


import os
from google import genai
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer  #baseline model class
from sentence_transformers import CrossEncoder


def run_config(config, query_text, client):
    """
    Run a single query through the pipeline defined by config.

    Args:
        config:     the config dict (all 7 keys)
        query_text: one question to answer
        client:     the genai.Client (already built by the caller)

    Returns:
        response that was generated, retrieval_result
    """

  
    #1. chunk
    build_records_results = chunk.build_records(
        chunk_size=config["chunk_size"], 
        chunking_strategy=config["chunking_strategy"]
    )


    #2. embed
    embedding_model = SentenceTransformer(config["embedding_model"])

    
    build_store_results = embed.build_store(
        records=build_records_results,
        model=embedding_model,
    )

    if config["query_transform"] != "raw":
        print("Query transform does not exist")
        return (None, None)


    #3. retrieve
    if config["retrieval_method"] == "dense":
        retrieval_result = retrieve.retrieve_dense(build_store_results, embedding_model, query_text, k=config["top_k"], season=None, show=None)
    elif config["retrieval_method"] == "bm25":
        bm25_index = retrieve.build_bm25_index(build_records_results)
        retrieval_result = retrieve.retrieve_bm25(bm25_index, build_records_results, query_text, k=config["top_k"], season=None, show=None)
    elif config["retrieval_method"] == "hybrid":
        bm25_index = retrieve.build_bm25_index(build_records_results)
        retrieval_result = retrieve.retrieve_hybrid(build_store_results, embedding_model, bm25_index, build_records_results, query_text, k=config["top_k"], season=None, rrf_k=60, show=None)


    #4. reranking (maybe)
    if config["reranker"] == "on":
        reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        retrieval_result = retrieve.rerank(retrieval_result, query_text, reranker_model, top_n=None)


    #5. generate
    response = generate.generate(query_text, retrieval_result, client, model_name="gemini-3.6-flash")

    #6. return
    return response, retrieval_result


config = {
    "chunk_size": 800,
    "chunking_strategy": "fixed",
    "embedding_model": "all-MiniLM-L6-v2",
    "top_k":5,
    "retrieval_method":"bm25",
    "reranker":"off",
    "query_transform":"something"
}



if __name__ == "__main__":
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")

    client = genai.Client(api_key=api_key)

    query_text="How many episodes are in Below Deck Mediterranean Season 1?"

    response, retrieval_result = run_config(config, query_text, client)
    print(response)
    print(retrieval_result)