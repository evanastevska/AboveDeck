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



def setup_pipeline(config):
    """
    Do the expensive config-dependent setup once.

    Args:
        config: the ablation config dict (all 7 keys)

    Returns:
        dict of pipeline objects the per-query function needs:
            - "records":         chunk records from build_records
            - "collection":      Chroma collection from build_store
            - "embedding_model": loaded SentenceTransformer
            - "bm25_index":      BM25 index (None if retrieval method doesn't need it)
            - "reranker_model":  loaded CrossEncoder (None if reranker is off)

    Steps:
        1. Chunk (build_records)
        2. Load the embedding model
        3. Embed + build store
        4. Build BM25 index if retrieval method needs it (bm25 or hybrid)
        5. Load reranker model if reranker is on
        6. Return everything in a dict

    TODO: implement
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


    #4. reranking (maybe)
    if config["reranker"] == "on":
        reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    else:
        reranker_model = None

    bm25_index = retrieve.build_bm25_index(build_records_results)

    pipeline = {"records":build_records_results,
        "collection":build_store_results,
        "embedding_model":embedding_model,
        "bm25_index":bm25_index,
        "reranker_model":reranker_model}

    return pipeline




def run_query(config, query_text, client, pipeline, return_generation=True):
    """
    Run a single query through an already-set-up pipeline.

    Args:
        config:     the config dict (retrieval_method, reranker, top_k, query_transform)
        query_text: one question to answer
        client:     the genai.Client
        pipeline:   the dict returned by setup_pipeline

    Returns:
        (response, retrieval_result) — same as run_config does now

    Steps:
        1. Guard clause for query_transform
        2. Retrieve (using the right method from pipeline)
        3. Rerank if reranker is on (using reranker_model from pipeline)
        4. Generate
        5. Return (response, retrieval_result)

    TODO: implement
    """
    if config["query_transform"] != "raw":
        print("Query transform does not exist")
        return (None, None)

    #3. retrieve
    if config["retrieval_method"] == "dense":
        retrieval_result = retrieve.retrieve_dense(pipeline["collection"], pipeline["embedding_model"], query_text, k=config["top_k"], season=None, show=None)
    elif config["retrieval_method"] == "bm25":
        retrieval_result = retrieve.retrieve_bm25(pipeline["bm25_index"], pipeline["records"], query_text, k=config["top_k"], season=None, show=None)
    elif config["retrieval_method"] == "hybrid":
        retrieval_result = retrieve.retrieve_hybrid(pipeline["collection"], pipeline["embedding_model"], pipeline["bm25_index"], pipeline["records"], query_text, k=config["top_k"], season=None, rrf_k=60, show=None)

        #4. reranking (maybe)
    if config["reranker"] == "on":
        retrieval_result = retrieve.rerank(retrieval_result, query_text, pipeline["reranker_model"], top_n=None)
    else:
        pass

    #5. generate
    if return_generation:
        response = None
        for attempt in range(3):
            try:
                response = generate.generate(query_text, retrieval_result, client, model_name="gemini-3.6-flash")
                break
            except Exception as e:
                print(f"Generation attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    import time
                    time.sleep(5)
    else:
        response = None

    #6. return
    return response, retrieval_result








if __name__ == "__main__":
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)

    config = {
        "chunk_size": 800,
        "chunking_strategy": "fixed",
        "embedding_model": "all-MiniLM-L6-v2",
        "top_k":5,
        "retrieval_method":"bm25",
        "reranker":"off",
        "query_transform":"raw"
    }

    query_text="How many episodes are in Below Deck Mediterranean Season 1?"

    pipeline = setup_pipeline(config)
    response, retrieval_result = run_query(config, query_text, client, pipeline)
    print(response)
    print(retrieval_result)