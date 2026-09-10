import os
from google import genai
from dotenv import load_dotenv

import chunk
import embed
import generate
import retrieve
from sentence_transformers import SentenceTransformer  #baseline model class
from sentence_transformers import CrossEncoder

def main():
    """Run the full v0 pipeline."""

    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")

    client = genai.Client(api_key=api_key)
    
    records = chunk.build_records()

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    query_text = "How many episodes are in Below Deck Mediterranean Season 1?"

    #dense retrieval
    collection = embed.build_store(records, model)
    dense_result = retrieve.retrieve_dense(collection, model, query_text, k=5, season=1, show="Below Deck Mediterranean")
    print(dense_result)

    #sparse retrieval
    bm25_index = retrieve.build_bm25_index(records)
    #sparese_result = retrieve.retrieve_bm25(bm25_index, records, query_text, k=5, season=None, show=None)
    #print(sparse_result)

    #hybrid_result = retrieve.retrieve_hybrid(collection, model, bm25_index, records, query_text, k=5, season=None, rrf_k=60, show=None)
    #print(hybrid_result)

    rerank_result = retrieve.rerank(dense_result, query_text, reranker_model, top_n=None)
    print(rerank_result)

    response = generate.generate(query_text, rerank_result, client, model_name="gemini-3.6-flash")
    print(response)


if __name__ == "__main__":
    main()