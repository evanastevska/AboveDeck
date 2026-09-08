import os
from google import genai
from dotenv import load_dotenv

import embed
import chunk
from sentence_transformers import SentenceTransformer  #baseline model class


def retrieve(collection, model, query_text, k=5, season=None):
    """
    Embed a question and RETURN the top-k hits for the next stage.
    Same retrieval as query_test, but hands data back instead of printing.

    Returns:
        list[dict], one per hit, carrying at least text + metadata
                     (so the caller can build the prompt AND trace provenance)
    """
    #embed query, build query_args (+ where filter iff season is not None),
    #collection.query(**query_args), [0]-unpack docs/metas/dists.

    #embed query_text. hand it to Chroma as a LIST (one query)
    query_embedding = model.encode([query_text]).tolist()
    
    query_args = {
        "query_embeddings": query_embedding,
        "n_results": k
    }
    if season is not None:
        query_args["where"] = {"season": season}

    results = collection.query(**query_args)

    #the result is a dict; each field (documents, metadatas, distances) comes
    #back nested one level deep — a list-of-lists, one inner list per query. You
    #sent ONE query, so take [0] of each to get this query's hits.
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]


    #package each (doc, meta, dist) into one dict, collect into a list.
    combined_list = [
        {"doc": doc, "meta": meta, "dist": dist} 
        for doc, meta, dist in zip(docs, metas, dists)
    ]

    
    return combined_list




def generate(query_text, retrieved, client, model_name):
    """
    Build a grounded prompt from retrieved chunks, call the LLM, return the answer.

    Args:
        query_text: the user's original question
        retrieved:  list of dicts from retrieve(), each has "doc", "meta", "dist"
        client:     the genai.Client you already built
        model_name: which model to call (e.g. "gemini-3.6-flash")

    Returns:
        str , the model's answer
    """
    #build a context block from retrieved.
    #loop through the hits. For each one, label it with its source(source_page + season from meta) and include the chunk text.
    #join into one big string the model can read. 

    context_segments = []

    #loop through combined list
    for i, item in enumerate(retrieved, start=1):
        doc_text = item["doc"]

        source_page = item["meta"].get("source_page", f"Document {i}")

        season = item["meta"].get("season", f"Document {i}")

        segment = f"[Doc {i}] Source Page: {source_page} Season: {season}\nContent: {doc_text}"
        context_segments.append(segment)

    #combine everything into a single string block
    context_block = "\n\n".join(context_segments)


    prompt = f"Here is the instruction: Answer the question ONLY from the context, and say so if the answer to the question is not there.\n\nHere is the context: {context_block}.\n\nHere is the question: {query_text}."

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )

    return response.text
