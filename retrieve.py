def retrieve_dense(collection, model, query_text, k=5, season=None):
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