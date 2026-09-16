from rank_bm25 import BM25Okapi

def retrieve_dense(collection, model, query_text, k=5, season=None, show=None):
    """
    Embed a question and RETURN the top-k hits for the next stage.

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
    
    
    if season is not None and show is not None:
        query_args["where"] = {"$and": [{"season": season}, {"show": show}]}
    elif season is not None:
            query_args["where"] = {"season": season}
    elif show is not None:
            query_args["where"] = {"show": show}


    results = collection.query(**query_args)

    #the result is a dict; each field (documents, metadatas, distances) comes
    #back nested one level deep — a list-of-lists, one inner list per query. You
    #sent ONE query, so take [0] of each to get this query's hits.
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    chunk_ids= results["ids"][0]


    #package each (doc, meta, dist) into one dict, collect into a list.
    combined_list = [
        {"doc": doc, "meta": meta, "dist": dist, "chunk_id": chunk_id} 
        for doc, meta, dist, chunk_id in zip(docs, metas, dists, chunk_ids)
    ]

    
    return combined_list



def build_bm25_index(records):
    """
    Build a BM25 index from the chunk records.

    Args:
        records: the same list of chunk dicts that build_store() uses

    Returns:
        the fitted BM25Okapi index object

    BM25Okapi wants a list of TOKENIZED documents, not raw strings.
    Each "tokenized document" is a list of lowercase words.
    Simplest tokenization: text.lower().split()

    IMPORTANT: the index only knows documents by POSITION (0, 1, 2...).
    It doesn't store your text or metadata. So the caller must keep
    the original records list alongside the index, position i in the
    index corresponds to records[i]. 
    """
    #pull the "text" out of each record and tokenize it
    tokenized_text = [record["text"].lower().split() for record in records]

    #build and return BM25Okapi from the tokenized list
    bm25 = BM25Okapi(tokenized_text)

    return bm25




def retrieve_bm25(bm25_index, records, query_text, k=5, season=None, show=None):
    """
    Score all chunks with BM25, optionally filter by season, return top-k.

    Args:
        bm25_index: the index from build_bm25_index()
        records:    the SAME records list (same order!) needed to look up
                    text and metadata by position
        query_text: the user's question (raw string)
        k:          how many results to return
        season:     if not None, only keep chunks matching this season

    Returns:
        list[dict]  same shape as retrieve_dense():
            [{"doc": ..., "meta": ..., "dist": ...}, ...]

        For "dist": BM25 returns SCORES (higher = more relevant),
        which is the opposite of cosine DISTANCE (lower = more similar).
        Store the score as-is just be aware it's not the same scale.

    Steps:
        1. Tokenize the query the same way you tokenized the docs
        2. Get scores for ALL documents (bm25_index.get_scores())
           returns one float per document, aligned with records
        3. Pair each score with its index position so know which
           record it belongs to
        4. If season is not None, filter out entries whose record
           doesn't match the season
        5. Sort by score descending (highest = best match)
        6. Take the top k
        7. Build the same list-of-dicts format as retrieve_dense
    """
    #tokenize query_text (same way as build_bm25_index)
    tokenized_query = query_text.lower().split()

    #get_scores from the index
    doc_scores = bm25_index.get_scores(tokenized_query)

    #pair each score with its position
    scored_pairs = []
    for i, score in enumerate(doc_scores):
        scored_pairs.append((i, score))

    #filter by season if season is not None, so can throw out any chunks from other seasons if user asks abt a specific season
    if season is not None and show is not None:
        filtered = [pair for pair in scored_pairs if records[pair[0]]["season"] == season and records[pair[0]]["show"] == show]
    elif season is not None:
        filtered = [pair for pair in scored_pairs if records[pair[0]]["season"] == season]
    elif show is not None:
        filtered = [pair for pair in scored_pairs if records[pair[0]]["show"] == show]
    else:
        filtered = scored_pairs

    #sort descending by score (which is at index 1 of each tuple)
    sorted_descending = sorted(filtered, key=lambda pair: pair[1], reverse=True)

    #slice top k
    top_k = sorted_descending[:k]

    #build and return list

    combined_list = []
    for position, score in top_k:
        doc = records[position]["text"]
        meta = {"source_page": records[position]["source_page"], "show": records[position]["show"], "season":records[position]["season"]}
        chunk_id = records[position]["chunk_id"]

        combined_list.append({"doc": doc, "meta": meta, "dist": score, "chunk_id": chunk_id})

    return combined_list


def retrieve_hybrid(collection, model, bm25_index, records, query_text, k=5, season=None, rrf_k=60, show=None):
    """
    Run both dense and BM25 retrieval, merge results with Reciprocal Rank Fusion.

    Args:
        collection:  Chroma collection (for dense)
        model:       SentenceTransformer (for dense)
        bm25_index:  BM25Okapi index (for BM25)
        records:     chunk records list (for BM25)
        query_text:  the user's question
        k:           how many results to return AFTER merging
        season:      season filter (passed to both retrievers)
        rrf_k:       RRF smoothing constant (default 60)

    Returns:
        list[dict] same shape: [{"doc": ..., "meta": ..., "dist": ...}, ...]
        "dist" here is the RRF score (higher = more relevant).

    How RRF works:
        Each retriever returns a RANKED list. A chunk's rank is its position
        in that list (rank 1 = best match, rank 2 = second best, etc.).

        For each chunk that appears in EITHER list, compute:
            rrf_score = sum of 1/(rrf_k + rank) across all lists it appears in

        ex: a chunk is rank 2 in dense and rank 5 in BM25:
            rrf_score = 1/(60+2) + 1/(60+5) = 1/62 + 1/65 ≈ 0.0315

        A chunk that appears in BOTH lists gets two terms added.
        A chunk that appears in only ONE list gets just one term.
        then sort by rrf_score descending and take top k.

    need a way to recognize "same chunk" across two lists.
    Each result has a "doc" (the chunk text). can use that as the
    dictionary key to accumulate scores (if the same text shows up in
    both lists, it's the same chunk)
    """

    ranked_results_dense = retrieve_dense(collection, model, query_text, k, season, show)
    ranked_results_sparse = retrieve_bm25(bm25_index, records, query_text, k, season, show)

    #a dict keyed by chunk text to accumulate RRF scores
    #for each chunk, store its meta so can return it later
    #start at 1 because rank = 1 best
    rrf_scores = {}
    
    for rank, chunk in enumerate(ranked_results_dense, start=1):
        key = chunk["doc"]
        if key not in rrf_scores:
            # first time seeing this chunk, set initial score, store meta
            rrf_scores[key] = {"score": 0.0, "meta": chunk["meta"], "chunk_id":chunk["chunk_id"]}
        rrf_scores[key]["score"] += 1.0 / (rrf_k + rank)


    #loop through BM25 results w RANK (same way)
    #for each: compute 1/(rrf_k + rank), ADD to existing score
    #(if it's already in the dict from dense, the scores accumulate
    #that's how chunks in both lists get boosted)
    for rank, chunk in enumerate(ranked_results_sparse, start=1):
            key = chunk["doc"]
            if key not in rrf_scores:
                # first time seeing this chunk, set initial score, store meta
                rrf_scores[key] = {"score": 0.0, "meta": chunk["meta"], "chunk_id":chunk["chunk_id"]}
            rrf_scores[key]["score"] += 1.0 / (rrf_k + rank)

    #sort descending by score
    sorted_descending = sorted(rrf_scores.items(), key=lambda pair: pair[1]["score"], reverse=True)

    #slice top k
    top_k = sorted_descending[:k]

    #build and return list

    combined_list = []
    for chunk_text, data in top_k:
        combined_list.append({"doc": chunk_text, "meta": data["meta"], "dist": data["score"], "chunk_id":data["chunk_id"]})

    return combined_list



def rerank(retrieved, query_text, reranker_model, top_n=None):
    """
    Re-score retrieved chunks with a cross-encoder, then re-sort.

    Args:
        retrieved:       list of dicts from ANY retriever (retrieve_dense,
                         retrieve_bm25, or retrieve_hybrid) — same
                         [{"doc": ..., "meta": ..., "dist": ...}, ...] shape
        query_text:      the user's original question
        reranker_model:  a loaded CrossEncoder model
        top_n:           how many to keep after reranking. If None, return
                         all of them (just re-sorted). This lets you
                         retrieve a bigger set (say k=20) and cut to the
                         best 5 after reranking.

    Returns:
        list[dict] same shape, but re-sorted by cross-encoder score
        (and possibly shorter if top_n is set).
        "dist" is replaced with the cross-encoder score.

    How a cross-encoder works:
        The embedding model (MiniLM) encodes the query and each chunk
        SEPARATELY, then compares vectors. Fast but rough.

        A cross-encoder reads the query and chunk TOGETHER as one input, like reading a question and a paragraph side by side. It
        outputs a single relevance score. Much slower (one model call
        per chunk), but more accurate because it sees both texts at once.

        That's why it's a RE-ranker , use fast retrieval first to
        get candidates, then the cross-encoder to re-score just those few.

    How to use it:
        reranker_model.predict() takes a list of [query, text] pairs
        and returns one score per pair. Higher = more relevant.
    """
    #a list of [query_text, chunk_text] pairs
    #one pair for each item in retrieved
    #(the query is the SAME every time , it's the chunk that changes)
    query_chunk_pairs = [[query_text, item["doc"]] for item in retrieved]


    #reranker_model.predict(pairs) to get scores
    #returns a list/array of floats, one per pair, same order
    scores = reranker_model.predict(query_chunk_pairs)

    #attach each score to its original result
    for i, hit in enumerate(retrieved):
        hit["score"] = scores[i]

    #sort by score descending
    sorted_descending = sorted(retrieved, key=lambda hit: hit["score"], reverse=True)

    #if top_n is set, slice to top_n
    if top_n is not None:
        sorted_descending = sorted_descending[:top_n]
    
    
    combined_list = []
    for hit in sorted_descending:
        combined_list.append({"doc": hit["doc"], "meta": hit["meta"], "dist": hit["score"], "chunk_id": hit["chunk_id"]})

    return combined_list



