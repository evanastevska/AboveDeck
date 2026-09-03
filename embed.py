"""
embed.py - turn the 258 chunk records into vectors and store them in a vector DB.

Pipeline for this step:
  chunk records -> embedding vectors -> Chroma collection (+ metadata) -> query it

embed each chunk's text with a local model (all-MiniLM-L6-v2), store the vectors + text + metadata in Chroma, and confirm can pull back sensible hits.

Housekeeping
  - add Chroma persist folder to .gitignore
  - after `pip install sentence-transformers chromadb`, re-freeze requirements.txt
"""

import chunk
from sentence_transformers import SentenceTransformer  #baseline model class
import chromadb #vector DB

def token_sanity_check(records, model):
    """
    Confirm model isn't silently truncating any chunk.

    why:
      all-MiniLM-L6-v2 only reads the first ~256 tokens of an input and silently
      drops the rest. chunks are sized in CHARS (800), not tokens, and text is markup-heavy. so a chunk could slip over 256.

      - for every record, count how many tokens its "text" becomes
      - track the largest count seen (and which chunk_id it was)
      - print that max next to the model's own limit, so see the headroom

    """

    model_max = model.max_seq_length #modle's limit
    max_count = 0
    biggest_chunk_id = None

    counter = 0

    for record in records:
        curr_count = len(model.tokenizer.encode(record["text"])) #number of tokens in this record's text
        if curr_count > max_count:
            max_count = curr_count
            biggest_chunk_id = record["chunk_id"]

        if curr_count > model_max:
            counter += 1

    print(max_count)
    print(biggest_chunk_id)
    print(model_max)
    print(counter)
            
    


def build_store(records, model):
    """
    Embed every chunk and load it into a persistent Chroma collection.

    Args:
        records: the 258 chunk dicts (each has chunk_id, text, show, season, source_page)
        model:   the already-loaded SentenceTransformer

    Returns:
        the Chroma collection, so the caller can query it.

    Chroma takes PARALLEL lists, so need to unpack records into aligned
    lists, then embed the documents, then hand all four over together.
    """

    ids = []
    documents = []
    metadatas = []

    for record in records:
        ids.append(record["chunk_id"])
        documents.append(record["text"])
        metadatas.append({"source_page": record["source_page"], "show": record["show"], "season":record["season"]})

    embeddings = model.encode(documents).tolist()

    #persistent Chroma client that writes to folder on disk
    client = chromadb.PersistentClient(path="chroma_store")

    #unit of storage and querying, acts as a container that groups related to embeddings, docs, metadata, ids
    #also called collection
    chunks = client.get_or_create_collection(name="chunks",metadata={"hnsw:space": "cosine"})

    chunks.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    return chunks


def query_test(collection, model, query_text, k=5, season=None):
    """
    Embed a question and print the top-k chunks Chroma returns.

    Args:
        collection: the collection from build_store
        model:      the SAME model (the query must be embedded the same way the
                    chunks were, or the vectors aren't comparable)
        query_text: the question, e.g. "who quit as chef?"
        k:          how many hits to return
        season:     if given, restrict the search to that season (metadata filter)

    This is for proof the store works: do the hits look relevant, and does the
    season filter actually narrow them?
    """
    # TODO: embed query_text the same way you embedded chunks. You'll hand it to
    #   Chroma as a LIST (one query), so shape it accordingly.
    

    # TODO: call collection.query(...). Arguments you'll use:
    #     query_embeddings = [ your query vector ]
    #     n_results        = k
    #     where            = {"season": season}   <-- ONLY when season is not None
    #   (build the call so the filter is included only when a season was passed.)

    # TODO: the result is a dict; each field (documents, metadatas, distances) comes
    #   back nested one level deep — a list-of-lists, one inner list per query. You
    #   sent ONE query, so take [0] of each to get this query's hits.

    # TODO: loop the hits and print, per hit:
    #     - the distance (Chroma cosine distance: LOWER = closer / more similar)
    #     - show + season from the metadata
    #     - the first ~150 chars of the document, to eyeball relevance
    pass


def main():
    """Run the whole embedding step end to end."""
    # TODO: model_name = "all-MiniLM-L6-v2"
    # TODO: records = ...     # your chunking module returns the list — use it
    # TODO: model   = ...     # construct the SentenceTransformer ONCE (loading is slow)
    # TODO: token_sanity_check(records, model)
    # TODO: collection = build_store(records, model)
    # TODO: query_test(collection, model, "who quit as chef?", k=5)            # plain
    # TODO: query_test(collection, model, "who quit as chef?", k=5, season=3)  # filtered
    
    records = chunk.main()
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    token_sanity_check(records, model)

    collection = build_store(records, model)


if __name__ == "__main__":
    main()