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

    counter = 0 #how many of the 258 chunks land over 266 token limit

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
        model:   the alr-loaded SentenceTransformer

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

    chunk_embeddings = model.encode(documents).tolist()

    #persistent Chroma client that writes to folder on disk
    client = chromadb.PersistentClient(path="chroma_store")

    #unit of storage and querying, acts as a container that groups related to embeddings, docs, metadata, ids
    #also called collection
    chunks = client.get_or_create_collection(name="chunks",metadata={"hnsw:space": "cosine"})

    chunks.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=chunk_embeddings)

    return chunks