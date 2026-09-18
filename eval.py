"""
eval.py, evaluation harness for the RAG pipeline.

Compares pipeline output against the gold set to produce two kinds of scores:
    1. Retrieval metrics (Recall@k): did the retriever find the right chunks?
    2. LLM-as-judge scoring: did the generated answer match the gold answer?

This file is the bridge between the pipeline (ablation.py) and the gold set (gold_set.json).
"""
import ablation
import json

def recall_at_k(retrieved_ids, gold_ids):
    """
    Compute Recall@k for a single question.

    Args:
        retrieved_ids:  list of chunk_id strings the retriever returned
        gold_ids:       list of chunk_id strings from the gold set

    Returns:
        float, the recall score (0.0 to 1.0)

    Recall@k = (gold chunks found in retrieved) / (total gold chunks)

    Edge case: if gold_ids is empty (negation questions), recall
    is not meaningful. Handle this so it doesn't crash.

    TODO: implement
    """
    hashmap_retrieved_id = set(retrieved_ids)
    score = 0

    if len(gold_ids) == 0:
        print("Negation")
        return None

    for gold_id in gold_ids:
        if gold_id in hashmap_retrieved_id:
            score += 1


    recall = score / len(gold_ids)

    return recall



def run_retrieval_eval(config, client):
    """
    Run Recall@k across the full gold set for one pipeline config.

    Args:
        config:  the ablation config dict (all 7 keys)
        client:  the genai.Client (needed by run_config)

    Returns:
        dict with:
            - "mean_recall": average Recall@k across all non-negation questions
            - "per_question": list of {"id": ..., "query_type": ..., "recall": ...}
              for every question (None for negation)

    Steps:
        1. Load gold_set.json
        2. Loop over every gold entry
        3. Run the pipeline for that entry's query (run_config)
        4. Extract the chunk_ids from the retrieval result
        5. Call recall_at_k with retrieved ids vs gold ids
        6. Collect per-question results
        7. Average the non-None scores

    TODO: implement
    """
    #1. load gold_set.json
    with open("gold_set.json", "r", encoding="utf-8") as file:
        gold_set = json.load(file)

    per_question = []


    for gold_entry in gold_set:
        query_text = gold_entry.get("query_text")

        response, retrieval_result = ablation.run_config(config, query_text, client)

        retrieved_chunk_ids = [chunk["chunk_id"] for chunk in retrieval_result]

        retrieved_recall = recall_at_k(retrieved_chunk_ids, gold_entry["gold_chunk_ids"])

        per_question.append({"id": gold_entry["id"], "query_type": gold_entry["query_type"], "recall": retrieved_recall})

    valid_scores = [q["recall"] for q in per_question if q.get("recall") is not None]

    if valid_scores:
        mean_recall = sum(valid_scores) / len(valid_scores)
    else:
        mean_recall  = 0.0

    return {
        "mean_recall": mean_recall,
        "per_question": per_question
    }

