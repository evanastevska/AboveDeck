"""
eval.py, evaluation harness for the RAG pipeline.

Compares pipeline output against the gold set to produce two kinds of scores:
    1. Retrieval metrics (Recall@k): did the retriever find the right chunks?
    2. LLM-as-judge scoring: did the generated answer match the gold answer?

This file is the bridge between the pipeline (ablation.py) and the gold set (gold_set.json).
"""
import ablation
import json

import os
from google import genai
from dotenv import load_dotenv

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
    """
    hashmap_retrieved_id = set(retrieved_ids)
    score = 0

    if len(gold_ids) == 0:
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

    """
    #1. load gold_set.json
    with open("gold_set.json", "r", encoding="utf-8") as file:
        gold_set = json.load(file)

    per_question = []

    pipeline = ablation.setup_pipeline(config)


    for gold_entry in gold_set:
        query_text = gold_entry.get("query_text")

        response, retrieval_result = ablation.run_query(config, query_text, client, pipeline, return_generation=False)

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




def judge_answer(query, generated_answer, gold_answer, context, openai_client):
    """
    Ask GPT-4o-mini to score one generated answer on a rubric.

    Args:
        query:             the original question
        generated_answer:  what the pipeline produced
        gold_answer:       the correct answer from the gold set
        context:           the retrieved chunks (joined as one string)
        openai_client:     an OpenAI client object

    Returns:
        dict with "faithfulness", "correctness", "completeness"
        (each an int 1–5), or None if the API call fails
    """

    prompt = f"""You are an evaluation judge for a RAG (Retrieval-Augmented Generation) system. You will be given a question, the context that was retrieved from a knowledge base, the answer the system generated, and the correct gold answer. Score the generated answer on three dimensions using the scales below.

    Faithfulness (1–5): Does the generated answer only use information from the retrieved context?
    1 = The answer contains major claims that are not in the retrieved context (hallucinated).
    2 = The answer contains some claims not supported by the retrieved context.
    3 = The answer mostly uses the context but includes minor unsupported details.
    4 = The answer is grounded in the context with only trivial rephrasing beyond it.
    5 = Every claim in the answer is directly supported by the retrieved context.

    Correctness (1–5): Does the generated answer match the gold answer?
    1 = The answer is completely wrong or contradicts the gold answer.
    2 = The answer gets the topic right but the key facts are wrong.
    3 = The answer is partially correct; some facts match, others are wrong or missing.
    4 = The answer is mostly correct with only minor inaccuracies.
    5 = The answer is fully correct and matches the gold answer.

    Completeness (1–5): Does the generated answer capture the full gold answer?
    1 = The answer misses the entire point of the gold answer.
    2 = The answer covers less than half of the gold answer.
    3 = The answer covers roughly half of the gold answer.
    4 = The answer covers most of the gold answer with minor gaps.
    5 = The answer fully covers everything in the gold answer.

    ---

    Query: {query}

    Retrieved Context: {context}

    Generated Answer: {generated_answer}

    Gold Answer: {gold_answer}

    ---

    Respond with ONLY a JSON object, no explanation, no markdown, no backticks:
    {{"faithfulness": <int>, "correctness": <int>, "completeness": <int>}}"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        result_text = response.choices[0].message.content
        scores = json.loads(result_text)
        return scores
    except Exception as e:
        print(f"Judge failed: {e}")
        return None






if __name__ == "__main__":
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")

    client = genai.Client(api_key=api_key)

    config = {
        "chunk_size": 800,
        "chunking_strategy": "fixed",
        "embedding_model": "all-MiniLM-L6-v2",
        "top_k":5,
        "retrieval_method":"dense",
        "reranker":"off",
        "query_transform":"raw"
    }

    retrieval_eval = run_retrieval_eval(config, client)
    print(retrieval_eval)

    #break down by query type
    from collections import defaultdict

    by_type = defaultdict(list)
    for q in retrieval_eval["per_question"]:
        if q["recall"] is not None:
            by_type[q["query_type"]].append(q["recall"])

    print(f"\nOverall mean Recall@k: {retrieval_eval['mean_recall']:.3f}")
    print(f"{'Query Type':<20} {'Count':>5} {'Mean Recall':>12}")
    print("-" * 40)
    for qtype, scores in sorted(by_type.items()):
        print(f"{qtype:<20} {len(scores):>5} {sum(scores)/len(scores):>12.3f}")

