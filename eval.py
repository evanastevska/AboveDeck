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
from openai import OpenAI
from dotenv import load_dotenv
import json as json_module

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






def run_retrieval_eval(config, client, pipeline=None):
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

    if pipeline is None:
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
    IMPORTANT: If the answer states that the information is not in the context, and the retrieved context indeed does not contain the answer, that is a FAITHFUL response (score 5). Do NOT penalize faithfulness for wrong or incomplete answers, that is what correctness and completeness measure. Faithfulness ONLY measures whether the answer stayed grounded in the retrieved context.

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



def run_judge_eval(config, client, openai_client, pipeline=None):
    """
    Run LLM-as-judge scoring across the full gold set for one pipeline config.

    Args:
        config:         the ablation config dict (all 7 keys)
        client:         the genai.Client (for generation)
        openai_client:  an OpenAI client (for the judge)

    Returns:
        dict with:
            - "mean_faithfulness": average across all scored questions
            - "mean_correctness": average across all scored questions
            - "mean_completeness": average across all scored questions
            - "per_question": list of {"id", "query_type", "scores"} for every question
              (scores is the dict from judge_answer, or None if it failed)

    Steps:
        1. Load gold_set.json
        2. Setup the pipeline once
        3. Loop over every gold entry:
            a. Run the pipeline WITH generation (return_generation=True)
            b. Join the retrieved chunk texts into one context string
            c. Call judge_answer with query, generated answer, gold answer, context
            d. Store the result
        4. Average each dimension (skip Nones)
        5. Return the results dict

    """
    with open("gold_set.json", "r", encoding="utf-8") as file:
            gold_set = json.load(file)


    per_question = []

    if pipeline is None:
        pipeline = ablation.setup_pipeline(config)

    for gold_entry in gold_set:
        query_text = gold_entry.get("query_text")
        response, retrieval_result = ablation.run_query(config, query_text, client, pipeline, return_generation=True)

        retrieved_chunk_ids = [chunk["doc"] for chunk in retrieval_result]

        context_string = "\n\n".join(retrieved_chunk_ids)

        if response is None:
            per_question.append({
                "id": gold_entry["id"],
                "query_type": gold_entry["query_type"],
                "scores": None,
                "context": context_string,
                "generated": None,
                "gold": gold_entry["answer_text"]
            })
            continue

        judge_answer_result = judge_answer(query=query_text, generated_answer=response, gold_answer=gold_entry["answer_text"], context=context_string, openai_client=openai_client)

        per_question.append({
            "id": gold_entry["id"],
            "query_type": gold_entry["query_type"],
            "scores": judge_answer_result,
            "context": context_string,
            "generated": response,
            "gold": gold_entry["answer_text"]
        })

    valid_faithful = [q["scores"]["faithfulness"] for q in per_question if q["scores"] is not None]
    valid_correct = [q["scores"]["correctness"] for q in per_question if q["scores"] is not None]
    valid_complete = [q["scores"]["completeness"] for q in per_question if q["scores"] is not None]


    #with open("judge_results_baseline.json", "w", encoding="utf-8") as f:
    #    json_module.dump({"mean_faithfulness": sum(valid_faithful) / len(valid_faithful) if valid_faithful else 0.0,
    #                      "mean_correctness": sum(valid_correct) / len(valid_correct) if valid_correct else 0.0,
    #                      "mean_completeness": sum(valid_complete) / len(valid_complete) if valid_complete else 0.0,
    #                      "per_question": per_question}, f, indent=2)
    #print("Saved to judge_results_baseline.json")

    return {
        "mean_faithfulness": sum(valid_faithful) / len(valid_faithful) if valid_faithful else 0.0,
        "mean_correctness": sum(valid_correct) / len(valid_correct) if valid_correct else 0.0,
        "mean_completeness": sum(valid_complete) / len(valid_complete) if valid_complete else 0.0,
        "per_question": per_question
    }




if __name__ == "__main__":
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")
    retrieval_client = genai.Client(api_key=api_key)
    judge_client = OpenAI()

    config = {
        "chunk_size": 800,
        "chunking_strategy": "fixed",
        "embedding_model": "all-MiniLM-L6-v2",
        "top_k": 5,
        "retrieval_method": "dense",
        "reranker": "off",
        "query_transform": "raw"
    }

    #Retrieval eval
    #retrieval_eval = run_retrieval_eval(config, retrieval_client)
    #print(retrieval_eval)

    #from collections import defaultdict
    #by_type = defaultdict(list)
    #for q in retrieval_eval["per_question"]:
    #    if q["recall"] is not None:
    #        by_type[q["query_type"]].append(q["recall"])
    #print(f"\nOverall mean Recall@k: {retrieval_eval['mean_recall']:.3f}")
    #print(f"{'Query Type':<20} {'Count':>5} {'Mean Recall':>12}")
    #print("-" * 40)
    #for qtype, scores in sorted(by_type.items()):
    #    print(f"{qtype:<20} {len(scores):>5} {sum(scores)/len(scores):>12.3f}")

    #Judge eval
    judge_eval = run_judge_eval(config, retrieval_client, judge_client)
    print(f"\nJudge Scores:")
    print(f"  Faithfulness: {judge_eval['mean_faithfulness']:.3f}")
    print(f"  Correctness:  {judge_eval['mean_correctness']:.3f}")
    print(f"  Completeness: {judge_eval['mean_completeness']:.3f}")



