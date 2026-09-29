# sweep.py final rerun: all 12 configs, per-question data + timing

import ablation
import eval as eval_module
import json
import os
import time
from google import genai
from openai import OpenAI
from dotenv import load_dotenv
from collections import defaultdict

BASELINE = {
    "chunk_size": 800,
    "chunking_strategy": "fixed",
    "embedding_model": "all-MiniLM-L6-v2",
    "top_k": 5,
    "retrieval_method": "dense",
    "reranker": "off",
    "query_transform": "raw"
}

def make_config(label, **overrides):
    """Copy baseline, swap in any overrides. Returns (label, config)."""
    config = BASELINE.copy()
    config.update(overrides)
    return (label, config)

def setup_key(config):
    """The three knobs that require rebuilding the pipeline (expensive)."""
    return (config["chunk_size"], config["chunking_strategy"], config["embedding_model"])


CONFIGS = [
    #baseline
    make_config("baseline"),

    #retrieval method axis
    make_config("bm25", retrieval_method="bm25"),
    make_config("hybrid", retrieval_method="hybrid"),

    #reranker axis
    make_config("dense+rerank", reranker="on"),

    #top-k axis
    make_config("topk2", top_k=2),
    make_config("topk10", top_k=10),

    #chunking axis (different setup_key triggers new pipeline)
    make_config("token128", chunk_size=128, chunking_strategy="token"),
    make_config("token256", chunk_size=256, chunking_strategy="token"),

    #combinations
    make_config("hybrid+rerank", retrieval_method="hybrid", reranker="on"),
    make_config("dense+topk10+rerank", top_k=10, reranker="on"),
    make_config("hybrid+topk10", retrieval_method="hybrid", top_k=10),
    make_config("hybrid+topk10+rerank", retrieval_method="hybrid", top_k=10, reranker="on"),
]


def slim_retrieval(per_question):
    """Keep only what's needed for query-type slicing. Drop nothing useful, skip nothing bulky."""
    return [{"id": q["id"], "query_type": q["query_type"], "recall": q["recall"]}
            for q in per_question]


def slim_judge(per_question):
    """Keep id, query_type, scores. Drop the bulky context/generated/gold strings."""
    return [{"id": q["id"], "query_type": q["query_type"], "scores": q["scores"]}
            for q in per_question]


def run_sweep():
    load_dotenv()
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    openai_client = OpenAI()

    #load completed configs so a crash doesn't lose progress
    all_results = []
    if os.path.exists("sweep_results.json"):
        with open("sweep_results.json", "r", encoding="utf-8") as f:
            all_results = json.load(f)
    done_labels = {r["label"] for r in all_results}

    #group configs by setup_key so expensive pipeline setup runs once per group
    config_map = defaultdict(list)
    for label, config in CONFIGS:
        key = setup_key(config)
        config_map[key].append((label, config))

    for key, config_list in config_map.items():
        #skip this group entirely if every config in it is done
        if all(label in done_labels for label, _ in config_list):
            print(f"\nGroup {key} — all done, skipping")
            continue

        #if any config in this group uses the reranker, load it during setup
        first_config = config_list[0][1].copy()
        if any(config["reranker"] == "on" for _, config in config_list):
            first_config["reranker"] = "on"
        pipeline = ablation.setup_pipeline(first_config)

        for label, config in config_list:
            if label in done_labels:
                print(f"\n--- {label} — already done, skipping ---")
                continue

            print(f"\n--- Running: {label} ---")

            #time retrieval eval (no generation pure retrieval latency)
            t0 = time.time()
            retrieval = eval_module.run_retrieval_eval(config, client, pipeline=pipeline)
            retrieval_time = time.time() - t0

            #time judge eval (includes generation + judging)
            t0 = time.time()
            judge = eval_module.run_judge_eval(config, client, openai_client, pipeline=pipeline)
            judge_time = time.time() - t0

            n_questions = len(retrieval["per_question"])
            avg_retrieval_ms = (retrieval_time / n_questions) * 1000

            print(f"  Recall@k:      {retrieval['mean_recall']:.3f}")
            print(f"  Faithfulness:  {judge['mean_faithfulness']:.3f}")
            print(f"  Correctness:   {judge['mean_correctness']:.3f}")
            print(f"  Completeness:  {judge['mean_completeness']:.3f}")
            print(f"  Avg retrieval: {avg_retrieval_ms:.0f}ms/query")

            all_results.append({
                "label": label,
                "config": config,
                "recall": retrieval["mean_recall"],
                "faithfulness": judge["mean_faithfulness"],
                "correctness": judge["mean_correctness"],
                "completeness": judge["mean_completeness"],
                "retrieval_time_s": round(retrieval_time, 2),
                "avg_retrieval_ms": round(avg_retrieval_ms, 1),
                "retrieval_per_question": slim_retrieval(retrieval["per_question"]),
                "judge_per_question": slim_judge(judge["per_question"]),
            })
            done_labels.add(label)

            #save after every config crash-safe
            with open("sweep_results.json", "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2)

    print(f"\nDone. {len(all_results)} configs in sweep_results.json")


if __name__ == "__main__":
    run_sweep()