# sweep.py, run the ablation sweep across all committed configs

import ablation
import eval as eval_module
import json
import os
from google import genai
from openai import OpenAI
from dotenv import load_dotenv
from collections import defaultdict

#baseline (everything else compared against this)
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

#all configs to sweep
#one axis at a time, everything else pinned to baseline
CONFIGS = [
    make_config("baseline"),

    #retrieval method axis
    make_config("bm25", retrieval_method="bm25"),
    make_config("hybrid", retrieval_method="hybrid"),

    #reranker axis (dense is the baseline retrieval method)
    make_config("dense+rerank", reranker="on"),

    #top-k axis
    make_config("topk2", top_k=2),
    make_config("topk10", top_k=10),
]


def run_sweep():
    """
    Run retrieval eval + judge eval for every config.

    Steps:
        1. Init API clients (genai + OpenAI)
        2. Group CONFIGS by setup_key so you only call setup_pipeline
           once per unique (chunk_size, chunking_strategy, embedding_model)
        3. Outer loop over groups:
             a. Call ablation.setup_pipeline once (pick any config in the group — they share setup)
           Inner loop over (label, config) pairs in that group:
             b. Run eval_module.run_retrieval_eval(config, client, pipeline=pipeline)
             c. Run eval_module.run_judge_eval(config, client, openai_client, pipeline=pipeline)
             d. Print label + scores so you can watch progress
             e. Append to all_results:
                {"label": label, "config": config,
                 "recall": retrieval_result["mean_recall"],
                 "faithfulness": judge_result["mean_faithfulness"],
                 "correctness": judge_result["mean_correctness"],
                 "completeness": judge_result["mean_completeness"]}
        4. Save all_results to sweep_results.json

    """
    load_dotenv()

    #1. clients
    api_key = os.getenv("GOOGLE_API_KEY")
    retrieval_client = genai.Client(api_key=api_key)
    judge_client = OpenAI()

    #group configs by setup_key
    #key = setup_key(config), value = list of (label, config) tuples
    config_map = defaultdict(list)
    for label, config in CONFIGS:
        key = setup_key(config)
        config_map[key].append((label, config))

    all_results = []

    #loop over groups
    #setup once per group, run all configs in that group
    for key, config_list in config_map.items():
        first_config = config_list[0][1]
        pipeline = ablation.setup_pipeline(first_config)

        for label, config in config_list:
            print(f"\n--- Running: {label} ---")

            retrieval_result = eval_module.run_retrieval_eval(config, retrieval_client, pipeline=pipeline)
            judge_result = eval_module.run_judge_eval(config, retrieval_client, judge_client, pipeline=pipeline)

            print(f"  Recall@k:      {retrieval_result['mean_recall']:.3f}")
            print(f"  Faithfulness:  {judge_result['mean_faithfulness']:.3f}")
            print(f"  Correctness:   {judge_result['mean_correctness']:.3f}")
            print(f"  Completeness:  {judge_result['mean_completeness']:.3f}")

            all_results.append({
                "label": label,
                "config": config,
                "recall": retrieval_result["mean_recall"],
                "faithfulness": judge_result["mean_faithfulness"],
                "correctness": judge_result["mean_correctness"],
                "completeness": judge_result["mean_completeness"],
            })

    #4.save
    with open("sweep_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDone. Saved {len(all_results)} configs to sweep_results.json")


if __name__ == "__main__":
    run_sweep()