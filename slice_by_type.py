# slice_by_type.py read sweep_results.json, slice by query type
# Two views:
#1. Per-config breakdown (each config's scores split by question type)
#2. Per-type comparison (for each question type, how do configs compare)

import json
from collections import defaultdict

def load_results(path="sweep_results.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def slice_one_config(result):
    """Return {query_type: {recall, correctness, completeness, n}} for one config."""
    #recall by type
    recall_by_type = defaultdict(list)
    for q in result["retrieval_per_question"]:
        if q["recall"] is not None:
            recall_by_type[q["query_type"]].append(q["recall"])

    #judge by type
    judge_by_type = defaultdict(lambda: {"correctness": [], "completeness": []})
    for q in result["judge_per_question"]:
        if q["scores"] is not None:
            judge_by_type[q["query_type"]]["correctness"].append(q["scores"]["correctness"])
            judge_by_type[q["query_type"]]["completeness"].append(q["scores"]["completeness"])

    #merge into one dict
    all_types = sorted(set(list(recall_by_type.keys()) + list(judge_by_type.keys())))
    sliced = {}
    for qtype in all_types:
        recalls = recall_by_type.get(qtype, [])
        judges = judge_by_type.get(qtype, {"correctness": [], "completeness": []})
        sliced[qtype] = {
            "n": max(len(recalls), len(judges["correctness"])),
            "recall": sum(recalls) / len(recalls) if recalls else None,
            "correctness": sum(judges["correctness"]) / len(judges["correctness"]) if judges["correctness"] else None,
            "completeness": sum(judges["completeness"]) / len(judges["completeness"]) if judges["completeness"] else None,
        }
    return sliced


def print_per_config(results):
    """View 1: each config's scores broken down by question type."""
    for result in results:
        label = result["label"]
        sliced = slice_one_config(result)
        avg_ms = result.get("avg_retrieval_ms", "—")

        print(f"\n{'='*60}")
        print(f"  {label}  (overall: Recall {result['recall']:.3f} | "
              f"Correct {result['correctness']:.3f} | "
              f"Complete {result['completeness']:.3f} | "
              f"Retrieval {avg_ms}ms)")
        print(f"{'='*60}")
        print(f"  {'Type':<20} {'N':>3} {'Recall':>8} {'Correct':>8} {'Complete':>9}")
        print(f"  {'-'*50}")
        for qtype, data in sorted(sliced.items()):
            r = f"{data['recall']:.3f}" if data["recall"] is not None else "   —"
            c = f"{data['correctness']:.3f}" if data["correctness"] is not None else "   —"
            comp = f"{data['completeness']:.3f}" if data["completeness"] is not None else "    —"
            print(f"  {qtype:<20} {data['n']:>3} {r:>8} {c:>8} {comp:>9}")


def print_per_type(results, labels=None):
    """View 2: for each question type, compare selected configs side by side.

    This is the view you need for the findings paragraph —
    'hybrid+topk10 improved multi-hop from X to Y'.
    """
    if labels is None:
        labels = [r["label"] for r in results]

    selected = [r for r in results if r["label"] in labels]
    all_sliced = {r["label"]: slice_one_config(r) for r in selected}

    #get all query types
    all_types = sorted(set().union(*(s.keys() for s in all_sliced.values())))

    for qtype in all_types:
        print(f"\n--- {qtype} ---")
        print(f"  {'Config':<28} {'Recall':>8} {'Correct':>8} {'Complete':>9}")
        print(f"  {'-'*55}")
        for label in labels:
            data = all_sliced[label].get(qtype, {})
            r = f"{data['recall']:.3f}" if data.get("recall") is not None else "   —"
            c = f"{data['correctness']:.3f}" if data.get("correctness") is not None else "   —"
            comp = f"{data['completeness']:.3f}" if data.get("completeness") is not None else "    —"
            print(f"  {label:<28} {r:>8} {c:>8} {comp:>9}")


if __name__ == "__main__":
    results = load_results()

    #view 1: every config broken down
    print("\n" + "="*60)
    print("  VIEW 1: PER-CONFIG BREAKDOWN")
    print("="*60)
    print_per_config(results)

    #view 2: compare key configs by question type
    #these are the ones that matter for the findings paragraph
    key_configs = ["baseline", "bm25", "topk10", "hybrid+topk10"]
    print("\n\n" + "="*60)
    print("  VIEW 2: PER-TYPE COMPARISON (key configs)")
    print("="*60)
    print_per_type(results, labels=key_configs)