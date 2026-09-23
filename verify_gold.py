"""
verify_gold.py one-off sanity check for gold_set.json.

For each gold entry, checks whether the answer_text (or a key phrase)
appears in the chunk(s) referenced by gold_chunk_ids. Flags mismatches
so you can review them manually.

Usage:
    python verify_gold.py
"""

import json
import chunk


def verify():
    #Build the same baseline chunks the gold set was written against
    records = chunk.build_records(chunk_size=800, chunking_strategy="fixed")

    #Make a lookup dict: chunk_id -> chunk text
    chunk_lookup = {r["chunk_id"]: r["text"] for r in records}

    #Load the gold set
    with open("gold_set.json", "r", encoding="utf-8") as f:
        gold_set = json.load(f)

    flagged = []

    for entry in gold_set:
        gold_ids = entry["gold_chunk_ids"]

        #Skip negation questions with empty gold_chunk_ids
        if not gold_ids:
            continue

        answer = entry["answer_text"].lower()

        #Check if ANY of the gold chunks contain the answer text
        found_in_any = False
        for chunk_id in gold_ids:
            if chunk_id not in chunk_lookup:
                flagged.append({
                    "id": entry["id"],
                    "reason": f"chunk_id {chunk_id} not found in records",
                    "query": entry["query_text"],
                    "answer": entry["answer_text"]
                })
                continue

            chunk_text = chunk_lookup[chunk_id].lower()
            if answer in chunk_text:
                found_in_any = True
                break

        #If full answer not found, try individual words (3+ chars)
        #to catch partial matches, flag for review but note it
        if not found_in_any:
            # Try key words from the answer (skip short words)
            answer_words = [w for w in answer.split() if len(w) >= 4]
            words_found = 0
            for word in answer_words:
                for chunk_id in gold_ids:
                    if chunk_id in chunk_lookup and word in chunk_lookup[chunk_id].lower():
                        words_found += 1
                        break

            word_pct = words_found / len(answer_words) if answer_words else 0

            flagged.append({
                "id": entry["id"],
                "query": entry["query_text"],
                "answer": entry["answer_text"],
                "gold_chunk_ids": gold_ids,
                "word_match": f"{words_found}/{len(answer_words)} key words found ({word_pct:.0%})"
            })

    #Report
    print(f"\nTotal entries checked: {len([e for e in gold_set if e['gold_chunk_ids']])}")
    print(f"Flagged for review: {len(flagged)}\n")

    for item in flagged:
        print(f"  ID {item['id']}: {item['query']}")
        print(f"    Answer: {item['answer']}")
        if 'reason' in item:
            print(f"    Issue: {item['reason']}")
        else:
            print(f"    Chunks: {item['gold_chunk_ids']}")
            print(f"    {item['word_match']}")
        print()


if __name__ == "__main__":
    verify()