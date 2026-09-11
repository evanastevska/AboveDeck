"""
gold_lookup.py helper for building the gold evaluation set.

Runs the baseline chunker and lets you search for a phrase
across all chunks to find which chunk_id(s) contain it.

Usage:
    python gold_lookup.py "Captain Lee"
"""

import sys
import chunk


def search_chunks(records, search_term):
    """
    Find every chunk that contains search_term (case-insensitive).

    Args:
        records:     list of chunk dicts from build_records()
        search_term: the phrase to look for

    Prints:
        For each match: the chunk_id, source_page, and a short
        snippet showing the search term in context.

    TODO:
        - loop through records
        - check if search_term appears in the chunk's text
        - for matches, print chunk_id and enough surrounding text
          to confirm it's the right hit (maybe ~80 or 40 chars around the match?)
    """
    for record in records:
        text = record["text"]

        text_content = text.lower()
        search_content = search_term.lower()

        margin = 40

        pos = text_content.find(search_content)
        if pos == -1:
            continue

        start = max(0, pos - margin)
        end = min(len(text), pos + len(search_term) + margin)
        
        print(f"Match found in Record ID {record['chunk_id']}: {text[start:end]}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit()

    search_term = sys.argv[1]

    records = chunk.build_records(chunk_size=800, chunking_strategy="fixed")

    search_chunks(records, search_term)