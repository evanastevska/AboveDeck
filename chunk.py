"""
chunk.py split corpus pages into retrievable chunks.

Reads the raw .txt files in corpus/ and turns each into a list of smaller overlapping chunks. 
Each chunk carries metadata (show, season, source page) so retrieval can filter by season later.
"""

def chunk_text(text, chunk_size, overlap):
    """
    Split ONE page's text into overlapping fixed-size chunks.

    Args:
        text:       the full page text (raw wikitext, uncleaned for now)
        chunk_size: characters per chunk
        overlap:    characters each chunk shares with the previous one

    Returns:
        list[str] - just the text pieces, no metadata yet

    The sliding window:
        - start at position 0
        - take a slice chunk_size characters long
        - move start forward by (chunk_size - overlap)   <-this is the "step"
        - repeat until run off the end
    """

    #for the window to move FORWARD each loop, overlap cannot be more than chunk_size
    #start can never advance
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk_size")

    start = 0
    list_chunks = []

    while start < len(text): #stop once start is past the end
        list_chunks.append(text[start : start + chunk_size]) #take the slice at current position, then keep it

        if start + chunk_size >= len(text): #if the slice took the last letter, do not create a new chunk, just keep in last slice. keeping "ghij" and "j" will pollute retrieval 
            break

        start += (chunk_size - overlap) #advance start by step

    return list_chunks


def metadata_for_file(filename):
    """
    Recover (show, season, source_page) from a corpus filename.

    Args:
        filename: e.g. "Below_Deck_Mediterranean_Season_1.txt"

    Returns:
        a dict: {"show": ..., "season": ..., "source_page": ...}
        (dict is handy - chunk_page will merge these keys straight into each chunk record)
    """

    meta = {} #my own dict

    meta["source_page"] = filename

    #split the filename on "_" into a list of wordpieces
    file_name_split = filename.split("_")

    if "Mediterranean" in file_name_split:
        meta["show"] = "Below Deck Mediterranean"
    else:
        meta["show"] = "Below Deck"

    #season grab the LAST piece ("1.txt"), drop the ".txt",convert to an int
    last_piece = file_name_split[-1]
    season_str = last_piece.replace(".txt", "")
    season = int(season_str)

    meta["season"] = season

    #returns "source_page": filename, "show": show, "season": season}
    return meta



def chunk_page(text, metadata, chunk_size, overlap):
    """
    Turn ONE page into a list of chunk RECORDS (dicts: text + metadata + id).

    Args:
        text:       the full page text
        metadata:   dict from metadata_for_file (show, season, source_page)
        chunk_size, overlap: passed straight through to chunk_text

    Returns:
        list[dict] - one record per chunk, each carrying its text, the page's
        metadata, and a unique chunk_id
    """
    # TODO: get the raw text pieces by calling chunk_text(...).

    # TODO: make an empty list to collect finished records.

    # TODO: loop over the pieces WITH their position number (0, 1, 2, ...).
    #       for each piece:
    #         - start a record from a COPY of metadata  (not metadata itself!)
    #         - add this piece's text under "text"
    #         - add a chunk_id: source_page + "-" + str(position)
    #         - append the record to your list

    # TODO: return the list of records.
    pass


def main():
    """Chunk every file in corpus/, then print a few to eyeball."""
    # TODO: list the .txt files in corpus/
    # TODO: for each file: read text -> metadata_for_file() -> chunk_page()
    #       -> add its records to one big list
    # TODO: print a handful. Look hard at: do the boundaries make sense?
    #       is the metadata correct? what do the two THIN pages (Med S10,
    #       BD S12) produce — one short chunk, or something weird?
    #       (you'll see raw wiki markup like [[ ]] and {{ }} inside chunks —
    #        that's EXPECTED, you chose not to clean yet. not a bug.)
    pass