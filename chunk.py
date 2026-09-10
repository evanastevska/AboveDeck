"""
chunk.py split corpus pages into retrievable chunks.

Reads the raw .txt files in corpus/ and turns each into a list of smaller overlapping chunks.
Each chunk carries metadata (show, season, source page) so retrieval can filter by season later.
"""

from pathlib import Path

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
        list[dict] - one record per chunk, each carrying its text, the page's metadata, and a unique chunk_id
    """
    #get the raw text pieces
    text_pieces = chunk_text(text, chunk_size, overlap)

    record_page_list = []

    #loop over pieces w their position number
    for i, piece in enumerate(text_pieces):
        record = metadata.copy() #
        record["text"] = piece #the current chunk, not the whole page (text)
        record["chunk_id"] = metadata["source_page"] + "-" + str(i)
        record_page_list.append(record)

    return record_page_list


def build_records(chunk_size, chunking_strategy):
    """
    Chunk every file in corpus/ into records, then print a few to eyeball.

    Returns:
        list[dict] — every chunk record from every page, all in one list
    """
    #point at the corpus folder and grab all .txt files (glob).

    corpus_dir = Path("corpus")
    files = corpus_dir.glob("*.txt") #glob=give me everything ending in .txt
    #each item comes back is a Path obj NOT str

    records_pages_list = [] #list to collect records from ALL pages

    overlap = chunk_size // 8

    if chunking_strategy != "fixed":
        print("This strategy does not exist")
        return []

    for file in files:
        text = file.read_text(encoding="utf-8") #read its text off disk
        metadata = metadata_for_file(file.name)
        record_page_list = chunk_page(text, metadata, chunk_size, overlap)
        records_pages_list.extend(record_page_list)


    #print(f"total records: {len(records_pages_list)}")
    #for r in records_pages_list[:3]:
    #    print(r)

    #for r in records_pages_list:
    #    if r["source_page"] in ("Below_Deck_Mediterranean_Season_10.txt", "Below_Deck_Season_12.txt"):
    #        print(r["chunk_id"], len(r["text"]))

    return records_pages_list
