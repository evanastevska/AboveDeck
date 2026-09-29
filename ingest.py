
"""
Pull Below Deck wiki pages via the MediaWiki API and save each as a .txt file.
"""

import requests
import time
from pathlib import Path

API_URL = "https://below-deck.fandom.com/api.php"
HEADERS = {"User-Agent": "BelowDeckRAG/0.1 (learning project)"}  #polite bot
REQUEST_DELAY_SECONDS = 0.5




#goal of these functions is to build a corpus


def call_api(params):
    """
    Send ONE request to the API and return the parsed JSON.
    Every call needs format=json, so enforce it here so never forgot.

    TODO:
      - make sure params includes format="json"
      - requests.get(API_URL, params=..., headers=HEADERS)
      - time.sleep(REQUEST_DELAY_SECONDS)  #be polite
      - return response.json()

    """

    params["format"] = "json"

    response = requests.get(API_URL, params=params, headers=HEADERS)
    data = response.json()
    time.sleep(REQUEST_DELAY_SECONDS)
    return data





#DOESNT WORK: first try, wiki does not tag categories reliably, switched to generating titles from a naming pattern
def get_pages_in_category(category):
    """
    Return a list of all page titles in a wiki category (e.g. "Episodes").

    Uses action=query, list=categorymembers. The catch: the API returns
    results in batches and hands you a 'continue' token when there's more.
    So this needs a LOOP that keeps calling until no continue token comes back.

    TODO:
      - build params: action, list, cmtitle (="Category:<category>"), cmlimit, format
      - call_api(params), pull titles out of the JSON
      - if the response has a 'continue' key, feed it back in and go again
    """


    titles = []

    PARAMS = {
        "action": "query",
        "cmtitle": f"Category:{category}",
        "cmlimit": "20",
        "list": "categorymembers",
        "format": "json"
    }

    data = call_api(PARAMS)
    print(data)

    PAGES = data["query"]["categorymembers"]
    for p in PAGES:
        titles.append(p["title"])


    while "continue" in data: #if theres more results since it sends in batches
        PARAMS.update(data["continue"]) #add results to params

        data = call_api(PARAMS) #fetch again so data is new batch

        PAGES = data["query"]["categorymembers"] #update pages
        for p in PAGES:
                titles.append(p["title"])


    return titles





def get_page_text(title):
    """
    Return the clean plain-text of a single page.

    TODO: build params, call_api, extract the text
    """


    PARAMS = {
        "action": "parse",
        "prop": "wikitext",
        "format": "json",
        "page": title
    }

    data = call_api(PARAMS)

    text = data["parse"]["wikitext"]["*"]
    return text





def save_page(title, text, out_dir="corpus"):
    """
    Write one page's text to disk as a .txt file.
    (Metadata like season/episode can parse from the title in a later step.)

    TODO:
      - Path(out_dir).mkdir(parents=True, exist_ok=True)
      - make a safe filename (spaces and slashes in titles break paths)
      - write the text
    """

    safe_title = title.replace(" ", "_").replace("/", "-").replace("'", "")

    out_dir = Path(out_dir) #turns folder into a Path object and stores it
    out_dir.mkdir(parents=True, exist_ok=True) #makes directory

    filepath = out_dir / f"{safe_title}.txt"

    #with means open file, do thing, close
    #w means write mode (creates file or overwrites if exists)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)



def main():
    list = []
    for n in range(1, 13):
        list.append(f"Below Deck Season {n}")

    for n in range(1, 11):
        list.append(f"Below Deck Mediterranean Season {n}")

    for title in list:
        text = get_page_text(title)
        save_page(title, text)
        print(f"Saved: {title}")

    #note: Med S10 and BD S12 were mid-air at scrape time, so they r sparse


if __name__ == "__main__":
    main()








