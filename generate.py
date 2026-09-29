import os
from google import genai
from dotenv import load_dotenv


def generate(query_text, retrieved, client, model_name):
    """
    Build a grounded prompt from retrieved chunks, call the LLM, return the answer.

    Args:
        query_text: the user's original question
        retrieved:  list of dicts from retrieve(), each has "doc", "meta", "dist"
        client:     the genai.Client you already built
        model_name: which model to call (e.g. "gemini-3.6-flash")

    Returns:
        str , the model's answer
    """
    #build a context block from retrieved.
    #loop through the hits. For each one, label it with its source(source_page + season from meta) and include the chunk text.
    #join into one big string the model can read.

    context_segments = []

    #loop through combined list
    for i, item in enumerate(retrieved, start=1):
        doc_text = item["doc"]

        source_page = item["meta"].get("source_page", f"Document {i}")

        season = item["meta"].get("season", f"Document {i}")

        segment = f"[Doc {i}] Source Page: {source_page} Season: {season}\nContent: {doc_text}"
        context_segments.append(segment)

    #combine everything into a single string block
    context_block = "\n\n".join(context_segments)


    prompt = f"Here is the instruction: Answer the question ONLY from the context, and say so if the answer to the question is not there.\n\nHere is the context: {context_block}.\n\nHere is the question: {query_text}."

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )

    return response.text
