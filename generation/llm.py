"""
llm.py — talk to local Llama 3 8B running via Ollama and build the answer

The flow here is:
  1. take the retrieved chunks from BM25 or dense retriever
  2. build a prompt that stuffs those chunks in as context
  3. send it to Ollama (which is just running locally on port 11434)
  4. return the answer + which chunks were cited as [1], [2], etc.

Why Ollama? It lets us run Llama 3 locally without needing an API key.
Just needs `ollama pull llama3` and `ollama serve` running in the background.
"""

import requests
import json

# Ollama runs a local HTTP server — this is the default endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"  # make sure you've done: ollama pull llama3


def build_prompt(query, retrieved_chunks):
    """
    Build the prompt we send to Llama. We number each chunk so the model
    can reference them as [1], [2], etc. in its answer.

    The instruction at the top tells the model to only use the provided
    context and to cite sources — keeps it grounded instead of hallucinating.
    """
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        vol = chunk["metadata"]["volume_name"]
        text = chunk["text"]
        # each chunk gets a numbered header so the model knows what to cite
        context_blocks.append(f"[{i}] (From: {vol})\n{text}")

    context_str = "\n\n".join(context_blocks)

    prompt = f"""You are a helpful assistant answering questions about ancient history.
Use ONLY the context passages provided below to answer the question.
After each claim in your answer, cite the source using its number like [1] or [2].
If the context doesn't contain enough information, say so honestly.

--- CONTEXT ---
{context_str}
--- END CONTEXT ---

Question: {query}

Answer:"""

    return prompt


def query_ollama(prompt, temperature=0.3):
    """
    Send the prompt to Ollama and stream the response back.
    temperature=0.3 keeps answers focused and factual — we don't want
    the model getting creative with historical facts.

    Returns the full response text as a string.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "temperature": temperature,
        "stream": True,  # stream so we can show output token by token in Streamlit
    }

    response = requests.post(OLLAMA_URL, json=payload, stream=True)

    if response.status_code != 200:
        raise RuntimeError(f"Ollama returned status {response.status_code}: {response.text}")

    # collect streamed chunks into a full response string
    full_response = ""
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            full_response += data.get("response", "")
            if data.get("done", False):
                break

    return full_response.strip()


def query_ollama_stream(prompt, temperature=0.3):
    """
    Generator version of query_ollama — yields tokens one by one.
    Streamlit can use this to show the answer as it's being generated,
    which feels much more responsive than waiting for the whole thing.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "temperature": temperature,
        "stream": True,
    }

    response = requests.post(OLLAMA_URL, json=payload, stream=True)

    if response.status_code != 200:
        raise RuntimeError(f"Ollama returned status {response.status_code}: {response.text}")

    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            token = data.get("response", "")
            if token:
                yield token
            if data.get("done", False):
                break


def generate_answer(query, retrieved_chunks):
    """
    Main entry point — takes a query + list of retrieved chunks,
    returns a dict with:
      - 'answer': the text response with [1][2] citations inline
      - 'sources': list of source metadata for rendering citation cards
    """
    prompt = build_prompt(query, retrieved_chunks)
    answer = query_ollama(prompt)

    # build the sources list so the frontend can render clickable citation cards
    sources = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        sources.append({
            "citation_number": i,
            "volume_name": chunk["metadata"]["volume_name"],
            "filename": chunk["metadata"]["filename"],
            "chunk_index": chunk["metadata"]["chunk_index"],
            "text_preview": chunk["text"][:300] + "...",
        })

    return {
        "answer": answer,
        "sources": sources,
    }


def generate_answer_stream(query, retrieved_chunks):
    """
    Streaming version of generate_answer. Yields tokens for the answer,
    then at the end yields a special sentinel dict with the sources list.

    In Streamlit you'd do something like:
        for chunk in generate_answer_stream(query, chunks):
            if isinstance(chunk, str):
                answer_so_far += chunk
                placeholder.write(answer_so_far)
            else:
                sources = chunk["sources"]
    """
    prompt = build_prompt(query, retrieved_chunks)

    for token in query_ollama_stream(prompt):
        yield token  # these are all strings

    # after streaming is done, yield the sources as a dict
    sources = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        sources.append({
            "citation_number": i,
            "volume_name": chunk["metadata"]["volume_name"],
            "filename": chunk["metadata"]["filename"],
            "chunk_index": chunk["metadata"]["chunk_index"],
            "text_preview": chunk["text"][:300] + "...",
        })

    yield {"sources": sources}  # sentinel — frontend checks isinstance(chunk, dict)


# quick test — run this to make sure Ollama is up and responding
if __name__ == "__main__":
    # pretend we got these from the retriever
    fake_chunks = [
        {
            "text": "The Battle of Marathon took place in 490 BC. The Athenians, though outnumbered, defeated the Persian army on the plain of Marathon and prevented the Persian invasion of Greece.",
            "metadata": {
                "volume_name": "Herodotus-The Histories_1",
                "filename": "Herodotus-The Histories_1.txt",
                "chunk_index": 42,
            }
        },
        {
            "text": "Miltiades commanded the Athenian forces at Marathon. He persuaded the other generals to attack rather than wait, arguing that delay would only hurt Athenian morale.",
            "metadata": {
                "volume_name": "Herodotus-The Histories_2",
                "filename": "Herodotus-The Histories_2.txt",
                "chunk_index": 17,
            }
        }
    ]

    query = "Who commanded the Athenians at Marathon and what was the outcome?"
    print(f"Query: {query}\n")
    print("Asking Ollama... (make sure `ollama serve` is running)")

    result = generate_answer(query, fake_chunks)
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources:")
    for s in result["sources"]:
        print(f"  [{s['citation_number']}] {s['volume_name']} (chunk {s['chunk_index']})")
