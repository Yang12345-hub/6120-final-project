"""
chunk.py — split all the historical texts into overlapping chunks
so the retriever has bite-sized pieces to search over.

Strategy: 300 tokens per chunk, 50-token overlap between consecutive chunks.
We use a simple whitespace tokenizer (split by spaces) because we just need
a rough token count — we're not doing anything fancy here.
"""

import os
import json

# where the raw txt files live
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# where we'll save the chunks as a JSON file
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "chunks.json")

CHUNK_SIZE = 200    # tokens per chunk
OVERLAP = 50        # how many tokens we repeat at the start of the next chunk


def tokenize(text):
    # super simple — just split on whitespace
    # good enough for counting and reassembling
    return text.split()


def detokenize(tokens):
    # put the tokens back into a string
    return " ".join(tokens)


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """
    Takes a big string and returns a list of overlapping text chunks.
    Each chunk is chunk_size tokens long, and consecutive chunks share
    'overlap' tokens at the boundary.
    """
    tokens = tokenize(text)
    chunks = []

    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunks.append(detokenize(chunk_tokens))

        # move forward by (chunk_size - overlap) so the next chunk
        # reuses the last 'overlap' tokens of this one
        start += chunk_size - overlap

    return chunks


def extract_volume_name(filename):
    """
    Pull a human-readable volume name out of the filename.
    Just strips the .txt extension — nothing clever needed.
    """
    return filename.replace(".txt", "")


def build_all_chunks():
    """
    Loop over every .txt file in data/, chunk it up, and collect
    metadata for each chunk: which file it came from, which chunk
    number it is within that file, and the volume name.
    """
    all_chunks = []

    txt_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]
    print(f"Found {len(txt_files)} text files in {DATA_DIR}")

    for filename in sorted(txt_files):
        filepath = os.path.join(DATA_DIR, filename)

        print(f"  Chunking: {filename} ...", end=" ", flush=True)

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        chunks = chunk_text(text)

        volume_name = extract_volume_name(filename)

        for i, chunk_text_str in enumerate(chunks):
            all_chunks.append({
                "text": chunk_text_str,
                "metadata": {
                    "filename": filename,
                    "chunk_index": i,
                    "volume_name": volume_name,
                }
            })

        print(f"{len(chunks)} chunks")

    return all_chunks


if __name__ == "__main__":
    chunks = build_all_chunks()

    # save to disk so retrieval code can load it without re-chunking every time
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"\nTotal chunks: {len(chunks)}")
    print(f"Saved to: {OUTPUT_PATH}")

    # quick sanity check — print the first chunk so we can eyeball it
    print("\n--- First chunk preview ---")
    first = chunks[0]
    print(f"Volume: {first['metadata']['volume_name']}")
    print(f"Chunk index: {first['metadata']['chunk_index']}")
    print(f"Text (first 200 chars): {first['text'][:200]}...")
