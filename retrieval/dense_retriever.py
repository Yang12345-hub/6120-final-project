"""
dense_retriever.py — semantic/embedding-based retrieval using SBERT + FAISS

Unlike BM25 which looks for exact word matches, dense retrieval converts text
into vectors (embeddings) and finds chunks that are semantically similar to
the query — even if they use different words.

We use:
  - all-MiniLM-L6-v2 from Sentence-Transformers to create the embeddings
  - FAISS to store them and do fast nearest-neighbor search

One gotcha: building the index takes a while (embedding 14k chunks), so we
save it to disk and reload it on subsequent runs instead of rebuilding every time.
"""

import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# paths
CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "..", "chunks.json")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "faiss.index")
METADATA_PATH = os.path.join(os.path.dirname(__file__), "..", "faiss_metadata.json")

# the model — small, fast, and surprisingly good for semantic similarity
MODEL_NAME = "all-MiniLM-L6-v2"


def load_chunks():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_index():
    """
    Embed all chunks and build a FAISS index. This only needs to run once —
    after that we load from disk.
    """
    print("Loading model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Loading chunks...")
    chunks = load_chunks()

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    print(f"Embedding {len(texts)} chunks... (this might take a minute)")
    # encode in batches — sentence-transformers handles this internally
    # show_progress_bar=True gives us a nice progress bar
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)

    # normalize so cosine similarity == dot product (makes FAISS search easier)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # build a flat L2 index — "flat" means exact search, no approximation
    # for 14k vectors this is totally fine, fast enough
    dim = embeddings.shape[1]  # should be 384 for MiniLM
    index = faiss.IndexFlatIP(dim)  # IP = inner product (= cosine sim after normalization)
    index.add(embeddings.astype(np.float32))

    # save both the index and the metadata separately
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadatas, f, ensure_ascii=False)

    print(f"Index built and saved. Total vectors: {index.ntotal}")
    return index, metadatas, model


def load_index():
    """
    Load the pre-built FAISS index from disk instead of rebuilding.
    """
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadatas = json.load(f)
    return index, metadatas


def get_index_and_model():
    """
    Check if we already have a saved index — if yes, load it.
    If not, build it from scratch. Returns (index, metadatas, model).
    """
    model = SentenceTransformer(MODEL_NAME)

    if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
        print("Loading existing FAISS index from disk...")
        index, metadatas = load_index()
    else:
        print("No saved index found — building from scratch...")
        index, metadatas, model = build_index()

    return index, metadatas, model


def retrieve(query, index, metadatas, model, k=5):
    """
    Embed the query and find the k most similar chunks.
    Returns a list of dicts with 'text' and 'metadata' keys,
    plus a 'score' so we can see how confident the retrieval is.
    """
    # embed + normalize the query the same way we did the chunks
    query_vec = model.encode([query])
    query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)

    # search — D = distances (scores), I = indices into our metadata list
    D, I = index.search(query_vec.astype(np.float32), k)

    # load the original chunks to get the text back
    chunks = load_chunks()

    results = []
    for score, idx in zip(D[0], I[0]):
        results.append({
            "text": chunks[idx]["text"],
            "metadata": metadatas[idx],
            "score": float(score)
        })

    return results


# quick test
if __name__ == "__main__":
    index, metadatas, model = get_index_and_model()

    test_query = "Persian invasion of Greece"
    print(f"\nQuery: {test_query}")
    print("-" * 50)

    results = retrieve(test_query, index, metadatas, model, k=5)
    for i, r in enumerate(results):
        print(f"\n[Result {i+1}] score={r['score']:.4f}")
        print(f"  Volume: {r['metadata']['volume_name']}")
        print(f"  Chunk: {r['metadata']['chunk_index']}")
        print(f"  Text: {r['text'][:200]}...")
