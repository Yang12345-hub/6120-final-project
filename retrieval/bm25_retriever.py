"""
bm25_retriever.py — keyword-based retrieval using BM25

BM25 is basically a smarter version of TF-IDF. It scores documents based on
how often query words appear in them, with some length normalization thrown in.
Great for exact keyword matches — like if the user asks about "Thermopylae",
BM25 will find chunks that literally contain that word.

We use LangChain's BM25Retriever which wraps the rank_bm25 library.
"""

import json
import os
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# path to the chunks file we built in chunk.py
CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "..", "chunks.json")


def load_chunks_as_documents():
    """
    Load chunks.json and convert each chunk into a LangChain Document.
    LangChain Document is just a wrapper around (page_content, metadata) —
    nothing fancy, but it's what all the LangChain retrievers expect.
    """
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    docs = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk["text"],
            metadata=chunk["metadata"]
        )
        docs.append(doc)

    print(f"Loaded {len(docs)} documents for BM25 index")
    return docs


def build_bm25_retriever(k=5):
    """
    Build the BM25 retriever from our chunks.
    k = how many top results to return per query.
    """
    docs = load_chunks_as_documents()

    # BM25Retriever.from_documents handles all the tokenization and index
    # building internally — we just hand it the docs and it figures it out
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = k

    return retriever


def retrieve(query, retriever):
    """
    Run a query and return the top-k matching chunks.
    Returns a list of LangChain Documents.
    """
    results = retriever.invoke(query)
    return results


# quick test when running this file directly
if __name__ == "__main__":
    print("Building BM25 index...")
    retriever = build_bm25_retriever(k=5)

    test_query = "What happened at the Battle of Marathon?"
    print(f"\nQuery: {test_query}")
    print("-" * 50)

    results = retrieve(test_query, retriever)
    for i, doc in enumerate(results):
        print(f"\n[Result {i+1}]")
        print(f"  Volume: {doc.metadata['volume_name']}")
        print(f"  Chunk: {doc.metadata['chunk_index']}")
        print(f"  Text: {doc.page_content[:200]}...")
