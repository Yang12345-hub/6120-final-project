"""
app.py — Streamlit frontend for the RAG system

The user types a question, picks a retrieval method (BM25, dense, or hybrid),
and gets an answer with numbered citations. Each citation expands into a card
showing the original passage so the user can verify the answer.

Run with: streamlit run app.py
"""

import streamlit as st
import sys
import os

# make sure our local modules are importable
sys.path.insert(0, os.path.dirname(__file__))

from retrieval.bm25_retriever import build_bm25_retriever, retrieve as bm25_retrieve
from retrieval.dense_retriever import get_index_and_model, retrieve as dense_retrieve
from generation.llm import generate_answer_stream

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ancient History RAG",
    page_icon="📜",
    layout="wide",
)

st.title("📜 Ancient History Q&A")
st.caption("Ask anything about Herodotus, Thucydides, Plutarch, and more.")

# ── load retrievers (cached so they only build once per session) ──────────────

@st.cache_resource(show_spinner="Building BM25 index...")
def load_bm25():
    # BM25 index is fast to build, no GPU needed
    return build_bm25_retriever(k=5)


@st.cache_resource(show_spinner="Loading FAISS index and embedding model...")
def load_dense():
    # this loads from disk if we already ran dense_retriever.py, otherwise builds it
    index, metadatas, model = get_index_and_model()
    return index, metadatas, model


# ── sidebar — retrieval settings ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    retrieval_mode = st.radio(
        "Retrieval method",
        options=["BM25 (keyword)", "Dense (semantic)", "Hybrid (both)"],
        index=2,  # default to hybrid
        help="BM25 is great for exact terms. Dense is better for conceptual questions. Hybrid combines both."
    )

    top_k = st.slider("Number of chunks to retrieve", min_value=2, max_value=10, value=5)

    st.divider()
    st.markdown("**About**")
    st.markdown(
        "This RAG system searches across ancient history texts including "
        "Herodotus, Thucydides, Plutarch, and The History of Antiquity series."
    )


# ── helper: run retrieval based on selected mode ──────────────────────────────

def run_retrieval(query, mode, k):
    """
    Run whichever retrieval method the user picked.
    For hybrid, we run both and merge — deduplicating by text content
    so we don't send the same chunk to the LLM twice.
    """
    if mode == "BM25 (keyword)":
        retriever = load_bm25()
        # BM25Retriever returns LangChain Documents, convert to our dict format
        docs = bm25_retrieve(query, retriever)
        docs = docs[:k]
        chunks = [{"text": d.page_content, "metadata": d.metadata} for d in docs]

    elif mode == "Dense (semantic)":
        index, metadatas, model = load_dense()
        chunks = dense_retrieve(query, index, metadatas, model, k=k)

    else:  # Hybrid
        # grab k results from each, then merge and deduplicate
        retriever = load_bm25()
        bm25_docs = bm25_retrieve(query, retriever)
        bm25_chunks = [{"text": d.page_content, "metadata": d.metadata} for d in bm25_docs]

        index, metadatas, model = load_dense()
        dense_chunks = dense_retrieve(query, index, metadatas, model, k=k)

        # merge: start with dense results, then add BM25 results that aren't duplicates
        seen_texts = {c["text"] for c in dense_chunks}
        merged = list(dense_chunks)
        for c in bm25_chunks:
            if c["text"] not in seen_texts:
                merged.append(c)
                seen_texts.add(c["text"])

        chunks = merged[:k]  # cap at k total

    return chunks


# ── main query UI ─────────────────────────────────────────────────────────────

query = st.text_input(
    "Ask a question",
    placeholder="e.g. What were the causes of the Peloponnesian War?",
)

ask_button = st.button("Ask", type="primary", disabled=not query)

if ask_button and query:
    # step 1: retrieve relevant chunks
    with st.spinner("Searching for relevant passages..."):
        chunks = run_retrieval(query, retrieval_mode, top_k)

    st.markdown(f"**Retrieved {len(chunks)} passages** using *{retrieval_mode}*")

    # step 2: stream the answer from Llama
    st.markdown("### Answer")
    answer_placeholder = st.empty()

    answer_text = ""
    sources = []

    # stream tokens in — update the placeholder on each token
    for chunk in generate_answer_stream(query, chunks):
        if isinstance(chunk, str):
            # it's a token — append and re-render
            answer_text += chunk
            answer_placeholder.markdown(answer_text + "▌")  # blinking cursor effect
        elif isinstance(chunk, dict):
            # it's the sentinel with sources — we're done streaming
            sources = chunk["sources"]

    # final render without the cursor
    answer_placeholder.markdown(answer_text)

    # step 3: render citation cards
    if sources:
        st.markdown("### Sources")
        st.caption("Click a source to expand the original passage.")

        for s in sources:
            label = f"[{s['citation_number']}] {s['volume_name']} — chunk {s['chunk_index']}"
            with st.expander(label):
                st.markdown(f"**File:** `{s['filename']}`")
                st.markdown(f"**Chunk index:** {s['chunk_index']}")
                st.divider()
                st.write(s["text_preview"])
