"""
Production-lean RAG backend (local indexes) with:
- PDF ingestion + chunking
- Dense retrieval (FAISS + sentence-transformers)
- Sparse retrieval (TF-IDF ~ BM25-ish)
- Pool merge -> MMR diversification -> Cross-encoder re-rank
- Token-budget selection
- Pretty prints + stage timings
- Optional Groq LLM call

Endpoints:
- GET  /health
- POST /index  (multipart/form-data file upload)
- POST /query  (JSON body: {"query": "...", "max_answer_tokens": 350})

Authoring notes (Windows + Anaconda + VSCode):
- Activate conda env in VSCode terminal: conda activate ragprod
- Run server: uvicorn app.main:app --reload
"""

import os
import time
import pickle
import hashlib
from typing import List, Dict, Any
import requests
from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from collections import defaultdict, deque

from app.core.budget import build_messages, ntoks
import os

# Text loading & splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Embeddings (dense) + re-ranker
from sentence_transformers import SentenceTransformer, CrossEncoder

# Dense ANN index
import faiss
import numpy as np

# Sparse retriever (TF-IDF ~ BM25-ish proxy for hybrid)
from sklearn.feature_extraction.text import TfidfVectorizer

# (Optional) load environment variables from .env for API keys, etc.
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# --------------------------
# FastAPI app
# --------------------------
app = FastAPI(title="RAG Retrieval Service", version="1.0")


# --------------------------
# Config (env-overridable)
# --------------------------
EMB_MODEL_NAME = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Retrieval knobs
K_FETCH = int(os.getenv("K_FETCH", "25"))      # initial pool size (dense + sparse combined)
K_MMR = int(os.getenv("K_MMR", "8"))           # diversified subset after MMR
M_FINAL = int(os.getenv("M_FINAL", "4"))       # chunks sent to LLM after re-rank

TOKEN_BUDGET = int(os.getenv("TOKEN_BUDGET", "1200"))   # max context tokens you allow

DATA_DIR = "data"      # incoming PDFs
INDEX_DIR = "indexes"  # persisted indexes (optional)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# --------------------------
# Models
# --------------------------
# Sentence embeddings model for dense retrieval (free, solid)
emb_model = SentenceTransformer(EMB_MODEL_NAME)
EMB_DIM = emb_model.get_sentence_embedding_dimension()

# Cross-encoder for accurate re-ranking (small & fast)
reranker = CrossEncoder(RERANK_MODEL_NAME)

# --------------------------
# In-memory stores
# --------------------------
TEXTS: List[str] = []   # chunk texts
METAS: List[Dict[str, Any]] = []  # metadata per chunk: page, source, id(hash)

faiss_index = None  # FAISS inner-product index on normalized embeddings
tfidf: TfidfVectorizer = None
tfidf_mat = None     # sparse matrix for TF-IDF scores
# Conversation history (per session_id)
SESSIONS: dict[str, deque[tuple[str, str]]] = defaultdict(lambda: deque(maxlen=30))

# --------------------------
# Utilities
# --------------------------
def hash_text(t: str) -> str:
    """Stable ID for a text chunk (handy for dedupe)."""
    return hashlib.md5(t.encode("utf-8")).hexdigest()


def chunk_pdf(path: str) -> List[Dict[str, Any]]:
    """
    Load PDF and split into overlapping chunks.
    Returns a list of dicts: {"text": str, "page": int, "source": str}
    """
    docs = PyPDFLoader(path).load()  # -> list[Document], typically one per page
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    items = []
    for d in chunks:
        items.append({
            "text": d.page_content,
            "page": d.metadata.get("page", -1),
            "source": d.metadata.get("source", os.path.basename(path))
        })
    return items


def build_dense_index(texts: List[str]) -> faiss.IndexFlatIP:
    """
    Create a FAISS inner-product index using normalized embeddings.
    Inner product on normalized vectors == cosine similarity.
    """
    X = emb_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    index = faiss.IndexFlatIP(EMB_DIM)
    index.add(np.array(X, dtype=np.float32))
    return index


def build_sparse_index(texts: List[str]):
    """
    Build a TF-IDF vectorizer + matrix.
    This serves as a quick, local, keyword-based retriever.
    For true BM25, you can use specialized libs; TF-IDF works well enough for hybrid demos.
    """
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50_000)
    mat = vectorizer.fit_transform(texts)
    return vectorizer, mat


def mmr_diversify(query_vec: np.ndarray,
                  cand_vecs: np.ndarray,
                  k: int = 8,
                  lambda_mult: float = 0.5) -> List[int]:
    """
    Max Marginal Relevance (MMR):
    Select k items balancing relevance to the query and novelty vs already selected.
    Returns indices into `cand_vecs`.

    query_vec: (d,)
    cand_vecs: (n, d) normalized embeddings
    lambda_mult: 1.0 -> pure relevance; 0.0 -> pure diversity
    """
    selected_idx: List[int] = []
    if cand_vecs.shape[0] == 0:
        return selected_idx

    # Relevance: cosine similarity (dot on normalized)
    sims = cand_vecs @ query_vec

    # 1) seed with most relevant
    first = int(np.argmax(sims))
    selected_idx.append(first)

    # 2) iteratively add items maximizing MMR score
    while len(selected_idx) < min(k, cand_vecs.shape[0]):
        remaining = [i for i in range(cand_vecs.shape[0]) if i not in selected_idx]
        best_score, best_i = -1e9, None
        for i in remaining:
            relevance = sims[i]
            # diversity: max similarity to any already selected
            diversity = 0.0
            for j in selected_idx:
                diversity = max(diversity, float(cand_vecs[i] @ cand_vecs[j]))
            # MMR objective
            score = lambda_mult * relevance - (1 - lambda_mult) * diversity
            if score > best_score:
                best_score, best_i = score, i
        selected_idx.append(best_i)

    return selected_idx


def estimate_tokens(text: str) -> int:
    """Crude token estimator (word ≈ token in English) sufficient for budgeting."""
    return len(text.split())


def select_with_token_budget(docs: List[Dict[str, Any]], budget: int) -> List[Dict[str, Any]]:
    """
    Greedy take from top until we hit the token budget.
    Assumes docs are already re-ranked best-first.
    """
    total, out = 0, []
    for d in docs:
        t = estimate_tokens(d["text"])
        if total + t <= budget:
            out.append(d)
            total += t
        else:
            break
    return out


# Helper for calling groq sdk
def call_groq_chat_sdk(model: str, prompt: str, max_tokens: int = 350) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set. Put it in your .env and ensure load_dotenv() runs before app starts.")
    client = Groq(api_key=api_key)

    # The SDK uses `max_completion_tokens` (NOT max_tokens)
    completion = client.chat.completions.create(
        model=model,  # e.g., "llama-3.3-70b-versatile"
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_completion_tokens=max_tokens,
        stream=False,
    )
    return completion.choices[0].message.content


# --------------------------
# API: Health
# --------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "chunks": len(TEXTS),
        "dense_index_built": faiss_index is not None,
        "sparse_index_built": tfidf_mat is not None
    }


# --------------------------
# API: Index a PDF (upload)
# --------------------------
@app.post("/index")
async def index_pdf(file: UploadFile):
    """
    Upload a PDF, chunk it, and rebuild indexes (dense + sparse).
    NOTE: for simplicity this rebuilds from scratch every time you upload.
          For prod, you would append & update indexes incrementally.
    """
    # 1) Save uploaded PDF
    path = os.path.join(DATA_DIR, file.filename)
    with open(path, "wb") as f:
        f.write(await file.read())

    # 2) Chunk
    items = chunk_pdf(path)  # list of {"text","page","source"}

    # 3) Append to in-memory stores
    for it in items:
        TEXTS.append(it["text"])
        METAS.append({
            "page": it["page"],
            "source": it["source"],
            "id": hash_text(it["text"])
        })

    # 4) Rebuild indexes
    global faiss_index, tfidf, tfidf_mat
    faiss_index = build_dense_index(TEXTS)
    tfidf, tfidf_mat = build_sparse_index(TEXTS)

    # 5) Persist a minimal snapshot (optional)
    with open(os.path.join(INDEX_DIR, "meta.pkl"), "wb") as f:
        pickle.dump(METAS, f)
    faiss.write_index(faiss_index, os.path.join(INDEX_DIR, "faiss.index"))
    with open(os.path.join(INDEX_DIR, "tfidf.pkl"), "wb") as f:
        pickle.dump(tfidf, f)
    with open(os.path.join(INDEX_DIR, "tfidf_mat.pkl"), "wb") as f:
        pickle.dump(tfidf_mat, f)

    return {"chunks_added": len(items), "total_chunks": len(TEXTS)}


# --------------------------
# API: Query
# --------------------------
class Query(BaseModel):
    query: str
    session_id: str = "default"   
    max_answer_tokens: int | None = 3500  # used during the LLM call


@app.post("/query")
def query(q: Query):
    """
    Retrieval pipeline:
      A) Dense search (FAISS) -> top K/2
      B) Sparse search (TF-IDF) -> top K/2
      C) Merge pools (unique)
      D) MMR diversification (-> K_MMR)
      E) Cross-encoder re-rank (-> M_FINAL)
      F) Token-budget selection
      G) (Optional) Call LLM with prompt; include timings
    """
    if faiss_index is None or tfidf_mat is None:
        raise HTTPException(status_code=400, detail="Index not built. Upload a PDF to /index first.")

    t0 = time.perf_counter()

    # ---- A) Dense search pool ----
    # Embed query (normalized); search FAISS inner-product
    q_vec = emb_model.encode([q.query], normalize_embeddings=True)[0].astype(np.float32)
    D, I = faiss_index.search(np.array([q_vec]), k=min(max(1, K_FETCH // 2), len(TEXTS)))
    dense_ids = I[0].tolist()
    tA = time.perf_counter()

    # ---- B) Sparse search pool ----
    # TF-IDF transform on query, score by dot product (cosine-ish if rows L2-normalized)
    q_sparse = tfidf.transform([q.query])
    scores = (tfidf_mat @ q_sparse.T).toarray().ravel()
    sparse_ids = np.argsort(-scores)[:max(1, K_FETCH // 2)].tolist()
    tB = time.perf_counter()

    # ---- C) Merge pools ----
    # Keep order preference: dense first, then sparse; dedupe via dict-from-keys trick
    pool_ids = list(dict.fromkeys(dense_ids + sparse_ids))[:K_FETCH]
    pool_texts = [TEXTS[i] for i in pool_ids]
    pool_metas = [METAS[i] for i in pool_ids]

    # ---- D) MMR diversification ----
    # Re-embed pool for MMR (cosine on normalized vectors)
    cand_vecs = emb_model.encode(pool_texts, normalize_embeddings=True, show_progress_bar=False)
    mmr_idx = mmr_diversify(q_vec, cand_vecs, k=min(K_MMR, len(pool_ids)), lambda_mult=0.5)
    mmr_texts = [pool_texts[i] for i in mmr_idx]
    mmr_metas = [pool_metas[i] for i in mmr_idx]
    tC = time.perf_counter()

    # ---- E) Cross-encoder re-rank -> choose M_FINAL ----
    pairs = [(q.query, t) for t in mmr_texts]
    rr_scores = reranker.predict(pairs)  # higher = more relevant
    top_idx = sorted(range(len(rr_scores)), key=lambda i: rr_scores[i], reverse=True)[:min(M_FINAL, len(mmr_texts))]
    docs = [{"text": mmr_texts[i], **mmr_metas[i]} for i in top_idx]
    tD = time.perf_counter()

    # ---- F) Token-budget selection ----
   
    docs_budgeted = select_with_token_budget(docs, TOKEN_BUDGET)
    tE = time.perf_counter()
    context = "\n\n---\n\n".join([f"[p{d['page']}] {d['text']}" for d in docs_budgeted])
    ...
    prompt = (
        "Use only the context to answer. Cite pages like [pX]. If unsure, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {q.query}"
    )
    ...
    # ---- F) Budget-aware packing (history + RAG) + LLM call ----
    rag_texts = [f"[p{d['page']}] {d['text']}" for d in docs]  # use re-ranked docs
    system_prompt = "You are a precise crypto research assistant. Cite sources with [pX]. Be concise."

    # Build history text from prior Q/A pairs for this session
    hist_pairs = list(SESSIONS[q.session_id])
    history_text = "\n".join(f"Q: {qq}\nA: {aa}" for qq, aa in hist_pairs)


    messages, budgets, kept_idx, prompt_tokens = build_messages(
        question=q.query,
        system=system_prompt,
        history=history_text,
        rag_chunks=rag_texts,
    )

    # Filter retrieved list to only the chunks actually packed (kept_idx)
    used_docs = [docs[i] for i in kept_idx]

    answer_t0 = time.perf_counter()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set.")
    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
        messages=messages,
        temperature=0.2,
        max_completion_tokens=min(budgets["output"], q.max_answer_tokens or budgets["output"]),
        stream=False,
    )
    answer = completion.choices[0].message.content
    answer_t1 = time.perf_counter()
    # Save this Q/A pair to session history
    SESSIONS[q.session_id].append((q.query, answer))



    # ---- Timings summary ----
    timings = {
        "dense_search_ms": round((tA - t0) * 1000, 1),
        "sparse_search_ms": round((tB - tA) * 1000, 1),
        "mmr_ms": round((tC - tB) * 1000, 1),
        "rerank_ms": round((tD - tC) * 1000, 1),
        "token_budget_ms": round((tE - tD) * 1000, 1),
        "total_retrieval_ms": round((tE - t0) * 1000, 1),
        "k_fetch": K_FETCH,
        "k_mmr": K_MMR,
        "m_final": M_FINAL,
        "context_tokens_est": sum(estimate_tokens(d["text"]) for d in docs_budgeted),
        "returned_chunks": len(docs_budgeted),
    }

    
    timings["llm_ms"] = round((answer_t1 - answer_t0) * 1000, 1)

    # Pretty debugging info
    retrieved = [
        {"page": d["page"], "source": d["source"], "preview": d["text"][:300].replace("\n", " ")}
        for d in docs_budgeted
    ]
    
    return {
        "answer": answer,
        "timings": timings,
        "retrieved_chunks": retrieved
    }


@app.post("/reset/{session_id}")
def reset_session(session_id: str):
    SESSIONS.pop(session_id, None)
    return {"ok": True}
