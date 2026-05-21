import os
import threading
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# --- PyTorch Memory Optimization for 512MB Render ---
# Force PyTorch and CPU math libraries to use exactly 1 thread per process
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import torch
# Disable memory-hungry gradient engines since we only do inference
torch.set_grad_enabled(False)
torch.set_num_threads(1)
# ----------------------------------------------------
# Authenticate with Hugging Face Hub if a token is provided.
# This suppresses the "unauthenticated requests" warning and raises rate limits.
_hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if _hf_token:
    try:
        from huggingface_hub import login as _hf_login
        _hf_login(token=_hf_token, add_to_git_credential=False)
    except Exception:
        pass  # non-fatal — model still loads anonymously

# Minimum cosine similarity between a user's profile embedding and a paper
# embedding for that user's expertise to count toward the paper's credibility.
# all-MiniLM-L6-v2: 0.5 ≈ meaningful domain overlap; tune down to 0.4 if too strict.
CREDIBILITY_SIMILARITY_THRESHOLD = 0.25

_model = None
_model_lock = threading.Lock()

def get_model():
    """Lazy loader for the SentenceTransformer model. Thread-safe."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:  # double-checked locking
            print("⏳ Loading SentenceTransformer model ('all-MiniLM-L6-v2')...")
            _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def is_model_ready() -> bool:
    """Non-blocking check — True only if model is already in memory."""
    return _model is not None


def paper_text(title: str, abstract: str) -> str:
    """Combines title and abstract for embedding."""
    if not abstract:
        return f"Title: {title}"
    return f"Title: {title}\nAbstract: {abstract}".strip()

def embed_text(text: str) -> np.ndarray:
    """Generates embedding for a given text."""
    return get_model().encode(text, convert_to_numpy=True)

def embed_paper(title: str, abstract: str) -> np.ndarray:
    """Generates a normalized embedding for a paper."""
    v = embed_text(paper_text(title, abstract))
    return normalize(v)

def normalize(v: np.ndarray) -> np.ndarray:
    """L2 normalizes a vector."""
    n = np.linalg.norm(v)
    return v if n == 0 else (v / n)

def build_user_profile_vec(
    user_papers_data: list[tuple[str, str]],
    keyword_weights: dict[str, float],
    alpha: float = 0.7,
) -> np.ndarray:
    """
    alpha near 1.0 = mostly based on user's papers
    alpha near 0.0 = mostly based on keywords
    """
    # 1) Paper component (mean of user paper embeddings)
    if user_papers_data:
        texts = [paper_text(t, a) for (t, a) in user_papers_data]
        vecs = get_model().encode(texts, convert_to_numpy=True)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        paper_vecs = vecs / norms
        paper_profile = np.mean(paper_vecs, axis=0)
    else:
        paper_profile = None

    # 2) Keyword component (weighted sum of keyword embeddings)
    if keyword_weights:
        kw_texts = [f"keyword: {k}" for k in keyword_weights.keys()]
        kw_vecs = get_model().encode(kw_texts, convert_to_numpy=True)
        weights = np.array([keyword_weights[k] for k in keyword_weights.keys()], dtype=float)

        # Weighted average of keyword vectors
        kw_profile = (kw_vecs * weights[:, None]).sum(axis=0) / max(weights.sum(), 1e-9)
    else:
        kw_profile = None

    # 3) Combine
    if paper_profile is None and kw_profile is None:
        return None

    if paper_profile is None:
        u = kw_profile
    elif kw_profile is None:
        u = paper_profile
    else:
        u = alpha * paper_profile + (1 - alpha) * kw_profile

    return normalize(u)

def recommend_topk(
    user_vec: np.ndarray,
    candidate_vecs: np.ndarray,   # shape: (N, D)
    candidate_ids: list[int],
    k: int = 5
):
    """
    Fast vectorized cosine similarity.
    Everything is assumed to be normalized, so similarity is dot product.
    """
    if candidate_vecs.size == 0:
        return []

    # Cosine sim is dot product for normalized vectors
    sims = candidate_vecs @ user_vec  # shape: (N,)
    
    # Get top K indices
    num_to_get = min(k, len(sims))
    top_idx = np.argpartition(-sims, kth=num_to_get-1)[:num_to_get]
    # Sort them by score
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    
    return [(candidate_ids[i], float(sims[i])) for i in top_idx]

def serialize_embedding(v: np.ndarray) -> bytes:
    """Serializes a numpy array to bytes for DB storage."""
    return v.tobytes()

def deserialize_embedding(b: bytes) -> np.ndarray:
    """Deserializes bytes back to a numpy array."""
    # all-MiniLM-L6-v2 produces float32 vectors of dimension 384
    return np.frombuffer(b, dtype=np.float32)
