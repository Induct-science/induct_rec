import os
import threading
import datetime
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

# The embedding model, driven by env so it can be swapped without a code
# change (the induct_be app sets EMBED_MODEL / EMBEDDING_DIM as the single
# source of truth — see app/utils/embedding_config.py). Changing the model
# requires re-embedding all papers + users and re-tuning the thresholds below.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")

# Minimum cosine similarity between a user's profile embedding and a paper
# embedding for that user's expertise to count toward the paper's credibility.
# NOTE: specific to EMBED_MODEL's cosine distribution — re-tune if you change it.
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
            print(f"⏳ Loading SentenceTransformer model ('{EMBED_MODEL}')...")
            _model = SentenceTransformer(EMBED_MODEL)
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

#: Half-life, in years, for weighting a researcher's own papers when building
#: their profile vector. A paper this many years old counts half as much as one
#: from the current year.
#:
#: ``None`` means an unweighted mean — every paper the researcher has ever
#: written pulls equally, which is the behaviour this module shipped with. That
#: is wrong for anyone whose interests moved: a physical oceanographer who spent
#: three years on sea ice a decade ago is still recommended sea ice at full
#: strength today.
#:
#: Left at ``None`` deliberately. Turning it on changes every feed, so it is a
#: decision to make after looking at real feeds — ``tools/rec_variants.py
#: --recency-halflife N`` in induct_be prints them. 4-5 years is the range worth
#: starting from: long enough that a productive researcher's current programme
#: dominates, short enough that one quiet year does not erase them.
DEFAULT_RECENCY_HALFLIFE_YEARS = None


def build_user_profile_vec(
    user_papers_data: list[tuple],
    keyword_weights: dict[str, float],
    alpha: float = 0.7,
    recency_halflife_years: float | None = DEFAULT_RECENCY_HALFLIFE_YEARS,
    current_year: int | None = None,
) -> np.ndarray:
    """
    alpha near 1.0 = mostly based on user's papers
    alpha near 0.0 = mostly based on keywords

    `user_papers_data` holds (title, abstract) or (title, abstract, year). The
    two forms may be mixed; a 2-tuple simply carries no year.

    `recency_halflife_years` weights each paper by ``0.5 ** (age / halflife)``
    so the profile tracks what someone works on *now* rather than everything
    they have ever published. ``None`` keeps the unweighted mean.

    A paper with no usable year gets weight 1.0 — treated as current rather
    than as maximally old. That matches `recommend_topk`, which sets
    ``age_factor = 1.0`` for unknown years on the same reasoning: an unknown
    date is an absence of evidence, not evidence of age. It does mean a corpus
    with many missing years dilutes the effect rather than skewing it.
    """
    # 1) Paper component (recency-weighted mean of user paper embeddings)
    if user_papers_data:
        texts = [paper_text(row[0], row[1]) for row in user_papers_data]
        vecs = get_model().encode(texts, convert_to_numpy=True)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        paper_vecs = vecs / norms

        if recency_halflife_years and recency_halflife_years > 0:
            year_now = current_year if current_year is not None else datetime.date.today().year
            years = [row[2] if len(row) > 2 else None for row in user_papers_data]
            w = np.array([
                1.0 if (y is None or y <= 0)
                else 0.5 ** (max(year_now - int(y), 0) / float(recency_halflife_years))
                for y in years
            ], dtype=float)
            total = w.sum()
            paper_profile = ((paper_vecs * w[:, None]).sum(axis=0) / total
                             if total > 0 else np.mean(paper_vecs, axis=0))
        else:
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
    k: int = 5,
    candidate_years: list[int] = None,
    candidate_credibilities: list[float] = None,
    credibility_weight: float = 0.15,
    candidate_saliences: list[float] = None,
    age_penalty_weight: float = 2.0,
    current_year: int = None,
    pool_size: int = 250
):
    """
    Fast vectorized cosine similarity with optional credibility re-ranking.
    Everything is assumed to be normalized, so similarity is dot product.
    `candidate_years` feeds the age penalty; results are returned in score order.
    """
    if candidate_vecs.size == 0:
        return []

    # Cosine sim is dot product for normalized vectors
    sims = candidate_vecs @ user_vec  # shape: (N,)

    # --- Age penalty (applied OUTSIDE the 15-paper tiers) -------------------
    #
    # Deliberately placed before the tiering, so it changes which tier a paper
    # lands in rather than only its position inside one. That makes it a
    # stronger lever than the credibility multiplier below, which can never
    # move a paper more than 14 positions. A recent paper can therefore
    # outrank a more topically relevant older one — that is the intent.
    #
    # `salience / age` is the age-adjusted attention a paper has attracted.
    # Measured on production 2026-09-20 (tools/README_paper_age_analysis.md):
    # raw citation counts span 40x across eras (1970s mean 127.6, 2020s 23.9)
    # purely because older papers have had longer to accrue them, and salience
    # is 3.0 + 0.1*citations. Dividing by age collapses that and reverses the
    # direction, favouring current work.
    #
    # CRITICAL: this rewrites the ranking score only. Paper.salience itself is
    # never modified, because author keyword scores -- and therefore user
    # credibility -- are computed from the raw stored value
    # (author_identification.calculate_merged_keywords). An age-adjusted
    # salience must never reach that path.
    #
    # log1p keeps the multiplier bounded and matches the credibility idiom:
    # a paper with no salience is not zeroed out, it simply gets no boost.
    if candidate_saliences is not None and candidate_years is not None:
        year_now = current_year if current_year is not None else datetime.date.today().year
        years_arr = np.asarray(candidate_years, dtype=float)
        sal_arr = np.maximum(np.asarray(candidate_saliences, dtype=float), 0.0)
    
        # Floor the age at 1 year: a paper published this year would otherwise
        # divide by zero, and a part-year rate extrapolates from almost nothing.
        ages = np.maximum(year_now - years_arr, 1.0)
        recency = sal_arr / ages
        age_factor = 1.0 + age_penalty_weight * np.log1p(recency)
    
        # A missing or nonsense year would otherwise read as maximally ancient
        # and bury the paper. Unknown age means no opinion, not a penalty.
        age_factor = np.where(years_arr > 0, age_factor, 1.0)
        sims = sims * age_factor

    
    if candidate_credibilities is not None:
        creds = np.array(candidate_credibilities)
        # Stage 1: Filter out negative similarity vectors (sims >= 0.0)
        relevance_mask = sims >= 0.0
        if not np.any(relevance_mask):
            relevance_mask = np.ones_like(sims, dtype=bool)

        valid_indices = np.where(relevance_mask)[0]
        valid_sims = sims[valid_indices]
        valid_creds = creds[valid_indices]

        sorted_order = np.argsort(-valid_sims)
        sorted_indices = valid_indices[sorted_order]

        # Stage 2: Process in 15-paper tiers (chunks) to enforce tight domain relevance boundaries
        tier_size = 15
        final_ordered_indices = []
        final_scores_list = []

        for i in range(0, len(sorted_indices), tier_size):
            tier_idx = sorted_indices[i:i+tier_size]
            tier_sims = sims[tier_idx]
            tier_creds = creds[tier_idx]

            # Bounded log-credibility multiplier within this 15-paper tier
            tier_sims_pos = np.maximum(tier_sims, 0)
            # `credibility_weight` was declared and documented but never read —
            # the coefficient was hardcoded at 0.15, so the parameter silently did
            # nothing. Default lowered from 0.2 to 0.15 to match what has actually
            # been running; wiring it up must not change anyone's feed.
            tier_scores = tier_sims_pos * (
                1.0 + credibility_weight * np.log1p(np.maximum(tier_creds, 0))
            )

            # Sort within tier descending by composite score
            sub_order = np.argsort(-tier_scores)
            final_ordered_indices.extend(tier_idx[sub_order])
            final_scores_list.extend(tier_scores[sub_order])

        top_idx = np.array(final_ordered_indices[:k])
        returned_scores = np.array(final_scores_list[:k])
    else:
        # Standard behavior without credibility
        num_to_get = min(k, len(sims))
        top_idx = np.argpartition(-sims, kth=num_to_get-1)[:num_to_get]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        returned_scores = sims[top_idx]
    
    # Results come back in score order, always.
    #
    # There used to be a hard newest-first re-sort of the top K here whenever
    # candidate_years was supplied. It was dropped for two reasons. It threw
    # away the score ordering wholesale inside the returned page, and it could
    # not survive the precompute path at all — that stores only (paper_id,
    # score) and the route re-reads ORDER BY score DESC, so any ordering not
    # encoded in the score evaporated. Recency is now carried by the age
    # penalty above, which *is* in the score, so both paths agree.
    return [(candidate_ids[i], float(returned_scores[idx])) for idx, i in enumerate(top_idx)]

def serialize_embedding(v: np.ndarray) -> bytes:
    """Serializes a numpy array to bytes for DB storage."""
    return v.tobytes()

def deserialize_embedding(b: bytes) -> np.ndarray:
    """Deserializes bytes back to a numpy array."""
    # all-MiniLM-L6-v2 produces float32 vectors of dimension 384
    return np.frombuffer(b, dtype=np.float32)


def mmr_select(scores, vecs, k, lam=0.7):
    """
    Pick `k` items by Maximal Marginal Relevance.

    Greedy: take the highest-scoring item, then repeatedly take whichever
    remaining item maximises

        lam * score - (1 - lam) * max_similarity_to_already_picked

    The second term is the whole point. Once a paper on mesoscale eddies is
    chosen, every other mesoscale-eddy paper is penalised in proportion to how
    much it resembles it, so a feed of eight near-duplicates from one group
    becomes one of them plus seven other things worth seeing. `lam = 1.0` is
    pure relevance, i.e. no diversification.

    Cost is O(k * n) dot products on an already-shortlisted pool, so it is
    negligible next to retrieval.

    Returns indices into `scores`, in selection order.
    """
    n = len(scores)
    if n == 0 or k <= 0:
        return []
    k = min(k, n)
    if lam >= 1.0:
        return list(np.argsort(-np.asarray(scores))[:k])

    scores = np.asarray(scores, dtype=float)
    # Diversity is measured on the similarity scale, so put relevance there too;
    # otherwise lam is comparing a score of ~8 against a cosine of ~0.8 and the
    # penalty never bites.
    span = float(scores.max() - scores.min())
    rel = (scores - scores.min()) / span if span > 0 else np.zeros_like(scores)

    chosen = [int(np.argmax(scores))]
    best_sim = vecs @ vecs[chosen[0]]
    for _ in range(k - 1):
        marginal = lam * rel - (1.0 - lam) * best_sim
        marginal[chosen] = -np.inf
        nxt = int(np.argmax(marginal))
        if not np.isfinite(marginal[nxt]):
            break
        chosen.append(nxt)
        best_sim = np.maximum(best_sim, vecs @ vecs[nxt])
    return chosen


#: How many of an author's works a reader has to have cited before that author
#: counts for as much as they ever will. Past this the strength is 1.0.
#:
#: The curve is logarithmic on purpose. The step from "never cited" to "cited
#: once" is the informative one — it says the reader knows this person's work
#: exists and found it worth building on. The step from the twentieth citation
#: to the twenty-first says nothing new, and a linear scale would hand the
#: whole effect to a reader's two or three closest collaborators.
CITED_AUTHOR_SATURATION = 10


def cited_author_strength(works_cited, saturation: int = CITED_AUTHOR_SATURATION) -> float:
    """
    How strongly a reader has drawn on one author, in [0, 1].

    `works_cited` is the number of distinct works by that author the reader's
    own papers cite. 0 gives 0.0; one gives about 0.29; `saturation` or more
    gives 1.0.
    """
    try:
        n = float(works_cited or 0)
    except (TypeError, ValueError):
        return 0.0
    if n <= 0:
        return 0.0
    return float(min(1.0, np.log1p(n) / np.log1p(max(saturation, 1))))


def cited_author_factor(strength, weight: float):
    """
    The multiplier for a strength (scalar or array): 1 + weight * strength.

    One definition, exported, because the host has to be able to apply the
    *same* factor outside the pipeline. Induct stores a reader's ranked feed
    and serves it for days; a bias that lived only inside `rank_candidates`
    would reach nobody until the next recompute. So the host multiplies the
    stored score by this at serve time — and divides it back out of what it
    stores, so the factor is never counted twice.

    Strengths are clipped to [0, 1] and anything unreadable is treated as 0,
    so a bad value can cost a paper its boost and never more than that.
    """
    s = np.nan_to_num(np.asarray(strength, dtype=float), nan=0.0,
                      posinf=1.0, neginf=0.0)
    s = np.clip(s, 0.0, 1.0)
    out = 1.0 + max(float(weight or 0.0), 0.0) * s
    return float(out) if out.ndim == 0 else out


# --------------------------------------------------------------------------- #
# A reader's interests                                                        #
# --------------------------------------------------------------------------- #
#
# A reader used to be one vector: the mean of everything they have written.
# That is a fair summary of somebody who works on one thing. For somebody who
# works on two it is a point *between* them, and the papers nearest that point
# are the ones that resemble both fields a little and neither closely — the
# broad surveys and the methods papers everybody cites. Their best matches in
# either field sit further away than those do.
#
# So a reader may also carry a few **interests**: groups of their own papers
# that belong together, each with a centre of its own and a share of the
# reader's work. Relevance is then the nearest of those centres (or the whole,
# whichever is closer — see `reader_relevance`), and positions in the feed are
# dealt to interests in proportion to their shares (`select_with_quota`).
#
# The whole-work vector is kept, and nothing below replaces it. A reader whose
# papers form one group has one interest, the host passes none, and their feed
# is computed exactly as it was.

#: The most interests a reader is given. Three, not ten: every interest takes
#: positions from the others, and a feed page is ten papers.
MAX_INTERESTS = 3

#: Two groups of a reader's papers are one interest when the mean cosine
#: between a paper in one and a paper in the other is at least this.
INTEREST_MERGE_SIMILARITY = 0.30

#: A group has to be at least this much of a reader's work to stand as an
#: interest of its own; a smaller one is folded into whichever it is nearest.
INTEREST_MIN_SHARE = 0.15

#: ...and at least this many papers. One paper is not an interest.
INTEREST_MIN_PAPERS = 3

#: Grouping compares every pair, so it is given at most this many papers,
#: taken evenly across the list; the rest join the group they are nearest to.
INTEREST_MAX_CLUSTER_INPUT = 400


def cluster_interests(
    vecs,
    weights=None,
    *,
    max_interests: int = MAX_INTERESTS,
    merge_similarity: float = INTEREST_MERGE_SIMILARITY,
    min_share: float = INTEREST_MIN_SHARE,
    min_papers: int = INTEREST_MIN_PAPERS,
):
    """
    Sort a reader's papers into interests. Returns one label per row of
    `vecs`: 0 for the largest interest by share, 1 for the next, and so on.

    The method is average-linkage agglomeration, chosen because it has no
    random start and no number of groups to guess. Every paper begins alone;
    the two closest groups are joined, repeatedly, until no two groups are as
    close as `merge_similarity`. "Close" is the mean cosine between a paper in
    one group and a paper in the other, weighted by `weights` when given.

    Then two tidying passes, in this order:

      * a group below `min_share` of the total weight, or with fewer than
        `min_papers` papers, is joined to the group it is closest to — so one
        odd paper from 2009 never becomes a third of somebody's feed;
      * while more than `max_interests` remain, the closest two are joined.

    Both passes join groups; neither discards a paper. Every paper the reader
    wrote still counts toward exactly one interest, which is what makes "one
    interest" the same thing as the single mean vector.

    Deterministic: the same papers in the same order give the same labels.
    """
    V = np.asarray(vecs, dtype=float)
    if V.ndim != 2 or V.shape[0] == 0:
        return np.zeros(0, dtype=int)
    n = V.shape[0]
    if n == 1 or max_interests <= 1:
        return np.zeros(n, dtype=int)

    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    V = V / norms

    if weights is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.nan_to_num(np.asarray(weights, dtype=float), nan=0.0,
                          posinf=0.0, neginf=0.0)
        # A weight of zero would make a group of such papers weigh nothing and
        # its mean undefined. Tiny rather than zero: it still cannot matter.
        w = np.maximum(w, 1e-9)

    if n > INTEREST_MAX_CLUSTER_INPUT:
        pick = np.unique(np.linspace(0, n - 1, INTEREST_MAX_CLUSTER_INPUT)
                         .round().astype(int))
    else:
        pick = np.arange(n)

    c = len(pick)
    # A group is held as the weighted sum of its vectors and its total weight.
    # Because every vector is unit length, sum_a . sum_b / (W_a W_b) *is* the
    # weighted mean cosine between the two groups — no pair list needed.
    sums = V[pick] * w[pick][:, None]
    W = w[pick].copy()
    counts = np.ones(c, dtype=int)
    alive = np.ones(c, dtype=bool)
    group = np.arange(c)

    S = (sums @ sums.T) / np.outer(W, W)
    np.fill_diagonal(S, -np.inf)

    def join(a, b):
        a, b = (a, b) if a < b else (b, a)
        sums[a] += sums[b]
        W[a] += W[b]
        counts[a] += counts[b]
        alive[b] = False
        group[group == b] = a
        row = (sums @ sums[a]) / (W * W[a])
        row[~alive] = -np.inf
        row[a] = -np.inf
        S[a, :] = row
        S[:, a] = row
        S[b, :] = -np.inf
        S[:, b] = -np.inf

    def closest_pair():
        flat = int(np.argmax(S))
        return divmod(flat, c)

    # 1. Join while any two groups are close enough to be one interest.
    while alive.sum() > 1:
        a, b = closest_pair()
        if S[a, b] < merge_similarity:
            break
        join(a, b)

    # 2. Fold in whatever is too small to stand alone, smallest first.
    while alive.sum() > 1:
        total = W[alive].sum()
        small = [g for g in np.flatnonzero(alive)
                 if W[g] / total < min_share or counts[g] < min_papers]
        if not small:
            break
        g = min(small, key=lambda j: (W[j], j))
        join(g, int(np.argmax(S[g])))

    # 3. Hold the ceiling.
    while alive.sum() > max(int(max_interests), 1):
        join(*closest_pair())

    labels = np.empty(n, dtype=int)
    labels[pick] = group
    if c < n:
        rest = np.setdiff1d(np.arange(n), pick)
        live = np.flatnonzero(alive)
        centres = sums[live] / np.linalg.norm(sums[live], axis=1, keepdims=True)
        labels[rest] = live[np.argmax(V[rest] @ centres.T, axis=1)]

    # Largest share first; a tie goes to the group whose first paper is earliest.
    ids = np.unique(labels)
    share = {g: w[labels == g].sum() for g in ids}
    first = {g: int(np.flatnonzero(labels == g)[0]) for g in ids}
    order = sorted(ids, key=lambda g: (-share[g], first[g]))
    rank = {g: r for r, g in enumerate(order)}
    return np.array([rank[g] for g in labels], dtype=int)


def interest_centres(vecs, labels, weights=None):
    """
    The centre and the share of each interest: `(centres, shares)`.

    `centres[j]` is the unit-length weighted mean of the papers labelled `j`;
    `shares[j]` is that interest's fraction of the total weight, and the
    shares sum to 1. With no weights a share is simply the fraction of the
    reader's papers that fall in the interest.
    """
    V = np.asarray(vecs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if V.ndim != 2 or V.shape[0] == 0:
        return np.zeros((0, 0)), np.zeros(0)
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    V = V / norms
    w = (np.ones(len(V)) if weights is None
         else np.maximum(np.nan_to_num(np.asarray(weights, dtype=float)), 1e-9))
    m = int(labels.max()) + 1
    centres = np.zeros((m, V.shape[1]))
    shares = np.zeros(m)
    for j in range(m):
        mask = labels == j
        centres[j] = normalize((V[mask] * w[mask][:, None]).sum(axis=0))
        shares[j] = w[mask].sum()
    return centres, shares / shares.sum()


def build_user_interests(
    user_papers_data: list[tuple],
    recency_halflife_years: float | None = DEFAULT_RECENCY_HALFLIFE_YEARS,
    current_year: int | None = None,
    *,
    paper_vecs=None,
    max_interests: int = MAX_INTERESTS,
    merge_similarity: float = INTEREST_MERGE_SIMILARITY,
    min_share: float = INTEREST_MIN_SHARE,
    min_papers: int = INTEREST_MIN_PAPERS,
) -> list[dict]:
    """
    A reader's interests, largest first. Each is

        {"vec": unit vector, "share": float, "papers": [row indices]}

    `user_papers_data` is what `build_user_profile_vec` takes — (title,
    abstract) or (title, abstract, year) — and `recency_halflife_years` means
    what it means there: a paper is weighted by ``0.5 ** (age / halflife)``,
    which here moves both an interest's centre and its share. A field somebody
    left ten years ago should not keep its historical share of their feed.
    ``None`` weights every paper equally.

    `paper_vecs` lets a host that already stores an embedding per paper pass
    them in, one row per entry of `user_papers_data`, instead of having the
    titles and abstracts encoded again. With it, this needs no model at all.

    **An interest is made of papers only.** `build_user_profile_vec` blends a
    reader's keywords into their one vector; that was tried here and dropped.
    A reader's heaviest keywords are the broad ones — "physics", "geology",
    "environmental science" — and mixing them into each interest pulled every
    interest back toward the middle of the reader's discipline, which is the
    thing interests exist to get away from. Measured on a three-field
    researcher, the keyword blend cut the corpus papers within reach of their
    smallest field from 392 to 171.

    Returns an empty list for a reader with no papers.
    """
    if not user_papers_data:
        return []

    if paper_vecs is None:
        texts = [paper_text(row[0], row[1]) for row in user_papers_data]
        vecs = get_model().encode(texts, convert_to_numpy=True)
    else:
        vecs = np.asarray(paper_vecs, dtype=float)
        if vecs.ndim != 2 or len(vecs) != len(user_papers_data):
            raise ValueError("paper_vecs must have one row per paper")

    weights = None
    if recency_halflife_years and recency_halflife_years > 0:
        year_now = current_year if current_year is not None else datetime.date.today().year
        years = [row[2] if len(row) > 2 else None for row in user_papers_data]
        weights = np.array([
            1.0 if (y is None or y <= 0)
            else 0.5 ** (max(year_now - int(y), 0) / float(recency_halflife_years))
            for y in years
        ], dtype=float)

    labels = cluster_interests(vecs, weights, max_interests=max_interests,
                               merge_similarity=merge_similarity,
                               min_share=min_share, min_papers=min_papers)
    centres, shares = interest_centres(vecs, labels, weights)
    return [{"vec": centres[j],
             "share": float(shares[j]),
             "papers": [int(i) for i in np.flatnonzero(labels == j)]}
            for j in range(len(shares))]


def reader_relevance(user_vec, candidate_vecs, interest_vecs=None):
    """
    How relevant each candidate is to a reader: `(relevance, nearest)`.

    With no interests this is the cosine to the reader's profile vector and
    `nearest` is None — what the pipeline has always computed.

    With interests, relevance is the **largest** of the cosine to the profile
    vector and the cosine to each interest, and `nearest[i]` says which
    interest candidate `i` is closest to.

    The profile vector stays in the maximum on purpose. It means relevance can
    only rise when a reader is given interests: no paper that qualified for
    their feed before stops qualifying, and a paper that sits between two of
    their fields — which the whole-work vector is closest to, and which is
    exactly what such a reader should see — keeps the relevance it had.

    Exported because a host has places outside `rank_candidates` that ask "is
    this paper relevant to this reader" (Induct checks its own papers against
    the similarity gate at serve time), and they must not use a different
    definition from the ranking.
    """
    # The candidate matrix keeps the precision it came in. A host holds every
    # paper in one float32 matrix, and forcing it to float64 here would copy
    # the whole corpus on every call.
    V = np.asarray(candidate_vecs)
    if V.dtype.kind != "f":
        V = V.astype(float)
    if V.ndim == 1:
        V = V.reshape(1, -1)
    rel = V @ np.asarray(user_vec, dtype=V.dtype)
    if interest_vecs is None or len(interest_vecs) == 0:
        return rel, None
    U = np.asarray(interest_vecs, dtype=V.dtype)
    if U.ndim == 1:
        U = U.reshape(1, -1)
    per = V @ U.T
    return np.maximum(rel, per.max(axis=1)), per.argmax(axis=1)


def rank_candidates(
    user_vec,
    candidate_vecs,
    candidate_ids,
    *,
    k: int = 10,
    candidate_years=None,
    candidate_credibilities=None,
    candidate_saliences=None,
    candidate_is_induct=None,
    min_similarity: float | None = None,
    scoring: str = "salience_age",
    age_penalty_weight: float = 2.0,
    recency_weight: float = 2.0,
    credibility_weight: float = 0.15,
    induct_boost: float = 1.0,
    mmr_lambda: float = 1.0,
    induct_low_credibility_max: float | None = None,
    slot_pattern=None,
    rerank_fn=None,
    rerank_top_n: int = 100,
    candidate_cited_author_strength=None,
    cited_author_weight: float = 0.0,
    interest_vecs=None,
    interest_shares=None,
):
    """
    The full recommendation pipeline, in one place, with explicit stages.

    Exists because `recommend_topk` fuses retrieval, scoring and tiering into
    one pass, which left no seam for a similarity gate, for diversification, or
    for a future cross-encoder. It also made a real failure hard to see: on
    2026-10-08 the ERA-Interim reanalysis paper (cosine 0.530 to the reader)
    topped an abyssal-turbulence researcher's feed, beating Large et al. 1994 at
    cosine 0.775, because salience/age of 176 citations a year outweighed a 46%
    deficit in topical fit. Nothing was wrong with the weights; the candidate
    pool was the entire corpus, so a paper that resembles all of climate science
    was allowed to compete.

    Stages, in this order, and the order matters:

      1. similarity      cosine against the reader's profile vector — or, for
                         a reader with interests, against the nearest of them
      2. gate            drop everything below `min_similarity` — the pool gate
      3. rerank (opt)    `rerank_fn` may replace relevance for the top
                         `rerank_top_n`; this is the cross-encoder seam
      4. score           relevance x recency x credibility x Induct boost
      5. diversify       MMR picks the final `k`

    **Scoring must precede MMR, not follow it.** Whatever runs last decides the
    order, so ranking by credibility after diversifying would re-sort the list
    and undo the spacing MMR just created.

    `scoring` selects the recency term:
      "salience_age" — (1 + w*log1p(salience/age)), the existing behaviour.
         Self-reinforcing: recommendation drives reads, reads drive salience.
      "recency"      — (1 + w*log1p(1/age)), no salience at all. Prefers a very
         relevant paper from this year over a famous one from 2011.
      "none"         — no recency term.

    `candidate_cited_author_strength` is the one term that is about *this
    reader* rather than about the paper: for each candidate, how strongly the
    reader has already drawn on the people who wrote it, in [0, 1], where 0 is
    "has never cited any of its authors". It multiplies the score by
    (1 + cited_author_weight * strength). See `cited_author_factor`.

    It is a factor in stage 4 and nowhere else, which is what keeps it honest:
    it is applied after the gate, so having cited somebody cannot bring one of
    their off-topic papers into the feed — it reorders papers that already
    qualified on relevance. And it is bounded, so the most-cited collaborator
    in a reader's record is worth at most (1 + weight) and cannot bury
    everybody else. With no strengths given, or a weight of 0, the pipeline
    is exactly what it was.

    `interest_vecs` is for a reader whose work falls into more than one
    field: one unit vector per interest, with `interest_shares` saying how much
    of the reader's work each one is (equal when omitted). See
    `build_user_interests`. It changes two stages and no formula:

      * stage 1 — relevance is the cosine to the nearest interest, or to
        `user_vec` when that is nearer (`reader_relevance`). It can only rise,
        so nothing that passed the gate without interests fails it with them.
      * stage 5 — positions are dealt to interests in proportion to their
        shares. A reader who is two-thirds one field and a third another gets
        about two papers in three from the first, on every page, rather than
        whichever field happens to have the better-cited literature.

    The score formula in stage 4 is untouched, and with no interests given the
    pipeline is exactly what it was.

    Returns [(paper_id, score)] with **strictly decreasing** scores. That is not
    cosmetic: the precompute path stores only (paper_id, score) and the serving
    route re-reads ORDER BY score DESC, so any ordering not encoded in the score
    is lost. An earlier newest-first re-sort was dropped for exactly that
    reason, and MMR's selection order would evaporate the same way.
    """
    import datetime as _dt

    user_vec = np.asarray(user_vec, dtype=float)
    V = np.asarray(candidate_vecs, dtype=float)
    if V.size == 0:
        return []

    n = V.shape[0]
    sims, nearest = reader_relevance(user_vec, V, interest_vecs)
    idx = np.arange(n)

    # --- 2. the pool gate -------------------------------------------------
    if min_similarity is not None:
        keep = sims >= min_similarity
        # Never hand a reader an empty feed because their profile is unusual.
        # Falling back to the nearest candidates keeps the gate a preference
        # rather than a cliff.
        if keep.sum() >= k:
            idx = idx[keep]
        else:
            idx = np.argsort(-sims)[: max(k, min(n, k))]

    rel = sims[idx]

    # --- 3. the cross-encoder seam ---------------------------------------
    # A cross-encoder scores (reader, paper) pairs jointly and cannot be
    # precomputed, so it is affordable in the nightly job and not in a request.
    # It plugs in here, over the shortlist only, and must return one relevance
    # value per id it was given.
    if rerank_fn is not None and len(idx) > 0:
        order = np.argsort(-rel)[:rerank_top_n]
        subset_ids = [candidate_ids[i] for i in idx[order]]
        new_rel = rerank_fn(subset_ids, rel[order])
        if new_rel is not None:
            rel = rel.copy()
            rel[order] = np.asarray(new_rel, dtype=float)

    # --- 4. scoring -------------------------------------------------------
    year_now = _dt.date.today().year
    factor = np.ones_like(rel)

    if scoring in ("salience_age", "recency") and candidate_years is not None:
        years = np.asarray([candidate_years[i] or 0 for i in idx], dtype=float)
        ages = np.maximum(year_now - years, 1.0)
        if scoring == "salience_age" and candidate_saliences is not None:
            sal = np.maximum(np.asarray([candidate_saliences[i] or 0.0 for i in idx],
                                        dtype=float), 0.0)
            f = 1.0 + age_penalty_weight * np.log1p(sal / ages)
        else:
            f = 1.0 + recency_weight * np.log1p(1.0 / ages)
        # Unknown year means no opinion, not maximal age.
        factor = factor * np.where(years > 0, f, 1.0)

    if candidate_credibilities is not None:
        creds = np.maximum(np.asarray([candidate_credibilities[i] or 0.0 for i in idx],
                                      dtype=float), 0.0)
        factor = factor * (1.0 + credibility_weight * np.log1p(creds))

    if candidate_is_induct is not None and induct_boost != 1.0:
        is_ind = np.asarray([bool(candidate_is_induct[i]) for i in idx])
        factor = factor * np.where(is_ind, induct_boost, 1.0)

    if candidate_cited_author_strength is not None and cited_author_weight:
        strength = [candidate_cited_author_strength[i] for i in idx]
        factor = factor * cited_author_factor(strength, cited_author_weight)

    scores = rel * factor

    # --- 5. diversify, and reserve slots for Induct papers ----------------
    #
    # Scoring above is universal: an Induct paper and an external paper are
    # judged by the same formula against the same similarity gate. Reservation
    # changes only *placement*, among papers that already qualified — a
    # reserved slot cannot surface an irrelevant paper, because an irrelevant
    # one was dropped at the gate.
    #
    # Why reserve at all: a paper published here needs established peers to
    # review it, read it, or ignore it, and that verdict is what decides
    # whether it publishes. None of those can happen to a paper nobody is
    # shown. Visibility is therefore part of the gatekeeping mechanism, not a
    # thumb on the scale — and it is what the `induct_boost` multiplier could
    # not deliver, since a new paper has credibility 0 and salience 0 and loses
    # to established literature at any plausible multiplier.
    if slot_pattern is not None and candidate_is_induct is not None:
        creds_for_stream = (
            [candidate_credibilities[i] or 0.0 for i in idx]
            if candidate_credibilities is not None else [0.0] * len(idx)
        )
        low_max = (induct_low_credibility_max
                   if induct_low_credibility_max is not None else 0.0)
        streams = [
            (STREAM_OPEN if not candidate_is_induct[i]
             else (STREAM_INDUCT_LOW if c <= low_max else STREAM_INDUCT_HIGH))
            for i, c in zip(idx, creds_for_stream)
        ]
        picked = select_with_quota(
            scores, V[idx], streams, k, lam=mmr_lambda, pattern=slot_pattern,
            groups=None if nearest is None else nearest[idx],
            group_shares=interest_shares)
    elif nearest is not None:
        # No slots reserved, but positions are still dealt to interests. An
        # empty pattern makes every position open.
        picked = select_with_quota(
            scores, V[idx], [STREAM_OPEN] * len(idx), k, lam=mmr_lambda,
            pattern=None, groups=nearest[idx], group_shares=interest_shares)
    else:
        picked = mmr_select(scores, V[idx], k, lam=mmr_lambda)
    out_ids = [candidate_ids[idx[p]] for p in picked]
    out_scores = [float(scores[p]) for p in picked]

    # Enforce strictly decreasing scores so ORDER BY score DESC reproduces this
    # exact order downstream. MMR's marginal scores are near-monotone but not
    # guaranteed so, and equal scores would re-sort arbitrarily.
    for i in range(1, len(out_scores)):
        if out_scores[i] >= out_scores[i - 1]:
            out_scores[i] = out_scores[i - 1] - 1e-9

    return list(zip(out_ids, out_scores))


#: Streams a candidate can belong to, for slot reservation.
STREAM_OPEN = 0        #: anything — filled by the universal ranking
STREAM_INDUCT_LOW = 1  #: an Induct paper whose credibility is still low
STREAM_INDUCT_HIGH = 2 #: an Induct paper that has accrued credibility
#: An Induct paper whose authors asked *this* reader to review it.
#:
#: A request is a signal inside the ranking, never an override of it. A
#: requested candidate must already have cleared the similarity gate to be in
#: the candidate set at all, so this stream can only reorder papers the reader
#: was eligible to see anyway — it cannot put an off-topic paper in a feed,
#: whoever asked. See Part V-C of CONSTITUTION.md.
STREAM_INDUCT_REQUESTED = 3

#: The ceiling on how much of one reader's attention the request mechanism may
#: claim: at most this many of the ten positions may prefer
#: STREAM_INDUCT_REQUESTED. A property of the feed rather than a per-requester
#: quota, so it holds however many people are asking. Enforced by
#: `validate_slot_pattern`, which `DEFAULT_SLOT_PATTERN` is checked against at
#: import.
MAX_REQUESTED_SLOTS = 3

#: What each position of the 10-slot window may be filled by, in preference
#: order. An empty tuple means **open**: the best remaining paper of any kind,
#: Induct included.
#:
#: Set by Bryan on 2026-10-11. Five positions open, three preferring
#: low-credibility Induct papers, two preferring high-credibility ones. The
#: window repeats: position 11 behaves like position 1, 12 like 2, and so on.
#:
#: Positions 1 and 2 are open so the two most visible slots always go to the
#: best papers available — an Induct paper takes them when it genuinely wins,
#: and reserving them would promote a worse paper where it is most obvious.
#:
#: The low-credibility slots **cascade**: low first, then high-credibility
#: Induct, then open. A newly submitted paper has no other route to a reader,
#: so its slots are tried first; but if there are none, the slot goes to another
#: Induct paper before it goes to the open pool. The high slots fall straight
#: through to open, because high-credibility Induct papers can also compete in
#: the open positions on merit.
#: Requested papers lead the cascade in the three low-credibility positions and
#: nowhere else, which is what holds the request mechanism to three of ten.
#: Positions 9 and 10 stay closed to requests: those are for Induct papers that
#: earned credibility, and letting a request take them would let an author buy a
#: position that is supposed to be won.
#:
#: Because `requested` leads a *cascade*, behaviour is unchanged when nobody has
#: requested anything — the slot falls through to low, then high, then open,
#: exactly as before.
DEFAULT_SLOT_PATTERN = (
    (),                                      # 1  open
    (),                                      # 2  open
    (STREAM_INDUCT_REQUESTED,
     STREAM_INDUCT_LOW, STREAM_INDUCT_HIGH), # 3
    (),                                      # 4  open
    (STREAM_INDUCT_REQUESTED,
     STREAM_INDUCT_LOW, STREAM_INDUCT_HIGH), # 5
    (),                                      # 6  open
    (STREAM_INDUCT_REQUESTED,
     STREAM_INDUCT_LOW, STREAM_INDUCT_HIGH), # 7
    (),                                      # 8  open
    (STREAM_INDUCT_HIGH,),                   # 9
    (STREAM_INDUCT_HIGH,),                   # 10
)


def validate_slot_pattern(pattern=None):
    """
    Check a slot pattern against the request ceiling. Returns the pattern.

    Raises ValueError if more than `MAX_REQUESTED_SLOTS` of its positions would
    accept a requested paper. `DEFAULT_SLOT_PATTERN` is validated at import, and
    any pattern coming from configuration should be passed through here, because
    the ceiling is the only thing standing between "authors may signal" and
    "authors may fill an expert's feed".
    """
    pattern = DEFAULT_SLOT_PATTERN if pattern is None else pattern
    n = sum(1 for slot in pattern
            if STREAM_INDUCT_REQUESTED in _slot_preferences(slot))
    if n > MAX_REQUESTED_SLOTS:
        raise ValueError(
            f"slot pattern allocates {n} positions to requested papers; "
            f"MAX_REQUESTED_SLOTS is {MAX_REQUESTED_SLOTS}"
        )
    return pattern


def _slot_preferences(slot):
    """
    Normalise one pattern entry to a tuple of streams, in preference order.

    Accepts an empty tuple or None for "open", a bare stream constant for a
    single preference, or a tuple for a cascade. The lenient forms exist so a
    hand-written pattern in a config or a test cannot silently mean the wrong
    thing.
    """
    if slot is None:
        return ()
    if isinstance(slot, (tuple, list)):
        return tuple(slot)
    return (slot,)


# Fail at import rather than at serve time: a pattern that over-allocates to
# requests is a policy breach, and discovering it from a reader's feed would
# mean it had already happened. Placed here because it needs
# `_slot_preferences`, defined above.
validate_slot_pattern(DEFAULT_SLOT_PATTERN)


def select_with_quota(scores, vecs, streams, k, *, lam=0.8, pattern=DEFAULT_SLOT_PATTERN,
                      groups=None, group_shares=None):
    """
    Choose `k` items, reserving slots for particular streams, with MMR inside.

    One ranking, one set of rules. Every candidate — Induct or external — is
    scored by the same formula and has to clear the same similarity gate; this
    step only decides *placement* among papers that already qualified. A
    reserved slot can never be filled by an irrelevant paper, because an
    irrelevant paper was removed before this function ever sees it.

    Each position carries a preference order (see `DEFAULT_SLOT_PATTERN`). The
    first preferred stream with anything left wins the slot; when every
    preference is exhausted the slot falls through to the best remaining
    candidate of any kind. So a feed is never padded: with no Induct papers
    worth showing, the page is exactly what the universal ranking produced.

    The pattern repeats every `len(pattern)` positions, and a paper already
    chosen is never offered again — within a page by construction here, and
    across pages because the callers exclude what they have already served.

    "Best" means the MMR marginal score, so diversification still applies
    *within* a stream — three reserved slots do not become three papers from
    the same group.

    `groups` deals the *open* positions among a reader's interests. It gives,
    for each candidate, the interest it is nearest to; `group_shares` gives
    each interest's share of the reader's work (equal when omitted). Each open
    position then goes to the interest that is furthest behind its share so
    far, and is filled by that interest's best remaining paper. Dealt this way
    the count for every interest stays within one paper of its share at every
    point down the list — so the first page is mixed, not only the ninth.

    A reserved slot is filled exactly as it is without groups, and the paper
    that fills it counts toward its own interest, so reservation and dealing
    do not compete: a reader's second field does not get an extra paper
    because one of its positions happened to be a reserved one.

    An interest with nothing left gives its positions to the others, in the
    same furthest-behind order. Nothing is padded and nothing is dropped: the
    set of candidates is the same, only the order of choosing changes.

    Returns indices into `scores`, in output order.
    """
    n = len(scores)
    if n == 0 or k <= 0:
        return []
    k = min(k, n)

    scores = np.asarray(scores, dtype=float)
    streams = np.asarray(streams, dtype=int)

    dealt = None
    if groups is not None:
        groups = np.asarray(groups, dtype=int)
        m = (int(groups.max()) + 1) if len(groups) else 0
        if group_shares is not None:
            m = max(m, len(group_shares))
            shares = np.zeros(m, dtype=float)
            given = np.maximum(np.nan_to_num(np.asarray(group_shares, dtype=float)), 0.0)
            shares[:len(given)] = given
        else:
            shares = np.ones(m, dtype=float)
        if m > 1 and shares.sum() > 0:
            shares = shares / shares.sum()
            dealt = np.zeros(m, dtype=int)

    span = float(scores.max() - scores.min())
    rel = (scores - scores.min()) / span if span > 0 else np.zeros_like(scores)

    chosen: list[int] = []
    best_sim = np.zeros(n, dtype=float)
    taken = np.zeros(n, dtype=bool)

    for pos in range(k):
        prefs = _slot_preferences(pattern[pos % len(pattern)]) if pattern else ()

        marginal = lam * rel - (1.0 - lam) * best_sim
        marginal = np.where(taken, -np.inf, marginal)

        # Try each preferred stream in order; fall through to open only after
        # every preference is exhausted. `taken` is what guarantees no paper is
        # ever chosen twice, whichever stream offers it.
        pick = None
        for want in prefs:
            eligible = np.where((streams == want) & ~taken, marginal, -np.inf)
            if np.isfinite(eligible.max()):
                pick = int(np.argmax(eligible))
                break
        if pick is None and dealt is not None:
            # Furthest behind its share first. `shares * (pos + 1)` is what an
            # interest is owed once this position is filled; a tie goes to the
            # larger interest, then to the lower index.
            owed = shares * (len(chosen) + 1) - dealt
            for g in sorted(range(len(shares)), key=lambda j: (-owed[j], -shares[j], j)):
                eligible = np.where(groups == g, marginal, -np.inf)
                if np.isfinite(eligible.max()):
                    pick = int(np.argmax(eligible))
                    break
        if pick is None:
            if not np.isfinite(marginal.max()):
                break
            pick = int(np.argmax(marginal))

        chosen.append(pick)
        taken[pick] = True
        if dealt is not None:
            dealt[groups[pick]] += 1
        best_sim = np.maximum(best_sim, vecs @ vecs[pick])

    return chosen
