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

      1. similarity      cosine against the reader's profile vector
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
    sims = V @ user_vec
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
        picked = select_with_quota(scores, V[idx], streams, k,
                                   lam=mmr_lambda, pattern=slot_pattern)
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
DEFAULT_SLOT_PATTERN = (
    (),                                      # 1  open
    (),                                      # 2  open
    (STREAM_INDUCT_LOW, STREAM_INDUCT_HIGH), # 3
    (),                                      # 4  open
    (STREAM_INDUCT_LOW, STREAM_INDUCT_HIGH), # 5
    (),                                      # 6  open
    (STREAM_INDUCT_LOW, STREAM_INDUCT_HIGH), # 7
    (),                                      # 8  open
    (STREAM_INDUCT_HIGH,),                   # 9
    (STREAM_INDUCT_HIGH,),                   # 10
)


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


def select_with_quota(scores, vecs, streams, k, *, lam=0.8, pattern=DEFAULT_SLOT_PATTERN):
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

    Returns indices into `scores`, in output order.
    """
    n = len(scores)
    if n == 0 or k <= 0:
        return []
    k = min(k, n)

    scores = np.asarray(scores, dtype=float)
    streams = np.asarray(streams, dtype=int)

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
        if pick is None:
            if not np.isfinite(marginal.max()):
                break
            pick = int(np.argmax(marginal))

        chosen.append(pick)
        taken[pick] = True
        best_sim = np.maximum(best_sim, vecs @ vecs[pick])

    return chosen
