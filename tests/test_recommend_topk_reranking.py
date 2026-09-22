# tests/test_recommend_topk_reranking.py
"""
Unit tests for the *reranking* paths of recommend_topk — the credibility tier
boost and the recency re-sort — plus the embedding (de)serialization round-trip.

These are the subtle, easy-to-break parts that the basic
test_recommendations_pure.py doesn't cover. All pure numpy, no model, no DB.
"""
import numpy as np
import pytest

from induct_rec.algorithm import recommend_topk, serialize_embedding, deserialize_embedding, normalize


def _vec_with_cosine(s: float, dim: int = 8) -> np.ndarray:
    """Unit vector whose cosine with e0 == s."""
    v = np.zeros(dim, dtype=np.float32)
    v[0] = s
    v[1] = float(np.sqrt(max(0.0, 1.0 - s * s)))
    return v


class TestCredibilityReranking:
    def test_high_credibility_overtakes_slightly_higher_similarity(self):
        """Within a tier, a strong credibility boost can outrank a marginally
        more-similar but zero-credibility paper (the tiered log-boost)."""
        user = np.zeros(8, dtype=np.float32); user[0] = 1.0
        # A: higher similarity, no credibility. B: slightly lower sim, high credibility.
        cand = np.vstack([_vec_with_cosine(0.80), _vec_with_cosine(0.78)])
        ids = [101, 202]
        creds = [0.0, 1000.0]
        recs = recommend_topk(user, cand, ids, k=2, candidate_credibilities=creds)
        assert [r[0] for r in recs][0] == 202  # credibility boost wins

    def test_credibility_does_not_resurrect_negative_similarity(self):
        """Papers with negative similarity are filtered before the boost, so a
        high-credibility but irrelevant paper doesn't get recommended first."""
        user = np.zeros(8, dtype=np.float32); user[0] = 1.0
        relevant = _vec_with_cosine(0.5)          # positive sim
        opposite = -user                          # cosine -1
        cand = np.vstack([opposite, relevant])
        recs = recommend_topk(user, cand, [1, 2], k=2, candidate_credibilities=[9999.0, 0.0])
        assert recs[0][0] == 2  # the relevant one leads despite far lower credibility

    def test_equal_similarity_higher_credibility_first(self):
        user = np.zeros(8, dtype=np.float32); user[0] = 1.0
        cand = np.vstack([_vec_with_cosine(0.7), _vec_with_cosine(0.7)])
        recs = recommend_topk(user, cand, [1, 2], k=2, candidate_credibilities=[1.0, 50.0])
        assert recs[0][0] == 2


class TestRecency:
    """
    Recency used to be a hard newest-first re-sort of the returned top-K. That
    was dropped: it discarded score ordering wholesale inside the page, and it
    could not survive the caller that matters. Induct's precompute job stores
    only (paper_id, score) and the route re-reads ORDER BY score DESC, so any
    ordering not encoded in the score evaporated — meaning the two
    recommendation paths could never agree while it existed.

    Recency is carried by the age penalty instead, which is part of the score.
    """

    def test_results_come_back_in_score_order(self):
        user = np.zeros(8, dtype=np.float32); user[0] = 1.0
        cand = np.vstack([_vec_with_cosine(0.9), _vec_with_cosine(0.8), _vec_with_cosine(0.7)])
        recs = recommend_topk(user, cand, [1, 2, 3], k=3, candidate_years=[2010, 2024, 2018])
        scores = [r[1] for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_years_alone_do_not_reorder_anything(self):
        """`candidate_years` is an age-penalty input; without salience it is inert."""
        user = np.zeros(8, dtype=np.float32); user[0] = 1.0
        cand = np.vstack([_vec_with_cosine(0.9), _vec_with_cosine(0.8), _vec_with_cosine(0.7)])
        with_years = recommend_topk(user, cand, [1, 2, 3], k=3, candidate_years=[2010, 2024, 2018])
        without = recommend_topk(user, cand, [1, 2, 3], k=3)
        assert [r[0] for r in with_years] == [r[0] for r in without] == [1, 2, 3]

    def test_the_age_penalty_is_what_promotes_recent_work(self):
        user = np.zeros(8, dtype=np.float32); user[0] = 1.0
        cand = np.vstack([_vec_with_cosine(0.9), _vec_with_cosine(0.8)])
        recs = dict(recommend_topk(
            user, cand, ["old", "new"], k=2,
            candidate_years=[1980, 2026], candidate_credibilities=[0.0, 0.0],
            candidate_saliences=[3.0, 3.0], current_year=2026,
        ))
        assert recs["new"] > recs["old"], "the newer, less similar paper should win"

    def test_recency_does_not_resurrect_an_irrelevant_paper(self):
        """A newer paper still has to be in the running on relevance."""
        user = np.zeros(8, dtype=np.float32); user[0] = 1.0
        cand = np.vstack([_vec_with_cosine(0.9), _vec_with_cosine(0.85), _vec_with_cosine(0.01)])
        recs = recommend_topk(
            user, cand, [1, 2, 3], k=2,
            candidate_years=[2000, 2001, 2030], candidate_credibilities=[0.0, 0.0, 0.0],
            candidate_saliences=[3.0, 3.0, 3.0], current_year=2026,
        )
        assert 3 not in {r[0] for r in recs}


class TestSerializationRoundTrip:
    def test_round_trip_preserves_vector(self):
        v = normalize(np.random.default_rng(0).random(384).astype(np.float32))
        restored = deserialize_embedding(serialize_embedding(v))
        assert restored.dtype == np.float32
        assert restored.shape == v.shape
        np.testing.assert_allclose(restored, v, atol=1e-7)

    def test_deserialize_infers_dimension(self):
        v = np.arange(384, dtype=np.float32)
        restored = deserialize_embedding(serialize_embedding(v))
        assert len(restored) == 384
