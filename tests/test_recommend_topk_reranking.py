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


class TestRecencyResort:
    def test_topk_resorted_youngest_first(self):
        """With candidate_years, the selected top-k is re-ordered newest-first."""
        user = np.zeros(8, dtype=np.float32); user[0] = 1.0
        # three clearly-positive candidates, descending similarity
        cand = np.vstack([_vec_with_cosine(0.9), _vec_with_cosine(0.8), _vec_with_cosine(0.7)])
        ids = [1, 2, 3]
        years = [2010, 2024, 2018]
        recs = recommend_topk(user, cand, ids, k=3, candidate_years=years)
        # returned order must be by year desc: 2024(id2), 2018(id3), 2010(id1)
        assert [r[0] for r in recs] == [2, 3, 1]

    def test_recency_only_reorders_the_selected_topk(self):
        """Recency re-sorts within the top-k, it doesn't pull in low-similarity
        papers just because they're newer."""
        user = np.zeros(8, dtype=np.float32); user[0] = 1.0
        cand = np.vstack([_vec_with_cosine(0.9), _vec_with_cosine(0.85), _vec_with_cosine(0.1)])
        ids = [1, 2, 3]
        years = [2000, 2001, 2030]   # id3 is newest but least similar
        recs = recommend_topk(user, cand, ids, k=2, candidate_years=years)
        returned = {r[0] for r in recs}
        assert 3 not in returned  # newest-but-irrelevant excluded from top-2
        assert returned == {1, 2}


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
