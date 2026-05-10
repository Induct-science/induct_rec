# tests/test_recommendations_pure.py
"""
Unit tests for the pure / stateless functions in induct_rec/algorithm.py.

These tests do NOT require a Flask app context or database.
    - normalize:          pure numpy
    - recommend_topk:     pure numpy
    - build_user_profile_vec: calls the SentenceTransformer model (pre-loaded
                          in the test process by the time this runs), but needs
                          no DB or external I/O.
"""

import numpy as np
import pytest

from induct_rec.algorithm import normalize, recommend_topk, build_user_profile_vec


# ─────────────────────────────────────────────────────────────────────────────
# normalize
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalize:
    def test_unit_norm(self):
        v = np.array([3.0, 4.0], dtype=np.float32)
        n = normalize(v)
        assert abs(np.linalg.norm(n) - 1.0) < 1e-6

    def test_already_normalized(self):
        v = np.array([1.0, 0.0], dtype=np.float32)
        n = normalize(v)
        np.testing.assert_allclose(n, v, atol=1e-6)

    def test_zero_vector_returns_zero(self):
        v = np.zeros(5, dtype=np.float32)
        n = normalize(v)
        np.testing.assert_array_equal(n, v)

    def test_negative_components(self):
        v = np.array([-3.0, 4.0], dtype=np.float32)
        n = normalize(v)
        assert abs(np.linalg.norm(n) - 1.0) < 1e-6

    def test_high_dimensional(self):
        rng = np.random.default_rng(42)
        v = rng.standard_normal(384).astype(np.float32)
        n = normalize(v)
        assert abs(np.linalg.norm(n) - 1.0) < 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# recommend_topk
# ─────────────────────────────────────────────────────────────────────────────

def _unit_vec(values):
    """Helper: make a unit vector from a list."""
    v = np.array(values, dtype=np.float32)
    return normalize(v)


class TestRecommendTopk:
    def test_returns_top_k(self):
        user = _unit_vec([1, 0, 0])
        cands = np.vstack([
            _unit_vec([1, 0, 0]),   # id=10 — most similar
            _unit_vec([0, 1, 0]),   # id=20
            _unit_vec([0, 0, 1]),   # id=30
        ])
        recs = recommend_topk(user, cands, [10, 20, 30], k=2)
        assert len(recs) == 2

    def test_most_similar_is_first(self):
        user = _unit_vec([1, 0, 0])
        cands = np.vstack([
            _unit_vec([0, 1, 0]),   # id=100 — less similar
            _unit_vec([1, 0, 0]),   # id=200 — identical
        ])
        recs = recommend_topk(user, cands, [100, 200], k=2)
        assert recs[0][0] == 200

    def test_scores_are_floats(self):
        user = _unit_vec([1, 0])
        cands = np.vstack([_unit_vec([1, 0]), _unit_vec([0, 1])])
        recs = recommend_topk(user, cands, [1, 2], k=2)
        for _id, score in recs:
            assert isinstance(score, float)

    def test_scores_bounded(self):
        """Cosine similarity for normalized vectors is in [-1, 1]."""
        user = _unit_vec([1, 0, 0])
        cands = np.vstack([
            _unit_vec([1, 0, 0]),
            _unit_vec([-1, 0, 0]),
        ])
        recs = recommend_topk(user, cands, [1, 2], k=2)
        for _id, score in recs:
            assert -1.0 - 1e-6 <= score <= 1.0 + 1e-6

    def test_k_greater_than_n(self):
        """Should return at most N results even when k > N."""
        user = _unit_vec([1, 0])
        cands = np.vstack([_unit_vec([1, 0])])
        recs = recommend_topk(user, cands, [1], k=10)
        assert len(recs) == 1

    def test_empty_candidates(self):
        user = _unit_vec([1, 0, 0])
        cands = np.empty((0, 3), dtype=np.float32)
        recs = recommend_topk(user, cands, [], k=5)
        assert recs == []

    def test_ids_align_with_scores(self):
        """Returned IDs must match the candidate_ids list."""
        user = _unit_vec([1, 0, 0])
        cands = np.vstack([_unit_vec([1, 0, 0]), _unit_vec([0, 1, 0])])
        ids = [42, 99]
        recs = recommend_topk(user, cands, ids, k=2)
        returned_ids = {r[0] for r in recs}
        assert returned_ids <= set(ids)

    def test_single_candidate(self):
        user = _unit_vec([1, 0])
        cands = np.vstack([_unit_vec([0.6, 0.8])])
        recs = recommend_topk(user, cands, [7], k=1)
        assert len(recs) == 1
        assert recs[0][0] == 7


# ─────────────────────────────────────────────────────────────────────────────
# build_user_profile_vec  (uses SentenceTransformer — model must be available)
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildUserProfileVec:
    """
    These tests call the real SentenceTransformer model (no mock).
    They run under the same conditions as previous integration tests
    which also uses embed_paper().
    """

    def test_returns_normalized_vector(self):
        vec = build_user_profile_vec(
            [("Ocean mixing", "turbulent mixing in the deep ocean")],
            {},
        )
        assert vec is not None
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5

    def test_returns_none_when_no_input(self):
        assert build_user_profile_vec([], {}) is None

    def test_keywords_only(self):
        vec = build_user_profile_vec([], {"turbulence": 5.0, "oceanography": 3.0})
        assert vec is not None
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5

    def test_papers_only(self):
        vec = build_user_profile_vec(
            [("Stratified flows", "boundary layer numerical simulation")],
            {},
        )
        assert vec is not None
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5

    def test_alpha_affects_combined_vector(self):
        """The combined vector (alpha=0.7) should differ from keywords-only (alpha=0)."""
        papers = [("Ocean turbulence", "mixing in deep water")]
        keywords = {"machine learning": 10.0}

        v_combined = build_user_profile_vec(papers, keywords, alpha=0.7)
        v_kw_only = build_user_profile_vec([], keywords)
        v_paper_only = build_user_profile_vec(papers, {})

        # Combined should be different from either extreme
        assert not np.allclose(v_combined, v_kw_only, atol=1e-3)
        assert not np.allclose(v_combined, v_paper_only, atol=1e-3)

    def test_similar_papers_produce_similar_vecs(self):
        """Two oceanography researchers should have more similar profiles than
        an oceanographer vs an NLP researcher."""
        v_ocean1 = build_user_profile_vec(
            [("Ocean heat transport", "meridional heat flux")], {}
        )
        v_ocean2 = build_user_profile_vec(
            [("Deep ocean mixing", "turbulent mixing processes")], {}
        )
        v_nlp = build_user_profile_vec(
            [("BERT for text classification", "transformer fine-tuning")], {}
        )

        sim_ocean = float(np.dot(v_ocean1, v_ocean2))
        sim_cross = float(np.dot(v_ocean1, v_nlp))
        assert sim_ocean > sim_cross
