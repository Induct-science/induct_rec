# tests/test_rank_candidates.py
"""
The staged pipeline: gate, score, diversify.

Built from a real failure. On 2026-10-08 the ERA-Interim reanalysis paper sat at
the top of an abyssal-turbulence researcher's feed with cosine 0.530, beating
Large et al. 1994 on oceanic vertical mixing at cosine 0.775, because
salience/age of 176 citations a year outweighed a 46% deficit in topical fit.
The weights were not wrong — the candidate pool was the whole corpus, so a paper
that resembles all of climate science was allowed to compete.

The fixture below is that situation in three dimensions: `offtopic` has low
similarity and enormous salience, the rest are close to the reader.
"""
import numpy as np
import pytest

from induct_rec.algorithm import mmr_select, rank_candidates


OFFTOPIC, CLASSIC, RECENT, NEWEST = 30993, 1, 2, 3


@pytest.fixture
def feed():
    user = np.array([1.0, 0.0, 0.0])
    user /= np.linalg.norm(user)
    V = np.array([
        [0.53, 0.85, 0.0],   # offtopic: cosine 0.53, salience 2645, from 2011
        [0.78, 0.63, 0.0],   # classic:  cosine 0.78, salience 459,  from 1994
        [0.77, 0.64, 0.0],   # recent:   cosine 0.77, salience 22,   from 2023
        [0.79, 0.61, 0.0],   # newest:   cosine 0.79, salience 4,    from 2025
    ], dtype=float)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return dict(
        user_vec=user, candidate_vecs=V,
        candidate_ids=[OFFTOPIC, CLASSIC, RECENT, NEWEST],
        candidate_years=[2011, 1994, 2023, 2025],
        candidate_saliences=[2645.0, 459.0, 22.0, 4.0],
        candidate_credibilities=[85.7, 139.8, 125.1, 88.7],
    )


def _ids(result):
    return [i for i, _ in result]


class TestPoolGate:
    def test_without_a_gate_the_famous_offtopic_paper_wins(self, feed):
        """The 2026-10-08 failure, reproduced. This is the regression baseline."""
        assert _ids(rank_candidates(k=4, **feed))[0] == OFFTOPIC

    def test_the_gate_removes_it(self, feed):
        out = _ids(rank_candidates(k=3, min_similarity=0.6, **feed))
        assert OFFTOPIC not in out
        assert out[0] == CLASSIC

    def test_gate_never_empties_a_feed(self, feed):
        """
        A floor above every candidate must fall back to the nearest, not return
        nothing: a reader with an unusual profile should still get a feed.
        """
        out = rank_candidates(k=3, min_similarity=0.99, **feed)
        assert len(out) == 3


class TestScoringModes:
    def test_recency_mode_prefers_the_newest(self, feed):
        out = _ids(rank_candidates(k=3, min_similarity=0.6, scoring="recency", **feed))
        assert out == [NEWEST, RECENT, CLASSIC]

    def test_salience_age_mode_prefers_the_most_cited(self, feed):
        out = _ids(rank_candidates(k=3, min_similarity=0.6, scoring="salience_age", **feed))
        assert out[0] == CLASSIC

    def test_none_mode_is_similarity_times_credibility(self, feed):
        out = rank_candidates(k=3, min_similarity=0.6, scoring="none", **feed)
        assert len(out) == 3

    def test_induct_boost_applies_only_to_induct_papers(self, feed):
        flags = [False, False, False, True]      # newest is the Induct paper
        boosted = _ids(rank_candidates(k=3, min_similarity=0.6, scoring="none",
                                       candidate_is_induct=flags, induct_boost=3.0, **feed))
        assert boosted[0] == NEWEST


class TestScoreContract:
    def test_scores_are_strictly_decreasing(self, feed):
        """
        Not cosmetic. The precompute path stores only (paper_id, score) and the
        serving route re-reads ORDER BY score DESC, so an order not encoded in
        the score is lost — which is why an earlier newest-first re-sort was
        removed. MMR's order has to survive that round trip.
        """
        for lam in (1.0, 0.7, 0.3):
            out = rank_candidates(k=4, mmr_lambda=lam, **feed)
            scores = [s for _, s in out]
            assert all(scores[i] < scores[i - 1] for i in range(1, len(scores))), lam

    def test_empty_candidates_return_empty(self, feed):
        assert rank_candidates(feed["user_vec"], np.empty((0, 3)), [], k=5) == []


class TestMMR:
    def test_lambda_one_is_pure_relevance(self):
        scores = [3.0, 1.0, 2.0]
        V = np.eye(3)
        assert mmr_select(scores, V, 3, lam=1.0) == [0, 2, 1]

    def test_diversity_displaces_a_near_duplicate(self):
        """
        Two near-identical high scorers and one distinct lower scorer: pure
        relevance takes both duplicates, MMR takes one and then the outlier.
        """
        V = np.array([[1.0, 0.0], [0.999, 0.045], [0.0, 1.0]])
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        scores = [1.0, 0.98, 0.5]
        assert mmr_select(scores, V, 2, lam=1.0) == [0, 1]
        assert mmr_select(scores, V, 2, lam=0.5) == [0, 2]

    def test_k_larger_than_pool(self):
        assert len(mmr_select([1.0, 2.0], np.eye(2), 10, lam=0.7)) == 2


class TestCrossEncoderSeam:
    def test_rerank_fn_can_override_relevance(self, feed):
        """
        The seam a cross-encoder will use: it receives the shortlisted ids and
        their current relevance, and returns replacement relevance. Here it
        inverts the order to prove the hook is actually consulted.
        """
        seen = {}

        def rerank(ids, rel):
            seen["ids"] = list(ids)
            return [-r for r in rel]

        out = rank_candidates(k=3, min_similarity=0.6, scoring="none",
                              rerank_fn=rerank, **feed)
        assert seen["ids"], "rerank_fn was never called"
        assert _ids(out)[0] != CLASSIC

    def test_rerank_fn_returning_none_is_ignored(self, feed):
        baseline = _ids(rank_candidates(k=3, min_similarity=0.6, **feed))
        out = _ids(rank_candidates(k=3, min_similarity=0.6,
                                   rerank_fn=lambda ids, rel: None, **feed))
        assert out == baseline
