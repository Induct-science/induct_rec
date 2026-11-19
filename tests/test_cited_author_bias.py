# tests/test_cited_author_bias.py
"""
A reader is shown more of the people they have already built on.

The term is a bounded multiplier in the scoring stage. What these pin down is
what it must not be able to do: bring an off-topic paper into a feed, hand the
whole feed to a reader's closest collaborators, or change anything at all for
a reader nobody has citation data for.
"""
import numpy as np
import pytest

from induct_rec.algorithm import (
    CITED_AUTHOR_SATURATION, cited_author_factor, cited_author_strength,
    rank_candidates,
)


OFFTOPIC, AHEAD, CLOSE, BEHIND = 9, 1, 2, 3


@pytest.fixture
def feed():
    """Four papers and no other signal: similarity decides, in this order."""
    user = np.array([1.0, 0.0, 0.0])
    V = np.array([
        [0.50, 0.87, 0.0],   # offtopic: below any sensible gate
        [0.90, 0.44, 0.0],   # ahead
        [0.85, 0.53, 0.0],   # close
        [0.70, 0.71, 0.0],   # behind
    ], dtype=float)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return dict(user_vec=user, candidate_vecs=V,
                candidate_ids=[OFFTOPIC, AHEAD, CLOSE, BEHIND],
                scoring="none")


def _ids(result):
    return [i for i, _ in result]


class TestStrength:
    def test_nothing_cited_is_no_strength(self):
        assert cited_author_strength(0) == 0.0
        assert cited_author_strength(None) == 0.0
        assert cited_author_strength(-3) == 0.0

    def test_it_rises_and_then_stops(self):
        one, three = cited_author_strength(1), cited_author_strength(3)
        assert 0.0 < one < three < 1.0
        assert cited_author_strength(CITED_AUTHOR_SATURATION) == pytest.approx(1.0)
        assert cited_author_strength(500) == 1.0

    def test_the_first_citation_is_the_big_step(self):
        """Knowing somebody's work at all says more than the tenth citation does."""
        first = cited_author_strength(1) - cited_author_strength(0)
        tenth = cited_author_strength(10) - cited_author_strength(9)
        assert first > 5 * tenth

    def test_an_unreadable_count_is_no_strength(self):
        assert cited_author_strength("many") == 0.0


class TestFactor:
    def test_bounded_by_the_weight(self):
        f = cited_author_factor([0.0, 0.5, 1.0, 7.0, -2.0, float("nan")], 0.5)
        assert list(f) == pytest.approx([1.0, 1.25, 1.5, 1.5, 1.0, 1.0])

    def test_a_scalar_comes_back_a_float(self):
        assert cited_author_factor(1.0, 0.5) == pytest.approx(1.5)
        assert isinstance(cited_author_factor(1.0, 0.5), float)

    def test_a_negative_weight_cannot_punish(self):
        """The term rewards a connection. It is not a way to demote anybody."""
        assert cited_author_factor(1.0, -3.0) == 1.0


class TestInThePipeline:
    def test_no_strengths_changes_nothing(self, feed):
        base = rank_candidates(k=4, **feed)
        assert rank_candidates(k=4, cited_author_weight=0.5, **feed) == base
        assert rank_candidates(
            k=4, candidate_cited_author_strength=[0, 0, 0, 0],
            cited_author_weight=0.5, **feed) == base

    def test_a_weight_of_zero_changes_nothing(self, feed):
        base = rank_candidates(k=4, **feed)
        assert rank_candidates(
            k=4, candidate_cited_author_strength=[1, 1, 1, 1],
            cited_author_weight=0.0, **feed) == base

    def test_a_cited_author_moves_up(self, feed):
        assert _ids(rank_candidates(k=3, min_similarity=0.6, **feed)) == [AHEAD, CLOSE, BEHIND]
        out = rank_candidates(
            k=3, min_similarity=0.6, cited_author_weight=0.5,
            candidate_cited_author_strength=[0.0, 0.0, 1.0, 0.0], **feed)
        assert _ids(out) == [CLOSE, AHEAD, BEHIND]

    def test_the_score_is_the_base_score_times_the_factor(self, feed):
        base = dict(rank_candidates(k=3, min_similarity=0.6, **feed))
        out = dict(rank_candidates(
            k=3, min_similarity=0.6, cited_author_weight=0.5,
            candidate_cited_author_strength=[0.0, 0.0, 1.0, 0.0], **feed))
        assert out[CLOSE] == pytest.approx(base[CLOSE] * 1.5)
        assert out[AHEAD] == pytest.approx(base[AHEAD])

    def test_it_cannot_carry_a_paper_past_a_much_better_match(self, feed):
        """
        Bounded: the strongest connection there is lifts a paper at cosine
        0.70 past one at 0.85, and not past one at 0.90 — the gap to the top
        is wider than the weight.
        """
        out = rank_candidates(
            k=3, min_similarity=0.6, cited_author_weight=0.25,
            candidate_cited_author_strength=[0.0, 0.0, 0.0, 1.0], **feed)
        assert _ids(out) == [AHEAD, BEHIND, CLOSE]

    def test_it_cannot_bring_an_offtopic_paper_through_the_gate(self, feed):
        """
        The gate runs first. Having cited somebody does not make their paper
        about something else relevant.
        """
        out = rank_candidates(
            k=3, min_similarity=0.6, cited_author_weight=5.0,
            candidate_cited_author_strength=[1.0, 0.0, 0.0, 0.0], **feed)
        assert OFFTOPIC not in _ids(out)

    def test_scores_still_strictly_decrease(self, feed):
        out = rank_candidates(
            k=4, cited_author_weight=0.5,
            candidate_cited_author_strength=[0.3, 0.0, 1.0, 0.3], **feed)
        scores = [s for _, s in out]
        assert all(a > b for a, b in zip(scores, scores[1:]))
