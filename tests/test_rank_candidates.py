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

from induct_rec.algorithm import (
    DEFAULT_SLOT_PATTERN, MAX_REQUESTED_SLOTS, STREAM_INDUCT_HIGH,
    STREAM_INDUCT_LOW, STREAM_INDUCT_REQUESTED, _slot_preferences, mmr_select,
    rank_candidates, select_with_quota, validate_slot_pattern,
)


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


class TestInductSlotReservation:
    """
    Visibility as part of the gatekeeping mechanism.

    A paper published on Induct needs established peers to review it, read it,
    or ignore it, and that verdict decides whether it publishes. None of those
    can happen to a paper nobody is shown — and a new Induct paper has
    credibility 0 and salience 0, so it loses to established literature at any
    plausible `induct_boost`. Reserving slots is the only mechanism that
    actually reaches readers.

    What reservation must NOT do is lower the relevance bar. Everything here is
    scored by the same formula and gated on the same similarity floor; only
    placement differs.
    """

    @pytest.fixture
    def mixed(self):
        """
        Enough of each kind to fill the pattern exactly: eight strong external
        papers, three brand-new Induct papers for slots 3/5/7, and two
        established Induct papers for slots 9/10. Sized this way so the test
        reads the pattern rather than the cascade — the cascade has its own test.
        """
        user = np.array([1.0, 0.0, 0.0]); user /= np.linalg.norm(user)
        rows, creds, induct = [], [], []
        for i in range(8):                       # external, high similarity
            rows.append([0.95 - i * 0.01, 0.1, 0.0]); creds.append(200.0); induct.append(False)
        for i in range(3):                       # Induct, brand new
            rows.append([0.70 - i * 0.01, 0.4, 0.0]); creds.append(0.0); induct.append(True)
        for i in range(2):                       # Induct, established
            rows.append([0.80 - i * 0.01, 0.3, 0.0]); creds.append(150.0); induct.append(True)
        V = np.asarray(rows, dtype=float)
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        n = len(creds)
        return dict(user_vec=user, candidate_vecs=V,
                    candidate_ids=list(range(n)),
                    candidate_years=[2024] * n,
                    candidate_credibilities=creds,
                    candidate_is_induct=induct)

    def _kw(self, **over):
        kw = dict(scoring="none", min_similarity=None, mmr_lambda=1.0,
                  slot_pattern=DEFAULT_SLOT_PATTERN, induct_low_credibility_max=0.0)
        kw.update(over)
        return kw

    def test_without_reservation_induct_papers_are_buried(self, mixed):
        """The baseline this exists to fix."""
        out = [i for i, _ in rank_candidates(k=6, **dict(
            scoring="none", min_similarity=None, mmr_lambda=1.0), **mixed)]
        assert all(i < 8 for i in out), out

    def test_reservation_surfaces_them(self, mixed):
        out = [i for i, _ in rank_candidates(k=10, **self._kw(), **mixed)]
        induct_positions = [pos for pos, i in enumerate(out, 1) if mixed["candidate_is_induct"][i]]
        assert induct_positions == [3, 5, 7, 9, 10], out

    def test_positions_one_and_two_are_never_reserved(self, mixed):
        """
        The two most visible slots go to the best papers. An Induct paper takes
        them when it genuinely wins; reserving them would promote a worse paper
        exactly where it is most obvious.
        """
        out = [i for i, _ in rank_candidates(k=10, **self._kw(), **mixed)]
        assert not mixed["candidate_is_induct"][out[0]]
        assert not mixed["candidate_is_induct"][out[1]]

    def test_low_slots_come_before_high_slots(self, mixed):
        """Positions 3, 5, 7 prefer low credibility; 9 and 10 prefer high."""
        out = [i for i, _ in rank_candidates(k=10, **self._kw(), **mixed)]
        creds = mixed["candidate_credibilities"]
        lows = [p for p, i in enumerate(out, 1) if mixed["candidate_is_induct"][i] and creds[i] == 0.0]
        highs = [p for p, i in enumerate(out, 1) if mixed["candidate_is_induct"][i] and creds[i] > 0.0]
        assert lows == [3, 5, 7], lows
        assert highs == [9, 10], highs

    def test_low_slots_cascade_to_high_before_open(self, mixed):
        """
        A low slot prefers a low-credibility Induct paper, then any Induct
        paper, and only then the open pool. With no low-credibility papers the
        low slots must still go to Induct work rather than to the open pool.
        """
        creds = list(mixed["candidate_credibilities"])
        for i in (8, 9, 10):                  # make every "new" Induct paper established
            creds[i] = 150.0
        no_low = dict(mixed, candidate_credibilities=creds)
        out = [i for i, _ in rank_candidates(k=10, **self._kw(), **no_low)]
        assert no_low["candidate_is_induct"][out[2]], "position 3 should cascade to an Induct paper"

    def test_no_paper_is_chosen_twice(self, mixed):
        out = [i for i, _ in rank_candidates(k=12, **self._kw(), **mixed)]
        assert len(out) == len(set(out)), out

    def test_the_window_repeats(self, mixed):
        """Position 13 behaves like 3: the pattern is a repeating window."""
        rows = list(mixed["candidate_vecs"])
        creds = list(mixed["candidate_credibilities"])
        induct = list(mixed["candidate_is_induct"])
        for i in range(8):                    # more low-credibility Induct papers
            rows.append([0.68 - i * 0.01, 0.45, 0.0]); creds.append(0.0); induct.append(True)
        V = np.asarray(rows, dtype=float); V /= np.linalg.norm(V, axis=1, keepdims=True)
        wide = dict(mixed, candidate_vecs=V, candidate_credibilities=creds,
                    candidate_is_induct=induct,
                    candidate_ids=list(range(len(creds))),
                    candidate_years=[2024] * len(creds))
        out = [i for i, _ in rank_candidates(k=14, **self._kw(), **wide)]
        assert wide["candidate_is_induct"][out[12]], "position 13 should be reserved like position 3"

    def test_empty_streams_fall_through_rather_than_padding(self, mixed):
        """
        With no Induct papers at all, the page must be exactly what the
        universal ranking produced — reservation never pads a feed.
        """
        n = len(mixed["candidate_ids"])
        no_induct = dict(mixed, candidate_is_induct=[False] * n)
        reserved = [i for i, _ in rank_candidates(k=6, **self._kw(), **no_induct)]
        plain = [i for i, _ in rank_candidates(k=6, **dict(
            scoring="none", min_similarity=None, mmr_lambda=1.0), **no_induct)]
        assert reserved == plain

    def test_the_gate_still_applies_to_reserved_slots(self, mixed):
        """
        Reservation must not become a bypass. An Induct paper below the
        similarity floor stays out however many slots are free.
        """
        far = dict(mixed)
        V = np.array(far["candidate_vecs"], copy=True)
        V[8] = np.array([0.1, 0.99, 0.0]); V[8] /= np.linalg.norm(V[8])
        far["candidate_vecs"] = V
        out = [i for i, _ in rank_candidates(k=10, **self._kw(min_similarity=0.5), **far)]
        assert 8 not in out, out


class TestRequestedSlots:
    """
    A review request is a signal inside the ranking, never an override of it
    (CONSTITUTION.md Part V-C). Three things must hold, and the third is the one
    the whole design rests on:

      * a requested paper leads the cascade in the three low-credibility
        positions, so an author's signal is worth something;
      * it is confined to those three of ten, so the mechanism can never claim
        more than a bounded share of one expert's attention;
      * it buys nothing at all below the similarity gate, so no amount of
        requesting puts an off-topic paper in front of anybody.
    """

    def test_the_default_pattern_allocates_exactly_three(self):
        eligible = [pos for pos, slot in enumerate(DEFAULT_SLOT_PATTERN, 1)
                    if STREAM_INDUCT_REQUESTED in _slot_preferences(slot)]
        assert eligible == [3, 5, 7]
        assert len(eligible) == MAX_REQUESTED_SLOTS

    def test_the_earned_positions_stay_closed_to_requests(self):
        """
        9 and 10 are for Induct papers that accrued credibility. A request must
        not be able to buy a position that is supposed to be won.
        """
        for pos in (9, 10):
            prefs = _slot_preferences(DEFAULT_SLOT_PATTERN[pos - 1])
            assert STREAM_INDUCT_REQUESTED not in prefs

    def test_an_over_allocating_pattern_is_refused(self):
        with pytest.raises(ValueError, match="MAX_REQUESTED_SLOTS"):
            validate_slot_pattern(((STREAM_INDUCT_REQUESTED,),) * (MAX_REQUESTED_SLOTS + 1))

    def test_the_default_pattern_is_valid(self):
        assert validate_slot_pattern(DEFAULT_SLOT_PATTERN) is DEFAULT_SLOT_PATTERN

    def _streams(self, requested_idx=(), low_idx=(), high_idx=(), n=12):
        s = np.zeros(n, dtype=int)
        for i in low_idx:
            s[i] = STREAM_INDUCT_LOW
        for i in high_idx:
            s[i] = STREAM_INDUCT_HIGH
        for i in requested_idx:
            s[i] = STREAM_INDUCT_REQUESTED
        return s

    def test_a_requested_paper_takes_the_slot_ahead_of_an_unrequested_one(self):
        """
        Same scores, same stream position — the only difference is that one was
        requested. It must win position 3.
        """
        n = 12
        scores = np.linspace(1.0, 0.0, n)
        vecs = np.eye(n)                       # mutually orthogonal: MMR is inert
        streams = self._streams(requested_idx=(11,), low_idx=(10,), n=n)
        out = select_with_quota(scores, vecs, streams, k=3)
        assert out[2] == 11, out

    def test_the_cascade_is_unchanged_when_nothing_is_requested(self):
        """
        `requested` leads a cascade rather than replacing it, so a feed with no
        requests must behave exactly as it did before the stream existed.
        """
        n = 12
        scores = np.linspace(1.0, 0.0, n)
        vecs = np.eye(n)
        streams = self._streams(low_idx=(9, 10, 11), n=n)
        with_requested = select_with_quota(scores, vecs, streams, k=10)
        legacy_pattern = (
            (), (),
            (STREAM_INDUCT_LOW, STREAM_INDUCT_HIGH), (),
            (STREAM_INDUCT_LOW, STREAM_INDUCT_HIGH), (),
            (STREAM_INDUCT_LOW, STREAM_INDUCT_HIGH), (),
            (STREAM_INDUCT_HIGH,), (STREAM_INDUCT_HIGH,),
        )
        before = select_with_quota(scores, vecs, streams, k=10, pattern=legacy_pattern)
        assert with_requested == before

    def test_a_fourth_requested_paper_cannot_claim_a_fourth_slot(self):
        """
        The ceiling, demonstrated rather than asserted about the pattern: with
        four requested papers competing, only three may be placed inside one
        ten-position window.
        """
        n = 14
        scores = np.linspace(1.0, 0.0, n)
        vecs = np.eye(n)
        streams = self._streams(requested_idx=(10, 11, 12, 13), n=n)
        out = select_with_quota(scores, vecs, streams, k=10)
        placed = [i for i in out if streams[i] == STREAM_INDUCT_REQUESTED]
        assert len(placed) == MAX_REQUESTED_SLOTS, placed

    def test_requests_do_not_pad_a_feed(self):
        """With nothing requested and no Induct papers, the page is untouched."""
        n = 10
        scores = np.linspace(1.0, 0.0, n)
        vecs = np.eye(n)
        streams = self._streams(n=n)
        assert select_with_quota(scores, vecs, streams, k=6) == list(range(6))
