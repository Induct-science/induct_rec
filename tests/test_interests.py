# tests/test_interests.py
"""
A reader who works in two fields is shown both.

One mean vector puts such a reader *between* their fields, where the nearest
papers are the ones that resemble each a little. Interests give them a centre
per field, and deal the feed between the fields in proportion to how much of
their work each one is.

What these pin down is mostly what interests must not do: change anything for
a reader with one field, cost anybody a paper that qualified before, let a
heavily cited field crowd out the reader's main one, or let an off-topic paper
in because there are now more centres to be near.
"""
import numpy as np
import pytest

from induct_rec import algorithm
from induct_rec.algorithm import (
    DEFAULT_SLOT_PATTERN, INTEREST_MAX_CLUSTER_INPUT, MAX_INTERESTS,
    STREAM_INDUCT_LOW, STREAM_OPEN,
    build_user_interests, build_user_profile_vec, cluster_interests,
    interest_centres, rank_candidates, reader_relevance, select_with_quota,
)

D = 16


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _axis(i, d=D):
    v = np.zeros(d)
    v[i] = 1.0
    return v


def _around(centre, n, spread=0.25, seed=0):
    """`n` unit vectors scattered about `centre`."""
    rng = np.random.default_rng(seed)
    pts = centre + spread * rng.normal(size=(n, len(centre))) / np.sqrt(len(centre))
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# Sorting a reader's papers into interests                                    #
# --------------------------------------------------------------------------- #

class TestClustering:
    def test_one_field_is_one_interest(self):
        labels = cluster_interests(_around(_axis(0), 20))
        assert set(labels) == {0}

    def test_two_fields_are_two_and_the_larger_comes_first(self):
        V = np.vstack([_around(_axis(1), 6, seed=1), _around(_axis(0), 14, seed=2)])
        labels = cluster_interests(V)
        assert list(labels[:6]) == [1] * 6
        assert list(labels[6:]) == [0] * 14

    def test_the_same_papers_give_the_same_answer(self):
        V = np.vstack([_around(_axis(0), 9, seed=3), _around(_axis(1), 7, seed=4),
                       _around(_axis(2), 5, seed=5)])
        assert np.array_equal(cluster_interests(V), cluster_interests(V.copy()))

    def test_one_stray_paper_is_not_an_interest(self):
        V = np.vstack([_around(_axis(0), 10), _axis(5)[None, :]])
        assert set(cluster_interests(V)) == {0}

    def test_two_stray_papers_are_not_one_either(self):
        """Below `min_papers`, however large a share of a short record."""
        V = np.vstack([_around(_axis(0), 4), _around(_axis(5), 2, seed=9)])
        assert set(cluster_interests(V)) == {0}
        assert set(cluster_interests(V, min_papers=2)) == {0, 1}

    def test_a_small_share_is_folded_into_the_nearest_field(self):
        # Three papers in forty: enough papers, not enough of the record.
        V = np.vstack([_around(_axis(0), 37), _around(_axis(5), 3, seed=9)])
        assert set(cluster_interests(V)) == {0}
        assert set(cluster_interests(V, min_share=0.05)) == {0, 1}

    def test_the_ceiling_holds_and_every_paper_is_still_placed(self):
        V = np.vstack([_around(_axis(i), 6, seed=i) for i in range(5)])
        labels = cluster_interests(V)
        assert len(labels) == 30
        assert set(labels) == set(range(MAX_INTERESTS))

    def test_a_ceiling_of_one_is_the_single_vector(self):
        V = np.vstack([_around(_axis(0), 6), _around(_axis(1), 6, seed=1)])
        assert set(cluster_interests(V, max_interests=1)) == {0}

    def test_weights_decide_which_interest_is_the_larger(self):
        V = np.vstack([_around(_axis(0), 10, seed=1), _around(_axis(1), 5, seed=2)])
        assert cluster_interests(V)[0] == 0
        heavy_second = np.array([1.0] * 10 + [5.0] * 5)
        assert cluster_interests(V, heavy_second)[0] == 1

    def test_a_field_somebody_has_left_can_fall_below_the_share(self):
        V = np.vstack([_around(_axis(0), 10, seed=1), _around(_axis(1), 5, seed=2)])
        assert set(cluster_interests(V)) == {0, 1}
        faded = np.array([1.0] * 10 + [0.05] * 5)
        assert set(cluster_interests(V, faded)) == {0}

    def test_a_long_record_is_sampled_and_every_paper_still_placed(self):
        n_a, n_b = INTEREST_MAX_CLUSTER_INPUT, INTEREST_MAX_CLUSTER_INPUT // 2
        V = np.vstack([_around(_axis(0), n_a, seed=1), _around(_axis(1), n_b, seed=2)])
        labels = cluster_interests(V)
        assert list(labels[:n_a]) == [0] * n_a
        assert list(labels[n_a:]) == [1] * n_b

    @pytest.mark.parametrize("V", [np.zeros((0, D)), np.zeros((1, D))])
    def test_nothing_and_one_paper(self, V):
        assert list(cluster_interests(V)) == [0] * len(V)

    def test_an_all_zero_embedding_does_not_break_it(self):
        V = np.vstack([_around(_axis(0), 6), np.zeros((1, D))])
        labels = cluster_interests(V)
        assert len(labels) == 7 and np.all(labels >= 0)

    def test_unreadable_weights_are_harmless(self):
        V = np.vstack([_around(_axis(0), 6, seed=1), _around(_axis(1), 6, seed=2)])
        w = np.array([1, 1, np.nan, 0, -2, 1, 1, 1, np.inf, 1, 1, 1], dtype=float)
        assert set(cluster_interests(V, w)) <= {0, 1}


class TestCentres:
    def test_shares_sum_to_one_and_centres_are_unit(self):
        V = np.vstack([_around(_axis(0), 12, seed=1), _around(_axis(1), 4, seed=2)])
        labels = cluster_interests(V)
        centres, shares = interest_centres(V, labels)
        assert shares == pytest.approx([0.75, 0.25])
        assert np.allclose(np.linalg.norm(centres, axis=1), 1.0)
        assert centres[0] @ _axis(0) > 0.9 and centres[1] @ _axis(1) > 0.9

    def test_one_interest_is_the_mean_of_everything(self):
        V = _around(_axis(0), 9)
        centres, shares = interest_centres(V, np.zeros(9, dtype=int))
        assert shares == pytest.approx([1.0])
        assert np.allclose(centres[0], _unit(V.mean(axis=0)))


# --------------------------------------------------------------------------- #
# From papers and keywords to interest vectors                                #
# --------------------------------------------------------------------------- #

class _FakeModel:
    """Maps a text to a fixed vector, and records what it was asked for."""

    def __init__(self, table):
        self.table = table
        self.asked = []

    def encode(self, texts, convert_to_numpy=True):
        self.asked.extend(texts)
        return np.array([self.table[t] for t in texts], dtype=float)


@pytest.fixture
def two_fields(monkeypatch):
    """Eight ocean papers from 2020 and four on sea ice from 2012."""
    ocean = _around(_axis(0), 8, seed=1)
    ice = _around(_axis(1), 4, seed=2)
    papers = ([(f"ocean {i}", "", 2020) for i in range(8)]
              + [(f"ice {i}", "", 2012) for i in range(4)])
    vecs = np.vstack([ocean, ice])
    # Unnormalised on purpose, as a model's output is.
    table = {algorithm.paper_text(t, a): 3.0 * v for (t, a, _), v in zip(papers, vecs)}
    model = _FakeModel(table)
    monkeypatch.setattr(algorithm, "get_model", lambda: model)
    return dict(papers=papers, vecs=vecs, model=model)


class TestBuildUserInterests:
    def test_no_papers_no_interests(self, two_fields):
        assert build_user_interests([]) == []

    def test_one_interest_is_the_mean_of_the_readers_papers(self, two_fields):
        """
        What `build_user_profile_vec` gives a reader with papers and no
        keywords — so "one interest" and "the one vector" are the same thing.
        """
        papers = two_fields["papers"]
        for halflife in (None, 4.0):
            (only,) = build_user_interests(papers, max_interests=1,
                                           recency_halflife_years=halflife,
                                           current_year=2026)
            expected = build_user_profile_vec(papers, {},
                                              recency_halflife_years=halflife,
                                              current_year=2026)
            assert np.allclose(only["vec"], expected, atol=1e-12)
            assert only["share"] == pytest.approx(1.0)
            assert only["papers"] == list(range(12))

    def test_two_fields_come_back_as_two(self, two_fields):
        ocean, ice = build_user_interests(two_fields["papers"])
        assert ocean["share"] == pytest.approx(8 / 12)
        assert ice["share"] == pytest.approx(4 / 12)
        assert ocean["papers"] == list(range(8)) and ice["papers"] == list(range(8, 12))
        vecs = two_fields["vecs"]
        assert np.allclose(ocean["vec"], _unit(vecs[:8].mean(axis=0)))
        assert np.allclose(ice["vec"], _unit(vecs[8:].mean(axis=0)))

    def test_every_paper_is_in_exactly_one_interest(self, two_fields):
        got = build_user_interests(two_fields["papers"])
        placed = sorted(i for interest in got for i in interest["papers"])
        assert placed == list(range(12))
        assert sum(interest["share"] for interest in got) == pytest.approx(1.0)

    def test_a_half_life_moves_the_shares(self, two_fields):
        plain = build_user_interests(two_fields["papers"], min_share=0.0)
        recent = build_user_interests(two_fields["papers"], min_share=0.0,
                                      recency_halflife_years=4.0, current_year=2026)
        # The ice papers are from 2012; with a half-life they count for less.
        assert recent[1]["share"] < plain[1]["share"]
        assert sum(i["share"] for i in recent) == pytest.approx(1.0)

    def test_stored_embeddings_mean_the_model_is_never_asked(self, two_fields):
        got = build_user_interests(two_fields["papers"], paper_vecs=two_fields["vecs"])
        assert two_fields["model"].asked == []
        encoded = build_user_interests(two_fields["papers"])
        assert len(two_fields["model"].asked) == 12
        for a, b in zip(got, encoded):
            assert np.allclose(a["vec"], b["vec"])

    def test_the_wrong_number_of_embeddings_is_refused(self, two_fields):
        with pytest.raises(ValueError):
            build_user_interests(two_fields["papers"], paper_vecs=two_fields["vecs"][:5])


# --------------------------------------------------------------------------- #
# Relevance                                                                   #
# --------------------------------------------------------------------------- #

class TestReaderRelevance:
    def test_without_interests_it_is_the_cosine_it_always_was(self):
        rng = np.random.default_rng(0)
        V, u = _around(_axis(0), 30, spread=2.0), _unit(rng.normal(size=D))
        rel, nearest = reader_relevance(u, V)
        assert nearest is None
        assert np.array_equal(rel, V @ u)
        assert np.array_equal(reader_relevance(u, V, [])[0], V @ u)

    def test_interests_can_only_raise_it(self):
        rng = np.random.default_rng(7)
        for _ in range(50):
            V = _around(_axis(0), 40, spread=4.0, seed=int(rng.integers(1e6)))
            a, b = _unit(rng.normal(size=D)), _unit(rng.normal(size=D))
            whole = _unit(2 * a + b)
            rel, _ = reader_relevance(whole, V, [a, b])
            assert np.all(rel >= V @ whole)

    def test_a_paper_between_two_fields_keeps_what_it_had(self):
        a, b = _axis(0), _axis(1)
        whole = _unit(a + b)
        bridging = whole.copy()
        rel, nearest = reader_relevance(whole, bridging, [a, b])
        assert rel[0] == pytest.approx(1.0)       # not the 0.707 either field gives it
        assert nearest[0] in (0, 1)

    def test_a_paper_in_one_field_is_measured_against_that_field(self):
        a, b = _axis(0), _axis(1)
        whole = _unit(a + b)
        rel, nearest = reader_relevance(whole, np.vstack([a, b]), [a, b])
        assert rel == pytest.approx([1.0, 1.0])
        assert list(nearest) == [0, 1]


# --------------------------------------------------------------------------- #
# The ranking                                                                 #
# --------------------------------------------------------------------------- #

A, B, BETWEEN, OFF = "a", "b", "between", "off"


@pytest.fixture
def reader():
    """
    Two-thirds field A, one third field B. Forty candidates in each, twenty
    that sit between the two, and ten that are nothing to do with either.
    """
    a, b = _axis(0), _axis(1)
    whole = _unit(2 * a + b)
    V = np.vstack([
        _around(a, 40, spread=0.5, seed=1),
        _around(b, 40, spread=0.5, seed=2),
        _around(_unit(a + b), 20, spread=0.3, seed=3),
        _around(_axis(7), 10, spread=0.3, seed=4),
    ])
    kinds = [A] * 40 + [B] * 40 + [BETWEEN] * 20 + [OFF] * 10
    return dict(
        base=dict(user_vec=whole, candidate_vecs=V,
                  candidate_ids=list(range(len(V))), scoring="none",
                  min_similarity=0.55),
        interests=dict(interest_vecs=np.vstack([a, b]), interest_shares=[2 / 3, 1 / 3]),
        kinds=kinds, a=a, b=b, V=V,
    )


def _kinds(result, kinds):
    return [kinds[i] for i, _ in result]


def _field(reader, result):
    """Which interest each returned paper is nearest to."""
    V = reader["V"]
    return ["a" if V[i] @ reader["a"] >= V[i] @ reader["b"] else "b" for i, _ in result]


class TestRankingWithInterests:
    def test_the_mean_alone_starves_the_second_field(self, reader):
        """The failure this exists for, shown on the fixture."""
        top = _kinds(rank_candidates(k=10, **reader["base"]), reader["kinds"])
        assert top.count(B) == 0

    def test_both_fields_are_on_the_first_page(self, reader):
        top = _field(reader, rank_candidates(k=10, **reader["base"], **reader["interests"]))
        assert 6 <= top.count("a") <= 7
        assert 3 <= top.count("b") <= 4

    def test_every_prefix_is_within_one_paper_of_the_shares(self, reader):
        got = _field(reader, rank_candidates(k=45, **reader["base"], **reader["interests"]))
        for t in range(1, len(got) + 1):
            assert abs(got[:t].count("a") - t * 2 / 3) < 1.0, t
            assert abs(got[:t].count("b") - t * 1 / 3) < 1.0, t

    def test_the_first_paper_is_from_the_larger_field(self, reader):
        got = _field(reader, rank_candidates(k=3, **reader["base"], **reader["interests"]))
        assert got[0] == "a"

    def test_a_heavily_cited_field_does_not_take_the_feed(self, reader):
        """
        Field B's literature is far better cited. Scored on one scale and left
        to compete, it would fill the page; dealt by share it gets its third.
        """
        n = len(reader["kinds"])
        saliences = [500.0 if kind == B else 1.0 for kind in reader["kinds"]]
        base = dict(reader["base"], scoring="salience_age",
                    candidate_saliences=saliences, candidate_years=[2020] * n)
        undealt = _field(reader, rank_candidates(
            k=10, **base, interest_vecs=reader["interests"]["interest_vecs"][1:]))
        assert undealt.count("b") >= 9
        dealt = _field(reader, rank_candidates(k=10, **base, **reader["interests"]))
        assert 6 <= dealt.count("a") <= 7

    def test_off_topic_papers_stay_out(self, reader):
        got = _kinds(rank_candidates(k=60, **reader["base"], **reader["interests"]),
                     reader["kinds"])
        assert OFF not in got

    def test_nothing_that_qualified_before_is_lost(self, reader):
        """Ask for the whole pool both ways: interests only ever add to it."""
        n = len(reader["kinds"])
        gate = reader["base"]["min_similarity"]
        V, whole = reader["V"], reader["base"]["user_vec"]
        passed_before = {i for i in range(n) if V[i] @ whole >= gate}
        rel, _ = reader_relevance(whole, V, reader["interests"]["interest_vecs"])
        passed_after = {i for i in range(n) if rel[i] >= gate}
        assert passed_before and passed_before < passed_after
        ranked = {i for i, _ in rank_candidates(
            k=len(passed_after), **reader["base"], **reader["interests"])}
        assert ranked == passed_after

    def test_an_exhausted_interest_hands_its_positions_on(self, reader):
        """Three papers in field B and no more: the page is still full."""
        keep = [i for i, kind in enumerate(reader["kinds"]) if kind == A] + [40, 41, 42]
        base = dict(reader["base"], candidate_vecs=reader["V"][keep],
                    candidate_ids=keep)
        got = rank_candidates(k=20, **base, **reader["interests"])
        assert len(got) == 20
        assert sum(1 for i, _ in got if i in (40, 41, 42)) == 3

    def test_scores_still_strictly_decrease(self, reader):
        scores = [s for _, s in rank_candidates(k=45, **reader["base"], **reader["interests"])]
        assert all(x > y for x, y in zip(scores, scores[1:]))

    def test_no_paper_is_chosen_twice(self, reader):
        ids = [i for i, _ in rank_candidates(k=90, **reader["base"], **reader["interests"])]
        assert len(ids) == len(set(ids))

    def test_shares_left_out_are_equal_shares(self, reader):
        got = _field(reader, rank_candidates(
            k=10, **reader["base"], interest_vecs=reader["interests"]["interest_vecs"]))
        assert got.count("a") == 5 and got.count("b") == 5

    def test_a_reader_with_no_interests_is_untouched(self, reader):
        plain = rank_candidates(k=30, **reader["base"])
        assert rank_candidates(k=30, **reader["base"], interest_vecs=None,
                               interest_shares=None) == plain

    def test_one_interest_equal_to_the_profile_changes_nothing_but_the_path(self, reader):
        """Same vector twice: relevance is identical, and one group deals nothing."""
        plain = rank_candidates(k=30, **reader["base"])
        same = rank_candidates(k=30, **reader["base"],
                               interest_vecs=[reader["base"]["user_vec"]])
        assert [i for i, _ in same] == [i for i, _ in plain]
        assert [s for _, s in same] == pytest.approx([s for _, s in plain])


class TestDealingAndReservedSlots:
    def test_a_reserved_slot_is_filled_as_before_and_counts_for_its_field(self, reader):
        """
        Mark four field-B papers as new Induct papers. Slots 3, 5 and 7 take
        them, as they would with no interests — and field B is not then handed
        extra open positions on top.
        """
        n = len(reader["kinds"])
        induct = [False] * n
        for i in (40, 41, 42, 43):
            induct[i] = True
        base = dict(reader["base"], candidate_is_induct=induct,
                    candidate_credibilities=[0.0] * n,
                    slot_pattern=DEFAULT_SLOT_PATTERN, induct_low_credibility_max=0.0)
        got = [i for i, _ in rank_candidates(k=10, **base, **reader["interests"])]
        for position in (3, 5, 7):
            assert induct[got[position - 1]], position
        fields = _field(reader, [(i, 0) for i in got])
        assert fields.count("b") <= 5

    def test_groups_left_out_is_the_selection_it_always_was(self):
        rng = np.random.default_rng(3)
        V = _around(_axis(0), 30, spread=1.0, seed=3)
        scores = rng.uniform(1, 5, size=30)
        streams = [STREAM_INDUCT_LOW if i % 7 == 0 else STREAM_OPEN for i in range(30)]
        plain = select_with_quota(scores, V, streams, 20)
        assert select_with_quota(scores, V, streams, 20, groups=None) == plain
        # One group is no dealing at all.
        assert select_with_quota(scores, V, streams, 20, groups=[0] * 30) == plain

    def test_dealing_stays_within_one_of_the_shares_for_any_split(self):
        rng = np.random.default_rng(11)
        for _ in range(200):
            m = int(rng.integers(2, 4))
            shares = np.maximum(rng.dirichlet(np.ones(m)), 0.15)
            shares /= shares.sum()
            groups = np.repeat(np.arange(m), 60)
            V = np.eye(m * 60)
            picked = select_with_quota(rng.uniform(1, 5, size=m * 60), V,
                                       [STREAM_OPEN] * (m * 60), 50, lam=1.0,
                                       pattern=None, groups=groups, group_shares=shares)
            counts = np.zeros(m)
            for t, p in enumerate(picked, 1):
                counts[groups[p]] += 1
                assert np.abs(counts - shares * t).max() < 1.0

    def test_within_an_interest_the_best_paper_goes_first(self):
        scores = np.array([5.0, 4.0, 3.0, 9.0, 8.0, 7.0])
        groups = [0, 0, 0, 1, 1, 1]
        picked = select_with_quota(scores, np.eye(6), [STREAM_OPEN] * 6, 6, lam=1.0,
                                   pattern=None, groups=groups, group_shares=[0.5, 0.5])
        assert [p for p in picked if groups[p] == 0] == [0, 1, 2]
        assert [p for p in picked if groups[p] == 1] == [3, 4, 5]
