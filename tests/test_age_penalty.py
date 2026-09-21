# tests/test_age_penalty.py
"""
The paper-age penalty in recommend_topk.

Induct's paper credibility floats upward for every paper, Induct and harvested
alike, because author credibility decades ago cannot be reconstructed for work
we merely ingested. The agreed counterweight was never a freeze — it was this:
a penalty on age, applied in the recommendation algorithm only.

The signal is `salience / age`. Salience is `3.0 + 0.1 x citations`, and raw
citation counts span 40x across eras purely because older papers have had
longer to accumulate them (measured on production 2026-09-20; see
induct_be/tools/README_paper_age_analysis.md). Dividing by age collapses that
and favours current work.

Two properties are load-bearing and both are pinned below:

1. It runs OUTSIDE the 15-paper tiers, so it changes which tier a paper lands
   in — not merely its rank within one. A recent paper can therefore beat a
   more topically relevant old one. That is the intent, and it is what makes
   this a stronger lever than the credibility multiplier.

2. It never modifies Paper.salience. Author keyword scores, and therefore user
   credibility, are computed from the raw stored value. An age-adjusted
   salience reaching that path would silently rewrite people's expertise.
"""
import numpy as np
import pytest

from induct_rec.algorithm import recommend_topk

YEAR = 2026


def _unit(x, y):
    v = np.array([x, y], dtype=np.float32)
    return v / np.linalg.norm(v)


def _rank(ids, scores):
    """Ranking by score, ignoring the newest-first display sort at the end."""
    return [i for i, _ in sorted(zip(ids, scores), key=lambda t: -t[1])]


class TestOutsideTheTiers:
    def test_a_recent_paper_can_beat_a_more_relevant_old_one(self):
        """
        The whole point of applying this before tiering. The 1975 paper is the
        better topical match; the 2025 paper still wins on score.
        """
        user = _unit(1, 0)
        vecs = np.array([_unit(1, 0), _unit(0.95, 0.05)])   # old matches better

        res = recommend_topk(
            user, vecs, ["old", "new"], k=2,
            candidate_years=[1975, 2025], candidate_credibilities=[0.0, 0.0],
            candidate_saliences=[3.0, 3.0], current_year=YEAR,
        )
        by_score = {pid: s for pid, s in res}
        assert by_score["new"] > by_score["old"], (
            "an age penalty applied inside the tiers could not achieve this"
        )

    def test_it_crosses_a_15_paper_tier_boundary(self):
        """
        Credibility can only reorder within a 15-paper block. Age must be able
        to lift a paper out of a lower tier entirely.
        """
        user = _unit(1, 0)
        # 20 papers of steadily decreasing relevance; the last is the newest.
        vecs = np.array([_unit(1.0, 0.02 * i) for i in range(20)])
        ids = list(range(20))
        years = [1980] * 19 + [2025]
        sal = [3.0] * 20

        without = recommend_topk(
            user, vecs, ids, k=20, candidate_years=years,
            candidate_credibilities=[0.0] * 20, current_year=YEAR,
        )
        with_age = recommend_topk(
            user, vecs, ids, k=20, candidate_years=years,
            candidate_credibilities=[0.0] * 20,
            candidate_saliences=sal, current_year=YEAR,
        )
        pos_without = _rank(*zip(*without)).index(19)
        pos_with = _rank(*zip(*with_age)).index(19)

        assert pos_without >= 15, "fixture should start the newest paper in a later tier"
        assert pos_with < pos_without - 5, (
            f"newest paper only moved {pos_without - pos_with} places; "
            "the penalty is not escaping its tier"
        )


class TestCredibilityIsUntouched:
    def test_the_inputs_are_not_mutated(self):
        """
        recommend_topk must not write back into the salience list it is handed.
        Author keyword scores read Paper.salience directly, so a mutated value
        here would propagate into user credibility.
        """
        user = _unit(1, 0)
        vecs = np.array([_unit(1, 0), _unit(0.9, 0.1)])
        sal = [3.0, 12.5]
        years = [1975, 2025]
        sal_before, years_before = list(sal), list(years)

        recommend_topk(
            user, vecs, [1, 2], k=2, candidate_years=years,
            candidate_credibilities=[0.0, 0.0],
            candidate_saliences=sal, current_year=YEAR,
        )

        assert sal == sal_before, "the salience list was mutated in place"
        assert years == years_before


class TestEdgeCases:
    def test_a_paper_from_this_year_does_not_divide_by_zero(self):
        user = _unit(1, 0)
        vecs = np.array([_unit(1, 0)])
        res = recommend_topk(
            user, vecs, [1], k=1, candidate_years=[YEAR],
            candidate_credibilities=[0.0], candidate_saliences=[3.0],
            current_year=YEAR,
        )
        assert np.isfinite(res[0][1])

    def test_an_unknown_year_is_neutral_not_ancient(self):
        """
        The route passes `year or 0` for papers with no year. Treating 0 as a
        publication date would make them 2026 years old and bury them.
        """
        user = _unit(1, 0)
        vecs = np.array([_unit(1, 0), _unit(1, 0)])
        res = dict(recommend_topk(
            user, vecs, ["unknown", "known"], k=2, candidate_years=[0, YEAR],
            candidate_credibilities=[0.0, 0.0], candidate_saliences=[3.0, 3.0],
            current_year=YEAR,
        ))
        # Neutral means the raw similarity, untouched.
        assert res["unknown"] == pytest.approx(1.0, abs=1e-6)
        assert res["known"] > res["unknown"], "a dated paper still gets its boost"

    def test_zero_salience_is_not_zeroed_out(self):
        """
        54% of the corpus has no credibility and many have no salience. A
        multiplicative `salience/age` would annihilate them; log1p(0) = 0 means
        they merely get no boost.
        """
        user = _unit(1, 0)
        vecs = np.array([_unit(1, 0)])
        res = recommend_topk(
            user, vecs, [1], k=1, candidate_years=[2020],
            candidate_credibilities=[0.0], candidate_saliences=[0.0],
            current_year=YEAR,
        )
        assert res[0][1] == pytest.approx(1.0, abs=1e-6)

    def test_weight_zero_disables_the_penalty(self):
        user = _unit(1, 0)
        vecs = np.array([_unit(1, 0), _unit(0.95, 0.05)])
        args = dict(
            candidate_years=[1975, 2025], candidate_credibilities=[0.0, 0.0],
            candidate_saliences=[3.0, 3.0], current_year=YEAR,
        )
        off = dict(recommend_topk(user, vecs, ["old", "new"], k=2,
                                  age_penalty_weight=0.0, **args))
        base = dict(recommend_topk(user, vecs, ["old", "new"], k=2,
                                   candidate_years=[1975, 2025],
                                   candidate_credibilities=[0.0, 0.0]))
        assert off == pytest.approx(base)

    def test_omitting_saliences_leaves_ranking_unchanged(self):
        """Existing callers that do not pass salience must be unaffected."""
        user = _unit(1, 0)
        vecs = np.array([_unit(1, 0), _unit(0.9, 0.1)])
        args = dict(candidate_years=[1990, 2024], candidate_credibilities=[10.0, 2.0])
        assert recommend_topk(user, vecs, [1, 2], k=2, **args) == \
               recommend_topk(user, vecs, [1, 2], k=2, candidate_saliences=None, **args)
