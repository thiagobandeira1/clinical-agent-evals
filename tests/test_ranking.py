"""Tests for ``clinevals.ranking``: pure ranking metrics that RAISE on empty gold.

Hand-computed fixtures; no I/O, no model. Contrast ``test_classify.py`` where empty gold is
legal by design.
"""

from collections.abc import Callable, Sequence
from collections.abc import Set as AbstractSet

import pytest

from clinevals.ranking import hit_at, mrr, precision_at, recall_at

KMetric = Callable[[Sequence[str], AbstractSet[str], int], float]

# The single gold id sits at rank 2 of three.
RANKED: list[str] = ["miss-a", "gold-1", "miss-b"]
GOLD: frozenset[str] = frozenset({"gold-1"})
K_METRICS: list[KMetric] = [hit_at, recall_at, precision_at]


# --- the hand-computed fixture: [miss, gold, miss] ------------------------------------


def test_mrr_is_reciprocal_rank_of_first_gold() -> None:
    assert mrr(RANKED, GOLD) == 0.5


@pytest.mark.parametrize(("k", "expected"), [(1, 0.0), (2, 1.0), (3, 1.0)])
def test_hit_at_k(k: int, expected: float) -> None:
    assert hit_at(RANKED, GOLD, k) == expected


@pytest.mark.parametrize(("k", "expected"), [(1, 0.0), (2, 1.0), (3, 1.0)])
def test_recall_at_k_single_gold(k: int, expected: float) -> None:
    assert recall_at(RANKED, GOLD, k) == expected


@pytest.mark.parametrize(("k", "expected"), [(1, 0.0), (2, 0.5), (3, 1 / 3)])
def test_precision_at_k(k: int, expected: float) -> None:
    assert precision_at(RANKED, GOLD, k) == pytest.approx(expected)


def test_precision_denominator_is_k_not_retrieved_length() -> None:
    """One gold hit in a one-item list still scores 1/5 at k=5 — the slot count is k."""
    assert precision_at(["gold-1"], GOLD, 5) == pytest.approx(0.2)


def test_k_beyond_ranked_length_is_fine() -> None:
    assert hit_at(RANKED, GOLD, 100) == 1.0
    assert recall_at(RANKED, GOLD, 100) == 1.0
    assert precision_at(RANKED, GOLD, 100) == pytest.approx(0.01)


def test_metrics_return_floats() -> None:
    assert isinstance(hit_at(RANKED, GOLD, 1), float)
    assert isinstance(recall_at(RANKED, GOLD, 1), float)
    assert isinstance(precision_at(RANKED, GOLD, 1), float)
    assert isinstance(mrr(RANKED, GOLD), float)


# --- empty gold is a dataset bug: every metric raises ---------------------------------


@pytest.mark.parametrize("metric", K_METRICS)
def test_empty_gold_raises_for_k_metrics(metric: KMetric) -> None:
    with pytest.raises(ValueError, match="gold set is empty"):
        metric(RANKED, frozenset(), 3)


def test_empty_gold_raises_for_mrr() -> None:
    with pytest.raises(ValueError, match="gold set is empty"):
        mrr(RANKED, frozenset())


@pytest.mark.parametrize("metric", K_METRICS)
def test_empty_gold_raises_even_when_nothing_was_ranked(metric: KMetric) -> None:
    with pytest.raises(ValueError, match="gold set is empty"):
        metric([], set(), 1)


def test_empty_gold_raises_for_mrr_even_when_nothing_was_ranked() -> None:
    with pytest.raises(ValueError, match="gold set is empty"):
        mrr([], set())


# --- k < 1 is a caller bug -------------------------------------------------------------


@pytest.mark.parametrize("metric", K_METRICS)
@pytest.mark.parametrize("k", [0, -1])
def test_k_below_one_raises(metric: KMetric, k: int) -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        metric(RANKED, GOLD, k)


# --- duplicates in the ranked list never double count ----------------------------------


def test_duplicate_gold_hits_count_once() -> None:
    ranked = ["gold-1", "gold-1", "gold-1"]

    assert recall_at(ranked, GOLD, 3) == 1.0  # not 3.0
    assert precision_at(ranked, GOLD, 3) == pytest.approx(1 / 3)  # not 1.0
    assert hit_at(ranked, GOLD, 3) == 1.0
    assert mrr(ranked, GOLD) == 1.0


def test_duplicate_misses_do_not_manufacture_hits() -> None:
    ranked = ["miss-a", "miss-a", "gold-1"]

    assert mrr(ranked, GOLD) == pytest.approx(1 / 3)
    assert hit_at(ranked, GOLD, 2) == 0.0
    assert recall_at(ranked, GOLD, 3) == 1.0
    assert precision_at(ranked, GOLD, 3) == pytest.approx(1 / 3)


# --- other boundaries -------------------------------------------------------------------


def test_nothing_ranked_with_gold_present_scores_zero_everywhere() -> None:
    assert hit_at([], GOLD, 1) == 0.0
    assert recall_at([], GOLD, 1) == 0.0
    assert precision_at([], GOLD, 1) == 0.0
    assert mrr([], GOLD) == 0.0


def test_multi_gold_partial_recall() -> None:
    ranked = ["g1", "x", "g2", "y"]
    gold = frozenset({"g1", "g2", "g3"})

    assert recall_at(ranked, gold, 1) == pytest.approx(1 / 3)
    assert recall_at(ranked, gold, 3) == pytest.approx(2 / 3)
    assert recall_at(ranked, gold, 4) == pytest.approx(2 / 3)  # g3 never retrieved
    assert precision_at(ranked, gold, 2) == 0.5
    assert precision_at(ranked, gold, 4) == 0.5
    assert hit_at(ranked, gold, 1) == 1.0
    assert mrr(ranked, gold) == 1.0


def test_mrr_uses_the_first_gold_hit_only() -> None:
    assert mrr(["x", "g2", "g1"], frozenset({"g1", "g2"})) == 0.5


def test_mrr_is_zero_when_no_gold_is_retrieved() -> None:
    assert mrr(["x", "y"], GOLD) == 0.0


def test_accepts_any_sequence_and_any_set() -> None:
    ranked: tuple[str, ...] = ("miss-a", "gold-1", "miss-b")
    gold: set[str] = {"gold-1"}

    assert mrr(ranked, gold) == 0.5
    assert hit_at(ranked, gold, 2) == 1.0
    assert recall_at(ranked, gold, 2) == 1.0
    assert precision_at(ranked, gold, 2) == 0.5
