"""Tests for ``clinevals.classify``: set-classification metrics with honest small-N semantics.

Doctrine under test: empty gold is LEGAL (never raises); undefined ratios are ``None``, never
a flattering 0.0/1.0; micro-averages come from SUMMED counts and differ from per-item means.
"""

import pytest
from pydantic import ValidationError

from clinevals.classify import ConfusionCounts, micro_prf, outside_universe_rate, set_confusion


def _defined(value: float | None) -> float:
    """Narrow a ratio the test expects to be defined."""
    assert value is not None
    return value


def _same_ratio(actual: float | None, expected: float | None) -> bool:
    """``None`` must match ``None`` exactly; floats match approximately."""
    if expected is None:
        return actual is None
    return actual is not None and actual == pytest.approx(expected)


# --- set_confusion ----------------------------------------------------------------------


def test_set_confusion_basic() -> None:
    counts = set_confusion({"a", "b", "c"}, {"b", "c", "d"})

    assert counts == ConfusionCounts(tp=2, fp=1, fn=1)


def test_set_confusion_perfect_match() -> None:
    counts = set_confusion(frozenset({"a", "b"}), {"a", "b"})

    assert counts == ConfusionCounts(tp=2, fp=0, fn=0)
    assert (counts.precision, counts.recall, counts.f1) == (1.0, 1.0, 1.0)


def test_set_confusion_disjoint_sets() -> None:
    assert set_confusion({"a"}, {"b"}) == ConfusionCounts(tp=0, fp=1, fn=1)


def test_empty_gold_and_empty_predicted_is_all_zero_and_all_none() -> None:
    counts = set_confusion(set(), set())

    assert counts == ConfusionCounts(tp=0, fp=0, fn=0)
    assert counts.precision is None
    assert counts.recall is None
    assert counts.f1 is None


def test_empty_gold_with_predictions_is_a_false_positive_probe() -> None:
    """A no-gap patient: precision is a real 0.0, recall/f1 are undefined — never 0.0."""
    counts = set_confusion({"x", "y"}, set())

    assert counts == ConfusionCounts(tp=0, fp=2, fn=0)
    assert counts.precision == 0.0
    assert counts.recall is None
    assert counts.f1 is None


def test_empty_predicted_with_gold_present() -> None:
    counts = set_confusion(set(), {"x", "y"})

    assert counts == ConfusionCounts(tp=0, fp=0, fn=2)
    assert counts.precision is None
    assert counts.recall == 0.0
    assert counts.f1 is None


# --- ConfusionCounts ratios ------------------------------------------------------------


@pytest.mark.parametrize(
    ("counts", "precision", "recall", "f1"),
    [
        (ConfusionCounts(tp=0, fp=0, fn=0), None, None, None),
        (ConfusionCounts(tp=0, fp=2, fn=0), 0.0, None, None),
        (ConfusionCounts(tp=0, fp=0, fn=2), None, 0.0, None),
        (ConfusionCounts(tp=0, fp=1, fn=1), 0.0, 0.0, 0.0),
        (ConfusionCounts(tp=2, fp=0, fn=0), 1.0, 1.0, 1.0),
        (ConfusionCounts(tp=1, fp=1, fn=3), 0.5, 0.25, 1 / 3),
        (ConfusionCounts(tp=9, fp=1, fn=0), 0.9, 1.0, 18 / 19),
    ],
    ids=["all-zero", "fp-only", "fn-only", "both-zero", "perfect", "harmonic", "dominant"],
)
def test_ratio_semantics(
    counts: ConfusionCounts, precision: float | None, recall: float | None, f1: float | None
) -> None:
    assert _same_ratio(counts.precision, precision)
    assert _same_ratio(counts.recall, recall)
    assert _same_ratio(counts.f1, f1)


def test_f1_is_zero_not_none_when_precision_and_recall_are_both_defined_and_zero() -> None:
    counts = ConfusionCounts(tp=0, fp=1, fn=1)

    assert counts.precision == 0.0
    assert counts.recall == 0.0
    assert counts.f1 == 0.0
    assert counts.f1 is not None


def test_f1_is_harmonic_mean() -> None:
    counts = ConfusionCounts(tp=1, fp=1, fn=3)  # precision 0.5, recall 0.25

    assert counts.f1 == pytest.approx(2 * 0.5 * 0.25 / (0.5 + 0.25))
    assert counts.f1 == pytest.approx(2 * 1 / (2 * 1 + 1 + 3))  # 2tp / (2tp + fp + fn)


def test_add_sums_each_count_and_returns_a_new_instance() -> None:
    left = ConfusionCounts(tp=1, fp=2, fn=3)
    right = ConfusionCounts(tp=4, fp=5, fn=6)

    total = left + right

    assert total == ConfusionCounts(tp=5, fp=7, fn=9)
    assert isinstance(total, ConfusionCounts)
    assert left == ConfusionCounts(tp=1, fp=2, fn=3)  # operands untouched
    assert right == ConfusionCounts(tp=4, fp=5, fn=6)


def test_add_with_zero_is_identity() -> None:
    counts = ConfusionCounts(tp=3, fp=1, fn=2)

    assert counts + ConfusionCounts(tp=0, fp=0, fn=0) == counts


def test_add_is_associative() -> None:
    a = ConfusionCounts(tp=1, fp=0, fn=2)
    b = ConfusionCounts(tp=0, fp=3, fn=0)
    c = ConfusionCounts(tp=4, fp=1, fn=1)

    assert (a + b) + c == a + (b + c) == ConfusionCounts(tp=5, fp=4, fn=3)


def test_confusion_counts_are_frozen() -> None:
    counts = ConfusionCounts(tp=1, fp=1, fn=1)

    with pytest.raises(ValidationError):
        counts.tp = 5


# --- micro_prf ----------------------------------------------------------------------------


def test_micro_prf_reads_ratios_off_the_summed_counts() -> None:
    out = micro_prf([ConfusionCounts(tp=1, fp=1, fn=0), ConfusionCounts(tp=1, fp=0, fn=1)])

    # summed: tp=2, fp=1, fn=1
    assert out == pytest.approx(
        {"micro_f1": 2 / 3, "micro_precision": 2 / 3, "micro_recall": 2 / 3}
    )


def test_micro_prf_is_key_sorted() -> None:
    out = micro_prf([ConfusionCounts(tp=1, fp=1, fn=1)])

    assert list(out) == ["micro_f1", "micro_precision", "micro_recall"]
    assert list(out) == sorted(out)


def test_micro_prf_omits_precision_and_f1_when_nothing_was_predicted() -> None:
    out = micro_prf([ConfusionCounts(tp=0, fp=0, fn=1), ConfusionCounts(tp=0, fp=0, fn=2)])

    assert out == {"micro_recall": 0.0}
    assert "micro_precision" not in out
    assert "micro_f1" not in out


def test_micro_prf_omits_recall_and_f1_when_there_was_no_gold() -> None:
    out = micro_prf([ConfusionCounts(tp=0, fp=3, fn=0)])

    assert out == {"micro_precision": 0.0}


def test_micro_prf_on_all_zero_counts_is_empty() -> None:
    assert micro_prf([ConfusionCounts(tp=0, fp=0, fn=0)]) == {}


def test_micro_prf_on_empty_iterable_is_empty() -> None:
    assert micro_prf([]) == {}


def test_micro_prf_accepts_any_iterable() -> None:
    rows = [ConfusionCounts(tp=1, fp=1, fn=0), ConfusionCounts(tp=1, fp=0, fn=1)]

    assert micro_prf(row for row in rows) == micro_prf(rows)


def test_micro_prf_keeps_a_defined_zero_f1() -> None:
    out = micro_prf([ConfusionCounts(tp=0, fp=1, fn=1)])

    assert out == {"micro_f1": 0.0, "micro_precision": 0.0, "micro_recall": 0.0}


def test_micro_differs_from_macro_on_an_asymmetric_fixture() -> None:
    """One dominant item (9 tp, 1 fp) and one whiffed item (1 fp, 1 fn).

    Macro (mean of per-item ratios) weighs the whiff as heavily as the 10-prediction item;
    micro (ratios of summed counts) weighs every prediction equally. The numbers must differ
    for precision, recall AND f1 — that gap is the honest headline on small slices.
    """
    dominant = ConfusionCounts(tp=9, fp=1, fn=0)  # p 0.9, r 1.0, f1 18/19
    whiff = ConfusionCounts(tp=0, fp=1, fn=1)  # p 0.0, r 0.0, f1 0.0

    micro = micro_prf([dominant, whiff])  # summed: tp=9, fp=2, fn=1
    macro_precision = (_defined(dominant.precision) + _defined(whiff.precision)) / 2
    macro_recall = (_defined(dominant.recall) + _defined(whiff.recall)) / 2
    macro_f1 = (_defined(dominant.f1) + _defined(whiff.f1)) / 2

    assert micro["micro_precision"] == pytest.approx(9 / 11)
    assert macro_precision == pytest.approx(0.45)
    assert micro["micro_precision"] != pytest.approx(macro_precision)

    assert micro["micro_recall"] == pytest.approx(9 / 10)
    assert macro_recall == pytest.approx(0.5)
    assert micro["micro_recall"] != pytest.approx(macro_recall)

    assert micro["micro_f1"] == pytest.approx(18 / 21)  # 2tp / (2tp + fp + fn)
    assert macro_f1 == pytest.approx((18 / 19 + 0.0) / 2)
    assert micro["micro_f1"] != pytest.approx(macro_f1)


def test_item_with_undefined_precision_still_moves_micro_recall() -> None:
    """An item that predicted nothing has no precision (a macro would drop it) but its fn
    still counts: micro recall falls to 0.9 while the mean of defined recalls is 0.5."""
    dominant = ConfusionCounts(tp=9, fp=1, fn=0)
    silent = ConfusionCounts(tp=0, fp=0, fn=1)
    assert silent.precision is None

    micro = micro_prf([dominant, silent])  # summed: tp=9, fp=1, fn=1

    assert micro == pytest.approx({"micro_f1": 0.9, "micro_precision": 0.9, "micro_recall": 0.9})
    macro_recall = (_defined(dominant.recall) + _defined(silent.recall)) / 2
    assert macro_recall == pytest.approx(0.5)
    assert micro["micro_recall"] != pytest.approx(macro_recall)


# --- outside_universe_rate ------------------------------------------------------------------


def test_outside_universe_rate_is_none_when_nothing_was_predicted() -> None:
    assert outside_universe_rate(set(), {"a", "b"}) is None
    assert outside_universe_rate(frozenset(), frozenset()) is None


def test_outside_universe_rate_is_zero_when_every_prediction_is_inside() -> None:
    assert outside_universe_rate({"a", "b"}, {"a", "b", "c"}) == 0.0


def test_outside_universe_rate_is_half_when_half_are_outside() -> None:
    assert outside_universe_rate({"a", "zzz"}, {"a", "b"}) == 0.5


def test_outside_universe_rate_is_one_when_every_prediction_is_outside() -> None:
    assert outside_universe_rate({"x", "y"}, {"a", "b"}) == 1.0
    assert outside_universe_rate({"x"}, set()) == 1.0  # empty universe: everything hallucinated


def test_outside_universe_rate_general_fraction() -> None:
    assert outside_universe_rate({"a", "b", "c", "x"}, {"a", "b", "c"}) == pytest.approx(0.25)
