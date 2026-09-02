"""Tests for ``clinevals.runner``: the generic scoring loop and ``EvalReport.test_overall``.

Keyless by construction: every ``score_fn`` here is table-driven post-hoc scoring over
recorded outputs, exactly the consumer contract (SPEC section 3: no live runs).
"""

import json
from collections.abc import Callable, Iterable, Mapping

import pytest

from clinevals.classify import ConfusionCounts, micro_prf, set_confusion
from clinevals.dataset import EvalItemBase, Split
from clinevals.runner import (
    AggregateMetrics,
    EvalReport,
    ItemResult,
    ItemScore,
    aggregate,
    score_items,
)


class Item(EvalItemBase):
    """A minimal consumer item: recorded prediction ids plus the gold ids."""

    predicted: frozenset[str] = frozenset()
    gold: frozenset[str] = frozenset()


Scorer = Callable[[Item], ItemScore | None]


class ScoreBoom(RuntimeError):
    """Raised by a deliberately broken ``score_fn``."""


def item(
    item_id: str,
    *,
    category: str = "ranking",
    split: Split = "test",
    predicted: Iterable[str] = (),
    gold: Iterable[str] = (),
) -> Item:
    return Item(
        item_id=item_id,
        category=category,
        split=split,
        predicted=frozenset(predicted),
        gold=frozenset(gold),
    )


def counts(tp: int, fp: int, fn: int) -> ConfusionCounts:
    return ConfusionCounts(tp=tp, fp=fp, fn=fn)


def table_scorer(table: Mapping[str, ItemScore | None]) -> Scorer:
    """Score by ``item_id`` lookup so every expected number is visible in the test."""

    def score(candidate: Item) -> ItemScore | None:
        return table[candidate.item_id]

    return score


def confusion_scorer(candidate: Item) -> ItemScore | None:
    """The P1 shape: counts only, no per-item metrics."""
    return ItemScore(counts=set_confusion(candidate.predicted, candidate.gold))


# --- skip-on-None and exception propagation -------------------------------------------------


def test_none_return_skips_item_everywhere() -> None:
    items = [
        item("a", category="ranking", split="test"),
        item("b", category="skipped", split="dev"),
        item("c", category="ranking", split="test"),
    ]
    scorer = table_scorer(
        {
            "a": ItemScore(metrics={"hit@1": 1.0}, counts=counts(1, 0, 0)),
            "b": None,
            "c": ItemScore(metrics={"hit@1": 0.0}, counts=counts(0, 1, 1)),
        }
    )

    report = score_items(items, scorer)

    assert [result.item_id for result in report.per_item] == ["a", "c"]
    assert report.overall == {"hit@1": 0.5}
    assert report.per_category == {"ranking": {"hit@1": 0.5}}
    assert "skipped" not in report.per_category
    assert report.per_split == {"test": {"hit@1": 0.5}}
    assert "dev" not in report.per_split
    assert report.counts_per_category == {"ranking": counts(1, 1, 1)}
    assert report.counts_per_split == {"test": counts(1, 1, 1)}


def test_all_items_skipped_yields_empty_report() -> None:
    items = [item("a"), item("b", split="dev")]

    report = score_items(items, lambda _: None)

    assert report.per_item == []
    assert report.per_category == {}
    assert report.overall == {}
    assert report.per_split == {}
    assert report.counts_per_category == {}
    assert report.counts_per_split == {}
    assert report.test_overall == {}


def test_zero_items_yields_empty_report_and_empty_test_overall() -> None:
    report = score_items([], table_scorer({}))

    assert report.per_item == []
    assert report.overall == {}
    assert report.per_split == {}
    assert report.test_overall == {}


def test_score_fn_exception_propagates_unchanged() -> None:
    items = [item("a"), item("b"), item("c")]

    def score(candidate: Item) -> ItemScore | None:
        if candidate.item_id == "b":
            raise ScoreBoom(f"cannot score {candidate.item_id}")
        return ItemScore(metrics={"hit@1": 1.0})

    with pytest.raises(ScoreBoom, match="cannot score b"):
        score_items(items, score)


def test_score_fn_is_called_once_per_item_in_input_order() -> None:
    items = [item("z"), item("a"), item("m")]
    seen: list[str] = []

    def score(candidate: Item) -> ItemScore | None:
        seen.append(candidate.item_id)
        return ItemScore(metrics={"x": 1.0})

    report = score_items(items, score)

    assert seen == ["z", "a", "m"]
    assert [result.item_id for result in report.per_item] == ["z", "a", "m"]


# --- per-item results -----------------------------------------------------------------------


def test_per_item_carries_identity_split_counts_and_sorted_metrics() -> None:
    items = [item("p1", category="measure-x", split="dev")]
    scorer = table_scorer(
        {"p1": ItemScore(metrics={"zeta": 1.0, "alpha": 0.0}, counts=counts(2, 1, 0))}
    )

    report = score_items(items, scorer)

    assert report.per_item == [
        ItemResult(
            item_id="p1",
            category="measure-x",
            split="dev",
            metrics={"alpha": 0.0, "zeta": 1.0},
            counts=counts(2, 1, 0),
        )
    ]
    assert list(report.per_item[0].metrics) == ["alpha", "zeta"]


def test_item_score_defaults_are_empty_metrics_and_no_counts() -> None:
    score = ItemScore()

    assert score.metrics == {}
    assert score.counts is None


def test_empty_metrics_item_is_recorded_but_contributes_no_means() -> None:
    items = [item("a"), item("b")]
    scorer = table_scorer({"a": ItemScore(metrics={"hit@1": 1.0}), "b": ItemScore()})

    report = score_items(items, scorer)

    assert [result.item_id for result in report.per_item] == ["a", "b"]
    assert report.per_item[1].metrics == {}
    assert report.overall == {"hit@1": 1.0}


# --- sparse-key aggregation -----------------------------------------------------------------


def test_sparse_keys_never_dilute_other_means() -> None:
    items = [
        item("r1", category="ranking"),
        item("r2", category="ranking"),
        item("r3", category="ranking"),
        item("r4", category="ranking"),
        item("f1", category="refusal"),
        item("f2", category="refusal"),
    ]
    scorer = table_scorer(
        {
            "r1": ItemScore(metrics={"hit@1": 1.0}),
            "r2": ItemScore(metrics={"hit@1": 0.0}),
            "r3": ItemScore(metrics={"hit@1": 1.0}),
            "r4": ItemScore(metrics={"hit@1": 0.0}),
            "f1": ItemScore(metrics={"refused": 1.0}),
            "f2": ItemScore(metrics={"refused": 0.0}),
        }
    )

    report = score_items(items, scorer)

    # 2/4 over ranking items only; a diluted mean would be 2/6.
    assert report.overall == {"hit@1": 0.5, "refused": 0.5}
    assert report.per_category == {
        "ranking": {"hit@1": 0.5},
        "refusal": {"refused": 0.5},
    }
    assert report.per_split == {"test": {"hit@1": 0.5, "refused": 0.5}}


def test_partially_shared_keys_average_only_over_carriers() -> None:
    items = [item("a"), item("b"), item("c")]
    scorer = table_scorer(
        {
            "a": ItemScore(metrics={"hit@1": 1.0, "mrr": 0.5}),
            "b": ItemScore(metrics={"hit@1": 0.0}),
            "c": ItemScore(metrics={"hit@1": 0.0, "mrr": 0.25}),
        }
    )

    report = score_items(items, scorer)

    assert report.overall == {"hit@1": 1.0 / 3.0, "mrr": 0.375}


def test_aggregate_means_per_category_and_overall() -> None:
    rows = [
        ("b", {"x": 1.0}),
        ("a", {"x": 0.0, "y": 1.0}),
        ("b", {"x": 0.0, "y": 0.5}),
    ]

    agg = aggregate(rows)

    assert agg == AggregateMetrics(
        per_category={"a": {"x": 0.0, "y": 1.0}, "b": {"x": 0.5, "y": 0.5}},
        overall={"x": 1.0 / 3.0, "y": 0.75},
    )
    assert list(agg.per_category) == ["a", "b"]


def test_aggregate_of_nothing_is_empty() -> None:
    assert aggregate([]) == AggregateMetrics(per_category={}, overall={})


# --- per-split means ------------------------------------------------------------------------


def test_per_split_means_are_computed_within_each_split() -> None:
    items = [
        item("d1", split="dev"),
        item("d2", split="dev"),
        item("t1", split="test"),
        item("t2", split="test"),
    ]
    scorer = table_scorer(
        {
            "d1": ItemScore(metrics={"hit@1": 1.0}),
            "d2": ItemScore(metrics={"hit@1": 1.0}),
            "t1": ItemScore(metrics={"hit@1": 1.0}),
            "t2": ItemScore(metrics={"hit@1": 0.0}),
        }
    )

    report = score_items(items, scorer)

    assert report.per_split == {"dev": {"hit@1": 1.0}, "test": {"hit@1": 0.5}}
    assert list(report.per_split) == ["dev", "test"]
    assert report.overall == {"hit@1": 0.75}


# --- count summation ------------------------------------------------------------------------


def test_counts_are_summed_per_category_and_per_split() -> None:
    items = [
        item("a", category="x", split="test"),
        item("b", category="x", split="dev"),
        item("c", category="y", split="test"),
        item("d", category="y", split="test"),
    ]
    scorer = table_scorer(
        {
            "a": ItemScore(counts=counts(2, 1, 0)),
            "b": ItemScore(counts=counts(1, 0, 3)),
            "c": ItemScore(counts=counts(0, 2, 1)),
            "d": ItemScore(counts=None),
        }
    )

    report = score_items(items, scorer)

    assert report.counts_per_category == {
        "x": counts(2, 1, 0) + counts(1, 0, 3),
        "y": counts(0, 2, 1),
    }
    assert report.counts_per_category == {"x": counts(3, 1, 3), "y": counts(0, 2, 1)}
    assert report.counts_per_split == {
        "dev": counts(1, 0, 3),
        "test": counts(2, 1, 0) + counts(0, 2, 1),
    }
    assert report.counts_per_split == {"dev": counts(1, 0, 3), "test": counts(2, 3, 1)}
    assert list(report.counts_per_category) == ["x", "y"]
    assert list(report.counts_per_split) == ["dev", "test"]


def test_items_without_counts_create_no_count_entries() -> None:
    items = [item("a", category="z"), item("b", category="z", split="dev")]
    scorer = table_scorer(
        {"a": ItemScore(metrics={"hit@1": 1.0}), "b": ItemScore(metrics={"hit@1": 0.0})}
    )

    report = score_items(items, scorer)

    assert report.counts_per_category == {}
    assert report.counts_per_split == {}
    assert report.per_category == {"z": {"hit@1": 0.5}}


def test_confusion_scorer_sums_set_confusion_including_no_gap_patients() -> None:
    items = [
        item("hit", category="m1", predicted={"g1", "g2"}, gold={"g1", "g2"}),
        item("miss", category="m1", predicted=set(), gold={"g3"}),
        item("no-gap-clean", category="m2", predicted=set(), gold=set()),
        item("no-gap-fp", category="m2", predicted={"phantom"}, gold=set()),
    ]

    report = score_items(items, confusion_scorer)

    assert report.counts_per_category == {"m1": counts(2, 0, 1), "m2": counts(0, 1, 0)}
    assert report.counts_per_split == {"test": counts(2, 1, 1)}
    assert report.per_item[2].counts == counts(0, 0, 0)


# --- test_overall: the publication block ----------------------------------------------------


def test_test_overall_merges_test_means_with_micro_prf_over_test_counts() -> None:
    items = [
        item("t1", category="a", split="test"),
        item("t2", category="a", split="test"),
        item("d1", category="a", split="dev"),
        item("d2", category="a", split="dev"),
    ]
    scorer = table_scorer(
        {
            "t1": ItemScore(metrics={"hit@1": 1.0, "precision@5": 0.25}, counts=counts(2, 0, 0)),
            "t2": ItemScore(metrics={"hit@1": 0.0, "precision@5": 0.5}, counts=counts(1, 1, 2)),
            "d1": ItemScore(metrics={"hit@1": 0.0, "precision@5": 1.0}, counts=counts(0, 10, 10)),
            "d2": ItemScore(metrics={"hit@1": 0.0, "precision@5": 1.0}, counts=counts(0, 10, 10)),
        }
    )

    report = score_items(items, scorer)
    published = report.test_overall

    test_counts = counts(3, 1, 2)
    assert report.counts_per_split["test"] == test_counts
    assert published == {
        "hit@1": 0.5,
        "micro_f1": micro_prf([test_counts])["micro_f1"],
        "micro_precision": 0.75,
        "micro_recall": 3 / 5,
        "precision@5": 0.375,
    }
    assert published == dict(
        sorted({**report.per_split["test"], **micro_prf([test_counts])}.items())
    )
    assert list(published) == [
        "hit@1",
        "micro_f1",
        "micro_precision",
        "micro_recall",
        "precision@5",
    ]
    # The diagnostic ``overall`` block DOES include dev; the publication block does not.
    assert report.overall["hit@1"] == 0.25
    assert published["hit@1"] == 0.5
    assert (
        published["micro_precision"]
        != micro_prf(report.counts_per_split.values())["micro_precision"]
    )


def test_test_overall_is_empty_when_only_dev_items_were_scored() -> None:
    items = [item("d1", split="dev"), item("d2", split="dev")]
    scorer = table_scorer(
        {
            "d1": ItemScore(metrics={"hit@1": 1.0}, counts=counts(5, 0, 0)),
            "d2": ItemScore(metrics={"hit@1": 1.0}, counts=counts(5, 0, 0)),
        }
    )

    report = score_items(items, scorer)

    assert report.overall == {"hit@1": 1.0}
    assert report.per_split == {"dev": {"hit@1": 1.0}}
    assert report.counts_per_split == {"dev": counts(10, 0, 0)}
    assert report.test_overall == {}


def test_test_overall_falls_back_to_overall_only_when_no_split_data_exists() -> None:
    # ``split`` is required on every item, so this branch is only reachable when a report is
    # constructed by hand (e.g. a consumer adapter); with split data present but no ``test``
    # key, ``overall`` must NOT leak into the publication block.
    no_splits = EvalReport(per_item=[], per_category={}, overall={"x": 0.5}, per_split={})
    dev_only = EvalReport(
        per_item=[], per_category={}, overall={"x": 0.5}, per_split={"dev": {"x": 0.5}}
    )

    assert no_splits.test_overall == {"x": 0.5}
    assert dev_only.test_overall == {}


def test_test_overall_without_test_counts_is_just_the_test_means() -> None:
    items = [item("t1", split="test"), item("d1", split="dev")]
    scorer = table_scorer(
        {
            "t1": ItemScore(metrics={"hit@1": 1.0}),
            "d1": ItemScore(metrics={"hit@1": 0.0}, counts=counts(1, 1, 1)),
        }
    )

    report = score_items(items, scorer)

    assert report.test_overall == {"hit@1": 1.0}
    assert "test" not in report.counts_per_split


def test_test_overall_omits_undefined_micro_ratios() -> None:
    items = [item("clean", split="test"), item("missed", split="test")]
    # tp=0, fp=0, fn=2: precision undefined (nothing predicted), recall 0.0, f1 undefined.
    scorer = table_scorer(
        {
            "clean": ItemScore(metrics={"refused": 0.0}, counts=counts(0, 0, 0)),
            "missed": ItemScore(metrics={"refused": 0.0}, counts=counts(0, 0, 2)),
        }
    )

    report = score_items(items, scorer)

    assert report.test_overall == {"micro_recall": 0.0, "refused": 0.0}


def test_test_overall_with_all_zero_test_counts_has_no_micro_keys() -> None:
    items = [item("no-gap", split="test")]
    scorer = table_scorer({"no-gap": ItemScore(metrics={"hitl": 0.0}, counts=counts(0, 0, 0))})

    report = score_items(items, scorer)

    assert report.counts_per_split == {"test": counts(0, 0, 0)}
    assert report.test_overall == {"hitl": 0.0}


def test_counts_only_scorer_publishes_micro_metrics_alone() -> None:
    items = [
        item("p1", split="test", predicted={"a", "b"}, gold={"a"}),
        item("p2", split="test", predicted={"c"}, gold={"c", "d"}),
        item("p3", split="dev", predicted={"x"}, gold=set()),
    ]

    report = score_items(items, confusion_scorer)

    assert report.overall == {}
    assert report.per_split == {"dev": {}, "test": {}}
    assert report.test_overall == micro_prf([counts(2, 1, 1)])
    assert report.test_overall == {
        "micro_f1": 2 * (2 / 3) * (2 / 3) / ((2 / 3) + (2 / 3)),
        "micro_precision": 2 / 3,
        "micro_recall": 2 / 3,
    }


# --- ordering and determinism ---------------------------------------------------------------


def test_every_report_mapping_is_key_sorted() -> None:
    items = [
        item("t", category="zeta", split="test"),
        item("d", category="alpha", split="dev"),
    ]
    scorer = table_scorer(
        {
            "t": ItemScore(metrics={"z": 1.0, "a": 0.0, "m": 0.5}, counts=counts(1, 1, 1)),
            "d": ItemScore(metrics={"z": 1.0, "a": 0.0}, counts=counts(1, 1, 1)),
        }
    )

    report = score_items(items, scorer)

    assert list(report.per_item[0].metrics) == ["a", "m", "z"]
    assert list(report.overall) == ["a", "m", "z"]
    assert list(report.per_category) == ["alpha", "zeta"]
    assert list(report.per_category["zeta"]) == ["a", "m", "z"]
    assert list(report.per_split) == ["dev", "test"]
    assert list(report.counts_per_category) == ["alpha", "zeta"]
    assert list(report.counts_per_split) == ["dev", "test"]
    assert list(report.test_overall) == [
        "a",
        "m",
        "micro_f1",
        "micro_precision",
        "micro_recall",
        "z",
    ]


def test_score_items_is_deterministic_across_runs_and_insertion_orders() -> None:
    items = [
        item("a", category="c1", split="test"),
        item("b", category="c2", split="dev"),
        item("c", category="c1", split="test"),
    ]

    def forward(candidate: Item) -> ItemScore | None:
        return ItemScore(metrics={"alpha": 1.0, "beta": 0.0}, counts=counts(1, 0, 2))

    def backward(candidate: Item) -> ItemScore | None:
        return ItemScore(metrics={"beta": 0.0, "alpha": 1.0}, counts=counts(1, 0, 2))

    first = score_items(items, forward)
    second = score_items(items, forward)
    reordered = score_items(items, backward)

    assert first.model_dump() == second.model_dump()
    assert first.model_dump() == reordered.model_dump()
    # ``json.dumps`` preserves insertion order, so equal strings prove identical key ORDER,
    # not merely equal contents.
    assert json.dumps(first.model_dump()) == json.dumps(second.model_dump())
    assert json.dumps(first.model_dump()) == json.dumps(reordered.model_dump())
    assert first.test_overall == second.test_overall == reordered.test_overall
