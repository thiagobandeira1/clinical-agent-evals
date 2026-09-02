"""The generic scoring loop and its report.

``score_fn`` is pure post-hoc scoring over outputs produced BEFORE the eval (recorded outputs
in keyless CI, live runs locally). Returning ``None`` skips an item; exceptions propagate —
there is no silent-partial-numbers mode.
"""

from collections.abc import Callable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from clinevals.classify import ConfusionCounts, micro_prf
from clinevals.dataset import ItemT


class ItemScore(BaseModel):
    """What a consumer's ``score_fn`` returns for one item."""

    model_config = ConfigDict(frozen=True)

    metrics: dict[str, float] = {}
    counts: ConfusionCounts | None = None


class ItemResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    item_id: str
    category: str
    split: str
    metrics: dict[str, float]
    counts: ConfusionCounts | None = None


class AggregateMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    per_category: dict[str, dict[str, float]]
    overall: dict[str, float]


def _means(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    means: dict[str, float] = {}
    for key in keys:
        values = [row[key] for row in rows if key in row]
        means[key] = sum(values) / len(values)
    return means


def aggregate(per_item: Sequence[tuple[str, Mapping[str, float]]]) -> AggregateMetrics:
    """Mean each metric per category and overall from ``(category, metrics)`` pairs.

    A metric key is averaged ONLY over the items that carry it (sparse-key means), so mixed
    slices (refusal items scored on ``refused`` alone) never dilute other means. Output dicts
    are key-sorted — byte-identical across runs for identical inputs.
    """
    by_category: dict[str, list[Mapping[str, float]]] = {}
    for category, metrics in per_item:
        by_category.setdefault(category, []).append(metrics)
    per_category = {cat: _means(rows) for cat, rows in sorted(by_category.items())}
    overall = _means([metrics for _, metrics in per_item])
    return AggregateMetrics(per_category=per_category, overall=overall)


def _sum_counts(rows: Sequence[ConfusionCounts]) -> ConfusionCounts:
    total = ConfusionCounts(tp=0, fp=0, fn=0)
    for row in rows:
        total = total + row
    return total


class EvalReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    per_item: list[ItemResult]
    per_category: dict[str, dict[str, float]]
    overall: dict[str, float]
    """Sparse-key means over ALL scored items (dev+test) — diagnostic only, never published."""
    per_split: dict[str, dict[str, float]]
    counts_per_category: dict[str, ConfusionCounts] = {}
    counts_per_split: dict[str, ConfusionCounts] = {}

    @property
    def test_overall(self) -> dict[str, float]:
        """THE publication block: test-split means (fallback: ``overall`` when no split data)
        merged with ``micro_prf`` over test-split counts. Key-sorted. ``build_artifact`` and
        the ratchet read ONLY this."""
        base = dict(self.per_split.get("test", self.overall if not self.per_split else {}))
        counts = self.counts_per_split.get("test")
        if counts is not None:
            base.update(micro_prf([counts]))
        return dict(sorted(base.items()))


def score_items(
    items: Sequence[ItemT], score_fn: Callable[[ItemT], ItemScore | None]
) -> EvalReport:
    """One deterministic pass in input order; ``None`` skips; exceptions propagate."""
    per_item: list[ItemResult] = []
    rows: list[tuple[str, Mapping[str, float]]] = []
    split_rows: dict[str, list[tuple[str, Mapping[str, float]]]] = {}
    counts_by_category: dict[str, list[ConfusionCounts]] = {}
    counts_by_split: dict[str, list[ConfusionCounts]] = {}
    for item in items:
        score = score_fn(item)
        if score is None:
            continue
        per_item.append(
            ItemResult(
                item_id=item.item_id,
                category=item.category,
                split=item.split,
                metrics=dict(sorted(score.metrics.items())),
                counts=score.counts,
            )
        )
        rows.append((item.category, score.metrics))
        split_rows.setdefault(item.split, []).append((item.category, score.metrics))
        if score.counts is not None:
            counts_by_category.setdefault(item.category, []).append(score.counts)
            counts_by_split.setdefault(item.split, []).append(score.counts)

    agg = aggregate(rows)
    return EvalReport(
        per_item=per_item,
        per_category=agg.per_category,
        overall=agg.overall,
        per_split={split: aggregate(r).overall for split, r in sorted(split_rows.items())},
        counts_per_category={cat: _sum_counts(r) for cat, r in sorted(counts_by_category.items())},
        counts_per_split={split: _sum_counts(r) for split, r in sorted(counts_by_split.items())},
    )
