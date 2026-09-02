"""Set-classification metrics with honest small-N semantics.

Empty gold is LEGAL here: a no-gap patient is a load-bearing false-positive probe. Undefined
ratios are ``None``, never a flattering 0.0/1.0. Micro-averages are derived from SUMMED
counts (they cannot be recovered from per-item means after the fact), which is why counts
travel through the runner and into artifacts as first-class values.
"""

from collections.abc import Iterable
from collections.abc import Set as AbstractSet

from pydantic import BaseModel, ConfigDict


class ConfusionCounts(BaseModel):
    model_config = ConfigDict(frozen=True)

    tp: int
    fp: int
    fn: int

    def __add__(self, other: "ConfusionCounts") -> "ConfusionCounts":
        return ConfusionCounts(tp=self.tp + other.tp, fp=self.fp + other.fp, fn=self.fn + other.fn)

    @property
    def precision(self) -> float | None:
        """tp / (tp + fp); None when nothing was predicted."""
        denominator = self.tp + self.fp
        return self.tp / denominator if denominator else None

    @property
    def recall(self) -> float | None:
        """tp / (tp + fn); None when there was nothing to find."""
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator else None

    @property
    def f1(self) -> float | None:
        """Harmonic mean; None when precision or recall is undefined; 0.0 when both are 0."""
        precision, recall = self.precision, self.recall
        if precision is None or recall is None:
            return None
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


def set_confusion(predicted: AbstractSet[str], gold: AbstractSet[str]) -> ConfusionCounts:
    """Confusion counts between two id sets. Never raises; empty sets are legal."""
    return ConfusionCounts(
        tp=len(predicted & gold), fp=len(predicted - gold), fn=len(gold - predicted)
    )


def micro_prf(counts: Iterable[ConfusionCounts]) -> dict[str, float]:
    """Sum the counts, then read the summed ratios.

    Keys ``micro_precision`` / ``micro_recall`` / ``micro_f1``; a key whose value is undefined
    is OMITTED, never coerced to 0.0. Key-sorted. Micro != macro on imbalanced slices — this
    is the honest headline for small per-category counts.
    """
    total = ConfusionCounts(tp=0, fp=0, fn=0)
    for item in counts:
        total = total + item
    out: dict[str, float] = {}
    if total.f1 is not None:
        out["micro_f1"] = total.f1
    if total.precision is not None:
        out["micro_precision"] = total.precision
    if total.recall is not None:
        out["micro_recall"] = total.recall
    return dict(sorted(out.items()))


def outside_universe_rate(predicted: AbstractSet[str], universe: AbstractSet[str]) -> float | None:
    """|predicted - universe| / |predicted|; None when nothing was predicted.

    A deterministic, keyless hallucination rate: predictions that name ids outside the
    legitimate universe (e.g. HCC codes no evidence could support).
    """
    if not predicted:
        return None
    return len(predicted - universe) / len(predicted)
