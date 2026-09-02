"""Pure ranking metrics over (ranked ids, gold id set).

Keyless, deterministic, no I/O. All functions raise ``ValueError`` on an empty gold set: in
a ranking eval that is always a dataset bug (refusal-style items are scored elsewhere, never
through these). Contrast :mod:`clinevals.classify`, where empty gold is legal by design.
"""

from collections.abc import Sequence
from collections.abc import Set as AbstractSet


def _check(gold: AbstractSet[str], k: int) -> None:
    if not gold:
        raise ValueError("gold set is empty — metric undefined (refusal items score elsewhere)")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")


def hit_at(ranked: Sequence[str], gold: AbstractSet[str], k: int) -> float:
    """1.0 if any gold id appears in the top-k, else 0.0."""
    _check(gold, k)
    return 1.0 if any(item in gold for item in ranked[:k]) else 0.0


def recall_at(ranked: Sequence[str], gold: AbstractSet[str], k: int) -> float:
    """Fraction of gold ids present in the top-k."""
    _check(gold, k)
    return len(set(ranked[:k]) & gold) / len(gold)


def precision_at(ranked: Sequence[str], gold: AbstractSet[str], k: int) -> float:
    """Fraction of the top-k slots occupied by gold ids (denominator is k, not len)."""
    _check(gold, k)
    return len(set(ranked[:k]) & gold) / k


def mrr(ranked: Sequence[str], gold: AbstractSet[str]) -> float:
    """Reciprocal rank of the first gold id; 0.0 when no gold id is retrieved."""
    if not gold:
        raise ValueError("gold set is empty — metric undefined (refusal items score elsewhere)")
    for rank, item in enumerate(ranked, start=1):
        if item in gold:
            return 1.0 / rank
    return 0.0
