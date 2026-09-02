"""Human spot-check primitives: reproducible stratified sampling and judge-human agreement."""

import random
from collections.abc import Callable, Sequence
from typing import TypeVar

T = TypeVar("T")


def stratified_sample(items: Sequence[T], *, n: int, key: Callable[[T], str], seed: int) -> list[T]:
    """Deterministic across runs, platforms, and Python builds.

    Strata are sorted by ``key``; order within a stratum is the input order; a seeded
    ``random.Random`` shuffles each stratum; selection is round-robin across strata up to
    ``n``. Same inputs + seed -> same sample, forever.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    strata: dict[str, list[T]] = {}
    for item in items:
        strata.setdefault(key(item), []).append(item)
    rng = random.Random(seed)  # noqa: S311 — sampling reproducibility, not security
    queues: list[list[T]] = []
    for stratum in sorted(strata):
        members = list(strata[stratum])
        rng.shuffle(members)
        queues.append(members)
    sample: list[T] = []
    while len(sample) < n and any(queues):
        for queue in queues:
            if queue and len(sample) < n:
                sample.append(queue.pop(0))
    return sample


def agreement_rate(pairs: Sequence[tuple[str, str]]) -> float:
    """Fraction of ``(judge_label, human_label)`` pairs that match exactly.

    Portfolio doctrine: a run with agreement below 0.80 is stamped untrusted — consumers
    compare against that bar and record the rate in ``RunStamp.judge_human_agreement``.
    Raises ``ValueError`` on an empty sequence.
    """
    if not pairs:
        raise ValueError("agreement rate is undefined over zero pairs")
    return sum(1 for judge, human in pairs if judge == human) / len(pairs)
