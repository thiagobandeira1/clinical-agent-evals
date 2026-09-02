"""Ratchet gates: regress against MEASURED baselines, never aspirational thresholds."""

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict


class Gate(BaseModel):
    model_config = ConfigDict(frozen=True)

    metric: str
    higher_is_better: bool = True
    """False for lower-is-better metrics such as a hallucination rate."""

    def __init__(self, metric: str, *, higher_is_better: bool = True) -> None:
        # Positional OR keyword metric name: ``Gate("micro_recall")`` reads like the gate it
        # declares, ``Gate(metric=...)`` stays valid for pydantic-style call sites.
        super().__init__(metric=metric, higher_is_better=higher_is_better)


def load_baseline(path: Path) -> dict[str, float]:
    """Flat metric floats from a committed baseline.json.

    Tolerates three shapes: flat ``{metric: value}``, ``{"metrics": {...}}``, or
    ``{"overall": {...}}`` (nested combinations too). Non-numeric and boolean values are
    ignored. Raises ``FileNotFoundError``; the vacuous-pass-when-absent policy belongs to
    consumer CLIs.
    """
    if not path.exists():
        raise FileNotFoundError(f"baseline not found at {path}")
    candidate: Any = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(candidate, dict) and isinstance(candidate.get("metrics"), dict):
        candidate = candidate["metrics"]
    if isinstance(candidate, dict) and isinstance(candidate.get("overall"), dict):
        candidate = candidate["overall"]
    if not isinstance(candidate, dict):
        raise ValueError(f"{path} does not contain a metric mapping")
    return {
        str(key): float(value)
        for key, value in candidate.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }


def compare_to_baseline(
    current: Mapping[str, float],
    baseline: Mapping[str, float],
    *,
    gates: Sequence[Gate],
    tolerance: float = 0.02,
) -> list[str]:
    """One message per gated metric worse than baseline by more than ``tolerance`` in the
    gate's direction. A gated metric missing from ``current`` is itself a regression; missing
    from ``baseline`` is skipped (the ratchet guards only measured numbers). ``[]`` == pass.
    """
    regressions: list[str] = []
    for gate in gates:
        if gate.metric not in baseline:
            continue
        expected = baseline[gate.metric]
        actual = current.get(gate.metric)
        if actual is None:
            regressions.append(
                f"{gate.metric}: missing from current results (baseline {expected:.4f})"
            )
            continue
        if math.isnan(actual):
            # NaN compares False against everything and would sail through the gate.
            regressions.append(f"{gate.metric}: current value is NaN (baseline {expected:.4f})")
            continue
        worse = (
            actual < expected - tolerance
            if gate.higher_is_better
            else actual > expected + tolerance
        )
        if worse:
            direction = "below" if gate.higher_is_better else "above"
            regressions.append(
                f"{gate.metric}: {actual:.4f} regressed {direction} baseline {expected:.4f} "
                f"(tolerance {tolerance:.2f})"
            )
    return regressions
