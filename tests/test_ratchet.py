"""Tests for ``clinevals.ratchet``: directional gates, tolerance edges, and baseline loading."""

import json
from pathlib import Path

import pytest

from clinevals.artifacts import RunStamp, build_artifact, write_artifact
from clinevals.classify import ConfusionCounts
from clinevals.dataset import EvalItemBase
from clinevals.ratchet import Gate, compare_to_baseline, load_baseline
from clinevals.runner import ItemScore, score_items

# Verbatim copy of hedis-spec-copilot/evals/baseline.json (P2's committed ratchet baseline as
# of 2026-09-01). The ``note`` key is a string and must be ignored by ``load_baseline``.
P2_BASELINE_TEXT = (
    "{\n"
    '  "note": "Measured ratchet baseline on the TEST split (never aspirational; dev is for '
    "tuning). CI fails if recall@8 or mrr drops more than 0.02 absolute below these. Moves "
    "only via PR with an ADR note. Source: evals/results/retrieval-2026-09-01.json after the "
    'TOC-phantom parser fix and year-inference tuning.",\n'
    '  "mrr": 0.6133,\n'
    '  "recall@8": 0.6459\n'
    "}\n"
)


def write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --- Gate ---------------------------------------------------------------------------------


def test_gate_defaults_to_higher_is_better() -> None:
    assert Gate(metric="mrr").higher_is_better is True
    assert Gate(metric="hallucination_rate", higher_is_better=False).higher_is_better is False


# --- compare_to_baseline: higher-is-better -------------------------------------------------


def test_higher_is_better_regression_below_tolerance_is_reported() -> None:
    regressions = compare_to_baseline(
        {"hit@1": 0.85}, {"hit@1": 0.90}, gates=(Gate(metric="hit@1"),)
    )

    assert regressions == ["hit@1: 0.8500 regressed below baseline 0.9000 (tolerance 0.02)"]


def test_higher_is_better_within_tolerance_passes() -> None:
    assert (
        compare_to_baseline({"hit@1": 0.89}, {"hit@1": 0.90}, gates=(Gate(metric="hit@1"),)) == []
    )


def test_higher_is_better_exactly_at_tolerance_passes() -> None:
    # Dyadic fractions: 0.75 - 0.125 == 0.625 exactly in binary floating point.
    assert (
        compare_to_baseline(
            {"mrr": 0.625}, {"mrr": 0.75}, gates=(Gate(metric="mrr"),), tolerance=0.125
        )
        == []
    )


def test_higher_is_better_exactly_at_default_tolerance_passes() -> None:
    expected = 0.6459
    tolerance = 0.02
    actual = expected - tolerance

    assert (
        compare_to_baseline(
            {"recall@8": actual}, {"recall@8": expected}, gates=(Gate(metric="recall@8"),)
        )
        == []
    )
    assert (
        compare_to_baseline(
            {"recall@8": actual},
            {"recall@8": expected},
            gates=(Gate(metric="recall@8"),),
            tolerance=tolerance,
        )
        == []
    )


def test_higher_is_better_just_below_tolerance_fails() -> None:
    # 1e-6 below the boundary: beyond the 1e-9 rounding epsilon, so a real regression.
    regressions = compare_to_baseline(
        {"mrr": 0.625 - 1e-6}, {"mrr": 0.75}, gates=(Gate(metric="mrr"),), tolerance=0.125
    )

    assert len(regressions) == 1
    assert regressions[0].startswith("mrr: ")
    assert "below" in regressions[0]


def test_higher_is_better_improvement_is_never_flagged() -> None:
    assert compare_to_baseline({"mrr": 0.99}, {"mrr": 0.5}, gates=(Gate(metric="mrr"),)) == []


def test_equal_values_pass_even_with_zero_tolerance() -> None:
    gates = (Gate(metric="mrr"),)

    assert compare_to_baseline({"mrr": 0.6133}, {"mrr": 0.6133}, gates=gates, tolerance=0.0) == []
    assert compare_to_baseline({"mrr": 0.6132}, {"mrr": 0.6133}, gates=gates, tolerance=0.0) == [
        "mrr: 0.6132 regressed below baseline 0.6133 (tolerance 0.00)"
    ]


# --- compare_to_baseline: lower-is-better --------------------------------------------------


def test_lower_is_better_regression_above_tolerance_is_reported() -> None:
    gate = Gate(metric="hallucination_rate", higher_is_better=False)

    regressions = compare_to_baseline(
        {"hallucination_rate": 0.15}, {"hallucination_rate": 0.10}, gates=(gate,)
    )

    assert regressions == [
        "hallucination_rate: 0.1500 regressed above baseline 0.1000 (tolerance 0.02)"
    ]


def test_lower_is_better_exactly_at_tolerance_passes() -> None:
    gate = Gate(metric="hallucination_rate", higher_is_better=False)

    # 0.125 + 0.125 == 0.25 exactly.
    assert (
        compare_to_baseline(
            {"hallucination_rate": 0.25},
            {"hallucination_rate": 0.125},
            gates=(gate,),
            tolerance=0.125,
        )
        == []
    )


def test_lower_is_better_just_above_tolerance_fails() -> None:
    gate = Gate(metric="hallucination_rate", higher_is_better=False)

    regressions = compare_to_baseline(
        {"hallucination_rate": 0.25 + 1e-9},
        {"hallucination_rate": 0.125},
        gates=(gate,),
        tolerance=0.125,
    )

    assert len(regressions) == 1
    assert "above" in regressions[0]


def test_lower_is_better_drop_is_an_improvement_not_a_regression() -> None:
    gate = Gate(metric="hallucination_rate", higher_is_better=False)

    assert (
        compare_to_baseline({"hallucination_rate": 0.0}, {"hallucination_rate": 0.5}, gates=(gate,))
        == []
    )


def test_direction_flips_the_verdict_for_the_same_numbers() -> None:
    current = {"m": 0.5}
    baseline = {"m": 0.9}

    assert compare_to_baseline(current, baseline, gates=(Gate(metric="m"),)) != []
    assert (
        compare_to_baseline(current, baseline, gates=(Gate(metric="m", higher_is_better=False),))
        == []
    )


# --- compare_to_baseline: missing metrics, gate selection ----------------------------------


def test_gated_metric_missing_from_current_is_a_regression() -> None:
    regressions = compare_to_baseline({}, {"mrr": 0.6133}, gates=(Gate(metric="mrr"),))

    assert len(regressions) == 1
    assert regressions[0].startswith("mrr: ")
    assert "missing" in regressions[0]
    assert "0.6133" in regressions[0]


def test_gated_metric_missing_from_baseline_is_skipped() -> None:
    regressions = compare_to_baseline(
        {"mrr": 0.0}, {"recall@8": 0.6459}, gates=(Gate(metric="mrr"),)
    )

    assert regressions == []


def test_no_gates_means_nothing_can_regress() -> None:
    assert compare_to_baseline({"mrr": 0.0}, {"mrr": 1.0}, gates=()) == []


def test_ungated_metrics_are_ignored() -> None:
    regressions = compare_to_baseline(
        {"mrr": 0.9, "recall@8": 0.0}, {"mrr": 0.9, "recall@8": 1.0}, gates=(Gate(metric="mrr"),)
    )

    assert regressions == []


def test_one_message_per_regressing_gate_in_gate_order() -> None:
    current = {"a": 0.0, "b": 1.0, "c": 0.0, "hall": 0.9}
    baseline = {"a": 1.0, "b": 1.0, "c": 1.0, "hall": 0.1}
    gates = (
        Gate(metric="c"),
        Gate(metric="b"),
        Gate(metric="hall", higher_is_better=False),
        Gate(metric="a"),
        Gate(metric="unmeasured"),
    )

    regressions = compare_to_baseline(current, baseline, gates=gates)

    assert [message.split(":")[0] for message in regressions] == ["c", "hall", "a"]


def test_ratchet_reads_a_report_test_overall_directly() -> None:
    class Item(EvalItemBase):
        pass

    items = [
        Item(item_id="t", category="m", split="test"),
        Item(item_id="d", category="m", split="dev"),
    ]

    def score(candidate: Item) -> ItemScore | None:
        if candidate.split == "dev":
            return ItemScore(counts=ConfusionCounts(tp=0, fp=9, fn=9))
        return ItemScore(metrics={"hitl_triggered": 1.0}, counts=ConfusionCounts(tp=3, fp=1, fn=1))

    report = score_items(items, score)
    gates = (
        Gate(metric="micro_recall"),
        Gate(metric="micro_precision"),
        Gate(metric="hitl_triggered"),
    )

    assert (
        compare_to_baseline(
            report.test_overall,
            {"micro_recall": 0.75, "micro_precision": 0.75, "hitl_triggered": 1.0},
            gates=gates,
        )
        == []
    )
    assert compare_to_baseline(report.test_overall, {"micro_recall": 0.80}, gates=gates) == [
        "micro_recall: 0.7500 regressed below baseline 0.8000 (tolerance 0.02)"
    ]


# --- load_baseline: shapes -----------------------------------------------------------------


def test_load_baseline_flat_shape(tmp_path: Path) -> None:
    path = write_json(tmp_path / "baseline.json", {"mrr": 0.6133, "recall@8": 0.6459})

    assert load_baseline(path) == {"mrr": 0.6133, "recall@8": 0.6459}


def test_load_baseline_metrics_wrapper(tmp_path: Path) -> None:
    path = write_json(tmp_path / "baseline.json", {"metrics": {"mrr": 0.5}, "item_count": 12})

    assert load_baseline(path) == {"mrr": 0.5}


def test_load_baseline_overall_wrapper(tmp_path: Path) -> None:
    path = write_json(tmp_path / "baseline.json", {"overall": {"mrr": 0.5}, "item_count": 12})

    assert load_baseline(path) == {"mrr": 0.5}


def test_load_baseline_nested_metrics_overall_artifact_shape(tmp_path: Path) -> None:
    artifact = {
        "tier": "keyless",
        "date": "2026-09-01",
        "git_sha": "abc123",
        "item_count": 12,
        "note": "metrics.overall is the test split; tuning uses dev only",
        "metrics": {
            "overall": {"micro_f1": 0.8, "micro_precision": 1.0, "micro_recall": 2 / 3},
            "per_category": {"m1": {"hitl": 1.0}},
            "per_split": {"dev": {"hitl": 0.0}, "test": {"hitl": 1.0}},
        },
    }
    path = write_json(tmp_path / "baseline.json", artifact)

    assert load_baseline(path) == {"micro_f1": 0.8, "micro_precision": 1.0, "micro_recall": 2 / 3}


def test_load_baseline_ignores_booleans_strings_nulls_and_containers(tmp_path: Path) -> None:
    payload = {
        "mrr": 0.5,
        "item_count": 12,
        "passed": True,
        "failed": False,
        "note": "0.99",
        "nothing": None,
        "list": [0.1, 0.2],
        "nested": {"x": 0.3},
    }
    path = write_json(tmp_path / "baseline.json", payload)

    loaded = load_baseline(path)

    assert loaded == {"item_count": 12.0, "mrr": 0.5}
    assert all(isinstance(value, float) for value in loaded.values())


def test_load_baseline_empty_object_is_an_empty_mapping(tmp_path: Path) -> None:
    assert load_baseline(write_json(tmp_path / "baseline.json", {})) == {}


def test_load_baseline_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="baseline not found"):
        load_baseline(tmp_path / "absent.json")


@pytest.mark.parametrize("text", ["[0.5, 0.6]", '"0.5"', "0.5", "null", "true"])
def test_load_baseline_non_mapping_json_raises(tmp_path: Path, text: str) -> None:
    path = tmp_path / "baseline.json"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="metric mapping"):
        load_baseline(path)


def test_load_baseline_reads_p2_real_baseline_ignoring_note(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    path.write_text(P2_BASELINE_TEXT, encoding="utf-8", newline="\n")

    assert load_baseline(path) == {"mrr": 0.6133, "recall@8": 0.6459}


def test_committed_artifact_round_trips_as_a_baseline(tmp_path: Path) -> None:
    class Item(EvalItemBase):
        pass

    items = [
        Item(item_id="t", category="m", split="test"),
        Item(item_id="d", category="m", split="dev"),
    ]

    def score(candidate: Item) -> ItemScore | None:
        return ItemScore(metrics={"hit@1": 1.0}, counts=ConfusionCounts(tp=1, fp=1, fn=0))

    report = score_items(items, score)
    artifact = build_artifact(report, RunStamp(git_sha="abc123", date="2026-09-01"), tier="keyless")
    path = tmp_path / "artifact.json"
    write_artifact(path, artifact)

    loaded = load_baseline(path)

    assert loaded == report.test_overall
    assert loaded == {"hit@1": 1.0, "micro_f1": 2 / 3, "micro_precision": 0.5, "micro_recall": 1.0}
    assert compare_to_baseline(report.test_overall, loaded, gates=(Gate(metric="micro_f1"),)) == []
