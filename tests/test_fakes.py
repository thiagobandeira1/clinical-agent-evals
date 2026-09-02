"""Tests for :mod:`clinevals.fakes`: ``scripted_judge`` round-trips through the real plumbing.

Every test drives ``invoke_judge`` with the fake model and parses the reply with
``parse_grounding_verdict`` -- the exact path a consumer's keyless judge tests take.
"""

import json
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel

from clinevals.fakes import scripted_judge
from clinevals.grounding import GroundingVerdict, parse_grounding_verdict
from clinevals.judge import JudgeParseError, Rubric, invoke_judge

RUBRIC = Rubric(name="test-rubric-v1", text="Grade the answer against the passages.")
PAYLOAD = "PASSAGES:\n[1] p\n\nQUESTION:\nq\n\nANSWER:\na"


def _verdict(
    supported: int = 0,
    unsupported: int = 0,
    contradicted: int = 0,
    *,
    ratio: float | None = None,
    notes: str = "",
) -> GroundingVerdict:
    return GroundingVerdict(
        claims_total=supported + unsupported + contradicted,
        claims_supported=supported,
        claims_unsupported=unsupported,
        claims_contradicted=contradicted,
        citation_valid_ratio=ratio,
        notes=notes,
    )


def _judge_once(verdict: GroundingVerdict) -> str:
    return invoke_judge(scripted_judge([verdict]), RUBRIC, PAYLOAD)


def test_scripted_judge_is_an_injectable_chat_model() -> None:
    assert isinstance(scripted_judge([]), BaseChatModel)


def test_round_trip_verdict_then_garbage() -> None:
    verdict = _verdict(3, 1, 1, notes="one hedge ignored")
    model = scripted_judge([verdict, "garbage"])

    first = invoke_judge(model, RUBRIC, PAYLOAD)
    parsed = parse_grounding_verdict(first)
    assert parsed == verdict
    assert parsed.claims_total == 5
    assert parsed.claims_supported == 3
    assert parsed.claims_unsupported == 1
    assert parsed.claims_contradicted == 1
    assert parsed.faithfulness == 0.6

    second = invoke_judge(model, RUBRIC, PAYLOAD)
    assert second == "garbage"
    with pytest.raises(JudgeParseError):
        parse_grounding_verdict(second)


def test_outputs_are_returned_in_script_order() -> None:
    only_supported = _verdict(1)
    only_unsupported = _verdict(0, 2)
    only_contradicted = _verdict(0, 0, 3)
    model = scripted_judge(["zero", only_supported, "two", only_unsupported, only_contradicted])

    outputs = [invoke_judge(model, RUBRIC, PAYLOAD) for _ in range(5)]

    assert outputs[0] == "zero"
    assert outputs[2] == "two"
    assert parse_grounding_verdict(outputs[1]) == only_supported
    assert parse_grounding_verdict(outputs[3]) == only_unsupported
    assert parse_grounding_verdict(outputs[4]) == only_contradicted


def test_string_entries_pass_through_verbatim() -> None:
    raw = '  ```json\n{"claims": []}\n```  '
    assert invoke_judge(scripted_judge([raw]), RUBRIC, PAYLOAD) == raw


def test_serialized_verdict_is_strict_schema_json() -> None:
    raw = _judge_once(_verdict(1, 1, 1, notes="n"))
    payload: dict[str, Any] = json.loads(raw)  # strict JSON: no prose, no fences
    assert set(payload) == {"claims", "notes"}
    assert payload["notes"] == "n"
    assert [claim["verdict"] for claim in payload["claims"]] == [
        "supported",
        "unsupported",
        "contradicted",
    ]
    assert all("claim" in claim for claim in payload["claims"])
    assert all("citation_valid" not in claim for claim in payload["claims"])


def test_citation_ratio_round_trips_half_over_two_cited_claims() -> None:
    verdict = _verdict(1, 1, ratio=0.5)
    parsed = parse_grounding_verdict(_judge_once(verdict))
    assert parsed.citation_valid_ratio == 0.5
    assert parsed == verdict


@pytest.mark.parametrize("ratio", [0.0, 1.0])
def test_citation_ratio_round_trips_at_extremes(ratio: float) -> None:
    parsed = parse_grounding_verdict(_judge_once(_verdict(2, 1, ratio=ratio)))
    assert parsed.citation_valid_ratio == ratio


def test_citation_ratio_round_trips_half_over_two_supported_claims() -> None:
    """Regression: buckets once aliased one dict via list multiplication (ratio collapsed)."""
    verdict = _verdict(2, ratio=0.5)
    parsed = parse_grounding_verdict(_judge_once(verdict))
    assert parsed.citation_valid_ratio == 0.5


def test_none_ratio_serializes_without_citation_flags() -> None:
    raw = _judge_once(_verdict(2, 1, ratio=None))
    assert "citation_valid" not in raw
    assert parse_grounding_verdict(raw).citation_valid_ratio is None


def test_zero_claims_serializes_to_empty_claims_list() -> None:
    raw = _judge_once(_verdict())  # claims_total 0
    assert json.loads(raw) == {"claims": [], "notes": ""}
    parsed = parse_grounding_verdict(raw)
    assert parsed.claims_total == 0
    assert parsed.faithfulness is None
    assert parsed.citation_valid_ratio is None


def test_notes_round_trip() -> None:
    verdict = _verdict(1, notes="judge saw a disclaimer footer; ignored it")
    assert parse_grounding_verdict(_judge_once(verdict)).notes == verdict.notes


def test_exhausted_script_is_loud_not_silent() -> None:
    model = scripted_judge(["only one"])
    assert invoke_judge(model, RUBRIC, PAYLOAD) == "only one"
    with pytest.raises((StopIteration, RuntimeError)):
        invoke_judge(model, RUBRIC, PAYLOAD)
