"""Tests for :mod:`clinevals.grounding`: rubric provenance golden, verdict parsing, metrics.

Keyless: no model is invoked here; the judge's raw text is canned.
"""

import hashlib
import json
from typing import Any

import pytest

from clinevals.grounding import (
    GroundingVerdict,
    build_judge_input,
    faithfulness_rubric,
    parse_grounding_verdict,
    verdict_metrics,
)
from clinevals.judge import JudgeParseError

HEDIS_DOMAIN = "Medicare Star Ratings / HEDIS measures"
HEDIS_RUBRIC_NAME = "hedis-grounding-v1"
# The ``judge_prompt_sha256`` P2 (hedis-spec-copilot) stamps into its committed eval artifacts.
P2_JUDGE_PROMPT_SHA256 = "31946648b81673911d373c4784496f672dc83f105e3a73d397a9e73f79915409"

CANNED_VERDICT: dict[str, Any] = {
    "claims": [
        {"claim": "The threshold is 7.0%.", "verdict": "supported", "citation_valid": True},
        {"claim": "The lookback is 10 years.", "verdict": "supported", "citation_valid": False},
        {"claim": "Members aged 18-75 qualify.", "verdict": "unsupported", "citation_valid": None},
        {"claim": "Exclusions never apply.", "verdict": "contradicted"},
    ],
    "notes": "answer cited [3] which does not exist",
}
CANNED_JSON = json.dumps(CANNED_VERDICT)


def _assert_canned(verdict: GroundingVerdict) -> None:
    assert verdict.claims_total == 4
    assert verdict.claims_supported == 2
    assert verdict.claims_unsupported == 1
    assert verdict.claims_contradicted == 1
    # Mean over the two CITED claims only (true, false); null/absent markers do not count.
    assert verdict.citation_valid_ratio == 0.5
    assert verdict.faithfulness == 0.5
    assert verdict.notes == "answer cited [3] which does not exist"


# --- faithfulness_rubric: provenance --------------------------------------------------------


def test_hedis_rubric_sha_matches_p2_stamped_judge_prompt() -> None:
    """GOLDEN PROVENANCE TEST: the most important test in this repo.

    P2 (hedis-spec-copilot) stamps ``judge_prompt_sha256`` into every committed eval artifact
    and pins that same hex string in its own golden test. SPEC goal 5 promises that
    ``faithfulness_rubric`` reproduces P2's frozen ``JUDGE_PROMPT`` byte-for-byte for the
    HEDIS domain phrase, so the identical sha must come out of THIS package.

    Editing ``_FAITHFULNESS_TEMPLATE`` in any way (a word, a space, the em-dash, the trailing
    newline) changes the sha and breaks CI in BOTH repos. That is by design: released rubric
    texts are immutable, and every stamped artifact's provenance must stay reproducible.
    Rubric evolution is a NEW name with a NEW golden sha, never an edit in place.
    """
    rubric = faithfulness_rubric(HEDIS_DOMAIN, name=HEDIS_RUBRIC_NAME)
    assert rubric.sha256 == P2_JUDGE_PROMPT_SHA256


def test_golden_sha_is_reproducible_from_the_rubric_text() -> None:
    # Independent route to the same digest, so the golden cannot hide behind Rubric.sha256.
    text = faithfulness_rubric(HEDIS_DOMAIN, name=HEDIS_RUBRIC_NAME).text
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == P2_JUDGE_PROMPT_SHA256


def test_rubric_name_is_carried() -> None:
    assert faithfulness_rubric(HEDIS_DOMAIN, name=HEDIS_RUBRIC_NAME).name == HEDIS_RUBRIC_NAME


def test_rubric_name_does_not_affect_sha() -> None:
    renamed = faithfulness_rubric(HEDIS_DOMAIN, name="hedis-grounding-v2-same-text")
    assert renamed.sha256 == P2_JUDGE_PROMPT_SHA256


def test_domain_phrase_is_substituted_exactly_once() -> None:
    text = faithfulness_rubric(HEDIS_DOMAIN, name=HEDIS_RUBRIC_NAME).text
    assert "__DOMAIN__" not in text
    assert text.count(HEDIS_DOMAIN) == 1
    assert text.startswith(f"You are grading a RAG answer about {HEDIS_DOMAIN} for faithfulness\n")


def test_different_domain_phrase_changes_sha_and_only_the_phrase() -> None:
    hedis = faithfulness_rubric(HEDIS_DOMAIN, name=HEDIS_RUBRIC_NAME)
    other = faithfulness_rubric("oncology care pathways", name="onc-grounding-v1")
    assert other.sha256 != P2_JUDGE_PROMPT_SHA256
    assert other.text == hedis.text.replace(HEDIS_DOMAIN, "oncology care pathways")


def test_rubric_is_deterministic_across_calls() -> None:
    first = faithfulness_rubric(HEDIS_DOMAIN, name=HEDIS_RUBRIC_NAME)
    second = faithfulness_rubric(HEDIS_DOMAIN, name=HEDIS_RUBRIC_NAME)
    assert first == second
    assert first.sha256 == second.sha256


# --- GroundingVerdict -----------------------------------------------------------------------


def test_faithfulness_is_supported_over_total() -> None:
    verdict = GroundingVerdict(
        claims_total=4,
        claims_supported=1,
        claims_unsupported=2,
        claims_contradicted=1,
        citation_valid_ratio=None,
    )
    assert verdict.faithfulness == 0.25


def test_faithfulness_is_none_when_no_claims() -> None:
    verdict = GroundingVerdict(
        claims_total=0,
        claims_supported=0,
        claims_unsupported=0,
        claims_contradicted=0,
        citation_valid_ratio=None,
    )
    assert verdict.faithfulness is None
    assert verdict.notes == ""


# --- parse_grounding_verdict ----------------------------------------------------------------


def test_parse_canned_json_gives_exact_counts_and_citation_ratio() -> None:
    _assert_canned(parse_grounding_verdict(CANNED_JSON))


def test_parse_tolerates_prose_and_fences_around_json() -> None:
    raw = f"Sure, here is my grading:\n```json\n{CANNED_JSON}\n```\nAnything else?"
    _assert_canned(parse_grounding_verdict(raw))


def test_parse_empty_claims_gives_none_faithfulness_and_ratio() -> None:
    verdict = parse_grounding_verdict('{"claims": [], "notes": ""}')
    assert verdict.claims_total == 0
    assert verdict.claims_supported == 0
    assert verdict.claims_unsupported == 0
    assert verdict.claims_contradicted == 0
    assert verdict.faithfulness is None
    assert verdict.citation_valid_ratio is None


def test_parse_uncited_claims_give_none_ratio() -> None:
    raw = (
        '{"claims": [{"claim": "a", "verdict": "supported", "citation_valid": null}, '
        '{"claim": "b", "verdict": "unsupported"}]}'
    )
    verdict = parse_grounding_verdict(raw)
    assert verdict.claims_total == 2
    assert verdict.citation_valid_ratio is None
    assert verdict.faithfulness == 0.5


@pytest.mark.parametrize(
    "raw",
    ['{"claims": []}', '{"claims": [], "notes": null}', '{"claims": [], "notes": 7}'],
)
def test_parse_missing_or_non_string_notes_become_empty(raw: str) -> None:
    assert parse_grounding_verdict(raw).notes == ""


@pytest.mark.parametrize(
    ("raw", "fragment"),
    [
        ("no json here at all", "no JSON object"),
        ('{"notes": "claims key missing"}', "'claims' list"),
        ('{"claims": null}', "'claims' list"),
        ('{"claims": {"claim": "x", "verdict": "supported"}}', "'claims' list"),
        ('{"claims": ["supported"]}', "claims[0] is not an object"),
        (
            '{"claims": [{"claim": "x", "verdict": "maybe"}]}',
            "claims[0] has invalid verdict 'maybe'",
        ),
        ('{"claims": [{"claim": "x"}]}', "claims[0] has invalid verdict None"),
        (
            '{"claims": [{"claim": "a", "verdict": "supported"}, '
            '{"claim": "b", "verdict": "nope"}]}',
            "claims[1] has invalid verdict 'nope'",
        ),
        (
            '{"claims": [{"claim": "x", "verdict": "supported", "citation_valid": "yes"}]}',
            "claims[0] citation_valid must be true/false/null",
        ),
        (
            '{"claims": [{"claim": "x", "verdict": "supported", "citation_valid": 1}]}',
            "claims[0] citation_valid must be true/false/null",
        ),
    ],
)
def test_parse_rejects_structurally_invalid_verdicts(raw: str, fragment: str) -> None:
    with pytest.raises(JudgeParseError) as excinfo:
        parse_grounding_verdict(raw)
    assert fragment in str(excinfo.value)


# --- build_judge_input ----------------------------------------------------------------------


def test_build_judge_input_formatting_is_exact() -> None:
    payload = build_judge_input(
        "What is the threshold?",
        "It is 7.0% [1].",
        ["The threshold is 7.0%.", "Members aged 18-75 qualify."],
    )
    assert payload == (
        "PASSAGES:\n"
        "[1] The threshold is 7.0%.\n"
        "\n"
        "[2] Members aged 18-75 qualify.\n"
        "\n"
        "QUESTION:\n"
        "What is the threshold?\n"
        "\n"
        "ANSWER:\n"
        "It is 7.0% [1]."
    )


def test_build_judge_input_with_no_passages_keeps_section_headers() -> None:
    assert build_judge_input("q", "a", []) == "PASSAGES:\n\n\nQUESTION:\nq\n\nANSWER:\na"


def test_build_judge_input_accepts_any_sequence() -> None:
    assert build_judge_input("q", "a", ("p",)) == build_judge_input("q", "a", ["p"])


# --- verdict_metrics ------------------------------------------------------------------------


def test_verdict_metrics_full_and_key_sorted() -> None:
    verdict = GroundingVerdict(
        claims_total=4,
        claims_supported=2,
        claims_unsupported=1,
        claims_contradicted=1,
        citation_valid_ratio=0.5,
    )
    out = verdict_metrics(verdict)
    assert out == {"citation_valid_ratio": 0.5, "contradiction_rate": 0.25, "faithfulness": 0.5}
    assert list(out) == sorted(out)
    assert all(isinstance(value, float) for value in out.values())


def test_verdict_metrics_omits_claim_ratios_when_no_claims() -> None:
    verdict = GroundingVerdict(
        claims_total=0,
        claims_supported=0,
        claims_unsupported=0,
        claims_contradicted=0,
        citation_valid_ratio=None,
    )
    out = verdict_metrics(verdict)
    assert "faithfulness" not in out
    assert "contradiction_rate" not in out
    assert out == {}


def test_verdict_metrics_omits_citation_ratio_when_nothing_cited() -> None:
    verdict = GroundingVerdict(
        claims_total=2,
        claims_supported=1,
        claims_unsupported=1,
        claims_contradicted=0,
        citation_valid_ratio=None,
    )
    out = verdict_metrics(verdict)
    assert "citation_valid_ratio" not in out
    assert out == {"contradiction_rate": 0.0, "faithfulness": 0.5}


def test_verdict_metrics_from_parsed_canned_verdict() -> None:
    out = verdict_metrics(parse_grounding_verdict(CANNED_JSON))
    assert out == {"citation_valid_ratio": 0.5, "contradiction_rate": 0.25, "faithfulness": 0.5}
