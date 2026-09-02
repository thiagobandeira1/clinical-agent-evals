"""The one shipped verdict family: per-claim faithfulness (grounding) of an answer to passages.

Extracted from P2's proven judge contract. :func:`faithfulness_rubric` reproduces P2's frozen
``JUDGE_PROMPT`` byte-for-byte for ``domain="Medicare Star Ratings / HEDIS measures"`` — a
golden sha test in each repo enforces it, so stamped provenance never moves.
"""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from clinevals.judge import JudgeParseError, Rubric, extract_json_object

_DOMAIN_SLOT = "__DOMAIN__"

_FAITHFULNESS_TEMPLATE = """\
You are grading a RAG answer about __DOMAIN__ for faithfulness
to its retrieved passages. Judge ONLY against the numbered passages provided — outside
knowledge must never rescue an unsupported claim.

Procedure:
1. Split the ANSWER into atomic factual claims (one independently checkable fact each).
   Ignore the machine-appended disclaimer footer and pure hedging language.
2. Assign each claim exactly one verdict:
   - "supported": stated by, or directly entailed by, at least one passage.
   - "unsupported": not present in any passage.
   - "contradicted": at least one passage states the opposite.
3. Numbers, ages, dates, and thresholds count as supported only when they match a passage
   verbatim.
4. For each claim carrying inline [n] citation markers, set "citation_valid" to true only
   if at least one *cited* passage [n] itself supports the claim, false otherwise. Use
   null for a claim with no citation marker.

Output STRICT JSON only — no markdown fences, no prose — exactly this shape:
{"claims": [{"claim": "<text>", "verdict": "supported|unsupported|contradicted",
"citation_valid": true|false|null}], "notes": "<one line on anything odd, or empty>"}
"""


def faithfulness_rubric(domain: str, *, name: str) -> Rubric:
    """The per-claim faithfulness rubric with exactly the domain phrase parameterized."""
    return Rubric(name=name, text=_FAITHFULNESS_TEMPLATE.replace(_DOMAIN_SLOT, domain))


class GroundingVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    claims_total: int
    claims_supported: int
    claims_unsupported: int
    claims_contradicted: int
    citation_valid_ratio: float | None
    """Mean of citation_valid over cited claims; None when no claim carried a citation."""
    notes: str = ""

    @property
    def faithfulness(self) -> float | None:
        """supported / total; None when the judge found no claims to grade."""
        if self.claims_total == 0:
            return None
        return self.claims_supported / self.claims_total


def build_judge_input(question: str, answer_text: str, passages: Sequence[str]) -> str:
    """Format the human-turn payload: numbered passages, then question, then answer."""
    numbered = "\n\n".join(f"[{i}] {passage}" for i, passage in enumerate(passages, start=1))
    return f"PASSAGES:\n{numbered}\n\nQUESTION:\n{question}\n\nANSWER:\n{answer_text}"


def parse_grounding_verdict(raw: str) -> GroundingVerdict:
    """Parse the judge's raw text; tolerant of surrounding prose, strict on structure.

    Raises :class:`JudgeParseError` on garbage: no JSON, a missing/non-list ``claims``, an
    unknown verdict value, or a non-boolean ``citation_valid``.
    """
    payload = extract_json_object(raw)
    claims_raw = payload.get("claims")
    if not isinstance(claims_raw, list):
        raise JudgeParseError("judge output JSON lacks a 'claims' list")
    supported = unsupported = contradicted = 0
    citation_flags: list[bool] = []
    for i, entry in enumerate(claims_raw):
        if not isinstance(entry, dict):
            raise JudgeParseError(f"claims[{i}] is not an object")
        verdict = entry.get("verdict")
        if verdict == "supported":
            supported += 1
        elif verdict == "unsupported":
            unsupported += 1
        elif verdict == "contradicted":
            contradicted += 1
        else:
            raise JudgeParseError(f"claims[{i}] has invalid verdict {verdict!r}")
        citation_valid = entry.get("citation_valid")
        if isinstance(citation_valid, bool):
            citation_flags.append(citation_valid)
        elif citation_valid is not None:
            raise JudgeParseError(
                f"claims[{i}] citation_valid must be true/false/null, got {citation_valid!r}"
            )
    notes_raw = payload.get("notes")
    ratio = sum(citation_flags) / len(citation_flags) if citation_flags else None
    return GroundingVerdict(
        claims_total=len(claims_raw),
        claims_supported=supported,
        claims_unsupported=unsupported,
        claims_contradicted=contradicted,
        citation_valid_ratio=ratio,
        notes=notes_raw if isinstance(notes_raw, str) else "",
    )


def verdict_metrics(verdict: GroundingVerdict) -> dict[str, float]:
    """Scorer-ready sparse metrics; each key ABSENT (not 0.0) when undefined."""
    out: dict[str, float] = {}
    if verdict.claims_total > 0:
        out["contradiction_rate"] = verdict.claims_contradicted / verdict.claims_total
        out["faithfulness"] = verdict.claims_supported / verdict.claims_total
    if verdict.citation_valid_ratio is not None:
        out["citation_valid_ratio"] = verdict.citation_valid_ratio
    return dict(sorted(out.items()))
