"""Keyless test doubles. With :mod:`clinevals.judge`, the only langchain-core importer."""

import json
from collections.abc import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from clinevals.grounding import GroundingVerdict


def _verdict_json(verdict: GroundingVerdict) -> str:
    """Serialize a verdict back into the judge's output schema (claims list + notes)."""
    # Fresh dicts per claim: list multiplication would alias one dict across a bucket and
    # the citation flags below would all collapse onto it.
    claims: list[dict[str, object]] = [
        {"claim": "c", "verdict": "supported"} for _ in range(verdict.claims_supported)
    ]
    claims += [{"claim": "c", "verdict": "unsupported"} for _ in range(verdict.claims_unsupported)]
    claims += [
        {"claim": "c", "verdict": "contradicted"} for _ in range(verdict.claims_contradicted)
    ]
    if verdict.citation_valid_ratio is not None and claims:
        # Encode the ratio over the available claims as closely as integer flags allow.
        valid_count = round(verdict.citation_valid_ratio * len(claims))
        for i, claim in enumerate(claims):
            claim["citation_valid"] = i < valid_count
    return json.dumps({"claims": claims, "notes": verdict.notes})


def scripted_judge(outputs: Sequence[str | GroundingVerdict]) -> BaseChatModel:
    """A ``GenericFakeChatModel`` returning each output in order.

    ``GroundingVerdict`` entries are serialized to correct verdict JSON; ``str`` entries pass
    through raw (so a garbage string exercises the ``JudgeParseError`` path). Keyless judge
    plumbing tests in every consumer, one line each.
    """
    scripted = [o if isinstance(o, str) else _verdict_json(o) for o in outputs]
    return GenericFakeChatModel(messages=iter(scripted))
