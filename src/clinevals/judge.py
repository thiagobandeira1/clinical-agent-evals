"""Generic LLM-judge machinery. With :mod:`clinevals.fakes`, the only langchain-core importer.

No network code lives here: the model arrives fully built as a ``BaseChatModel`` (CI injects
a fake). This module owns the immutable rubric type, one invocation function, and the
tolerant JSON extraction every verdict parser is built on.
"""

import hashlib
import json
from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict


class Rubric(BaseModel):
    """A frozen judging prompt. Released texts are immutable; evolution is a new ``name``."""

    model_config = ConfigDict(frozen=True)

    name: str
    """Carries the version: ``'hedis-grounding-v1'``."""
    text: str

    @property
    def sha256(self) -> str:
        """Stamped into artifacts so prompt drift is visible, never silent."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


class JudgeParseError(ValueError):
    """The judge's output carried no parseable, structurally valid verdict JSON."""


def extract_json_object(raw: str) -> dict[str, Any]:
    """Return the first valid JSON object embedded anywhere in ``raw``.

    Tolerates prose or markdown fences around the JSON by attempting a decode at every
    ``{`` until one parses. Raises :class:`JudgeParseError` when nothing does.
    """
    decoder = json.JSONDecoder()
    idx = raw.find("{")
    while idx != -1:
        try:
            obj, _ = decoder.raw_decode(raw, idx)
        except json.JSONDecodeError:
            idx = raw.find("{", idx + 1)
            continue
        if isinstance(obj, dict):
            return cast(dict[str, Any], obj)
        idx = raw.find("{", idx + 1)
    raise JudgeParseError("no JSON object found in judge output")


def invoke_judge(model: BaseChatModel, rubric: Rubric, payload: str) -> str:
    """``model.invoke([system=rubric.text, human=payload])`` -> response text.

    No retries, no batching, no parsing — consumers loop and parse (with
    :func:`clinevals.grounding.parse_grounding_verdict` or their own model over
    :func:`extract_json_object`).
    """
    response = model.invoke([SystemMessage(content=rubric.text), HumanMessage(content=payload)])
    content = response.content
    if isinstance(content, str):
        return content
    # Multi-part content: concatenate the text parts.
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and isinstance(part.get("text"), str):
            parts.append(part["text"])
    return "".join(parts)
