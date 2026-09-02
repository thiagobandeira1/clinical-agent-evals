"""Tests for :mod:`clinevals.judge`: rubric provenance, tolerant JSON extraction, invocation.

Keyless by construction: every model here is a langchain fake. No network, no API keys.
"""

import hashlib
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, ValidationError

from clinevals.fakes import scripted_judge
from clinevals.judge import JudgeParseError, Rubric, extract_json_object, invoke_judge


class _RecordingModel(BaseChatModel):
    """A minimal ``BaseChatModel`` that records every message list it is invoked with."""

    calls: list[list[BaseMessage]] = Field(default_factory=list)
    reply: str = "ok"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])

    @property
    def _llm_type(self) -> str:
        return "recording-fake"


# --- Rubric ---------------------------------------------------------------------------------


def test_rubric_sha256_is_sha256_of_text() -> None:
    rubric = Rubric(name="r-v1", text="Grade the answer.")
    assert rubric.sha256 == hashlib.sha256(b"Grade the answer.").hexdigest()
    assert len(rubric.sha256) == 64


def test_rubric_sha256_depends_on_text_only() -> None:
    same_text_a = Rubric(name="a", text="same")
    same_text_b = Rubric(name="b", text="same")
    trailing_space = Rubric(name="a", text="same ")
    assert same_text_a.sha256 == same_text_b.sha256
    assert same_text_a.sha256 != trailing_space.sha256


def test_rubric_sha256_hashes_utf8_bytes() -> None:
    text = "naïve façade: judge only the passages"
    expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert Rubric(name="u", text=text).sha256 == expected


def test_rubric_name_is_carried() -> None:
    assert Rubric(name="hedis-grounding-v1", text="t").name == "hedis-grounding-v1"


def test_rubric_is_frozen() -> None:
    rubric = Rubric(name="r-v1", text="immutable")
    with pytest.raises(ValidationError):
        rubric.text = "edited in place"
    assert rubric.text == "immutable"


# --- extract_json_object --------------------------------------------------------------------


def test_extract_plain_json_object() -> None:
    raw = '{"claims": [], "notes": "x"}'
    assert extract_json_object(raw) == {"claims": [], "notes": "x"}


def test_extract_tolerates_surrounding_whitespace() -> None:
    assert extract_json_object('  \n\t {"a": 1}\n\n') == {"a": 1}


def test_extract_json_wrapped_in_prose() -> None:
    raw = 'Sure! Here is my grading: {"claims": [], "notes": ""} Let me know if you need more.'
    assert extract_json_object(raw) == {"claims": [], "notes": ""}


def test_extract_json_wrapped_in_markdown_fences() -> None:
    raw = 'Here you go:\n```json\n{"claims": [{"claim": "c", "verdict": "supported"}]}\n```\n'
    assert extract_json_object(raw) == {"claims": [{"claim": "c", "verdict": "supported"}]}


def test_extract_first_valid_object_wins_when_two_present() -> None:
    assert extract_json_object('{"first": 1} {"second": 2}') == {"first": 1}


def test_extract_skips_an_undecodable_brace_before_a_valid_object() -> None:
    raw = 'note {not json} and then {"ok": true}'
    assert extract_json_object(raw) == {"ok": True}


def test_extract_returns_outermost_object_not_nested_one() -> None:
    assert extract_json_object('{"outer": {"inner": 1}}') == {"outer": {"inner": 1}}


def test_extract_object_embedded_in_array_is_found() -> None:
    # "embedded anywhere": the first decodable object wins even inside an enclosing array.
    assert extract_json_object('[{"a": 1}, {"b": 2}]') == {"a": 1}


def test_extract_returns_a_plain_dict() -> None:
    assert type(extract_json_object('{"a": [1, {"b": null}]}')) is dict


@pytest.mark.parametrize(
    "raw",
    ["", "no json here", "{", "{broken", '{"unterminated": ', "}{", "{'single': 'quotes'}"],
)
def test_extract_garbage_raises_judge_parse_error(raw: str) -> None:
    with pytest.raises(JudgeParseError, match="no JSON object"):
        extract_json_object(raw)


@pytest.mark.parametrize("raw", ["[1, 2, 3]", '["a", "b"]', "[]", "42", '"just a string"'])
def test_extract_non_object_json_alone_raises(raw: str) -> None:
    with pytest.raises(JudgeParseError):
        extract_json_object(raw)


def test_judge_parse_error_is_a_value_error() -> None:
    assert issubclass(JudgeParseError, ValueError)


# --- invoke_judge ---------------------------------------------------------------------------


def test_invoke_judge_returns_scripted_string_verbatim() -> None:
    scripted = "  not json at all {oops}\n"
    model = scripted_judge([scripted])
    assert invoke_judge(model, Rubric(name="r", text="t"), "payload") == scripted


def test_invoke_judge_sends_rubric_text_as_system_and_payload_as_human() -> None:
    model = _RecordingModel(reply="verdict text")
    rubric = Rubric(name="r-v1", text="You are grading a RAG answer.")
    payload = "PASSAGES:\n[1] p\n\nQUESTION:\nq\n\nANSWER:\na"

    assert invoke_judge(model, rubric, payload) == "verdict text"

    assert len(model.calls) == 1
    assert len(model.calls[0]) == 2
    system, human = model.calls[0]
    assert isinstance(system, SystemMessage)
    assert system.content == rubric.text
    assert isinstance(human, HumanMessage)
    assert human.content == payload


def test_invoke_judge_does_not_parse_or_retry() -> None:
    model = _RecordingModel(reply="garbage, not JSON")
    assert invoke_judge(model, Rubric(name="r", text="t"), "p") == "garbage, not JSON"
    assert len(model.calls) == 1


def test_invoke_judge_concatenates_multi_part_text_content() -> None:
    parts: list[str | dict[Any, Any]] = [
        {"type": "text", "text": '{"claims": '},
        "[]",
        {"type": "image_url", "image_url": {"url": "ignored"}},
        {"type": "text", "text": 7},
        {"type": "text", "text": "}"},
    ]
    model = GenericFakeChatModel(messages=iter([AIMessage(content=parts)]))
    assert invoke_judge(model, Rubric(name="r", text="t"), "p") == '{"claims": []}'
