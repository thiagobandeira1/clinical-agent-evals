"""Tests for ``clinevals.dataset``: JSONL -> typed items, all violations reported at once.

Keyless: pure file I/O against ``tmp_path``. ``QItem`` plays the consumer-subclass role.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import get_args

import pytest
from pydantic import Field, ValidationError

from clinevals.dataset import DatasetError, EvalItemBase, Split, load_jsonl


class QItem(EvalItemBase):
    """A consumer-style item: base plumbing plus task fields (one required, one defaulted)."""

    question: str
    gold: list[str] = Field(default_factory=list)


LINE_Q1 = (
    '{"item_id": "q1", "category": "dosing", "split": "dev", '
    '"question": "How much?", "gold": ["p1", "p2"]}'
)
LINE_Q2 = '{"item_id": "q2", "category": "screening", "split": "test", "question": "When?"}'
LINE_Q2_DUPLICATING_Q1 = LINE_Q2.replace('"q2"', '"q1"')
LINE_BAD_JSON = "{not json"
LINE_MISSING_SPLIT = '{"item_id": "q3", "category": "dosing", "question": "How much?"}'
LINE_HOLDOUT_SPLIT = '{"item_id": "q4", "category": "dosing", "split": "holdout", "question": "?"}'


def _write(path: Path, *lines: str) -> Path:
    """Write ``lines`` LF-joined as explicit bytes so the fixture is identical on every OS."""
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    return path


# --- happy path -----------------------------------------------------------------------


def test_load_jsonl_happy_path(tmp_path: Path) -> None:
    path = _write(tmp_path / "gold.jsonl", LINE_Q1, LINE_Q2)

    items = load_jsonl(path, QItem)

    assert isinstance(items, list)
    assert all(isinstance(item, QItem) for item in items)
    assert items == [
        QItem(
            item_id="q1", category="dosing", split="dev", question="How much?", gold=["p1", "p2"]
        ),
        QItem(item_id="q2", category="screening", split="test", question="When?"),
    ]
    assert items[1].gold == []  # defaulted task field


def test_empty_file_yields_empty_list(tmp_path: Path) -> None:
    assert load_jsonl(_write(tmp_path / "gold.jsonl"), QItem) == []


def test_blank_and_whitespace_only_lines_are_skipped(tmp_path: Path) -> None:
    path = _write(tmp_path / "gold.jsonl", "", LINE_Q1, "   ", "\t", LINE_Q2, "")

    assert [item.item_id for item in load_jsonl(path, QItem)] == ["q1", "q2"]


def test_crlf_file_loads_identically_to_lf_file(tmp_path: Path) -> None:
    """A gold set authored on Windows (CRLF) must load exactly like the LF-authored one."""
    crlf = tmp_path / "crlf.jsonl"
    crlf.write_bytes(("\r\n".join([LINE_Q1, LINE_Q2]) + "\r\n").encode("utf-8"))
    lf = _write(tmp_path / "lf.jsonl", LINE_Q1, LINE_Q2)

    assert load_jsonl(crlf, QItem) == load_jsonl(lf, QItem)


def test_loaded_items_are_frozen(tmp_path: Path) -> None:
    item = load_jsonl(_write(tmp_path / "gold.jsonl", LINE_Q1), QItem)[0]

    with pytest.raises(ValidationError):
        item.category = "changed"


def test_split_literal_is_exactly_dev_and_test() -> None:
    assert get_args(Split) == ("dev", "test")


def test_dataset_error_is_a_value_error() -> None:
    assert issubclass(DatasetError, ValueError)


# --- phase 1: per-line failures with name:lineno context ------------------------------


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"nope\.jsonl"):
        load_jsonl(tmp_path / "nope.jsonl", QItem)


def test_invalid_json_line_reports_name_and_lineno(tmp_path: Path) -> None:
    path = _write(tmp_path / "gold.jsonl", LINE_Q1, LINE_BAD_JSON)

    with pytest.raises(DatasetError) as excinfo:
        load_jsonl(path, QItem)

    assert "gold.jsonl:2: invalid JSON: " in str(excinfo.value)


def test_all_bad_lines_are_reported_at_once(tmp_path: Path) -> None:
    """All-violations-at-once holds for line-level problems too, not just validators."""
    path = _write(tmp_path / "gold.jsonl", LINE_BAD_JSON, LINE_Q1, LINE_MISSING_SPLIT)

    with pytest.raises(DatasetError) as excinfo:
        load_jsonl(path, QItem)

    message = str(excinfo.value)
    assert message.startswith("gold set invalid:")
    assert "gold.jsonl:1: invalid JSON: " in message
    assert "gold.jsonl:3: " in message and "split" in message


def test_lineno_counts_physical_lines_including_blank_ones(tmp_path: Path) -> None:
    """Line numbers are physical file lines, so an editor's goto-line lands on the culprit."""
    path = _write(tmp_path / "gold.jsonl", LINE_Q1, "", "", LINE_BAD_JSON)

    with pytest.raises(DatasetError) as excinfo:
        load_jsonl(path, QItem)

    assert "gold.jsonl:4: invalid JSON: " in str(excinfo.value)


def test_model_invalid_line_missing_split_reports_lineno(tmp_path: Path) -> None:
    path = _write(tmp_path / "gold.jsonl", LINE_Q1, LINE_MISSING_SPLIT)

    with pytest.raises(DatasetError) as excinfo:
        load_jsonl(path, QItem)

    message = str(excinfo.value)
    assert "gold.jsonl:2: " in message
    assert "split" in message


def test_split_literal_rejects_holdout_in_file(tmp_path: Path) -> None:
    path = _write(tmp_path / "gold.jsonl", LINE_Q1, LINE_HOLDOUT_SPLIT)

    with pytest.raises(DatasetError) as excinfo:
        load_jsonl(path, QItem)

    message = str(excinfo.value)
    assert "gold.jsonl:2: " in message
    assert "split" in message
    assert "holdout" in message


def test_split_literal_rejects_holdout_on_the_model() -> None:
    with pytest.raises(ValidationError) as excinfo:
        QItem.model_validate(
            {"item_id": "q1", "category": "dosing", "split": "holdout", "question": "?"}
        )

    assert [error["loc"] for error in excinfo.value.errors()] == [("split",)]


@pytest.mark.parametrize("split", ["dev", "test"])
def test_split_literal_accepts_dev_and_test(split: str) -> None:
    item = QItem.model_validate(
        {"item_id": "q1", "category": "dosing", "split": split, "question": "?"}
    )

    assert item.split == split


def test_phase_one_failure_stops_before_any_validator_runs(tmp_path: Path) -> None:
    """A malformed line aborts the load; no validator ever sees a partial list."""
    calls: list[int] = []

    def spy(items: Sequence[QItem]) -> list[str]:
        calls.append(len(items))
        return []

    path = _write(tmp_path / "gold.jsonl", LINE_Q1, LINE_BAD_JSON)

    with pytest.raises(DatasetError):
        load_jsonl(path, QItem, validators=[spy])

    assert calls == []


# --- phase 2: whole-file invariants, aggregated into ONE error ------------------------


def test_duplicate_item_id_is_a_dataset_error(tmp_path: Path) -> None:
    path = _write(tmp_path / "gold.jsonl", LINE_Q1, LINE_Q2_DUPLICATING_Q1)

    with pytest.raises(DatasetError) as excinfo:
        load_jsonl(path, QItem)

    assert str(excinfo.value) == "gold set invalid:\n  q1: duplicate item_id"


def test_duplicate_and_validator_problems_aggregate_into_one_error(tmp_path: Path) -> None:
    def categories_must_be_known(items: Sequence[QItem]) -> list[str]:
        known = {"dosing"}
        return [
            f"{item.item_id}: unknown category {item.category!r}"
            for item in items
            if item.category not in known
        ]

    path = _write(tmp_path / "gold.jsonl", LINE_Q1, LINE_Q2_DUPLICATING_Q1)

    with pytest.raises(DatasetError) as excinfo:
        load_jsonl(path, QItem, validators=[categories_must_be_known])

    # BOTH problems, ONE error: the duplicate check and the custom validator each report.
    assert str(excinfo.value) == (
        "gold set invalid:\n  q1: duplicate item_id\n  q1: unknown category 'screening'"
    )


def test_every_validator_runs_and_every_problem_is_listed(tmp_path: Path) -> None:
    def always_a(items: Sequence[QItem]) -> list[str]:
        return ["problem A"]

    def always_b(items: Sequence[QItem]) -> list[str]:
        return ["problem B", "problem C"]

    path = _write(tmp_path / "gold.jsonl", LINE_Q1, LINE_Q2)

    with pytest.raises(DatasetError) as excinfo:
        load_jsonl(path, QItem, validators=(always_a, always_b))

    assert str(excinfo.value) == "gold set invalid:\n  problem A\n  problem B\n  problem C"


def test_validator_receives_the_full_typed_list(tmp_path: Path) -> None:
    received: list[Sequence[QItem]] = []

    def capture(items: Sequence[QItem]) -> list[str]:
        received.append(items)
        return []

    path = _write(tmp_path / "gold.jsonl", LINE_Q1, "", LINE_Q2)

    items = load_jsonl(path, QItem, validators=[capture])

    assert len(received) == 1
    seen = received[0]
    assert isinstance(seen, list)
    assert len(seen) == 2
    assert all(isinstance(item, QItem) for item in seen)
    assert [item.item_id for item in seen] == ["q1", "q2"]
    assert list(seen) == items


def test_passing_validators_leave_the_items_unchanged(tmp_path: Path) -> None:
    def ok(items: Sequence[QItem]) -> list[str]:
        return []

    path = _write(tmp_path / "gold.jsonl", LINE_Q1, LINE_Q2)

    assert load_jsonl(path, QItem, validators=[ok, ok]) == load_jsonl(path, QItem)
