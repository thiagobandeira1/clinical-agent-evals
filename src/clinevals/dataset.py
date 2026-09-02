"""Gold-set loading: JSONL -> typed items, with all-violations-at-once validation.

Consumers subclass :class:`EvalItemBase` to add task fields and pass domain invariants as
plain validator functions. ``split`` is REQUIRED with no default: silent split assignment is
a publication hazard (test-split-only publication is the portfolio's honesty story).
"""

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

Split = Literal["dev", "test"]


class DatasetError(ValueError):
    """The gold-set file is malformed or violates an invariant; the message lists EVERY problem."""


class EvalItemBase(BaseModel):
    """The plumbing every eval item carries; task fields live on consumer subclasses."""

    model_config = ConfigDict(frozen=True)

    item_id: str
    category: str
    """Free per-project vocabulary; consumers restrict it via a validator."""
    split: Split


ItemT = TypeVar("ItemT", bound=EvalItemBase)
Validator = Callable[[Sequence[ItemT]], list[str]]
"""A dataset-level check returning problem strings (``[]`` means the check passed)."""


def load_jsonl(
    path: Path,
    item_type: type[ItemT],
    *,
    validators: Sequence[Validator[ItemT]] = (),
) -> list[ItemT]:
    """Load one ``item_type`` per non-blank line and enforce every invariant at once.

    Every problem — unparseable lines (``{name}:{lineno}:`` context), model-invalid lines,
    duplicate ``item_id``s, and each ``validators`` finding — aggregates into ONE
    :class:`DatasetError`, so a broken gold set is fixed in one pass, not one line per run.
    Validators only run when every line parsed (they receive a fully typed list). Raises
    ``FileNotFoundError`` when the file is missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"gold set not found at {path}")
    items: list[ItemT] = []
    problems: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"{path.name}:{lineno}: invalid JSON: {exc}")
            continue
        try:
            items.append(item_type.model_validate(raw))
        except ValidationError as exc:
            problems.append(f"{path.name}:{lineno}: {exc}")
    if problems:
        raise DatasetError("gold set invalid:\n  " + "\n  ".join(problems))

    seen: set[str] = set()
    for item in items:
        if item.item_id in seen:
            problems.append(f"{item.item_id}: duplicate item_id")
        seen.add(item.item_id)
    for validator in validators:
        problems.extend(validator(items))
    if problems:
        raise DatasetError("gold set invalid:\n  " + "\n  ".join(problems))
    return items
