"""Stamped, byte-deterministic eval artifacts and README table sync.

Numbers are generated from committed artifacts, never hand-typed: ``build_artifact`` writes
only ``EvalReport.test_overall`` as ``metrics.overall``, ``write_artifact`` serializes
deterministically, and ``readme_in_sync`` is the CI byte-check as one pure call. Every file
write uses ``newline="\\n"`` so Windows and Linux produce identical bytes.
"""

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from clinevals.runner import EvalReport

EVAL_BEGIN = "<!-- EVAL:BEGIN -->"
EVAL_END = "<!-- EVAL:END -->"


class ReportError(RuntimeError):
    """The artifact or README does not have the shape reporting requires."""


def write_artifact(path: Path, payload: Mapping[str, object]) -> None:
    """Deterministic JSON: sorted keys, 2-space indent, non-ASCII preserved, trailing newline,
    LF line endings on every OS. Never invents data — stamps come from the caller."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def sha256_file(path: Path, *, short: int | None = None) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest[:short] if short else digest


def detect_git_sha(repo_root: Path | None = None) -> str:
    """``git rev-parse HEAD``; RuntimeError outside a repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607 — git resolved via PATH by design
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError("not inside a git repository (or git unavailable)") from exc
    return result.stdout.strip()


class RunStamp(BaseModel):
    model_config = ConfigDict(frozen=True)

    git_sha: str
    date: str
    """ISO date, no time-of-day — reruns within a day stay byte-identical."""
    config_hash: str | None = None
    dataset_hash: str | None = None
    models: dict[str, str] = {}
    """e.g. ``{"answer_model": ..., "judge_model": ...}``."""
    rubric_sha256: str | None = None
    judge_human_agreement: float | None = None
    extra: dict[str, str] = {}


def _counts_block(counts: Mapping[str, object]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for key, value in counts.items():
        tp = getattr(value, "tp", None)
        fp = getattr(value, "fp", None)
        fn = getattr(value, "fn", None)
        if isinstance(tp, int) and isinstance(fp, int) and isinstance(fn, int):
            out[key] = {"fn": fn, "fp": fp, "tp": tp}
    return out


def build_artifact(
    report: EvalReport,
    stamp: RunStamp,
    *,
    tier: str,
    extra: Mapping[str, object] | None = None,
    include_per_item: bool = False,
) -> dict[str, object]:
    """Fixed artifact shape. ``metrics.overall`` IS ``report.test_overall`` — publishing
    dev-contaminated numbers would require hand-building a payload."""
    payload: dict[str, object] = {
        "tier": tier,
        "date": stamp.date,
        "git_sha": stamp.git_sha,
        "item_count": len(report.per_item),
        "note": "metrics.overall is the test split; tuning uses dev only",
        "metrics": {
            "overall": report.test_overall,
            "per_category": report.per_category,
            "per_split": report.per_split,
        },
    }
    if stamp.config_hash is not None:
        payload["config_hash"] = stamp.config_hash
    if stamp.dataset_hash is not None:
        payload["dataset_hash"] = stamp.dataset_hash
    for model_key, model_id in sorted(stamp.models.items()):
        payload[model_key] = model_id
    if stamp.rubric_sha256 is not None:
        payload["rubric_sha256"] = stamp.rubric_sha256
    if stamp.judge_human_agreement is not None:
        payload["judge_human_agreement"] = stamp.judge_human_agreement
    for key, value in sorted(stamp.extra.items()):
        payload[key] = value
    counts: dict[str, object] = {}
    if report.counts_per_category:
        counts["per_category"] = _counts_block(report.counts_per_category)
    if report.counts_per_split:
        counts["per_split"] = _counts_block(report.counts_per_split)
    if counts:
        payload["counts"] = counts
    if include_per_item:
        payload["per_item"] = [item.model_dump() for item in report.per_item]
    if extra:
        for extra_key, extra_value in extra.items():
            payload[extra_key] = extra_value
    return payload


def _format_value(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return str(value)
    return f"{value:.3f}"


def render_metrics_table(
    artifact: Mapping[str, object],
    *,
    comparison: tuple[str, str] | None = None,
    primary_label: str = "Value",
) -> str:
    """Markdown table from ``artifact["metrics"]`` (the ``overall`` block or a flat mapping)
    plus an italic stamp line.

    ``comparison=(column_label, metrics_key)`` renders a second column from
    ``artifact["metrics"][metrics_key]`` (``primary_label`` names the first column; P2's
    two-column table is reproduced byte-for-byte with ``("Dense-only", "dense_only_overall")``
    and ``primary_label="Hybrid"``).
    """
    metrics = artifact.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ReportError("artifact has no 'metrics' mapping")
    overall_raw = metrics.get("overall", metrics)
    if not isinstance(overall_raw, Mapping) or not overall_raw:
        raise ReportError("artifact metrics carry no overall values")
    comparison_raw: Mapping[str, object] | None = None
    if comparison is not None:
        candidate = metrics.get(comparison[1])
        if isinstance(candidate, Mapping) and candidate:
            comparison_raw = candidate
    if comparison is not None and comparison_raw is not None:
        lines = [f"| Metric (test split) | {primary_label} | {comparison[0]} |", "|---|---|---|"]
        lines.extend(
            f"| {key} | {_format_value(overall_raw[key])} | "
            f"{_format_value(comparison_raw.get(key, '—'))} |"
            for key in sorted(overall_raw)
        )
    else:
        lines = [f"| Metric | {primary_label} |", "|---|---|"]
        lines.extend(
            f"| {key} | {_format_value(overall_raw[key])} |" for key in sorted(overall_raw)
        )
    stamp_keys = ("git_sha", "config_hash", "dataset_hash", "answer_model", "judge_model")
    stamps = [f"{key}={artifact[key]}" for key in stamp_keys if key in artifact]
    if stamps:
        lines.extend(["", f"_Stamps: {' · '.join(stamps)}_"])
    return "\n".join(lines)


def _synced_text(
    text: str,
    artifact: Mapping[str, object],
    comparison: tuple[str, str] | None,
    primary_label: str,
    readme_path: Path,
) -> str:
    begin = text.find(EVAL_BEGIN)
    end = text.find(EVAL_END)
    if begin == -1 or end == -1 or end < begin:
        raise ReportError(
            f"README markers {EVAL_BEGIN} / {EVAL_END} missing or out of order in {readme_path}"
        )
    table = render_metrics_table(artifact, comparison=comparison, primary_label=primary_label)
    return text[: begin + len(EVAL_BEGIN)] + "\n" + table + "\n" + text[end:]


def sync_readme(
    readme_path: Path,
    artifact: Mapping[str, object],
    *,
    comparison: tuple[str, str] | None = None,
    primary_label: str = "Value",
) -> str:
    """Replace the region between the EVAL markers; idempotent; LF endings; returns the text."""
    text = readme_path.read_text(encoding="utf-8")
    updated = _synced_text(text, artifact, comparison, primary_label, readme_path)
    with readme_path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(updated)
    return updated


def readme_in_sync(
    readme_path: Path,
    artifact: Mapping[str, object],
    *,
    comparison: tuple[str, str] | None = None,
    primary_label: str = "Value",
) -> bool:
    """Pure check: would :func:`sync_readme` change the file? The CI byte-check as one call."""
    text = readme_path.read_text(encoding="utf-8")
    return _synced_text(text, artifact, comparison, primary_label, readme_path) == text
