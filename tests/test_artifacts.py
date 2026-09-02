"""Tests for ``clinevals.artifacts``: deterministic writes, stamps, tables, README sync."""

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from clinevals.artifacts import (
    EVAL_BEGIN,
    EVAL_END,
    ReportError,
    RunStamp,
    build_artifact,
    detect_git_sha,
    readme_in_sync,
    render_metrics_table,
    sha256_file,
    sync_readme,
    write_artifact,
)
from clinevals.classify import set_confusion
from clinevals.dataset import EvalItemBase
from clinevals.runner import EvalReport, ItemScore, score_items

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HEX40 = re.compile(r"^[0-9a-f]{40}$")

# --- fixtures ----------------------------------------------------------------------------


class _Item(EvalItemBase):
    predicted: tuple[str, ...] = ()
    gold: tuple[str, ...] = ()
    value: float = 0.0


_ITEMS: tuple[_Item, ...] = (
    _Item(item_id="d1", category="gap", split="dev", predicted=("a",), gold=("a", "b"), value=0.2),
    _Item(
        item_id="t1", category="gap", split="test", predicted=("a", "b"), gold=("a", "b"), value=0.8
    ),
    _Item(item_id="t2", category="no-gap", split="test", predicted=("c",), gold=(), value=1.0),
)


def _score_with_counts(item: _Item) -> ItemScore | None:
    return ItemScore(
        metrics={"value": item.value},
        counts=set_confusion(set(item.predicted), set(item.gold)),
    )


def _score_without_counts(item: _Item) -> ItemScore | None:
    return ItemScore(metrics={"value": item.value})


def _report() -> EvalReport:
    return score_items(_ITEMS, _score_with_counts)


_FULL_STAMP = RunStamp(
    git_sha="deadbeef",
    date="2026-01-01",
    config_hash="cfg-0001",
    dataset_hash="ds-0001",
    models={"judge_model": "judge-x", "answer_model": "answer-x"},
    rubric_sha256="ab" * 32,
    judge_human_agreement=0.9,
    extra={"run_label": "unit"},
)
_MINIMAL_STAMP = RunStamp(git_sha="deadbeef", date="2026-01-01")

_STAMPED_ARTIFACT: dict[str, object] = {
    "tier": "keyless",
    "git_sha": "deadbeef",
    "answer_model": "answer-x",
    "metrics": {"overall": {"z_metric": 1.0, "a_metric": 0.5}},
}

_SINGLE_COLUMN_TABLE = (
    "| Metric | Value |\n"
    "|---|---|\n"
    "| a_metric | 0.500 |\n"
    "| z_metric | 1.000 |\n"
    "\n"
    "_Stamps: git_sha=deadbeef · answer_model=answer-x_"
)

_README_TEMPLATE = (
    f"# Consumer\n\nIntro.\n\n{EVAL_BEGIN}\n| stale | table |\n{EVAL_END}\n\nTrailing prose.\n"
)


def _write_readme(tmp_path: Path, text: str = _README_TEMPLATE) -> Path:
    readme = tmp_path / "README.md"
    readme.write_bytes(text.encode("utf-8"))
    return readme


def _region(text: str) -> str:
    """The text strictly between the two markers."""
    begin = text.index(EVAL_BEGIN) + len(EVAL_BEGIN)
    end = text.index(EVAL_END)
    return text[begin:end]


# --- write_artifact ----------------------------------------------------------------------


def test_write_artifact_sorted_keys_indent_unicode_and_trailing_newline(tmp_path: Path) -> None:
    payload = {"b": 1, "a": {"z": [1, 2], "y": "é"}, "c": True, "d": None, "e": 0.5}
    expected = (
        "{\n"
        '  "a": {\n'
        '    "y": "é",\n'
        '    "z": [\n'
        "      1,\n"
        "      2\n"
        "    ]\n"
        "  },\n"
        '  "b": 1,\n'
        '  "c": true,\n'
        '  "d": null,\n'
        '  "e": 0.5\n'
        "}\n"
    )
    out = tmp_path / "artifact.json"
    write_artifact(out, payload)
    data = out.read_bytes()
    assert data == expected.encode("utf-8")
    assert data.endswith(b"}\n")
    assert b"\\u00e9" not in data  # ensure_ascii=False


def test_write_artifact_has_no_carriage_returns_on_this_os(tmp_path: Path) -> None:
    out = tmp_path / "artifact.json"
    write_artifact(out, {"metrics": {"overall": {"x": 1.0}}, "note": "multi\nline"})
    data = out.read_bytes()
    assert b"\r" not in data  # the latent Windows \r\n bug fixed by newline="\n"
    assert data.count(b"\n") >= 5


def test_write_artifact_creates_parent_directories(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "deeper" / "artifact.json"
    write_artifact(out, {"k": "v"})
    assert out.read_bytes() == b'{\n  "k": "v"\n}\n'


def test_write_artifact_rewrite_is_byte_identical(tmp_path: Path) -> None:
    payload = build_artifact(_report(), _FULL_STAMP, tier="keyless", include_per_item=True)
    out = tmp_path / "artifact.json"
    write_artifact(out, payload)
    first = out.read_bytes()
    rebuilt = build_artifact(_report(), _FULL_STAMP, tier="keyless", include_per_item=True)
    write_artifact(out, rebuilt)
    assert out.read_bytes() == first


# --- sha256_file -------------------------------------------------------------------------


def test_sha256_file_full_and_short(tmp_path: Path) -> None:
    target = tmp_path / "blob.bin"
    target.write_bytes(b"hello\n")
    full = "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"
    assert hashlib.sha256(b"hello\n").hexdigest() == full
    assert sha256_file(target) == full
    assert len(sha256_file(target)) == 64
    short = sha256_file(target, short=16)
    assert len(short) == 16
    assert short == full[:16]


# --- detect_git_sha ----------------------------------------------------------------------


def test_detect_git_sha_inside_this_repo_returns_40_hex_chars() -> None:
    if shutil.which("git") is None:
        pytest.skip("git is not on PATH")
    sha = detect_git_sha(_REPO_ROOT)
    assert _HEX40.match(sha), sha


def test_detect_git_sha_raises_in_an_empty_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Stop git's upward search at tmp_path's parent so a repository that happens to enclose
    # the temp directory (e.g. a dotfiles repo in $HOME) cannot leak a sha into this test.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", tmp_path.parent.as_posix())
    with pytest.raises(RuntimeError, match="not inside a git repository"):
        detect_git_sha(tmp_path)


def test_detect_git_sha_raises_in_a_repo_without_commits(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git is not on PATH")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    with pytest.raises(RuntimeError, match="not inside a git repository"):
        detect_git_sha(tmp_path)


def test_detect_git_sha_honours_repo_root_argument(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git is not on PATH")
    git = [
        "git",
        "-c",
        "user.name=clinevals-tests",
        "-c",
        "user.email=tests@example.invalid",
        "-c",
        "commit.gpgsign=false",
    ]
    subprocess.run([*git, "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "init"], cwd=tmp_path, check=True)
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True, check=True
    ).stdout.strip()
    sha = detect_git_sha(tmp_path)
    assert _HEX40.match(sha), sha
    assert sha == expected


# --- build_artifact ----------------------------------------------------------------------


def test_build_artifact_overall_is_test_overall_not_dev_contaminated_overall() -> None:
    report = _report()
    artifact = build_artifact(report, _MINIMAL_STAMP, tier="keyless")
    metrics = artifact["metrics"]
    assert isinstance(metrics, dict)
    # The fixture mixes dev (0.2) and test (0.8, 1.0) items, so the two blocks must differ.
    assert report.overall["value"] == pytest.approx(2.0 / 3.0)
    assert report.test_overall["value"] == pytest.approx(0.9)
    assert report.overall != report.test_overall
    assert metrics["overall"] == report.test_overall
    assert metrics["overall"] != report.overall
    # test_overall merges micro P/R/F1 over the test-split counts.
    assert metrics["overall"]["micro_precision"] == pytest.approx(2.0 / 3.0)
    assert metrics["overall"]["micro_recall"] == 1.0
    assert metrics["overall"]["micro_f1"] == pytest.approx(0.8)
    assert list(metrics["overall"]) == ["micro_f1", "micro_precision", "micro_recall", "value"]
    # The per-split block still carries the test block (and dev, for diagnostics).
    assert metrics["per_split"] == report.per_split
    assert set(metrics["per_split"]) == {"dev", "test"}
    assert metrics["per_split"]["test"] == {"value": 0.9}
    assert metrics["per_category"] == report.per_category


def test_build_artifact_fixed_header_fields() -> None:
    artifact = build_artifact(_report(), _MINIMAL_STAMP, tier="keyless")
    assert artifact["tier"] == "keyless"
    assert artifact["date"] == "2026-01-01"
    assert artifact["git_sha"] == "deadbeef"
    assert artifact["item_count"] == 3
    assert artifact["note"] == "metrics.overall is the test split; tuning uses dev only"


def test_build_artifact_flattens_every_stamp_field_to_top_level() -> None:
    artifact = build_artifact(_report(), _FULL_STAMP, tier="judged")
    assert artifact["config_hash"] == "cfg-0001"
    assert artifact["dataset_hash"] == "ds-0001"
    assert artifact["answer_model"] == "answer-x"
    assert artifact["judge_model"] == "judge-x"
    assert artifact["rubric_sha256"] == "ab" * 32
    assert artifact["judge_human_agreement"] == 0.9
    assert artifact["run_label"] == "unit"
    assert "models" not in artifact
    assert "extra" not in artifact


def test_build_artifact_omits_unset_stamp_fields() -> None:
    artifact = build_artifact(_report(), _MINIMAL_STAMP, tier="keyless")
    for key in ("config_hash", "dataset_hash", "rubric_sha256", "judge_human_agreement"):
        assert key not in artifact
    assert "answer_model" not in artifact
    assert "judge_model" not in artifact


def test_build_artifact_counts_block_is_tp_fp_fn_only() -> None:
    artifact = build_artifact(_report(), _MINIMAL_STAMP, tier="keyless")
    assert artifact["counts"] == {
        "per_category": {
            "gap": {"tp": 3, "fp": 0, "fn": 1},
            "no-gap": {"tp": 0, "fp": 1, "fn": 0},
        },
        "per_split": {
            "dev": {"tp": 1, "fp": 0, "fn": 1},
            "test": {"tp": 2, "fp": 1, "fn": 0},
        },
    }


def test_build_artifact_has_no_counts_block_when_no_item_carried_counts() -> None:
    report = score_items(_ITEMS, _score_without_counts)
    artifact = build_artifact(report, _MINIMAL_STAMP, tier="keyless")
    assert "counts" not in artifact
    metrics = artifact["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["overall"] == {"value": 0.9}


def test_build_artifact_include_per_item() -> None:
    report = _report()
    assert "per_item" not in build_artifact(report, _MINIMAL_STAMP, tier="keyless")
    artifact = build_artifact(report, _MINIMAL_STAMP, tier="keyless", include_per_item=True)
    per_item = artifact["per_item"]
    assert isinstance(per_item, list)
    assert len(per_item) == 3 == artifact["item_count"]
    assert per_item[0] == {
        "item_id": "d1",
        "category": "gap",
        "split": "dev",
        "metrics": {"value": 0.2},
        "counts": {"tp": 1, "fp": 0, "fn": 1},
    }


def test_build_artifact_extra_mapping_lands_at_top_level() -> None:
    artifact = build_artifact(
        _report(), _MINIMAL_STAMP, tier="keyless", extra={"dense_only_overall": {"value": 0.4}}
    )
    assert artifact["dense_only_overall"] == {"value": 0.4}
    metrics = artifact["metrics"]
    assert isinstance(metrics, dict)
    assert "dense_only_overall" not in metrics


def test_build_artifact_is_json_serializable_by_write_artifact(tmp_path: Path) -> None:
    artifact = build_artifact(_report(), _FULL_STAMP, tier="keyless", include_per_item=True)
    out = tmp_path / "artifact.json"
    write_artifact(out, artifact)
    text = out.read_text(encoding="utf-8")
    assert text.startswith('{\n  "answer_model": "answer-x",\n')
    assert text.endswith("}\n")


# --- render_metrics_table ----------------------------------------------------------------


def test_render_single_column_table() -> None:
    assert render_metrics_table(_STAMPED_ARTIFACT) == _SINGLE_COLUMN_TABLE


def test_render_single_column_custom_primary_label_and_no_stamps() -> None:
    artifact: dict[str, object] = {"metrics": {"overall": {"m": 0.25}}}
    assert render_metrics_table(artifact, primary_label="Keyless") == (
        "| Metric | Keyless |\n|---|---|\n| m | 0.250 |"
    )


def test_render_stamp_line_follows_fixed_key_order() -> None:
    artifact: dict[str, object] = {
        "judge_model": "j",
        "answer_model": "a",
        "dataset_hash": "d",
        "config_hash": "c",
        "git_sha": "g",
        "rubric_sha256": "ignored-not-a-stamp-key",
        "metrics": {"overall": {"m": 1.0}},
    }
    assert render_metrics_table(artifact).splitlines()[-1] == (
        "_Stamps: git_sha=g · config_hash=c · dataset_hash=d · answer_model=a · judge_model=j_"
    )


def test_render_comparison_two_column_table_with_em_dash_placeholder() -> None:
    artifact: dict[str, object] = {
        "metrics": {
            "overall": {"z_metric": 1.0, "a_metric": 0.5},
            "dense_only_overall": {"a_metric": 0.4},
        }
    }
    table = render_metrics_table(
        artifact, comparison=("Dense-only", "dense_only_overall"), primary_label="Hybrid"
    )
    assert table.splitlines()[0] == "| Metric (test split) | Hybrid | Dense-only |"
    assert table == (
        "| Metric (test split) | Hybrid | Dense-only |\n"
        "|---|---|---|\n"
        "| a_metric | 0.500 | 0.400 |\n"
        "| z_metric | 1.000 | — |"
    )


def test_render_comparison_rows_follow_primary_keys_only() -> None:
    artifact: dict[str, object] = {
        "metrics": {
            "overall": {"a": 0.5},
            "dense_only_overall": {"a": 0.4, "only_in_comparison": 0.9},
        }
    }
    table = render_metrics_table(artifact, comparison=("Dense-only", "dense_only_overall"))
    assert "only_in_comparison" not in table
    assert table.splitlines()[0] == "| Metric (test split) | Value | Dense-only |"


def test_render_comparison_block_absent_or_empty_falls_back_to_single_column() -> None:
    # Documents current behaviour: a missing/empty comparison block is not an error.
    absent: dict[str, object] = {"metrics": {"overall": {"a": 0.5}}}
    empty: dict[str, object] = {"metrics": {"overall": {"a": 0.5}, "dense_only_overall": {}}}
    for artifact in (absent, empty):
        table = render_metrics_table(artifact, comparison=("Dense-only", "dense_only_overall"))
        assert table == "| Metric | Value |\n|---|---|\n| a | 0.500 |"


def test_render_uses_flat_metrics_mapping_when_no_overall_block() -> None:
    artifact: dict[str, object] = {"metrics": {"b": 2.0, "a": 1.0}}
    assert render_metrics_table(artifact) == (
        "| Metric | Value |\n|---|---|\n| a | 1.000 |\n| b | 2.000 |"
    )


def test_render_value_formatting() -> None:
    artifact: dict[str, object] = {
        "metrics": {"overall": {"flag": True, "count": 3, "ratio": 0.12345, "label": "n/a"}}
    }
    assert render_metrics_table(artifact).splitlines()[2:] == [
        "| count | 3.000 |",
        "| flag | True |",
        "| label | n/a |",
        "| ratio | 0.123 |",
    ]


@pytest.mark.parametrize(
    "artifact",
    [
        {},
        {"metrics": "not-a-mapping"},
        {"metrics": {}},
        {"metrics": {"overall": {}}},
        {"metrics": {"overall": "not-a-mapping"}},
    ],
    ids=["no-metrics", "metrics-not-mapping", "metrics-empty", "overall-empty", "overall-str"],
)
def test_render_raises_report_error_on_missing_metrics(artifact: dict[str, object]) -> None:
    with pytest.raises(ReportError):
        render_metrics_table(artifact)


# --- sync_readme / readme_in_sync --------------------------------------------------------


def test_sync_readme_replaces_only_the_marked_region(tmp_path: Path) -> None:
    readme = _write_readme(tmp_path)
    returned = sync_readme(readme, _STAMPED_ARTIFACT)
    text = readme.read_text(encoding="utf-8")
    assert returned == text
    assert text == (
        "# Consumer\n\nIntro.\n\n"
        f"{EVAL_BEGIN}\n{_SINGLE_COLUMN_TABLE}\n{EVAL_END}\n\n"
        "Trailing prose.\n"
    )
    assert _region(text) == "\n" + _SINGLE_COLUMN_TABLE + "\n"
    assert "| stale | table |" not in text


def test_sync_readme_writes_lf_only(tmp_path: Path) -> None:
    readme = _write_readme(tmp_path)
    sync_readme(readme, _STAMPED_ARTIFACT)
    assert b"\r" not in readme.read_bytes()


def test_sync_readme_normalises_a_crlf_readme_to_lf(tmp_path: Path) -> None:
    readme = _write_readme(tmp_path, _README_TEMPLATE.replace("\n", "\r\n"))
    assert b"\r\n" in readme.read_bytes()
    sync_readme(readme, _STAMPED_ARTIFACT)
    data = readme.read_bytes()
    assert b"\r" not in data
    assert data.decode("utf-8").endswith("Trailing prose.\n")


def test_sync_readme_is_idempotent(tmp_path: Path) -> None:
    readme = _write_readme(tmp_path)
    sync_readme(readme, _STAMPED_ARTIFACT)
    first = readme.read_bytes()
    sync_readme(readme, _STAMPED_ARTIFACT)
    assert readme.read_bytes() == first
    sync_readme(readme, _STAMPED_ARTIFACT)
    assert readme.read_bytes() == first


def test_sync_readme_passes_comparison_and_label_through(tmp_path: Path) -> None:
    artifact: dict[str, object] = {
        "metrics": {"overall": {"a": 0.5}, "dense_only_overall": {"b": 0.1}},
    }
    readme = _write_readme(tmp_path)
    text = sync_readme(
        readme, artifact, comparison=("Dense-only", "dense_only_overall"), primary_label="Hybrid"
    )
    assert _region(text) == (
        "\n| Metric (test split) | Hybrid | Dense-only |\n|---|---|---|\n| a | 0.500 | — |\n"
    )


@pytest.mark.parametrize(
    "text",
    [
        "# No markers at all\n",
        f"# Only begin\n{EVAL_BEGIN}\n",
        f"# Only end\n{EVAL_END}\n",
        f"# Out of order\n{EVAL_END}\nstuff\n{EVAL_BEGIN}\n",
    ],
    ids=["none", "begin-only", "end-only", "out-of-order"],
)
def test_sync_and_check_raise_report_error_on_bad_markers(tmp_path: Path, text: str) -> None:
    readme = _write_readme(tmp_path, text)
    with pytest.raises(ReportError, match="missing or out of order"):
        sync_readme(readme, _STAMPED_ARTIFACT)
    assert readme.read_bytes() == text.encode("utf-8")  # nothing written on failure
    with pytest.raises(ReportError, match="missing or out of order"):
        readme_in_sync(readme, _STAMPED_ARTIFACT)


def test_sync_readme_raises_report_error_when_artifact_has_no_metrics(tmp_path: Path) -> None:
    readme = _write_readme(tmp_path)
    with pytest.raises(ReportError):
        sync_readme(readme, {"git_sha": "deadbeef"})
    assert readme.read_bytes() == _README_TEMPLATE.encode("utf-8")


def test_readme_in_sync_true_after_sync_and_pure(tmp_path: Path) -> None:
    readme = _write_readme(tmp_path)
    assert readme_in_sync(readme, _STAMPED_ARTIFACT) is False  # stale placeholder table
    sync_readme(readme, _STAMPED_ARTIFACT)
    synced = readme.read_bytes()
    assert readme_in_sync(readme, _STAMPED_ARTIFACT) is True
    assert readme.read_bytes() == synced  # the check never writes


def test_readme_in_sync_false_after_changing_one_digit_in_readme(tmp_path: Path) -> None:
    readme = _write_readme(tmp_path)
    sync_readme(readme, _STAMPED_ARTIFACT)
    text = readme.read_text(encoding="utf-8")
    assert text.count("| a_metric | 0.500 |") == 1
    readme.write_bytes(text.replace("| a_metric | 0.500 |", "| a_metric | 0.501 |").encode("utf-8"))
    assert readme_in_sync(readme, _STAMPED_ARTIFACT) is False


def test_readme_in_sync_false_after_changing_one_digit_in_artifact(tmp_path: Path) -> None:
    readme = _write_readme(tmp_path)
    sync_readme(readme, _STAMPED_ARTIFACT)
    drifted: dict[str, object] = {
        **_STAMPED_ARTIFACT,
        "metrics": {"overall": {"z_metric": 1.0, "a_metric": 0.501}},
    }
    assert readme_in_sync(readme, drifted) is False
    assert readme_in_sync(readme, _STAMPED_ARTIFACT) is True


def test_readme_in_sync_respects_comparison_and_label(tmp_path: Path) -> None:
    artifact: dict[str, object] = {
        "metrics": {"overall": {"a": 0.5}, "dense_only_overall": {"a": 0.4}},
    }
    readme = _write_readme(tmp_path)
    comparison = ("Dense-only", "dense_only_overall")
    sync_readme(readme, artifact, comparison=comparison, primary_label="Hybrid")
    assert readme_in_sync(readme, artifact, comparison=comparison, primary_label="Hybrid") is True
    assert readme_in_sync(readme, artifact, comparison=comparison) is False  # label differs
    assert readme_in_sync(readme, artifact) is False  # single-column differs
