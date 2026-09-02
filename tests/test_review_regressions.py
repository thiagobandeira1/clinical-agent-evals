"""Regression tests for defects confirmed by the adversarial review; each pins one fix."""

import json
import math
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import clinevals
from clinevals import (
    DatasetError,
    EvalItemBase,
    Gate,
    compare_to_baseline,
    detect_git_sha,
    load_baseline,
    load_jsonl,
    write_artifact,
)


class TestRatchetBoundary:
    """A drop of EXACTLY the tolerance must pass, regardless of float rounding."""

    def test_exact_tolerance_drop_passes_higher_is_better(self) -> None:
        # 0.17 - 0.02 == 0.15000000000000002 in binary floating point.
        assert compare_to_baseline({"m": 0.15}, {"m": 0.17}, gates=(Gate("m"),)) == []

    def test_exact_tolerance_rise_passes_lower_is_better(self) -> None:
        # 0.18 + 0.02 == 0.19999999999999998.
        gate = Gate("h", higher_is_better=False)
        assert compare_to_baseline({"h": 0.20}, {"h": 0.18}, gates=(gate,)) == []

    def test_fresh_ratio_at_boundary_passes(self) -> None:
        assert compare_to_baseline({"m": 15 / 100}, {"m": 17 / 100}, gates=(Gate("m"),)) == []

    def test_beyond_tolerance_still_regresses(self) -> None:
        assert compare_to_baseline({"m": 0.149}, {"m": 0.17}, gates=(Gate("m"),))

    def test_nan_baseline_is_a_regression_not_a_free_pass(self) -> None:
        messages = compare_to_baseline({"m": 0.0}, {"m": math.nan}, gates=(Gate("m"),))
        assert messages and "NaN" in messages[0]


class TestArtifactsNaN:
    def test_write_artifact_refuses_nan(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            write_artifact(tmp_path / "a.json", {"metrics": {"overall": {"x": math.nan}}})
        assert not (tmp_path / "a.json").exists() or (tmp_path / "a.json").read_text() == ""


class TestBom:
    def test_load_jsonl_accepts_utf8_bom(self, tmp_path: Path) -> None:
        class Item(EvalItemBase):
            pass

        path = tmp_path / "gold.jsonl"
        line = json.dumps({"item_id": "a", "category": "c", "split": "test"})
        path.write_bytes(b"\xef\xbb\xbf" + line.encode("utf-8") + b"\n")
        assert [i.item_id for i in load_jsonl(path, Item)] == ["a"]

    def test_load_jsonl_without_bom_unchanged(self, tmp_path: Path) -> None:
        class Item(EvalItemBase):
            pass

        path = tmp_path / "gold.jsonl"
        path.write_text('{"item_id": "a", "category": "c", "split": "test"}\n', encoding="utf-8")
        assert len(load_jsonl(path, Item)) == 1
        bad = tmp_path / "bad.jsonl"
        bad.write_text("{nope\n", encoding="utf-8")
        with pytest.raises(DatasetError):
            load_jsonl(bad, Item)

    def test_load_baseline_accepts_utf8_bom(self, tmp_path: Path) -> None:
        path = tmp_path / "baseline.json"
        path.write_bytes(b"\xef\xbb\xbf" + b'{"mrr": 0.5}')
        assert load_baseline(path) == {"mrr": 0.5}


class TestGitSha:
    def test_bad_repo_root_raises_runtime_error_on_every_os(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError):
            detect_git_sha(tmp_path / "does-not-exist")


class TestLazyExportsStayTyped:
    def test_dir_lists_lazy_names(self) -> None:
        listed = dir(clinevals)
        assert "Rubric" in listed and "scripted_judge" in listed

    def test_package_root_imports_type_check_under_mypy_strict(self, tmp_path: Path) -> None:
        """Consumers import from bare ``clinevals``; those names must not degrade to Any."""
        consumer = tmp_path / "consumer.py"
        consumer.write_text(
            textwrap.dedent(
                """
                from clinevals import GroundingVerdict, Rubric, faithfulness_rubric, invoke_judge
                from clinevals import parse_grounding_verdict, scripted_judge

                RUBRIC: Rubric = faithfulness_rubric("x", name="x-v1")


                def rubric_sha() -> str:
                    return RUBRIC.sha256


                class MyVerdict(GroundingVerdict):
                    pass


                def faith(raw: str) -> float | None:
                    return parse_grounding_verdict(raw).faithfulness


                def run() -> str:
                    return invoke_judge(scripted_judge(["{}"]), RUBRIC, "payload")
                """
            ),
            encoding="utf-8",
        )
        result = subprocess.run(
            [sys.executable, "-m", "mypy", "--strict", str(consumer)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
