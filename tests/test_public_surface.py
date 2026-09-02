"""Public-surface contract: ``__all__`` + module paths ARE the compatibility surface (SPEC §4).

Every name below is transcribed from the SPEC §4 table, module by module. Changing this file
is a deliberate API change and belongs in the CHANGELOG (SPEC §7).
"""

import ast
import importlib
import importlib.metadata
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import clinevals

# SPEC §4, verbatim: module path -> exported names.
SPEC_SURFACE: dict[str, frozenset[str]] = {
    "clinevals.dataset": frozenset(
        {"Split", "DatasetError", "EvalItemBase", "ItemT", "Validator", "load_jsonl"}
    ),
    "clinevals.ranking": frozenset({"hit_at", "recall_at", "precision_at", "mrr"}),
    "clinevals.classify": frozenset(
        {"ConfusionCounts", "set_confusion", "micro_prf", "outside_universe_rate"}
    ),
    "clinevals.runner": frozenset(
        {"ItemScore", "ItemResult", "AggregateMetrics", "aggregate", "EvalReport", "score_items"}
    ),
    "clinevals.ratchet": frozenset({"Gate", "load_baseline", "compare_to_baseline"}),
    "clinevals.judge": frozenset(
        {"Rubric", "JudgeParseError", "extract_json_object", "invoke_judge"}
    ),
    "clinevals.grounding": frozenset(
        {
            "GroundingVerdict",
            "parse_grounding_verdict",
            "build_judge_input",
            "faithfulness_rubric",
            "verdict_metrics",
        }
    ),
    "clinevals.human": frozenset({"stratified_sample", "agreement_rate"}),
    "clinevals.artifacts": frozenset(
        {
            "EVAL_BEGIN",
            "EVAL_END",
            "ReportError",
            "write_artifact",
            "sha256_file",
            "detect_git_sha",
            "RunStamp",
            "build_artifact",
            "render_metrics_table",
            "sync_readme",
            "readme_in_sync",
        }
    ),
    "clinevals.fakes": frozenset({"scripted_judge"}),
}
SPEC_NAMES: frozenset[str] = frozenset().union(*SPEC_SURFACE.values())

# Package metadata exported alongside the SPEC §4 table (it lives in ``__init__``, not in any
# of the ten modules, so the per-module table cannot list it).
PACKAGE_METADATA_NAMES: frozenset[str] = frozenset({"__version__"})

# Modules that must stay pure: no langchain-core and no intra-package imports.
LEAF_MODULES: tuple[str, ...] = (
    "clinevals.ranking",
    "clinevals.classify",
    "clinevals.dataset",
    "clinevals.human",
    "clinevals.ratchet",
)
# SPEC §8: "langchain-core as a hard dep for one type (contained to two modules)".
LANGCHAIN_IMPORTERS: frozenset[str] = frozenset({"judge", "fakes"})

assert clinevals.__file__ is not None
PACKAGE_DIR: Path = Path(clinevals.__file__).parent


def _run_python(code: str) -> str:
    """Run ``code`` in a fresh interpreter (same venv) and return its stdout."""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _top_level_imports(module_path: Path) -> set[str]:
    """Top-level package names imported anywhere in the module source."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(module_path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found


def test_all_is_exactly_the_spec_section_4_surface() -> None:
    assert len(clinevals.__all__) == len(set(clinevals.__all__)), "duplicate names in __all__"
    assert set(clinevals.__all__) == SPEC_NAMES | PACKAGE_METADATA_NAMES
    # The ONLY name beyond the SPEC §4 table is the version string.
    assert set(clinevals.__all__) - SPEC_NAMES == PACKAGE_METADATA_NAMES


def test_every_all_name_is_importable_from_the_bare_package() -> None:
    missing = [name for name in clinevals.__all__ if not hasattr(clinevals, name)]
    assert missing == []
    # ``from clinevals import *`` is driven by ``__all__``: every entry must resolve.
    namespace: dict[str, object] = {}
    exec("from clinevals import *", namespace)
    assert set(namespace) - {"__builtins__"} == set(clinevals.__all__)


def test_every_spec_name_lives_at_its_spec_module_path() -> None:
    """Module paths are part of the surface: ``clinevals.X`` IS ``clinevals.<module>.X``."""
    problems: list[str] = []
    for module_name, names in SPEC_SURFACE.items():
        module = importlib.import_module(module_name)
        for name in sorted(names):
            if not hasattr(module, name):
                problems.append(f"{module_name}.{name}: missing from module")
            elif getattr(clinevals, name, None) is not getattr(module, name):
                problems.append(f"{module_name}.{name}: package re-export is a different object")
    assert problems == []


def test_version_is_frozen_at_0_1_0_and_matches_distribution_metadata() -> None:
    assert clinevals.__version__ == "0.1.0"
    assert importlib.metadata.version("clinical-agent-evals") == clinevals.__version__


def test_langchain_core_is_imported_only_by_judge_and_fakes() -> None:
    importers = {
        path.stem
        for path in sorted(PACKAGE_DIR.glob("*.py"))
        if "langchain_core" in _top_level_imports(path)
    }
    assert importers == LANGCHAIN_IMPORTERS


@pytest.mark.parametrize("module_name", LEAF_MODULES)
def test_leaf_module_source_imports_neither_langchain_core_nor_clinevals(module_name: str) -> None:
    imports = _top_level_imports(PACKAGE_DIR / f"{module_name.rsplit('.', 1)[1]}.py")
    assert "langchain_core" not in imports
    assert "clinevals" not in imports, "a leaf module must not depend on the package"


def test_leaf_modules_load_standalone_with_langchain_core_blocked() -> None:
    """Runtime form of leaf purity: each leaf executes (with its transitive deps) while any
    ``import langchain_core`` would raise. Bypasses the package ``__init__`` on purpose."""
    leaves = [
        (name.rsplit(".", 1)[1], str(PACKAGE_DIR / f"{name.rsplit('.', 1)[1]}.py"))
        for name in LEAF_MODULES
    ]
    code = textwrap.dedent(
        f"""
        import importlib.util, sys
        sys.modules["langchain_core"] = None  # any `import langchain_core` now raises
        loaded = []
        for name, path in {leaves!r}:
            spec = importlib.util.spec_from_file_location("leaf_" + name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            loaded.append(name)
        print(",".join(loaded))
        """
    )
    assert _run_python(code) == ",".join(name for name, _ in leaves)


def test_importing_leaf_modules_through_the_package_does_not_load_langchain_core() -> None:
    """Runtime leaf purity: __init__ loads the langchain-dependent names lazily (PEP 562)."""
    code = "import sys\n" + "".join(f"import {name}\n" for name in LEAF_MODULES)
    code += "print('langchain_core' in sys.modules)\n"
    assert _run_python(code) == "False"
