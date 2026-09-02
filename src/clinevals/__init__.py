"""clinevals — keyless-CI-first evaluation harness for clinical LLM agents.

``__all__`` (plus the module paths) IS the compatibility surface.

The langchain-core-dependent names (judge, grounding, fakes) are loaded lazily via PEP 562
so that importing the pure-metrics modules (``clinevals.ranking`` etc.) through the package
never pulls in langchain — the leaf-purity guarantee holds at runtime, not just in source.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Static imports so mypy/pyright type the lazily loaded names precisely; TYPE_CHECKING is
    # False at runtime, so the leaf-purity guarantee (no langchain import) still holds.
    from clinevals.fakes import scripted_judge
    from clinevals.grounding import (
        GroundingVerdict,
        build_judge_input,
        faithfulness_rubric,
        parse_grounding_verdict,
        verdict_metrics,
    )
    from clinevals.judge import JudgeParseError, Rubric, extract_json_object, invoke_judge

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
from clinevals.classify import ConfusionCounts, micro_prf, outside_universe_rate, set_confusion
from clinevals.dataset import DatasetError, EvalItemBase, ItemT, Split, Validator, load_jsonl
from clinevals.human import agreement_rate, stratified_sample
from clinevals.ranking import hit_at, mrr, precision_at, recall_at
from clinevals.ratchet import Gate, compare_to_baseline, load_baseline
from clinevals.runner import (
    AggregateMetrics,
    EvalReport,
    ItemResult,
    ItemScore,
    aggregate,
    score_items,
)

__version__ = "0.1.0"

#: name -> module for the lazily loaded (langchain-core-importing) surface.
_LAZY: dict[str, str] = {
    "Rubric": "clinevals.judge",
    "JudgeParseError": "clinevals.judge",
    "extract_json_object": "clinevals.judge",
    "invoke_judge": "clinevals.judge",
    "GroundingVerdict": "clinevals.grounding",
    "build_judge_input": "clinevals.grounding",
    "faithfulness_rubric": "clinevals.grounding",
    "parse_grounding_verdict": "clinevals.grounding",
    "verdict_metrics": "clinevals.grounding",
    "scripted_judge": "clinevals.fakes",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module 'clinevals' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)


def __dir__() -> list[str]:
    """Expose the lazy names to dir()/IDE completion alongside the eager ones."""
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "EVAL_BEGIN",
    "EVAL_END",
    "AggregateMetrics",
    "ConfusionCounts",
    "DatasetError",
    "EvalItemBase",
    "EvalReport",
    "Gate",
    "GroundingVerdict",
    "ItemResult",
    "ItemScore",
    "ItemT",
    "JudgeParseError",
    "ReportError",
    "Rubric",
    "RunStamp",
    "Split",
    "Validator",
    "__version__",
    "aggregate",
    "agreement_rate",
    "build_artifact",
    "build_judge_input",
    "compare_to_baseline",
    "detect_git_sha",
    "extract_json_object",
    "faithfulness_rubric",
    "hit_at",
    "invoke_judge",
    "load_baseline",
    "load_jsonl",
    "micro_prf",
    "mrr",
    "outside_universe_rate",
    "parse_grounding_verdict",
    "precision_at",
    "readme_in_sync",
    "recall_at",
    "render_metrics_table",
    "score_items",
    "scripted_judge",
    "set_confusion",
    "sha256_file",
    "stratified_sample",
    "sync_readme",
    "verdict_metrics",
    "write_artifact",
]
