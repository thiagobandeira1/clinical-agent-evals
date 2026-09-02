"""clinevals — keyless-CI-first evaluation harness for clinical LLM agents.

``__all__`` (plus the module paths) IS the compatibility surface.
"""

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
from clinevals.fakes import scripted_judge
from clinevals.grounding import (
    GroundingVerdict,
    build_judge_input,
    faithfulness_rubric,
    parse_grounding_verdict,
    verdict_metrics,
)
from clinevals.human import agreement_rate, stratified_sample
from clinevals.judge import JudgeParseError, Rubric, extract_json_object, invoke_judge
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
