# clinical-agent-evals

A keyless-CI-first evaluation harness for clinical LLM agents: typed gold sets, honest
small-N metrics, injected judges, ratcheted gates, and byte-deterministic artifacts — with no
network code anywhere in the package.

[![CI](https://github.com/thiagobandeira1/clinical-agent-evals/actions/workflows/ci.yml/badge.svg)](https://github.com/thiagobandeira1/clinical-agent-evals/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Import name: `clinevals`.

## Why this exists

The portfolio's clinical agents — a care-gap detector, a HEDIS spec copilot, a code-suggestion
assistant — all need the same eval doctrine: CI that runs with zero secrets, numbers that are
measured rather than typed, and publication from the test split only. That doctrine was proven
inside [hedis-spec-copilot](https://github.com/thiagobandeira1/hedis-spec-copilot) (its
ADR-005 and ADR-007) and kept getting re-implemented, slightly differently, per repo. This
package extracts it once, as library law: the defaults make the honest thing the easy thing
and the dishonest thing require a deliberate, visible override.

## Install

```bash
uv add "clinical-agent-evals @ git+https://github.com/thiagobandeira1/clinical-agent-evals@v0.1.0"
```

Python 3.12+. Runtime dependencies: `pydantic>=2.8`, `langchain-core>=0.3` (the latter is
used for exactly one type — see [Limitations](#limitations)). The package ships `py.typed`.

## Quickstart — a gap-detection eval in the keyless tier

Two input files. The gold set is JSONL, one patient per line; `split` is required with no
default, because silent split assignment is a publication hazard. `pt-002` has an empty gold
set — a no-gap patient is a legal, load-bearing false-positive probe.

`evals/gold/gaps.jsonl`

```jsonl
{"item_id": "pt-001", "category": "BCS", "split": "test", "gold_gaps": ["bcs-due"]}
{"item_id": "pt-002", "category": "COL", "split": "test", "gold_gaps": []}
{"item_id": "pt-003", "category": "CBP", "split": "test", "gold_gaps": ["cbp-high"], "needs_review": true}
{"item_id": "pt-004", "category": "CBP", "split": "dev", "gold_gaps": ["cbp-high"]}
```

`README.md` (the consumer's own README) carries the two markers the table is written between:

```markdown
<!-- EVAL:BEGIN -->
<!-- EVAL:END -->
```

The eval itself:

```python
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from clinevals import (
    EvalItemBase,
    Gate,
    ItemScore,
    RunStamp,
    build_artifact,
    compare_to_baseline,
    detect_git_sha,
    load_baseline,
    load_jsonl,
    readme_in_sync,
    score_items,
    set_confusion,
    sha256_file,
    sync_readme,
    write_artifact,
)


class GapCaseItem(EvalItemBase):
    gold_gaps: frozenset[str]  # empty == a no-gap patient: a legal false-positive probe
    needs_review: bool = False


def known_measures(items: Sequence[GapCaseItem]) -> list[str]:
    bad = [i for i in items if i.category not in {"BCS", "COL", "CBP"}]
    return [f"{i.item_id}: unknown measure {i.category}" for i in bad]


gold = Path("evals/gold/gaps.jsonl")
items = load_jsonl(gold, GapCaseItem, validators=[known_measures])

# Outputs the agent produced BEFORE the eval: (gaps flagged, escalated to a human).
# score_fn is pure post-hoc scoring over recorded outputs; it never runs the agent.
recorded: dict[str, tuple[set[str], bool]] = {
    "pt-001": ({"bcs-due"}, False),
    "pt-002": ({"col-due"}, False),  # false alarm on the no-gap patient
    "pt-003": ({"cbp-high"}, True),
    "pt-004": (set(), False),
}


def score(item: GapCaseItem) -> ItemScore:
    gaps, escalated = recorded[item.item_id]
    hitl = {"hitl_triggered": float(escalated)} if item.needs_review else {}  # sparse key
    return ItemScore(metrics=hitl, counts=set_confusion(gaps, item.gold_gaps))


report = score_items(items, score)
stamp = RunStamp(
    git_sha=detect_git_sha(),
    date=date.today().isoformat(),
    dataset_hash=sha256_file(gold, short=12),
)
artifact = build_artifact(report, stamp, tier="keyless")  # metrics.overall IS report.test_overall
write_artifact(Path("evals/results/latest.json"), artifact)  # LF bytes on every OS
sync_readme(Path("README.md"), artifact)  # rewrites only the region between the EVAL markers
assert readme_in_sync(Path("README.md"), artifact)  # the CI byte-check, as one pure call

baseline = Path("evals/baseline.json")  # a previous MEASURED artifact; moves only by PR
if baseline.exists():  # absent on the very first run — bootstrap by copying results/latest.json
    regressions = compare_to_baseline(
        report.test_overall,
        load_baseline(baseline),
        gates=(
            Gate(metric="micro_recall"),
            Gate(metric="micro_precision"),
            Gate(metric="hitl_triggered"),
        ),
    )
    if regressions:
        raise SystemExit("\n".join(regressions))
```

What this produces: `metrics.overall` in the artifact (and the table between the markers)
holds `micro_precision`, `micro_recall` and `micro_f1` computed from the *summed* test-split
counts, plus `hitl_triggered` averaged over the one case that carries the key. `pt-004` is
dev and never reaches `metrics.overall`; its raw counts still land in `counts.per_split.dev`
so nothing is hidden. `load_baseline` reads `metrics.overall` from a copied artifact, so the
baseline is literally a previous run, never a typed-in target.

### The judged tier

The single model-touching function takes an injected `BaseChatModel`. In CI that is a
scripted fake; a local full run passes a real model built by the consumer.

```python
from clinevals import (
    GroundingVerdict,
    JudgeParseError,
    build_judge_input,
    faithfulness_rubric,
    invoke_judge,
    parse_grounding_verdict,
    scripted_judge,
    verdict_metrics,
)

# P2's frozen JUDGE_PROMPT, byte-for-byte, for this domain phrase (golden sha test in both repos).
rubric = faithfulness_rubric("Medicare Star Ratings / HEDIS measures", name="hedis-grounding-v1")

judge = scripted_judge(  # CI: keyless. Locally: any BaseChatModel takes this slot.
    [
        GroundingVerdict(
            claims_total=2,
            claims_supported=1,
            claims_unsupported=1,
            claims_contradicted=0,
            citation_valid_ratio=None,
        ),
        "Sure! Here is my grade, with no JSON anywhere.",
    ]
)
payload = build_judge_input(
    question="How often is colorectal screening due?",
    answer_text="Every 10 years with colonoscopy [1]. Coverage starts at 40.",
    passages=["Colonoscopy every 10 years satisfies the measure.", "Ages 45-75 are eligible."],
)
verdict = parse_grounding_verdict(invoke_judge(judge, rubric, payload))
metrics = verdict_metrics(verdict)  # {"contradiction_rate": 0.0, "faithfulness": 0.5}
# A key is ABSENT (not 0.0) when undefined: drop `metrics` into ItemScore(metrics=...) inside
# a score_fn, and stamp provenance with RunStamp(rubric_sha256=rubric.sha256, ...).

try:  # the second scripted output is garbage: it raises, it never becomes a silent 0.0
    parse_grounding_verdict(invoke_judge(judge, rubric, payload))
except JudgeParseError as exc:
    print(f"unparseable verdict, item skipped: {exc}")
```

For the human spot-check, `stratified_sample(items, n=15, key=lambda i: i.category, seed=7)`
picks a reproducible sample and `agreement_rate(pairs)` scores the judge against the human
labels; the rate is stamped via `RunStamp(judge_human_agreement=...)`.

## Doctrine (why the defaults are the way they are)

- **Keyless by construction.** There is no provider client, key handling, retry, or network
  I/O in the package. `invoke_judge(model, rubric, payload)` is the only function that talks to
  a model and it takes one fully built by the caller; `clinevals.fakes.scripted_judge` makes
  every consumer's judge plumbing testable with zero secrets. Only `judge` and `fakes` import
  `langchain-core`.
- **Test-split-only publication, via `test_overall`.** `EvalReport.test_overall` is the sole
  block `build_artifact` writes as `metrics.overall` and the sole block the ratchet is meant to
  read. It is the test-split sparse-key means merged with `micro_prf` over test-split counts.
  If only dev items were scored it is `{}`, and `render_metrics_table` refuses to render an
  empty block — you cannot publish a table from dev data without hand-building the payload.
  `metrics.per_category` and `metrics.per_split` are diagnostic and span both splits.
- **Measured, never aspirational, with a skip/fail asymmetry.** `compare_to_baseline` gates
  only against numbers the baseline actually recorded: a gated metric missing from the
  *baseline* is skipped, a gated metric missing from the *current* run is a regression. The
  default `tolerance=0.02` is P2's measured noise floor. `Gate(metric=...,
  higher_is_better=False)` handles lower-is-better metrics such as a hallucination rate.
  `Gate` is a pydantic model: construct it with keywords.
- **`None`, not zero, and raw counts travel.** `ConfusionCounts.precision/.recall/.f1` return
  `None` when undefined; `micro_prf` and `verdict_metrics` omit undefined keys rather than
  coercing to 0.0; `aggregate` averages each key only over the items that carry it. Counts
  ride `ItemScore.counts` into `counts.per_category` / `counts.per_split` in the artifact
  because micro-averages cannot be recovered from per-item means after the fact. `ranking`
  is the one deliberate exception: empty gold there is a dataset bug and raises.
- **Byte-deterministic artifacts, LF on every OS.** `write_artifact` uses
  `sort_keys=True, indent=2, ensure_ascii=False`, a trailing newline and `newline="\n"`;
  `sync_readme` writes the same way. Identical inputs produce identical bytes on Windows and
  Linux (this fixed a latent `\r\n` bug in P2's `report.py`). `stratified_sample` is seeded
  and stratum-sorted for the same reason.
- **Rubric immutability and sha stamping.** `Rubric` is frozen; a released text is never
  edited — evolution is a new `name`. `Rubric.sha256` is stamped into artifacts so prompt
  drift is visible, never silent. `faithfulness_rubric` reproduces P2's `JUDGE_PROMPT`
  byte-for-byte for the HEDIS domain phrase (see `docs/adr/002-rubric-immutability.md`).
- **Human spot-check agreement below 0.80 means untrusted.** `agreement_rate` computes the
  number; `RunStamp.judge_human_agreement` carries it into the artifact. The 0.80 bar is
  doctrine, enforced by consumer CI, not by the library.

Consumers' READMEs carry the generated eval tables; this library has no eval artifact of its
own, and the doctrine forbids typing numbers into a README by hand.

## Module map

| Module | One line |
|---|---|
| `clinevals.dataset` | `EvalItemBase` (`item_id`, `category`, required `split`), `load_jsonl` with all-invariants-at-once `DatasetError`, `Validator` |
| `clinevals.ranking` | Pure `hit_at` / `recall_at` / `precision_at` / `mrr`; **raise** on empty gold |
| `clinevals.classify` | `ConfusionCounts` (`+`, `None`-valued ratios), `set_confusion`, `micro_prf`, `outside_universe_rate`; empty gold **legal** |
| `clinevals.runner` | `score_items(items, score_fn)` → `EvalReport` with `.test_overall`; `None` skips, exceptions propagate; sparse-key `aggregate` |
| `clinevals.ratchet` | `Gate`, `load_baseline` (tolerant shapes), `compare_to_baseline` (skip/fail asymmetry, tolerance) |
| `clinevals.judge` | Frozen `Rubric(.sha256)`, `invoke_judge(model, rubric, payload)`, tolerant `extract_json_object`, `JudgeParseError` |
| `clinevals.grounding` | Per-claim faithfulness family: `faithfulness_rubric`, `build_judge_input`, `parse_grounding_verdict`, `GroundingVerdict`, `verdict_metrics` |
| `clinevals.human` | `stratified_sample` (seeded, round-robin across sorted strata), `agreement_rate` |
| `clinevals.artifacts` | `RunStamp`, `build_artifact`, `write_artifact`, `render_metrics_table`, `sync_readme`, `readme_in_sync`, `sha256_file`, `detect_git_sha` |
| `clinevals.fakes` | `scripted_judge(outputs)` — a `GenericFakeChatModel` that replays verdicts or raw strings |

`clinevals.__all__` plus the module paths above is the compatibility surface.

## Consumers

- **P1 — care-gap agent (day one).** `GapCaseItem(EvalItemBase)` plus validators; keyless tier
  scores recorded outputs with `set_confusion` per patient (category = measure id, so
  per-measure counts come for free) and category-branched HITL/refusal booleans as sparse keys;
  `build_artifact` → `write_artifact` → `sync_readme`; CI gates on `micro_recall`,
  `micro_precision` and `hitl_triggered` plus `readme_in_sync`. Judged tier through
  `invoke_judge` with its own verdict model over `extract_json_object`; spot-check via
  `stratified_sample` + `agreement_rate`.
- **P2 — hedis-spec-copilot (adoption PR).** Shims keep `hedis_copilot.evals.*` imports
  working; `retrieval_metrics`, `judge` and `report` become re-exports; `run_retrieval_eval`
  becomes a ~40-line adapter over `score_items`. The acceptance bar is a byte-identical artifact
  and README region against the pre-PR outputs, with a golden test pinning the stamped rubric
  sha.
- **P4 — code-suggestion assistant.** P1's shape plus `outside_universe_rate` as a
  deterministic hallucination rate, gated with `Gate(metric="hallucination_rate",
  higher_is_better=False)`.

## Versioning

SemVer with an explicit 0.x contract: a patch release is additive or a bugfix; a minor release
may break and carries a Migration section in `CHANGELOG.md`. Consumers pin exact tags
(`@v0.1.0`) and commit their `uv.lock`. v0.1.0 is frozen before P1's first commit; 1.0.0 comes
when P1 and P2 are green on the same tag.

## Limitations

- **Small-N presentation.** The README table shows means; the per-item N behind each mean is
  visible only in the artifact JSON (`counts.*`, `per_item` when requested). Showing counts in
  the table is a 0.2 candidate.
- **`langchain-core` is a hard dependency** for a single type (`BaseChatModel`) and the fake it
  ships. It is contained to `judge` and `fakes`; every other module is pydantic-only.
- **No trajectory primitives.** `score_fn` is post-hoc scoring over recorded outputs; there is
  no agent harness, retriever, multi-turn state, or LangSmith integration in 0.1.
- **No statistics beyond means and summed counts.** No confidence intervals, no kappa.
- `readme_in_sync` reads the README in text mode, so it compares content after universal
  newline translation; `sync_readme` then writes LF. A CRLF checkout is normalized once.

## License

MIT — see `LICENSE`.
