# SPEC — clinical-agent-evals (P5)

> Status: **approved v0.1** · Import name `clinevals` · Produced by a 2-lens design panel
> (library API, consumer fit) + adversarial synthesis, extracting the PROVEN eval patterns
> from `hedis-spec-copilot` (P2). Decisions are recorded in `docs/adr/`.

## 1. Purpose

A reusable, installable, keyless-CI-first evaluation harness for the portfolio's clinical
LLM agents. Consumers: **P1** (LangGraph care-gap agent — gap-detection P/R/F1 with legal
empty-gold "no-gap" patients, outreach-quality judging, HITL/refusal slices, CI gates),
**P2** (adopts the judge/report/runner/ratchet layers via one shim-based PR with byte-identical
artifacts as the acceptance bar), **P4** (code-suggestion P/R + deterministic hallucination
rate via `outside_universe_rate`, lower-is-better gate).

## 2. Goals

1. Generalize P2's proven patterns: JSONL gold sets with all-violations-at-once validation,
   pure ranking metrics (verbatim), **new** confusion-count classification metrics, one
   generic scoring loop with dev/test discipline, injected-`BaseChatModel` judging with
   tolerant verdict parsing, directional ratchet gates, stamped byte-deterministic artifacts
   with README marker sync.
2. **Keyless by construction**: no network code exists in the package; the single
   model-touching function takes an injected model; `clinevals.fakes.scripted_judge` makes
   every consumer's judge plumbing testable with zero secrets.
3. Institutionalize hard-won review findings as defaults: **test-split-only publication**
   (`EvalReport.test_overall` is the only block `build_artifact` and the ratchet read),
   measured-never-aspirational baselines (missing-from-baseline skipped, missing-from-current
   fails), README numbers generated between markers with a pure `readme_in_sync()` CI check,
   honest small-N reporting (raw tp/fp/fn published; micro-averages from summed counts;
   `None`-not-zero for undefined ratios).
4. Zero domain leakage: no HEDIS vocabulary, no chunk/label resolution, no gate-A semantics.
5. Rubric provenance: `faithfulness_rubric(domain, name=...)` reproduces P2's frozen
   `JUDGE_PROMPT` byte-for-byte for the HEDIS domain phrase (golden sha tests in both repos).
6. Small and excellent: 10 flat modules, ~40 exported names, deps `pydantic>=2.7` +
   `langchain-core>=0.3`, Python 3.12+, mypy --strict + `py.typed`, MIT, installable via
   git+https exact-tag pins; **v0.1.0 frozen before P1's first commit**.

## 3. Non-goals

- No LLM provider code, API clients, key handling, retries, or network I/O anywhere.
- No orchestration of the system under test (no agent harness, no retriever). `score_fn` is
  pure post-hoc scoring over recorded outputs.
- No domain vocabulary in core; no generic gold-resolution machinery (zero committed callers).
- No CLI, config format, plugin system, scorer registry, or abstract scorer classes.
- No statistics beyond means and summed counts (no CIs, kappa, numpy/scipy).
- No LangSmith in 0.1; no PyPI in 0.1; no multi-turn trajectory primitives.

## 4. Public API (the compatibility surface = `__all__` + module paths)

| Module | Names |
|---|---|
| `clinevals.dataset` | `Split`, `DatasetError`, `EvalItemBase`, `ItemT`, `Validator`, `load_jsonl(path, item_type, *, validators=())` |
| `clinevals.ranking` | `hit_at`, `recall_at`, `precision_at`, `mrr` — raise on empty gold (dataset bug) |
| `clinevals.classify` | `ConfusionCounts(tp, fp, fn)` with `+`, `.precision/.recall/.f1 -> float \| None`; `set_confusion`, `micro_prf`, `outside_universe_rate` — empty gold **legal** |
| `clinevals.runner` | `ItemScore`, `ItemResult`, `AggregateMetrics`, `aggregate`, `EvalReport(.test_overall)`, `score_items(items, score_fn)` — `None` return skips the item; exceptions propagate |
| `clinevals.ratchet` | `Gate(metric, higher_is_better=True)`, `load_baseline`, `compare_to_baseline(current, baseline, *, gates, tolerance=0.02)` |
| `clinevals.judge` | `Rubric(name, text).sha256`, `JudgeParseError`, `extract_json_object`, `invoke_judge(model, rubric, payload) -> str` |
| `clinevals.grounding` | `GroundingVerdict(.faithfulness)`, `parse_grounding_verdict`, `build_judge_input`, `faithfulness_rubric(domain, *, name)`, `verdict_metrics` |
| `clinevals.human` | `stratified_sample(items, *, n, key, seed)`, `agreement_rate(pairs)` |
| `clinevals.artifacts` | `EVAL_BEGIN/EVAL_END`, `ReportError`, `write_artifact`, `sha256_file`, `detect_git_sha`, `RunStamp`, `build_artifact`, `render_metrics_table`, `sync_readme`, `readme_in_sync` |
| `clinevals.fakes` | `scripted_judge(outputs)` |

Semantics that matter:
- **Two metric modules with opposite empty-gold doctrine**: `ranking` raises (empty gold is a
  dataset bug); `classify` never raises (a no-gap patient is a load-bearing false-positive
  probe) and returns `None` for undefined ratios, never a fake 0.0/1.0.
- `EvalReport.test_overall` = test-split sparse-key means (fallback: overall when no splits)
  merged with `micro_prf` over test-split counts. `build_artifact` writes it as
  `metrics.overall` with the note "metrics.overall is the test split; tuning uses dev only".
- `compare_to_baseline` takes flat mappings; a gated metric missing from `current` is a
  regression; missing from `baseline` is skipped.
- All file writes use `newline="\n"` (fixes a latent Windows `\r\n` bug found in P2's
  `report.py`), and artifacts are `sort_keys=True, indent=2, ensure_ascii=False` + trailing
  newline — byte-identical across OSes.
- `render_metrics_table(artifact, *, comparison=(label, metrics_key) | None,
  primary_label=...)` reproduces P2's two-column table byte-for-byte for
  `comparison=("Dense-only", "dense_only_overall"), primary_label="Hybrid"`.
- Released `Rubric` texts are immutable; evolution is a new name.

## 5. Consumption

- **P1 (day one)**: `GapCaseItem(EvalItemBase)` + validators; keyless tier scores recorded
  agent outputs with `set_confusion` per patient (category = measure id → per-measure counts
  for free) and category-branched HITL/refusal booleans; `build_artifact` → `write_artifact`
  → `sync_readme`; CI gate `compare_to_baseline(report.test_overall, load_baseline(...),
  gates=(Gate("micro_recall"), Gate("micro_precision"), Gate("hitl_triggered")))` +
  `readme_in_sync`. Judged tier via `invoke_judge` + own verdict model over
  `extract_json_object`; spot-check via `stratified_sample` + `agreement_rate` stamped into
  `RunStamp.judge_human_agreement` (<0.80 ⇒ run marked untrusted).
- **P2 adoption PR**: shims preserve `hedis_copilot.evals.*` imports; `retrieval_metrics`,
  `judge`, `report` become re-exports; `runner.run_retrieval_eval` becomes a ~40-line adapter
  over `score_items` (pops `refused` from overall/per_split for byte-compat); golden test pins
  the pre-PR `judge_prompt_sha256`; acceptance = byte-identical artifact + README region.
- **P4**: P1's shape + `outside_universe_rate` + `Gate("hallucination_rate",
  higher_is_better=False)`.

## 6. Testing

Keyless everywhere (no secrets configured). Ported P2 seed tests per module; classify
boundary semantics (never raises; `None` on undefined; micro ≠ macro on an asymmetric
fixture); runner (skip-on-None, sparse-key aggregation, count summation, `test_overall`
merge/fallback, exception propagation); **determinism suite** (byte-identical rewrites, no
`\r` in artifacts/README, idempotent sync, seeded sampling stability) on ubuntu **and**
windows; rubric provenance golden (HEDIS domain sha == P2's stamped sha); judge round-trip
with `scripted_judge` incl. the garbage → `JudgeParseError` path; ratchet directions and
tolerance edges; **consumer-contract tests** (mini-P2 and mini-P1 end-to-end — the mini-P1
fixture doubles as the README quickstart); public-surface test (`__all__` matches spec);
release job installs the built wheel into a clean venv and imports it.

## 7. Versioning

SemVer with an explicit 0.x contract: patch = additive/bugfix; minor may break with a
CHANGELOG Migration section. Consumers pin exact tags + uv.lock commit hashes. v0.1.0 tagged
from this spec before P1's first commit; 1.0.0 when P1 and P2 are green on the same tag.

## 8. Risks (accepted)

Rubric byte-identity (mitigated by golden sha tests in both repos; fallback: P2 wraps its
literal in `Rubric`); P2 byte-compat sharp edge (`refused` key popping — loud if forgotten);
one-time `\r\n` normalization in P2 artifacts; API churn while P1 is mid-flight (freeze +
batch breaks into 0.2); score_fn misuse for live runs (documented hard; contract fixture
models recorded outputs); small-N table presentation (counts in JSON, means in README — 0.2
candidate); langchain-core as a hard dep for one type (contained to two modules).
