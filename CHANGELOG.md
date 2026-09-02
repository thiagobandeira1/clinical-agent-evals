# Changelog

All notable changes to `clinical-agent-evals` (import name `clinevals`) are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) with the 0.x
contract stated at the bottom of this file.

## [Unreleased]

## [0.1.0] - 2026-09-01

First release. The eval doctrine proven in `hedis-spec-copilot` (its ADR-005 and ADR-007),
extracted as an installable, keyless-CI-first library. Frozen before the care-gap agent's
first commit so consumers can pin it.

### Added

- `clinevals.dataset` — `Split`, `DatasetError`, `EvalItemBase` (`item_id`, `category`,
  required `split`), `ItemT`, `Validator`, `load_jsonl(path, item_type, *, validators=())`
  with all-invariants-at-once error reporting.
- `clinevals.ranking` — `hit_at`, `recall_at`, `precision_at`, `mrr`; all raise on an empty
  gold set (a dataset bug in a ranking eval).
- `clinevals.classify` — `ConfusionCounts(tp, fp, fn)` with `+` and `None`-valued
  `.precision/.recall/.f1`; `set_confusion`, `micro_prf` (undefined keys omitted),
  `outside_universe_rate` (deterministic hallucination rate); empty gold is legal.
- `clinevals.runner` — `ItemScore`, `ItemResult`, `AggregateMetrics`, sparse-key `aggregate`,
  `EvalReport` with the `test_overall` publication block, `score_items(items, score_fn)`
  (`None` skips an item; exceptions propagate).
- `clinevals.ratchet` — `Gate(metric=..., higher_is_better=True)` (keyword-only, a pydantic
  model), `load_baseline` (flat,
  `metrics`, or `overall` shapes), `compare_to_baseline(current, baseline, *, gates,
  tolerance=0.02)` with the skip-if-missing-from-baseline / fail-if-missing-from-current
  asymmetry.
- `clinevals.judge` — frozen `Rubric(name, text)` with `.sha256`, `JudgeParseError`,
  tolerant `extract_json_object`, `invoke_judge(model, rubric, payload) -> str` (the only
  model-touching function; the model is injected).
- `clinevals.grounding` — `faithfulness_rubric(domain, *, name)` reproducing P2's frozen
  `JUDGE_PROMPT` byte-for-byte for the HEDIS domain phrase, `GroundingVerdict`,
  `build_judge_input`, `parse_grounding_verdict`, `verdict_metrics`.
- `clinevals.human` — seeded `stratified_sample(items, *, n, key, seed)`, `agreement_rate`.
- `clinevals.artifacts` — `EVAL_BEGIN`/`EVAL_END`, `ReportError`, `RunStamp`,
  `build_artifact` (writes only `test_overall` as `metrics.overall`), byte-deterministic
  `write_artifact` (sorted keys, 2-space indent, non-ASCII preserved, trailing newline, LF on
  every OS), `render_metrics_table` (P2's two-column table reproduced with
  `comparison=("Dense-only", "dense_only_overall")`, `primary_label="Hybrid"`),
  `sync_readme`, pure `readme_in_sync`, `sha256_file`, `detect_git_sha`.
- `clinevals.fakes` — `scripted_judge(outputs)`, a `GenericFakeChatModel` replaying
  `GroundingVerdict`s (serialized to verdict JSON) or raw strings (for the parse-error path).
- `py.typed`; mypy `--strict` clean; Python 3.12+; MIT.

### Doctrine encoded as defaults

- Keyless by construction: no provider, key, retry, or network code exists in the package.
- Test-split-only publication: `EvalReport.test_overall` is the only block `build_artifact`
  publishes as `metrics.overall`; `split` has no default.
- Measured-never-aspirational ratchet with the skip/fail asymmetry and a 0.02 tolerance.
- `None`-not-zero for undefined ratios; raw tp/fp/fn counts travel into artifacts so
  micro-averages are computed from summed counts.
- Byte-identical artifacts and README regions across Ubuntu and Windows.
- Rubric immutability plus sha stamping (`docs/adr/002-rubric-immutability.md`).
- Judge-human agreement below 0.80 marks a run untrusted (stamped, enforced by consumers).

### Versioning contract (0.x)

- **Patch** (`0.1.x`): additive or bugfix only; no public name is removed or changes meaning.
- **Minor** (`0.2.0`): may break; the release notes carry a `### Migration` section.
- Consumers pin exact tags (`@v0.1.0`) and commit their lockfile. `1.0.0` is tagged when P1
  and P2 are green on the same tag.

[Unreleased]: https://github.com/thiagobandeira1/clinical-agent-evals/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/thiagobandeira1/clinical-agent-evals/releases/tag/v0.1.0
