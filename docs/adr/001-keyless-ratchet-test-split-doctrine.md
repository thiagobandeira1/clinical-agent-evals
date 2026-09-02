# ADR-001: Keyless CI, measured ratchet and test-split publication as library law

Status: Accepted
Date: 2026-09-01

## Context

`hedis-spec-copilot` (P2) learned its eval doctrine the hard way and recorded it twice:
ADR-005 (keyless CI with a two-tier harness, a ratchet against a *measured* baseline, README
numbers generated never typed, judge-human agreement below 0.80 marks a run untrusted) and
ADR-007 (a gate that could not separate refusal traps from real questions was demoted to a
degenerate-input guard; the honest number stayed published *with context*). Both hold the
doctrine in prose and a CLI. The next consumers (P1, P4) would re-implement it, and every
re-implementation is a chance to drop a clause. The question was not *whether* to inherit
the doctrine but *where* it lives: in prose consumers are asked to follow, or in the shape
of the API so that following it is the default path.

## Decision

Each doctrine point is encoded structurally; the structure is the contract.

1. **Keyless by construction.** No module contains provider, key, retry, or network code.
   `judge.invoke_judge(model, rubric, payload)` is the only function that touches a model and
   takes a fully built `BaseChatModel` from the caller; `fakes.scripted_judge` returns a
   `GenericFakeChatModel`, so every consumer's judge path runs with no secret configured. Only
   `judge` and `fakes` import `langchain-core`. `artifacts.build_artifact(report, stamp, *,
   tier)` requires the tier keyword, so an artifact always names the tier that produced it.
2. **Test-split-only publication.** `dataset.EvalItemBase.split` is required with no default.
   `runner.EvalReport.test_overall` (test-split sparse-key means merged with `micro_prf` over
   test-split counts) is the only block `build_artifact` writes as `metrics.overall` and the
   block consumers hand to the ratchet; `EvalReport.overall` is documented as diagnostic. With
   no test item scored, `test_overall` is `{}` and `render_metrics_table` raises `ReportError`
   rather than rendering an empty table (so a dev-only smoke test must not call `sync_readme`).
3. **Measured, never aspirational.** `ratchet.compare_to_baseline` skips a gated metric
   absent from the baseline and fails one absent from the current run, so the ratchet only
   guards numbers that were actually measured. `load_baseline` reads a previous artifact's
   `metrics.overall` and raises `FileNotFoundError`: vacuous-pass-when-absent is a consumer
   CLI policy, not a library default. `tolerance=0.02` is P2's noise floor;
   `Gate(metric=..., higher_is_better=False)` covers lower-is-better metrics in the same path.
4. **Numbers generated, never typed.** `artifacts.sync_readme` rewrites only the region
   between `EVAL_BEGIN`/`EVAL_END` and raises `ReportError` when the markers are missing, so a
   table is never appended elsewhere; `readme_in_sync` is the same computation as a pure
   boolean — the entire CI check.
5. **Honest small-N.** `classify.ConfusionCounts` ratios are `None` when undefined;
   `micro_prf` and `grounding.verdict_metrics` omit undefined keys; `runner.aggregate`
   averages a key only over the items carrying it. Counts ride `ItemScore.counts` through
   `EvalReport.counts_per_split` into the artifact's `counts` block, because micro-averages
   cannot be rebuilt from means.
6. **ADR-007's lesson, generalized.** `ranking` raises on empty gold (a dataset bug there);
   `classify` treats empty gold as a legal false-positive probe; no gate-A or refusal
   semantics exist in core — refusal and HITL outcomes are category-branched sparse keys in
   the consumer's `score_fn`, published next to the counts that give them context.
7. **Spot-check agreement.** `human.agreement_rate` computes the rate; `RunStamp` stamps it.

## Alternatives considered

- **Documentation only**: rejected — P2 had that and still needed two ADRs; prose cannot fail CI.
- **A scorer registry / abstract scorer base class**: rejected as a SPEC non-goal;
  `score_items` plus a plain callable admits no plugin drift.
- **Publishing `overall` (dev + test) behind a flag**: rejected; the honesty story is that
  the published block *cannot* contain dev numbers, not that a flag says whether it does.
- **Absolute thresholds instead of a ratchet**: rejected per P2 ADR-005 — too lax or
  unjustifiable; a measured baseline is self-calibrating.

## Consequences

- CI still cannot catch LLM-quality regressions; a prompt change that hurts faithfulness
  surfaces only when someone pays for a judged run. This gap is inherited, not solved.
- The guarantee is "the dishonest thing needs a deliberate override", not "impossible":
  `build_artifact`'s `extra` and `RunStamp.extra` write at the top level and can shadow fixed
  keys such as `metrics`. P2's adapter needs that hatch (it pops `refused`), so it stays; a
  colliding `extra` key is a review red flag.
- A newly added `Gate` is a no-op until the baseline is regenerated (the price of never gating
  on an unmeasured number), and the 0.02 tolerance absorbs regressions below the noise floor.
- `readme_in_sync` compares text after universal-newline reading; on a CRLF checkout it is a
  content check, and `sync_readme` normalizes to LF once (P2 will see a one-time diff).
- The 0.80 agreement bar is a stamped field that consumers enforce; refusing to build an
  artifact below it was rejected — an untrusted run still deserves an inspectable record.
- Sparse-key means hide each metric's N in the README table (counts live in the JSON only):
  an accepted small-N presentation gap and a 0.2 candidate.
