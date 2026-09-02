# Architecture Decision Records

Decisions that shaped `clinical-agent-evals` (import name `clinevals`), in the order they were
made. Each record is self-contained: the context that forced a choice, the choice, the
alternatives rejected, and the trade-offs accepted. `docs/SPEC.md` states *what* the library
is; these records preserve *why* it is built this way. Two of the decisions were first made in
`hedis-spec-copilot` (its ADR-005 and ADR-007) and are inherited here as library law.

## Index

| ADR | Decision | Enforced / implemented in |
|---|---|---|
| [001](001-keyless-ratchet-test-split-doctrine.md) | Keyless CI, measured ratchet with skip/fail asymmetry, and test-split-only publication encoded structurally in the API | `judge.invoke_judge`, `fakes.scripted_judge`, `runner.EvalReport.test_overall`, `artifacts.build_artifact`, `ratchet.compare_to_baseline`, `artifacts.readme_in_sync` |
| [002](002-rubric-immutability.md) | Rubrics are frozen, sha-stamped, and reproduce P2's `JUDGE_PROMPT` byte-for-byte for the HEDIS domain phrase | `judge.Rubric`, `grounding.faithfulness_rubric`, `artifacts.RunStamp.rubric_sha256`, golden sha tests in both repos |

## Conventions

- **Format**: Status (Accepted / Superseded by ADR-XXX / Deprecated), Date, Context,
  Decision, Alternatives considered, Consequences — with honest trade-offs, not sales copy.
- **Numbering**: sequential, zero-padded (`NNN-short-slug.md`). Numbers are never reused.
- **Lifecycle**: ADRs are immutable history. A changed decision gets a *new* ADR that
  references and supersedes the old one; the old record keeps the context that made the
  original choice sensible at the time.
- **Doctrine coupling**: a change to any default named in ADR-001 (the 0.02 tolerance, the
  skip/fail asymmetry, what `build_artifact` publishes) or to a released rubric text (ADR-002)
  requires a new ADR and, under the 0.x contract, a minor release with a Migration section.
