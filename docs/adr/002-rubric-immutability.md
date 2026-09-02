# ADR-002: Rubrics are immutable, sha-stamped, and byte-identical to P2's frozen prompt

Status: Accepted
Date: 2026-09-01

## Context

P2 froze its per-claim faithfulness judge prompt as a module literal (`JUDGE_PROMPT`) and
stamps `judge_prompt_sha256()` — the SHA-256 of the prompt's UTF-8 bytes — into every judged
artifact so that prompt drift is visible in the provenance chain rather than silently
changing what a number means. Committed P2 artifacts already carry the value
`31946648b81673911d373c4784496f672dc83f105e3a73d397a9e73f79915409`. P2's adoption of this
library is gated on byte-identical artifacts and README regions against its pre-PR outputs,
so the library must produce exactly that sha for exactly that prompt. At the same time the
library must be domain-free: the prompt names "Medicare Star Ratings / HEDIS measures", and
P1/P4 need the same rubric for their own domains.

## Decision

- **`Rubric(name, text)` is a frozen pydantic model** (`clinevals.judge`). `name` carries the
  version (`hedis-grounding-v1`); `text` is the system prompt. `Rubric.sha256` is
  `sha256(text.encode("utf-8"))` — the text only, matching P2's function exactly.
- **Released texts are immutable.** A rubric whose text has shipped in an artifact is never
  edited, not even for whitespace; a change is a new `Rubric` under a new `name`
  (`hedis-grounding-v2`). The old name and its sha stay valid provenance for old artifacts.
- **`faithfulness_rubric(domain, *, name)`** (`clinevals.grounding`) builds the rubric from a
  private template that equals P2's literal with exactly one substitution token, `__DOMAIN__`,
  in place of the domain phrase. `faithfulness_rubric("Medicare Star Ratings / HEDIS
  measures", name=...)` therefore reproduces `JUDGE_PROMPT` byte-for-byte and its `.sha256`
  equals `31946648…9409`. A golden sha test pins this in *both* repositories.
- **Stamping.** `RunStamp.rubric_sha256` writes the artifact key `rubric_sha256`. P2's
  artifacts use the key `judge_prompt_sha256`; the adoption adapter carries that key through
  `RunStamp.extra` so the *value* is shared and the *key* stays byte-compatible on P2's side.
- **Parsing is part of the contract.** `parse_grounding_verdict` and `build_judge_input` are
  P2's implementations verbatim (renamed), so the judge sees the same human turn and its
  output is read the same way.

## Alternatives considered

- **A rubric registry or versioned prompt store**: rejected — SPEC non-goal (no registry, no
  plugin system); a frozen value type plus a naming convention is sufficient and has no
  runtime state.
- **Hashing `name + text`**: rejected — P2's stamp hashes the text alone; changing the input
  would break the byte-identity bar for no provenance gain.
- **A templating engine (Jinja, `str.format`)**: rejected — braces already appear in the
  prompt's JSON shape, and any engine risks whitespace or escaping drift; one literal token
  and `str.replace` is auditable by eye.
- **Mutable prompt plus a CHANGELOG entry per edit**: rejected — drift would be documented,
  not visible in the artifact that carries the number.

## Consequences

- Any edit to the template, including whitespace or punctuation, fails the golden sha test in
  both repos. That is the intended failure mode: prompt changes are loud and require a new
  name.
- Immutability is enforced by tests and convention, not by the type system: `Rubric` accepts
  any text. Reviewers must treat a changed template or a reused name as a breaking change.
- The sha covers the system prompt only. `build_judge_input` (the human turn) and the parser
  are frozen by tests but not by the stamp; a change there alters judge behaviour without
  moving `rubric_sha256`. Accepted for 0.1 because both are P2-verbatim and pinned.
- Two rubrics with different names but identical text share a sha; the sha is provenance of
  the *text*, the name is provenance of the *version*. Consumers stamp both when they need to
  distinguish (`RunStamp.extra`).
- A domain phrase that itself contains `__DOMAIN__` would be substituted once, correctly, but
  the token's presence in a template is the only structural guard against a second, silent
  substitution; template authors must not introduce more tokens.
- The key-name split (`rubric_sha256` here, `judge_prompt_sha256` in P2) is a sharp edge the
  adapter owns. It is loud if forgotten — P2's byte-identity test fails — but it is a second
  place to keep in sync until P2 migrates its artifact key in a future minor release.
