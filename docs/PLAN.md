# Build plan — clinical-agent-evals

- [ ] **A. Package**: 10 flat modules per SPEC §4 + `__init__` re-exports + `py.typed`.
- [ ] **B. Tests**: per-module seeds ported from P2, classify boundaries, runner semantics,
      determinism suite (ubuntu + windows), rubric-sha golden, judge round-trip with fakes,
      ratchet edges, consumer-contract (mini-P2, mini-P1), public-surface, coverage ≥90%.
- [ ] **C. Docs**: README (executable mini-P1 quickstart, dogfooded eval table), CHANGELOG,
      ADR-001 (doctrine as library law), ADR-002 (rubric immutability).
- [ ] **D. Review gate**: adversarial multi-agent review; fix confirmed findings via PR.
- [ ] **E. Ship**: repo + CI green (incl. wheel-install job), tag **v0.1.0**.
- [ ] **F. P2 adoption PR**: shims + golden sha test + byte-identical artifact/README check.
