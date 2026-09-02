"""Tests for ``clinevals.human``: seeded stratified sampling and judge-human agreement."""

import pytest

from clinevals.human import agreement_rate, stratified_sample

# Input order deliberately puts "beta" first so the tests can show strata are SORTED by key,
# not visited in first-seen order. Three strata of four items each.
_ITEMS: tuple[str, ...] = tuple(
    f"{stratum}-{i}" for stratum in ("beta", "alpha", "gamma") for i in range(4)
)

# Uneven strata: alpha has 1 member, beta 3, gamma 2.
_UNEVEN: tuple[str, ...] = ("beta-0", "gamma-0", "alpha-0", "beta-1", "gamma-1", "beta-2")


def _stratum(item: str) -> str:
    return item.split("-")[0]


# --- stratified_sample -------------------------------------------------------------------


def test_same_inputs_and_seed_give_identical_output() -> None:
    first = stratified_sample(_ITEMS, n=7, key=_stratum, seed=42)
    second = stratified_sample(_ITEMS, n=7, key=_stratum, seed=42)
    assert first == second
    assert len(first) == 7


def test_n_equal_to_stratum_count_yields_exactly_one_per_stratum() -> None:
    sample = stratified_sample(_ITEMS, n=3, key=_stratum, seed=1)
    # Round-robin over strata sorted by key: alpha, beta, gamma — regardless of input order.
    assert [_stratum(item) for item in sample] == ["alpha", "beta", "gamma"]


def test_every_stratum_is_covered_before_any_stratum_repeats() -> None:
    sample = stratified_sample(_ITEMS, n=8, key=_stratum, seed=3)
    strata_in_order = [_stratum(item) for item in sample]
    # First pass covers all three strata; the second pass covers all three again; the
    # remaining two are a partial third pass. No stratum appears twice within a pass.
    assert strata_in_order[0:3] == ["alpha", "beta", "gamma"]
    assert strata_in_order[3:6] == ["alpha", "beta", "gamma"]
    assert strata_in_order[6:8] == ["alpha", "beta"]


def test_round_robin_skips_exhausted_strata_and_keeps_going() -> None:
    sample = stratified_sample(_UNEVEN, n=6, key=_stratum, seed=5)
    expected = ["alpha", "beta", "gamma", "beta", "gamma", "beta"]
    assert [_stratum(item) for item in sample] == expected


def test_n_zero_returns_empty_list() -> None:
    assert stratified_sample(_ITEMS, n=0, key=_stratum, seed=0) == []


def test_empty_items_return_empty_list_for_any_n() -> None:
    assert stratified_sample([], n=5, key=_stratum, seed=0) == []


def test_n_greater_than_len_returns_every_item_exactly_once() -> None:
    sample = stratified_sample(_ITEMS, n=len(_ITEMS) + 10, key=_stratum, seed=9)
    assert len(sample) == len(_ITEMS)
    assert set(sample) == set(_ITEMS)
    assert len(set(sample)) == len(sample)


def test_sample_never_contains_duplicates() -> None:
    for n in range(len(_ITEMS) + 1):
        sample = stratified_sample(_ITEMS, n=n, key=_stratum, seed=11)
        assert len(sample) == n
        assert len(set(sample)) == n


def test_different_seed_keeps_membership_when_sampling_everything() -> None:
    # Order is (very likely) different across seeds, so only membership is asserted.
    seed_a = stratified_sample(_ITEMS, n=len(_ITEMS), key=_stratum, seed=1)
    seed_b = stratified_sample(_ITEMS, n=len(_ITEMS), key=_stratum, seed=2)
    assert set(seed_a) == set(seed_b) == set(_ITEMS)


def test_different_seed_keeps_per_stratum_quota_for_partial_samples() -> None:
    seed_a = stratified_sample(_ITEMS, n=7, key=_stratum, seed=1)
    seed_b = stratified_sample(_ITEMS, n=7, key=_stratum, seed=2)
    quota_a = sorted(_stratum(item) for item in seed_a)
    quota_b = sorted(_stratum(item) for item in seed_b)
    assert quota_a == quota_b == ["alpha", "alpha", "alpha", "beta", "beta", "gamma", "gamma"]


def test_negative_n_raises_value_error() -> None:
    with pytest.raises(ValueError, match="n must be >= 0"):
        stratified_sample(_ITEMS, n=-1, key=_stratum, seed=0)


def test_input_is_not_mutated() -> None:
    items = list(_ITEMS)
    stratified_sample(items, n=5, key=_stratum, seed=4)
    assert items == list(_ITEMS)


# --- agreement_rate ----------------------------------------------------------------------


def test_agreement_rate_is_the_exact_matching_fraction() -> None:
    pairs = [
        ("supported", "supported"),
        ("unsupported", "supported"),
        ("contradicted", "contradicted"),
        ("supported", "supported"),
    ]
    assert agreement_rate(pairs) == 0.75


def test_agreement_rate_bounds() -> None:
    assert agreement_rate([("a", "a"), ("b", "b")]) == 1.0
    assert agreement_rate([("a", "b"), ("b", "a")]) == 0.0


def test_agreement_rate_is_exact_string_match() -> None:
    assert agreement_rate([("Supported", "supported")]) == 0.0
    assert agreement_rate([("supported ", "supported")]) == 0.0


def test_agreement_rate_empty_raises_value_error() -> None:
    with pytest.raises(ValueError, match="zero pairs"):
        agreement_rate([])
