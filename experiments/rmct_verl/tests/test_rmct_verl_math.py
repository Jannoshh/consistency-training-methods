"""Gate A for the verl port: the row-level RMCT math must agree, bit for bit,
with calling the slime port's pipeline directly.

The arithmetic is not duplicated here — ``rmct_core.compute_row_advantages``
delegates to ``slime_port.pipeline``. What these tests actually pin down is the
part that IS new in the verl port and could silently corrupt a run:

* the flat-row -> (group, variant) -> Rollout packing, and
* the advantage-back-to-row mapping,

both checked against an independently constructed call to
``build_batch_item`` / ``compute_batch_advantages`` on the same populations.
Plus the masking semantics (reference rows, unparsed training rows, zero-signal
batches) and the KL term.

Run: uv run --no-sync python -m pytest experiments/rmct_verl/tests -q
"""

import random
import sys
from pathlib import Path

import pytest
import torch

_RMCT_VERL = Path(__file__).resolve().parents[1]
if str(_RMCT_VERL) not in sys.path:
    sys.path.insert(0, str(_RMCT_VERL))

from recipe.rmct.rmct_core import (
    REFERENCE_VARIANT,
    TRAINING_VARIANT,
    AdvantageConfig,
    RMCTRow,
    centered_kl_penalty,
    compute_row_advantages,
)
from slime_port.pipeline import build_batch_item, compute_batch_advantages
from slime_port.types import Rollout

TOL = 0.0  # identical arithmetic on identical floats — exact equality


# ── Synthetic batches ────────────────────────────────────────────────────────


def make_rows(rng, n_groups=5, n_ref=16, n_train=16, parse_rate=0.9, interleave=True):
    """A flat row list shaped like a verl batch (rollout.n repeats per variant)."""
    rows = []
    per_group = []
    for group_idx in range(n_groups):
        group_id = f"g{group_idx}"
        p_ref = rng.uniform(0.05, 0.95)
        p_hat = rng.uniform(0.05, 0.95)
        group_rows = []
        for variant, count, rate in (
            (REFERENCE_VARIANT, n_ref, p_ref),
            (TRAINING_VARIANT, n_train, p_hat),
        ):
            for _ in range(count):
                group_rows.append(
                    RMCTRow(
                        group_id=group_id,
                        variant=variant,
                        parse_ok=rng.random() < parse_rate,
                        trait=float(rng.random() < rate),
                    )
                )
        per_group.append(group_rows)
    if interleave:
        # verl's repeat(interleave=True) keeps a row's n copies contiguous, but
        # balance_batch can reorder groups arbitrarily. Shuffle group order to
        # prove the mapping does not depend on it.
        rng.shuffle(per_group)
    for group_rows in per_group:
        rows.extend(group_rows)
    return rows


def reference_pipeline(rows, config):
    """Call the slime port directly on the same populations, independently of
    ``rmct_core``. Returns {(group_id, row_index_within_group): advantage}."""
    group_order = []
    populations = {}
    row_index = {}  # id(rollout) -> flat row index
    for idx, row in enumerate(rows):
        pert = 0 if row.variant == REFERENCE_VARIANT else 1
        if row.group_id not in populations:
            populations[row.group_id] = {0: [], 1: []}
            group_order.append(row.group_id)
        rollout = Rollout(
            tokens=[],
            logprobs=[],
            text="",
            trait_value=float(row.trait),
            perturbation_idx=pert,
            parsed_successfully=bool(row.parse_ok),
            answer_parsed=bool(row.parse_ok),
            has_logprobs=True,
        )
        populations[row.group_id][pert].append(rollout)
        row_index[id(rollout)] = idx

    items = []
    dropped_groups = set()
    for group_idx, group_id in enumerate(group_order):
        item = build_batch_item(
            datapoint_idx=group_idx,
            rollouts=populations[group_id],
            reference_indices=[0],
            training_indices=[1],
            n_gradient=None,
        )
        if item is None:
            dropped_groups.add(group_id)
        else:
            items.append(item)
    result = compute_batch_advantages(items, config)

    advantages = [0.0] * len(rows)
    for rollout, advantage in zip(result.grad_rollouts, result.advantages):
        advantages[row_index[id(rollout)]] = float(advantage) if result.has_signal else 0.0
    return advantages, result, dropped_groups


# ── Gate A: bitwise parity with the slime port ───────────────────────────────


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize(
    "config",
    [
        AdvantageConfig(advantage_estimator="grpo_normalized", normalization="per_item"),
        AdvantageConfig(advantage_estimator="grpo_normalized", normalization="pooled"),
        AdvantageConfig(advantage_estimator="snr_scaling", normalization="per_item"),
        AdvantageConfig(advantage_estimator="snr_scaling", snr_mode="hard", snr_z=1.0),
        AdvantageConfig(advantage_estimator="matched_pair"),
        AdvantageConfig(advantage_estimator="matched_pair", snr_normalizer="none"),
    ],
    ids=lambda c: f"{c.advantage_estimator}-{c.normalization}-{c.snr_mode}-{c.snr_normalizer}",
)
def test_row_advantages_match_slime_pipeline(seed, config):
    rng = random.Random(seed)
    rows = make_rows(rng)
    expected, expected_result, _ = reference_pipeline(rows, config)

    got = compute_row_advantages(rows, config)

    assert got.has_signal == expected_result.has_signal
    assert len(got.advantages) == len(rows)
    for idx, (a, b) in enumerate(zip(got.advantages, expected)):
        assert a == pytest.approx(b, abs=TOL), f"row {idx}: {a} != {b}"


@pytest.mark.parametrize("seed", range(4))
def test_row_advantages_invariant_to_row_order(seed):
    """balance_batch may reorder the batch; the per-row advantage must not move
    with it. (Only true because ``normalization='per_item'`` standardizes within
    a group — the pooled mode is order-invariant too, being a batch statistic.)"""
    rng = random.Random(seed)
    rows = make_rows(rng)
    config = AdvantageConfig()
    base = compute_row_advantages(rows, config)

    permutation = list(range(len(rows)))
    random.Random(seed + 100).shuffle(permutation)
    shuffled = [rows[i] for i in permutation]
    other = compute_row_advantages(shuffled, config)

    for new_idx, old_idx in enumerate(permutation):
        assert other.advantages[new_idx] == pytest.approx(base.advantages[old_idx], abs=TOL)
        assert other.trainable[new_idx] == base.trainable[old_idx]


# ── Masking semantics ────────────────────────────────────────────────────────


def test_reference_rows_are_never_trainable():
    rng = random.Random(0)
    rows = make_rows(rng, n_groups=3)
    result = compute_row_advantages(rows)
    for row, trainable, advantage, reason in zip(rows, result.trainable, result.advantages, result.skip_reasons):
        if row.variant == REFERENCE_VARIANT:
            assert not trainable
            assert advantage == 0.0
            assert reason == "reference_row"


def test_unparsed_training_rows_are_zero_masked():
    rows = (
        [RMCTRow("g0", REFERENCE_VARIANT, True, float(i % 2)) for i in range(8)]
        # mixed traits, so the group has within-group variance and real signal
        + [RMCTRow("g0", TRAINING_VARIANT, True, float(i < 4)) for i in range(6)]
        + [RMCTRow("g0", TRAINING_VARIANT, False, 0.0) for _ in range(2)]
    )
    result = compute_row_advantages(rows)
    for row, trainable, advantage, reason in zip(rows, result.trainable, result.advantages, result.skip_reasons):
        if row.variant == TRAINING_VARIANT and not row.parse_ok:
            assert not trainable
            assert advantage == 0.0
            assert reason == "answer_parse_failure"
    # The parsed training rows still carry gradient.
    assert any(result.trainable)


def test_group_without_a_parsed_reference_is_dropped():
    rows = [RMCTRow("g0", REFERENCE_VARIANT, False, 0.0) for _ in range(4)] + [
        RMCTRow("g0", TRAINING_VARIANT, True, 1.0) for _ in range(4)
    ]
    result = compute_row_advantages(rows)
    assert not any(result.trainable)
    assert result.p_ref["g0"] is None
    assert set(result.skip_reasons) == {"reference_row", "no_reference_rate"}


def test_zero_signal_batch_masks_everything():
    """Every training rollout in a group sharing one trait value gives zero
    within-group variance -> zero advantage -> no signal -> nothing trains."""
    rows = [RMCTRow("g0", REFERENCE_VARIANT, True, 0.0) for _ in range(4)] + [
        RMCTRow("g0", TRAINING_VARIANT, True, 1.0) for _ in range(4)
    ]
    result = compute_row_advantages(rows)
    assert not result.has_signal
    assert not any(result.trainable)
    assert all(a == 0.0 for a in result.advantages)
    assert "zero_advantage_batch" in result.skip_reasons

    # The trainer's guard: masking every row leaves no tokens, and token-mean
    # loss aggregation would divide by that zero.
    response_mask = torch.ones(len(rows), 5, dtype=torch.int64)
    trainable = torch.tensor(result.trainable, dtype=torch.int64).unsqueeze(-1)
    assert int((response_mask * trainable).sum().item()) == 0


def test_anchor_weight_is_rejected():
    with pytest.raises(NotImplementedError):
        compute_row_advantages([], AdvantageConfig(anchor_weight=0.5))


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError, match="unknown variant"):
        compute_row_advantages([RMCTRow("g0", "anchor", True, 1.0)])


# ── KL term ──────────────────────────────────────────────────────────────────


def test_centered_kl_matches_the_slime_formula():
    torch.manual_seed(0)
    policy = torch.randn(6, 9)
    ref = torch.randn(6, 9)
    mask = (torch.rand(6, 9) > 0.3).to(torch.int64)
    kl_coef = 0.05

    penalty, avg_diff = centered_kl_penalty(policy, ref, mask, kl_coef)

    # slime_port/rmct_advantage.py, per-sample form.
    diffs = [(policy[i] - ref[i]) * mask[i] for i in range(6)]
    total_mask = sum(mask[i].sum() for i in range(6))
    total_diff = sum(d.sum() for d in diffs)
    expected_avg = total_diff / torch.clamp(total_mask.float(), min=1e-8)
    expected = torch.stack([kl_coef * mask[i] * (expected_avg - diffs[i]) for i in range(6)])

    assert torch.equal(penalty, expected)
    assert avg_diff.item() == pytest.approx(expected_avg.item(), abs=1e-12)


def test_centered_kl_is_mean_zero_over_the_masked_population():
    torch.manual_seed(1)
    policy, ref = torch.randn(4, 7), torch.randn(4, 7)
    mask = (torch.rand(4, 7) > 0.5).to(torch.float32)
    penalty, _ = centered_kl_penalty(policy, ref, mask, 0.05)
    assert penalty.sum().item() == pytest.approx(0.0, abs=1e-5)
    # Masked-out tokens get exactly zero, never a stray centering constant.
    assert torch.equal(penalty[mask == 0], torch.zeros_like(penalty[mask == 0]))


def test_centered_kl_survives_an_all_masked_batch():
    policy, ref = torch.randn(2, 3), torch.randn(2, 3)
    mask = torch.zeros(2, 3)
    penalty, avg_diff = centered_kl_penalty(policy, ref, mask, 0.05)
    assert torch.equal(penalty, torch.zeros_like(penalty))
    assert avg_diff.item() == 0.0
