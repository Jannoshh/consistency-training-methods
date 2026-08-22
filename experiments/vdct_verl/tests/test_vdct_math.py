"""Hand-computed tests for the VDCT math core, Gate-A style on CPU.

Every closed-form expectation below is written out from the reward
definitions directly (independent of the implementation), and the per-group
standardization is cross-checked against an independent call to the
parity-tested ``slime_port.advantages`` functions on the same reward lists.

Run: uv run --no-sync python -m pytest experiments/vdct_verl/tests -q
"""

import math

import numpy as np
import pytest
from recipe.vdct.vdct_core import (
    ANSWER_KIND,
    DISTRIBUTION_KIND,
    JS_MAX,
    REFERENCE_VARIANT,
    TRAINING_VARIANT,
    VDCTConfig,
    VDCTRow,
    build_rows,
    compute_row_advantages,
    dump_columns,
    entropy,
    js_divergence,
    log_score,
    mean_distribution,
    merge_dump_fields,
    total_variation,
    worst_case_reward,
)
from slime_port.advantages import normalize_advantages, normalize_grouped

APPROX = 1e-12


# ── Distribution arithmetic, hand-computed ───────────────────────────────────


def test_js_of_identical_distributions_is_zero():
    assert js_divergence([0.2, 0.5, 0.3], [0.2, 0.5, 0.3]) == pytest.approx(0.0, abs=APPROX)


def test_js_of_disjoint_one_hots_is_ln2():
    assert js_divergence([1.0, 0.0], [0.0, 1.0]) == pytest.approx(JS_MAX, abs=APPROX)
    assert JS_MAX == pytest.approx(math.log(2.0), abs=0.0)


def test_js_hand_computed_value():
    # p=[0.5,0.5], q=[1,0], m=[0.75,0.25]
    p_term = 0.5 * math.log(0.5 / 0.75) + 0.5 * math.log(0.5 / 0.25)
    q_term = 1.0 * math.log(1.0 / 0.75)
    expected = 0.5 * p_term + 0.5 * q_term
    assert js_divergence([0.5, 0.5], [1.0, 0.0]) == pytest.approx(expected, abs=APPROX)


def test_js_is_symmetric_and_bounded():
    p, q = [0.7, 0.2, 0.1], [0.1, 0.1, 0.8]
    assert js_divergence(p, q) == pytest.approx(js_divergence(q, p), abs=APPROX)
    assert 0.0 <= js_divergence(p, q) <= JS_MAX


def test_js_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        js_divergence([0.5, 0.5], [1.0, 0.0, 0.0])


def test_log_score_hand_computed():
    q = [0.7, 0.3]
    expected = (2 * math.log(0.7) + math.log(0.3)) / 3
    assert log_score(q, [0, 0, 1], epsilon=1e-3) == pytest.approx(expected, abs=APPROX)


def test_log_score_floor_applies():
    assert log_score([1.0, 0.0], [1], epsilon=1e-3) == pytest.approx(math.log(1e-3), abs=APPROX)


def test_log_score_needs_answers_and_valid_indices():
    with pytest.raises(ValueError, match="at least one answer"):
        log_score([0.5, 0.5], [], epsilon=1e-3)
    with pytest.raises(ValueError, match="outside distribution"):
        log_score([0.5, 0.5], [2], epsilon=1e-3)


def test_worst_case_reward_formula():
    config = VDCTConfig(lambda_log_score=2.0, epsilon=1e-3)
    assert worst_case_reward(config) == pytest.approx(-JS_MAX + 2.0 * math.log(1e-3), abs=APPROX)


def test_entropy_uniform_and_one_hot():
    assert entropy([0.25] * 4) == pytest.approx(math.log(4.0), abs=APPROX)
    assert entropy([1.0, 0.0, 0.0]) == pytest.approx(0.0, abs=APPROX)


def test_total_variation_hand_computed():
    assert total_variation([0.6, 0.4], [0.4, 0.6]) == pytest.approx(0.2, abs=APPROX)
    assert total_variation([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0, abs=APPROX)


def test_mean_distribution():
    assert mean_distribution([(0.2, 0.8), (0.6, 0.4)]) == pytest.approx([0.4, 0.6], abs=APPROX)
    with pytest.raises(ValueError, match="length mismatch"):
        mean_distribution([(0.5, 0.5), (1.0,)])


# ── Batch assembly ───────────────────────────────────────────────────────────


def dist_row(variant, dist, group="g0", parse_ok=True):
    return VDCTRow(
        group_id=group,
        variant=variant,
        kind=DISTRIBUTION_KIND,
        parse_ok=parse_ok,
        option_distribution=tuple(dist) if dist is not None else None,
    )


def answer_row(variant, index, group="g0", parse_ok=True):
    return VDCTRow(
        group_id=group,
        variant=variant,
        kind=ANSWER_KIND,
        parse_ok=parse_ok,
        answer_index=index,
    )


def test_basic_group_rewards_hand_computed():
    """Two parsed distributions per side, answers on both sides: every reward
    reproduced from the formulas, advantages from slime's per-item math."""
    ref_d1, ref_d2 = (0.6, 0.3, 0.1), (0.4, 0.5, 0.1)
    train_d1, train_d2 = (0.2, 0.7, 0.1), (0.1, 0.8, 0.1)
    rows = [
        dist_row(REFERENCE_VARIANT, ref_d1),
        dist_row(REFERENCE_VARIANT, ref_d2),
        dist_row(TRAINING_VARIANT, train_d1),
        dist_row(TRAINING_VARIANT, train_d2),
        answer_row(REFERENCE_VARIANT, 0),
        answer_row(REFERENCE_VARIANT, 1),
        answer_row(TRAINING_VARIANT, 1),
    ]
    config = VDCTConfig(lambda_log_score=1.0, epsilon=1e-3)
    result = compute_row_advantages(rows, config)

    target = [(0.6 + 0.4) / 2, (0.3 + 0.5) / 2, 0.1]  # mean parsed reference dists
    # Reference side: proper-scoring term only (no anchor, 2026-08-22 decision).
    expected_ref = [log_score(d, [0, 1], 1e-3) for d in (ref_d1, ref_d2)]
    expected_train = [-js_divergence(d, target) + log_score(d, [1], 1e-3) for d in (train_d1, train_d2)]
    assert result.rewards[0] == pytest.approx(expected_ref[0], abs=APPROX)
    assert result.rewards[1] == pytest.approx(expected_ref[1], abs=APPROX)
    assert result.rewards[2] == pytest.approx(expected_train[0], abs=APPROX)
    assert result.rewards[3] == pytest.approx(expected_train[1], abs=APPROX)
    assert result.q_ref_target["g0"] == pytest.approx(target, abs=APPROX)

    # Advantages: per-side standardization, exactly slime's normalize_advantages.
    exp_ref_adv = normalize_advantages(expected_ref)
    exp_train_adv = normalize_advantages(expected_train)
    assert result.advantages[0] == pytest.approx(exp_ref_adv[0], abs=APPROX)
    assert result.advantages[1] == pytest.approx(exp_ref_adv[1], abs=APPROX)
    assert result.advantages[2] == pytest.approx(exp_train_adv[0], abs=APPROX)
    assert result.advantages[3] == pytest.approx(exp_train_adv[1], abs=APPROX)
    assert result.has_signal
    assert result.trainable[:4] == [True] * 4
    # Answer rows: no gradient, zero advantage.
    assert result.trainable[4:] == [False] * 3
    assert result.advantages[4:] == [0.0] * 3
    assert all(reason == "answer_row" for reason in result.skip_reasons[4:])


def test_advantages_match_independent_normalize_grouped_call():
    rows = [
        dist_row(REFERENCE_VARIANT, (0.6, 0.3, 0.1)),
        dist_row(REFERENCE_VARIANT, (0.4, 0.5, 0.1)),
        dist_row(TRAINING_VARIANT, (0.2, 0.7, 0.1)),
        dist_row(TRAINING_VARIANT, (0.1, 0.8, 0.1)),
        answer_row(REFERENCE_VARIANT, 0),
        answer_row(TRAINING_VARIANT, 1),
        dist_row(REFERENCE_VARIANT, (0.9, 0.05, 0.05), group="g1"),
        dist_row(REFERENCE_VARIANT, (0.7, 0.2, 0.1), group="g1"),
        dist_row(TRAINING_VARIANT, (0.3, 0.3, 0.4), group="g1"),
        dist_row(TRAINING_VARIANT, (0.5, 0.25, 0.25), group="g1"),
        answer_row(REFERENCE_VARIANT, 0, group="g1"),
        answer_row(TRAINING_VARIANT, 2, group="g1"),
    ]
    result = compute_row_advantages(rows)
    dist_indices = [i for i, row in enumerate(rows) if row.kind == DISTRIBUTION_KIND]
    flat_rewards = [result.rewards[i] for i in dist_indices]
    slices = [(0, 2), (2, 4), (4, 6), (6, 8)]  # (group, side) populations in order
    expected = normalize_grouped(flat_rewards, slices, "per_item")
    for flat_idx, row_idx in enumerate(dist_indices):
        assert result.advantages[row_idx] == pytest.approx(expected[flat_idx], abs=APPROX)


def test_unparsed_distribution_gets_worst_case_and_trains():
    rows = [
        dist_row(REFERENCE_VARIANT, (0.6, 0.3, 0.1)),
        dist_row(TRAINING_VARIANT, (0.2, 0.7, 0.1)),
        dist_row(TRAINING_VARIANT, None, parse_ok=False),
        answer_row(TRAINING_VARIANT, 1),
        answer_row(REFERENCE_VARIANT, 0),
    ]
    config = VDCTConfig()
    result = compute_row_advantages(rows, config)
    assert result.rewards[2] == pytest.approx(worst_case_reward(config), abs=APPROX)
    assert result.trainable[2]  # format compliance is trained
    assert result.skip_reasons[2] is None
    assert result.metrics["vdct/n_unparsed_dist_rows"] == 1.0


def test_unparsed_reference_distributions_are_excluded_from_target():
    rows = [
        dist_row(REFERENCE_VARIANT, (0.6, 0.3, 0.1)),
        dist_row(REFERENCE_VARIANT, None, parse_ok=False),
        dist_row(TRAINING_VARIANT, (0.2, 0.7, 0.1)),
        answer_row(TRAINING_VARIANT, 1),
        answer_row(REFERENCE_VARIANT, 0),
    ]
    result = compute_row_advantages(rows)
    # Target is the single parsed reference distribution, not a mean with the failure.
    assert result.q_ref_target["g0"] == pytest.approx([0.6, 0.3, 0.1], abs=APPROX)


def test_all_reference_unparsed_drops_the_training_side():
    """No parsed reference distributions -> no consistency target -> training
    rows skipped whole (the analog of RMCT's no_reference_rate)."""
    rows = [
        dist_row(REFERENCE_VARIANT, None, parse_ok=False),
        dist_row(TRAINING_VARIANT, (0.2, 0.7, 0.1)),
        answer_row(TRAINING_VARIANT, 1),
    ]
    result = compute_row_advantages(rows)
    assert result.q_ref_target["g0"] is None
    assert not result.trainable[1]
    assert result.advantages[1] == 0.0
    assert result.skip_reasons[1] == "no_reference_target"
    # Skip counters are core-owned metrics (one per skipped row).
    assert result.metrics["vdct/skip/no_reference_target"] == 1.0
    assert result.metrics["vdct/skip/answer_row"] == 1.0


def test_reference_side_has_no_consistency_term():
    """No anchor (2026-08-22 decision): reference rewards are the
    proper-scoring term alone, regardless of where the distribution sits."""
    current = (0.1, 0.1, 0.8)
    rows = [
        dist_row(REFERENCE_VARIANT, current),
        dist_row(REFERENCE_VARIANT, (0.1, 0.2, 0.7)),
        answer_row(REFERENCE_VARIANT, 2),
    ]
    config = VDCTConfig()
    result = compute_row_advantages(rows, config)
    assert result.rewards[0] == pytest.approx(log_score(current, [2], config.epsilon), abs=APPROX)


def test_missing_answer_samples_omit_log_score_term():
    rows = [
        dist_row(TRAINING_VARIANT, (0.2, 0.7, 0.1)),
        dist_row(TRAINING_VARIANT, (0.3, 0.6, 0.1)),
        dist_row(REFERENCE_VARIANT, (0.5, 0.3, 0.2)),
        answer_row(TRAINING_VARIANT, 1, parse_ok=False),  # no PARSED answers
        answer_row(REFERENCE_VARIANT, 0),
    ]
    result = compute_row_advantages(rows)
    target = [0.5, 0.3, 0.2]
    assert result.rewards[0] == pytest.approx(-js_divergence((0.2, 0.7, 0.1), target), abs=APPROX)
    assert result.metrics["vdct/sides_missing_answer_samples"] == 1.0


def test_lambda_zero_drops_the_proper_scoring_term():
    rows = [
        dist_row(TRAINING_VARIANT, (0.2, 0.7, 0.1)),
        dist_row(REFERENCE_VARIANT, (0.5, 0.3, 0.2)),
        answer_row(TRAINING_VARIANT, 1),
        answer_row(REFERENCE_VARIANT, 0),
    ]
    result = compute_row_advantages(rows, VDCTConfig(lambda_log_score=0.0))
    assert result.rewards[0] == pytest.approx(-js_divergence((0.2, 0.7, 0.1), [0.5, 0.3, 0.2]), abs=APPROX)
    assert result.rewards[1] == 0.0  # reference side: no consistency term, lambda=0 drops the log score


def test_zero_signal_batch_masks_everything():
    """Identical rewards within every slice -> zero advantages -> no signal."""
    d = (0.5, 0.3, 0.2)
    rows = [
        dist_row(REFERENCE_VARIANT, d),
        dist_row(REFERENCE_VARIANT, d),
        dist_row(TRAINING_VARIANT, d),
        dist_row(TRAINING_VARIANT, d),
        answer_row(REFERENCE_VARIANT, 0),
        answer_row(TRAINING_VARIANT, 0),
    ]
    result = compute_row_advantages(rows)
    assert not result.has_signal
    assert not any(result.trainable)
    assert all(a == 0.0 for a in result.advantages)
    assert "zero_advantage_batch" in result.skip_reasons


def test_row_order_invariance():
    rows = [
        dist_row(REFERENCE_VARIANT, (0.6, 0.3, 0.1)),
        dist_row(REFERENCE_VARIANT, (0.4, 0.5, 0.1)),
        dist_row(TRAINING_VARIANT, (0.2, 0.7, 0.1)),
        dist_row(TRAINING_VARIANT, (0.1, 0.8, 0.1)),
        answer_row(REFERENCE_VARIANT, 0),
        answer_row(TRAINING_VARIANT, 1),
    ]
    base = compute_row_advantages(rows)
    permutation = [3, 5, 0, 4, 2, 1]
    shuffled = [rows[i] for i in permutation]
    other = compute_row_advantages(shuffled)
    for new_idx, old_idx in enumerate(permutation):
        assert other.advantages[new_idx] == pytest.approx(base.advantages[old_idx], rel=1e-9)
        assert other.trainable[new_idx] == base.trainable[old_idx]
        assert other.rewards[new_idx] == pytest.approx(base.rewards[old_idx], rel=1e-9)


# ── Validation errors ────────────────────────────────────────────────────────


def test_unknown_variant_and_kind_are_rejected():
    with pytest.raises(ValueError, match="unknown variant"):
        compute_row_advantages([VDCTRow("g0", "bogus", DISTRIBUTION_KIND, True, (1.0,))])
    with pytest.raises(ValueError, match="unknown kind"):
        compute_row_advantages([VDCTRow("g0", REFERENCE_VARIANT, "logits", True, (1.0,))])


def test_parsed_distribution_row_requires_a_distribution():
    with pytest.raises(ValueError, match="without option_distribution"):
        compute_row_advantages([VDCTRow("g0", REFERENCE_VARIANT, DISTRIBUTION_KIND, True, None)])


# ── verl-batch adapters ──────────────────────────────────────────────────────


def test_build_rows_from_object_arrays():
    non_tensor = {
        "group_id": np.array(["g0", "g0"], dtype=object),
        "variant": np.array([REFERENCE_VARIANT, TRAINING_VARIANT], dtype=object),
        "kind": np.array([DISTRIBUTION_KIND, ANSWER_KIND], dtype=object),
        "parse_ok": np.array([True, False]),
        "option_distribution": np.array([[0.5, 0.5], None], dtype=object),
        "answer_index": np.array([None, None], dtype=object),
    }
    rows = build_rows(non_tensor)
    assert rows[0] == VDCTRow("g0", REFERENCE_VARIANT, DISTRIBUTION_KIND, True, (0.5, 0.5))
    assert rows[1].kind == ANSWER_KIND and rows[1].option_distribution is None


def test_build_rows_requires_vdct_fields():
    with pytest.raises(KeyError, match="missing VDCT fields"):
        build_rows({"group_id": ["g0"], "variant": [REFERENCE_VARIANT]})


def test_dump_columns_align_with_rows():
    rows = [
        dist_row(REFERENCE_VARIANT, (0.6, 0.4, 0.0)[:3]),
        dist_row(TRAINING_VARIANT, (0.2, 0.7, 0.1)),
        answer_row(TRAINING_VARIANT, 1),
    ]
    result = compute_row_advantages(rows)
    columns = dump_columns(rows, result)
    assert columns["vdct_reward"] == result.rewards
    assert columns["advantage"] == result.advantages
    assert columns["trainable"] == result.trainable
    assert columns["skip_reason"] == result.skip_reasons
    # Every row carries its group's target, answer rows included.
    assert columns["q_ref_target"] == [result.q_ref_target["g0"]] * 3


def test_merge_dump_fields():
    merged = merge_dump_fields({"reward": [0.0, 0.0]}, {"kind": ["distribution", "answer"]}, 2)
    assert merged["kind"] == ["distribution", "answer"]
    # never overwrites, never accepts a misaligned column
    assert merge_dump_fields({"kind": ["a", "b"]}, {"kind": ["c", "d"]}, 2)["kind"] == ["a", "b"]
    with pytest.raises(ValueError, match="dump field"):
        merge_dump_fields({}, {"kind": ["only-one"]}, 2)
