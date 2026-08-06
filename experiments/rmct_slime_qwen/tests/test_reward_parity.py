"""Gate A: reward/advantage parity between the slime port and ctm's RMCT loop.

Two layers:

1. Synthetic parity (always runs, no GPU): identical rollout populations are
   pushed through ``ctm.training.rl.RLTrainer._build_training_batch`` (the
   original, datum construction stubbed out) and through
   ``slime_port.pipeline.compute_batch_advantages``. Rewards and advantages
   must be bitwise equal across estimators, normalization modes, and anchor
   weights.

2. Replay parity (runs when ``RMCT_ROLLOUT_DIR`` points at a real run's
   ``rollouts/`` directory of ``step_*.jsonl.zst`` files): reconstructs each
   step's populations from the persisted ``RolloutRecord``s and checks the
   ported pipeline reproduces the logged ``p_hat``/``p_ref``/``reward``/
   ``advantage`` values. Logged advantages are pre-KL (the backend's
   ``incorporate_kl_penalty`` mutates datums after logging), so no KL term
   belongs in this comparison.

Run: uv run python -m pytest experiments/rmct_slime_qwen/tests -q
"""

import json
import os
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slime_port.pipeline import (
    AdvantageConfig,
    build_batch_item,
    compute_batch_advantages,
)
from slime_port.pipeline import (
    BatchItem as PortBatchItem,
)
from slime_port.types import Rollout as PortRollout

TOL = 0.0  # identical arithmetic on identical floats — exact equality


# ── Synthetic population construction ────────────────────────────────────────


def _make_population(rng, n_perts=2, n_per_pert=24, parse_rate=0.9):
    """Rollout specs per perturbation: (trait, parsed, pert_idx)."""
    population = {}
    for pert in range(n_perts):
        p_bias = rng.uniform(0.05, 0.95)
        rollouts = []
        for _ in range(n_per_pert):
            parsed = rng.random() < parse_rate
            trait = float(rng.random() < p_bias)
            rollouts.append((trait, parsed, pert))
        population[pert] = rollouts
    return population


def _to_rollouts(population, cls):
    return {
        pert: [
            cls(
                tokens=[1, 2, 3],
                logprobs=[-0.1, -0.2, -0.3],
                text="x",
                trait_value=trait,
                perturbation_idx=pert,
                parsed_successfully=parsed,
            )
            for trait, parsed, pert in specs
        ]
        for pert, specs in population.items()
    }


def _ctm_reference(batch_items, *, advantage_estimator, normalization, anchor_weight, snr_normalizer="trait_std"):
    """Run the ORIGINAL ``_build_training_batch`` without a backend."""
    from ctm.training.rl import RLConfig, RLTrainer

    config = RLConfig(
        advantage_estimator=advantage_estimator,
        anchor_weight=anchor_weight,
        snr_normalizer=snr_normalizer,
    )
    config.loop.normalize = normalization
    trainer = RLTrainer.__new__(RLTrainer)
    trainer.config = config
    from ctm.core.rewards import ConsistencyReward

    trainer.reward_function = ConsistencyReward()
    trainer._rollout_logger = None
    trainer._snr_metrics = {}
    trainer._pending_rollout_meta = []
    trainer._create_rl_datum = lambda prompt, rollout, adv: (rollout, adv)
    _datums, consistency_rewards, anchor_rewards, advantages, _data = trainer._build_training_batch(batch_items)
    return consistency_rewards, anchor_rewards, advantages


def _make_ctm_item(port_item: PortBatchItem, ctm_rollout_map):
    """Mirror a port BatchItem into ctm's BatchItem with ctm Rollout twins."""
    from ctm.core.types import BatchItem as CtmBatchItem

    return CtmBatchItem(
        datapoint_idx=port_item.datapoint_idx,
        datapoint={},
        train_rollouts=[ctm_rollout_map[id(r)] for r in port_item.train_rollouts],
        anchor_rollouts=[ctm_rollout_map[id(r)] for r in port_item.anchor_rollouts],
        sampled_rollouts=[],
        initial_rollouts=[],
        p_hat=dict(port_item.p_hat),
        p_hat_counts=dict(port_item.p_hat_counts),
        p_ref=port_item.p_ref,
        p_ref_init=None,
        reference_rates=dict(port_item.reference_rates),
        reference_rate_counts=dict(port_item.reference_rate_counts),
        initial_reference_rates=dict(port_item.initial_reference_rates),
        initial_reference_rate_counts=dict(port_item.initial_reference_rate_counts),
        n_total=0,
        n_parsed=0,
        n_ref_parsed=port_item.n_ref_parsed,
        n_training_parsed=0,
    )


def _build_pair_batch(seed, n_items, anchor: bool):
    """Identical synthetic batch as (ctm BatchItems, port BatchItems)."""
    from ctm.core.types import Rollout as CtmRollout

    rng = random.Random(seed)
    port_items, ctm_items = [], []
    for i in range(n_items):
        population = _make_population(rng)
        port_rollouts = _to_rollouts(population, PortRollout)
        ctm_rollouts = _to_rollouts(population, CtmRollout)
        ctm_map = {}
        for pert in port_rollouts:
            for pr, cr in zip(port_rollouts[pert], ctm_rollouts[pert]):
                ctm_map[id(pr)] = cr
        initial_rates = {0: rng.uniform(0.1, 0.9)} if anchor else None
        initial_counts = {0: 128} if anchor else None
        port_item = build_batch_item(
            datapoint_idx=i,
            rollouts=port_rollouts,
            reference_indices=[0],
            training_indices=[1],
            initial_reference_rates=initial_rates,
            initial_reference_rate_counts=initial_counts,
        )
        assert port_item is not None
        port_items.append(port_item)
        ctm_items.append(_make_ctm_item(port_item, ctm_map))
    return ctm_items, port_items


@pytest.mark.parametrize("estimator", ["grpo_normalized", "snr_scaling", "matched_pair"])
@pytest.mark.parametrize("normalization", ["per_item", "pooled"])
@pytest.mark.parametrize("anchor_weight", [0.0, 0.5])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_synthetic_parity(estimator, normalization, anchor_weight, seed):
    ctm_items, port_items = _build_pair_batch(seed, n_items=4, anchor=anchor_weight > 0)
    expected = _ctm_reference(
        ctm_items,
        advantage_estimator=estimator,
        normalization=normalization,
        anchor_weight=anchor_weight,
    )
    got = compute_batch_advantages(
        port_items,
        AdvantageConfig(
            advantage_estimator=estimator,
            normalization=normalization,
            anchor_weight=anchor_weight,
        ),
    )
    exp_consistency, exp_anchor, exp_adv = expected
    assert got.consistency_rewards == exp_consistency
    assert got.anchor_rewards == exp_anchor
    assert got.advantages == exp_adv


def test_matched_pair_bare_normalizer():
    ctm_items, port_items = _build_pair_batch(7, n_items=3, anchor=False)
    expected = _ctm_reference(
        ctm_items,
        advantage_estimator="matched_pair",
        normalization="per_item",
        anchor_weight=0.0,
        snr_normalizer="none",
    )
    got = compute_batch_advantages(
        port_items,
        AdvantageConfig(advantage_estimator="matched_pair", snr_normalizer="none"),
    )
    assert got.advantages == expected[2]


# ── Replay parity against a real run's rollout log ───────────────────────────


def _iter_step_records(rollout_dir: Path):
    import zstandard

    for path in sorted(rollout_dir.glob("step_*.jsonl.zst")):
        raw = zstandard.ZstdDecompressor().decompress(path.read_bytes())
        records = [json.loads(line) for line in raw.decode().splitlines() if line]
        yield path.name, records


@pytest.mark.skipif(
    not os.environ.get("RMCT_ROLLOUT_DIR"),
    reason="set RMCT_ROLLOUT_DIR to a run's rollouts/ directory to replay it",
)
def test_replay_parity():
    """Reproduce logged p_hat / p_ref / reward / advantage from raw records.

    Assumes the mcq-bias sycophancy layout (reference index 0, one training
    index 1) and the paper estimator settings unless overridden via
    RMCT_REPLAY_{ESTIMATOR,NORMALIZATION,ANCHOR_WEIGHT}.
    """
    rollout_dir = Path(os.environ["RMCT_ROLLOUT_DIR"])
    config = AdvantageConfig(
        advantage_estimator=os.environ.get("RMCT_REPLAY_ESTIMATOR", "grpo_normalized"),
        normalization=os.environ.get("RMCT_REPLAY_NORMALIZATION", "per_item"),
        anchor_weight=float(os.environ.get("RMCT_REPLAY_ANCHOR_WEIGHT", "0.0")),
    )
    n_steps = 0
    n_compared = 0
    for name, records in _iter_step_records(rollout_dir):
        policy_records = [r for r in records if r["sample_source"] == "policy"]
        by_dp: dict[int, list[dict]] = {}
        for record in policy_records:
            by_dp.setdefault(record["datapoint_idx"], []).append(record)

        port_items = []
        train_records_per_item = []
        for dp_idx, dp_records in by_dp.items():  # insertion order = file order = batch order
            rollouts: dict[int, list[PortRollout]] = {}
            train_records = []
            for record in dp_records:
                rollout = PortRollout(
                    tokens=[],
                    logprobs=[],
                    text=record["completion_text"],
                    trait_value=record["trait_value"],
                    perturbation_idx=record["perturbation_idx"],
                    parsed_successfully=record["parsed_successfully"],
                )
                rollouts.setdefault(record["perturbation_idx"], []).append(rollout)
                if record["role"] == "train":
                    train_records.append((record, rollout))
            item = build_batch_item(
                datapoint_idx=dp_idx,
                rollouts=rollouts,
                reference_indices=[0],
                training_indices=sorted(k for k in rollouts if k != 0),
                p_ref_init_fallback=dp_records[0].get("p_ref_init"),
            )
            if item is None:
                assert not train_records, f"{name} dp{dp_idx}: item dropped but has train records"
                continue
            # The gradient set must be exactly the records logged as role=train.
            item.train_rollouts = [rollout for _, rollout in train_records]
            port_items.append(item)
            train_records_per_item.append(train_records)

            for record, _ in train_records:
                assert record["p_hat"] == pytest.approx(
                    item.p_hat[record["perturbation_idx"]], abs=1e-9
                ), f"{name} dp{dp_idx}: p_hat mismatch"
                assert record["p_ref"] == pytest.approx(item.p_ref, abs=1e-9), f"{name} dp{dp_idx}: p_ref mismatch"

        if not port_items:
            continue
        got = compute_batch_advantages(port_items, config)
        flat_records = [record for train_records in train_records_per_item for record, _ in train_records]
        assert len(flat_records) == len(got.consistency_rewards)
        for record, reward, advantage in zip(flat_records, got.consistency_rewards, got.advantages):
            if record["reward"] is not None:
                assert record["reward"] == pytest.approx(reward, abs=1e-9), f"{name}: reward mismatch"
                n_compared += 1
            if record["advantage"] is not None:
                assert record["advantage"] == pytest.approx(advantage, abs=1e-9), f"{name}: advantage mismatch"
        n_steps += 1
    assert n_steps > 0, "no step files found"
    print(f"replayed {n_steps} steps, {n_compared} rewards compared")
