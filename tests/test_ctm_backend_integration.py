"""End-to-end RL + SFT loops against a fake in-memory backend.

Unlike test_rl_pipelining (which mocks _collect_rollouts and measures overlap),
these tests run the REAL rollout collection, trait classification, reward /
advantage construction, datum creation, and rollout persistence — only the
compute substrate is faked at the TrainingBackend seam. This is the contract
test any new backend (local torch/PEFT, vLLM) must also satisfy.
"""

import asyncio
import json
import math
from unittest.mock import MagicMock, patch

import pytest
import torch
from tinker import types

from ctm.backends.base import ForwardBackwardOutput
from ctm.core.rewards import ConsistencyReward
from ctm.evals.analysis.rollouts import iter_rollouts, load_index
from ctm.training.rl import (
    GenerationConfig,
    RateEstimationConfig,
    RLConfig,
    RLTrainer,
    TrainingLoopConfig,
    TrainingSamplingConfig,
    _classify_traits,
)
from ctm.training.sft import SFTConfig, train_sft
from ctm.core.config import AdamConfig, CheckpointConfig, LoRAConfig

A_TOKEN, B_TOKEN = 65, 66  # completions: "(A)" (biased) vs "(B)"


@pytest.mark.parametrize("value", [math.nan, math.inf, -0.01, 1.01])
def test_trait_classification_rejects_non_probability_values(value):
    raw = [([], [], "full", "answer")]
    with pytest.raises(ValueError, match=r"finite \[0, 1\]"):
        asyncio.run(_classify_traits(lambda _answer, _datapoint, _messages: value, raw, {}, []))


def test_trait_classification_preserves_abstention():
    raw = [([], [], "full", "answer")]
    assert asyncio.run(_classify_traits(lambda _answer, _datapoint, _messages: None, raw, {}, [])) == [None]


class FakeTokenizer:
    def decode(self, tokens):
        if not tokens:
            return ""
        if tokens[0] == A_TOKEN:
            return "The best answer is: (A)"
        if tokens[0] == B_TOKEN:
            return "The best answer is: (B)"
        return " ".join(str(t) for t in tokens)


class FakeRenderer:
    """Encodes which perturbation a prompt came from in its first token."""

    def build_generation_prompt(self, messages):
        cued = any("bias" in m["content"] for m in messages)
        return types.ModelInput.from_ints(tokens=[1 if cued else 0])

    def build_supervised_example(self, messages):
        return types.ModelInput.from_ints(tokens=[1, 2, 3, 4]), torch.tensor([0.0, 0.0, 1.0, 1.0])

    def parse_response(self, tokens):
        raise RuntimeError("no structured parse in fake")  # decode_response falls back to tokenizer

    def get_stop_sequences(self):
        return []


class FakeSampler:
    """Deterministic trait mix: cued prompts answer (A) 75% of the time, neutral 25%."""

    async def sample(self, prompt, *, max_tokens, temperature, stop, num_samples):
        from ctm.backends.base import SampledSequence

        cued = prompt.to_ints()[0] == 1
        seqs = []
        for i in range(num_samples):
            biased = (i % 4 != 0) if cued else (i % 4 == 0)  # 3/4 vs 1/4
            seqs.append(SampledSequence(tokens=[A_TOKEN if biased else B_TOKEN], logprobs=[-0.1]))
        return seqs


class FakeBackend:
    """In-memory TrainingBackend that records every call."""

    renderer_source = "tinker"

    def __init__(self):
        self.fb_datums: list[list] = []
        self.fb_loss_fns: list[str] = []
        self.optim_lrs: list[float] = []
        self.kl_calls = 0
        self.refresh_calls = 0
        self.checkpoints: list[str] = []
        self.shutdown_calls = 0

    def shutdown(self):
        self.shutdown_calls += 1

    def setup(self, **kwargs):
        pass

    def policy_sampler(self, name):
        return FakeSampler()

    async def refresh_policy_sampler(self, name):
        self.refresh_calls += 1
        return FakeSampler()

    def base_sampler(self):
        return FakeSampler()

    async def submit_forward_backward(self, datums, loss_fn):
        self.fb_datums.append(list(datums))
        self.fb_loss_fns.append(loss_fn)
        lengths = [d.loss_fn_inputs["target_tokens"].to_torch().shape[0] for d in datums]

        class Pending:
            async def result(self):
                return ForwardBackwardOutput(logprobs=[-0.1 * torch.ones(n) for n in lengths], metrics={"loss": 0.5})

        return Pending()

    async def submit_optim_step(self, *, learning_rate, adam):
        self.optim_lrs.append(learning_rate)

        class Pending:
            async def result(self):
                return None

        return Pending()

    async def incorporate_kl_penalty(self, datums, *, kl_coef, kl_discount_factor):
        self.kl_calls += 1
        return {"kl_penalty": 0.0}

    async def save_checkpoint(self, *, name, log_dir, loop_state, kind):
        self.checkpoints.append(name)
        return {"sampler_path": f"fake://checkpoint/{name}", "state_path": None}


def _perturbations():
    return [
        lambda dp: {"messages": [{"role": "user", "content": dp["question"]}]},
        lambda dp: {"messages": [{"role": "user", "content": "bias " + dp["question"]}]},
    ]


def _trait(answer_text, dp, realized_messages):
    return 1.0 if "(A)" in answer_text else 0.0


class TestRLEndToEnd:
    def _run(self, tmp_path, *, trait_classifier=_trait, **config_overrides):
        cfg = dict(
            experiment_name="itest",
            run_name="rl",
            lora=LoRAConfig(rank=4, seed=0),
            optimizer=AdamConfig(learning_rate=1e-4),
            reference_rate=RateEstimationConfig(perturbation_indices=[0], n_rollouts=8),
            # n_rollouts_for_consistency=None → gradient set = ALL 8 parsed rollouts, so the
            # 6:2 trait mix (and hence nonzero advantage variance) is deterministic — no
            # random.sample subset can collapse a batch to all-equal rewards and skip it.
            training=TrainingSamplingConfig(
                perturbation_indices=[1], n_rollouts_for_rate=8, n_rollouts_for_consistency=None
            ),
            loop=TrainingLoopConfig(batch_size=2, refresh_policy_every_n_steps=1, n_epochs=1),
            generation=GenerationConfig(max_new_tokens=16, temperature=0.7),
            checkpoint=CheckpointConfig(),
            kl_coef=0.1,
            anchor_weight=0.5,
            log_base_dir=str(tmp_path / "logs"),
            rollout_dir=str(tmp_path / "rollouts"),
        )
        cfg.update(config_overrides)
        config = RLConfig(**cfg)
        backend = FakeBackend()
        trainer = RLTrainer(config=config, reward_function=ConsistencyReward(), backend=backend)
        # Wire fakes directly instead of setup() (no real model/renderer offline)
        trainer.setup_done = True
        trainer.renderer = FakeRenderer()
        trainer.tokenizer = FakeTokenizer()
        trainer.sampling_client = FakeSampler()
        trainer.base_sampling_client = FakeSampler()
        trainer.anchor_sampling_client = FakeSampler()

        datapoints = [{"question": f"Q{i}"} for i in range(4)]
        with patch("ctm.training.rl.setup_logging") as mock_logging:
            logger = MagicMock()
            mock_logging.return_value = logger
            final = asyncio.run(
                trainer.train(
                    datapoints=datapoints,
                    perturbation_fns=_perturbations(),
                    trait_classifier=trait_classifier,
                )
            )
        return final, backend, trainer, logger

    def test_preloop_failure_still_shuts_down_backend(self, tmp_path):
        config = RLConfig(
            experiment_name="itest",
            run_name="preloop-failure",
            lora=LoRAConfig(rank=4, seed=0),
            optimizer=AdamConfig(learning_rate=1e-4),
            reference_rate=RateEstimationConfig(perturbation_indices=[0], n_rollouts=2),
            training=TrainingSamplingConfig(perturbation_indices=[1], n_rollouts_for_rate=2),
            loop=TrainingLoopConfig(batch_size=1, n_epochs=1),
            generation=GenerationConfig(max_new_tokens=4, temperature=0.7),
            checkpoint=CheckpointConfig(),
            log_base_dir=str(tmp_path / "logs"),
        )
        backend = FakeBackend()
        trainer = RLTrainer(config=config, reward_function=ConsistencyReward(), backend=backend)
        trainer.setup_done = True

        with patch("ctm.training.rl.setup_logging", side_effect=RuntimeError("logger failed")):
            with pytest.raises(RuntimeError, match="logger failed"):
                asyncio.run(trainer.train([], [], _trait))

        assert backend.shutdown_calls == 1

    def test_full_loop_trains_and_checkpoints(self, tmp_path):
        final, backend, _, _ = self._run(tmp_path)
        assert final == "fake://checkpoint/itest_rl"
        # provenance manifest written into the run's log dir
        from ctm.training.manifest import read_run_manifest

        manifest = read_run_manifest(tmp_path / "logs" / "itest" / "rl")
        assert manifest["kind"] == "rl" and manifest["backend"] == "FakeBackend"
        assert manifest["n_datapoints"] == 4 and manifest["n_perturbations"] == 2
        # 4 datapoints / batch 2 = 2 steps, all with a 0.5 trait gap → no skips
        assert len(backend.fb_datums) == 2
        assert backend.fb_loss_fns == ["ppo", "ppo"]
        assert len(backend.optim_lrs) == 2
        assert backend.refresh_calls == 2  # refresh_every=1 remains strict after actual updates
        assert backend.kl_calls == 2  # kl_coef > 0 routes through the backend
        assert backend.shutdown_calls == 1
        # real tinker datums with RL loss inputs
        datum = backend.fb_datums[0][0]
        assert {"advantages", "logprobs", "mask", "target_tokens"} <= set(datum.loss_fn_inputs)
        advs = datum.loss_fn_inputs["advantages"].to_torch()
        assert advs.abs().sum() > 0

    def test_refresh_waits_for_optimizer_when_accumulating(self, tmp_path):
        _, backend, _, _ = self._run(
            tmp_path,
            loop=TrainingLoopConfig(
                batch_size=1,
                gradient_accumulation_steps=2,
                refresh_policy_every_n_steps=1,
                n_epochs=1,
            ),
        )

        assert len(backend.fb_datums) == 4
        assert len(backend.optim_lrs) == 2
        assert backend.refresh_calls == 2

    def test_rollouts_persisted_with_context(self, tmp_path):
        _, _, trainer, _ = self._run(tmp_path)
        rollout_dir = tmp_path / "rollouts"
        index = load_index(rollout_dir)
        assert [e["step"] for e in index] == [1, 2]

        records = list(iter_rollouts(rollout_dir))
        roles = {r.role for r in records}
        assert roles == {"train", "anchor", "initial_reference"}
        assert len(records) == 80  # 64 policy samples + 16 lazy initial-reference samples
        for r in records:
            assert "(A)" in r.completion_text or "(B)" in r.completion_text
            assert 0.0 <= r.p_ref <= 1.0
            assert r.trait_value in (0.0, 1.0)
            if r.role == "train":
                assert r.perturbation_idx == 1 and r.p_hat is not None
            elif r.role in {"anchor", "initial_reference"}:
                assert r.perturbation_idx == 0 and r.p_hat is None
        # advantage/reward context survives: cued trait=1 rollouts are discouraged (adv < 0)
        cued_biased = [r for r in records if r.role == "train" and r.trait_value == 1.0]
        assert cued_biased and all(r.advantage < 0 for r in cued_biased)

    def test_rollout_log_none_writes_nothing(self, tmp_path):
        self._run(tmp_path, rollout_log="none")
        assert not (tmp_path / "rollouts").exists()

    def test_rate_only_surplus_is_persisted(self, tmp_path):
        training = TrainingSamplingConfig(
            perturbation_indices=[1],
            n_rollouts_for_rate=8,
            n_rollouts_for_consistency=2,
            n_rollouts_for_anchor=0,
        )
        self._run(tmp_path, training=training, anchor_weight=0.0)

        records = list(iter_rollouts(tmp_path / "rollouts"))
        assert len(records) == 64
        selected = [record for record in records if record.role == "train"]
        rate_only = [record for record in records if record.role == "rate"]
        assert len(selected) == 8  # 2 selected variants × 4 datapoints
        assert len(rate_only) == 56
        assert all(record.skipped_from_training for record in rate_only)
        assert all(record.skip_reason == "rate_only" for record in rate_only)

    def test_zero_signal_batch_keeps_rollouts_and_full_diagnostics(self, tmp_path):
        _, backend, _, logger = self._run(
            tmp_path,
            trait_classifier=lambda _answer, _dp, _messages: 0.0,
            anchor_weight=0.0,
        )

        assert backend.fb_datums == []
        assert backend.refresh_calls == 0  # unchanged weights do not need a new snapshot
        records = list(iter_rollouts(tmp_path / "rollouts"))
        assert len(records) == 64
        assert all(record.skipped_from_training for record in records)
        train_records = [record for record in records if record.role == "train"]
        rate_records = [record for record in records if record.role == "rate"]
        assert train_records and all(record.advantage == 0.0 for record in train_records)
        assert all(record.skip_reason == "zero_advantage_batch" for record in train_records)
        assert rate_records and all(record.advantage is None for record in rate_records)
        assert all(record.skip_reason == "rate_only" for record in rate_records)

        step_metrics = [call.args[0] for call in logger.log_metrics.call_args_list if "train/p_ref" in call.args[0]]
        assert len(step_metrics) == 2
        assert all(metrics["train/skipped_empty_batch"] == 1 for metrics in step_metrics)
        assert all(metrics["train/advantage_abs_mean"] == 0.0 for metrics in step_metrics)
        assert all(metrics["rollout/grader_failure_count"] == 0 for metrics in step_metrics)
        assert all(metrics["rollout/grader_failure_rate"] == 0.0 for metrics in step_metrics)
        assert all(metrics["rollout/grader_sample_count"] == 32 for metrics in step_metrics)

    def test_wandb_is_opt_in(self):
        assert RLConfig().wandb_project is None
        assert SFTConfig().wandb_project is None


class TestSFTEndToEnd:
    def test_full_loop(self, tmp_path):
        data = tmp_path / "train.jsonl"
        samples = [
            {"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]}
            for i in range(5)
        ]
        data.write_text("".join(json.dumps(s) + "\n" for s in samples))

        cfg = SFTConfig(
            experiment_name="itest",
            run_name="sft",
            optimizer=AdamConfig(learning_rate=1e-4, lr_schedule="linear"),
            batch_size=2,
            n_epochs=1,
            log_base_dir=str(tmp_path / "logs"),
        )
        backend = FakeBackend()
        with (
            patch("ctm.training.sft.setup_logging") as mock_logging,
            patch("ctm.training.sft.get_renderer_and_tokenizer", return_value=(FakeRenderer(), FakeTokenizer())),
        ):
            mock_logging.return_value = MagicMock()
            final = asyncio.run(train_sft(data, config=cfg, backend=backend))

        assert final == "fake://checkpoint/itest_sft"
        from ctm.training.manifest import read_run_manifest

        manifest = read_run_manifest(tmp_path / "logs" / "itest" / "sft")
        assert manifest["kind"] == "sft" and manifest["n_samples"] == 5
        assert backend.fb_loss_fns == ["cross_entropy"] * 3  # ceil(5/2) steps
        assert len(backend.optim_lrs) == 3
        # linear schedule from full LR, strictly decreasing
        assert backend.optim_lrs[0] == pytest.approx(1e-4)
        assert backend.optim_lrs[0] > backend.optim_lrs[1] > backend.optim_lrs[2]
        # SFT datums carry supervised loss inputs
        datum = backend.fb_datums[0][0]
        assert {"target_tokens", "weights"} <= set(datum.loss_fn_inputs)

    def test_submit_ahead_queues_next_microbatch_before_waiting(self, tmp_path):
        data = tmp_path / "train.jsonl"
        samples = [
            {"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]}
            for i in range(3)
        ]
        data.write_text("".join(json.dumps(sample) + "\n" for sample in samples))
        events = []

        class SubmitAheadBackend(FakeBackend):
            training_submit_ahead = 1

            async def submit_forward_backward(self, datums, loss_fn):
                index = len(self.fb_datums)
                events.append(("submit_fwd", index))
                pending = await super().submit_forward_backward(datums, loss_fn)

                class Pending:
                    async def result(self):
                        events.append(("wait_fwd", index))
                        return await pending.result()

                return Pending()

        cfg = SFTConfig(
            experiment_name="itest",
            run_name="submit-ahead",
            optimizer=AdamConfig(learning_rate=1e-4),
            batch_size=1,
            log_base_dir=str(tmp_path / "logs"),
        )
        with (
            patch("ctm.training.sft.setup_logging") as mock_logging,
            patch("ctm.training.sft.get_renderer_and_tokenizer", return_value=(FakeRenderer(), FakeTokenizer())),
        ):
            mock_logging.return_value = MagicMock()
            asyncio.run(train_sft(data, config=cfg, backend=SubmitAheadBackend()))

        assert events.index(("submit_fwd", 1)) < events.index(("wait_fwd", 0))
        assert events.index(("submit_fwd", 2)) < events.index(("wait_fwd", 1))

    def test_submit_ahead_stops_at_intermediate_checkpoint_boundary(self, tmp_path):
        data = tmp_path / "train.jsonl"
        samples = [
            {"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]}
            for i in range(3)
        ]
        data.write_text("".join(json.dumps(sample) + "\n" for sample in samples))
        events = []

        class CheckpointingBackend(FakeBackend):
            training_submit_ahead = 1

            async def submit_forward_backward(self, datums, loss_fn):
                events.append(("submit_fwd", len(self.fb_datums)))
                return await super().submit_forward_backward(datums, loss_fn)

            async def save_checkpoint(self, *, name, log_dir, loop_state, kind):
                events.append(("checkpoint", loop_state["step"]))
                return await super().save_checkpoint(
                    name=name,
                    log_dir=log_dir,
                    loop_state=loop_state,
                    kind=kind,
                )

        cfg = SFTConfig(
            experiment_name="itest",
            run_name="checkpoint-barrier",
            optimizer=AdamConfig(learning_rate=1e-4),
            checkpoint=CheckpointConfig(save_every_n_steps=1),
            batch_size=1,
            log_base_dir=str(tmp_path / "logs"),
        )
        with (
            patch("ctm.training.sft.setup_logging") as mock_logging,
            patch("ctm.training.sft.get_renderer_and_tokenizer", return_value=(FakeRenderer(), FakeTokenizer())),
        ):
            mock_logging.return_value = MagicMock()
            asyncio.run(train_sft(data, config=cfg, backend=CheckpointingBackend()))

        assert events.index(("checkpoint", 1)) < events.index(("submit_fwd", 1))
        assert events.index(("checkpoint", 2)) < events.index(("submit_fwd", 2))

    def test_gradient_accumulation_groups_microbatches(self, tmp_path):
        data = tmp_path / "train.jsonl"
        samples = [
            {"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]}
            for i in range(5)
        ]
        data.write_text("".join(json.dumps(sample) + "\n" for sample in samples))
        cfg = SFTConfig(
            experiment_name="itest",
            run_name="accum",
            optimizer=AdamConfig(learning_rate=1e-4, lr_schedule="linear"),
            batch_size=1,
            gradient_accumulation_steps=2,
            n_epochs=1,
            log_base_dir=str(tmp_path / "logs"),
        )
        backend = FakeBackend()
        with (
            patch("ctm.training.sft.setup_logging") as mock_logging,
            patch("ctm.training.sft.get_renderer_and_tokenizer", return_value=(FakeRenderer(), FakeTokenizer())),
        ):
            mock_logging.return_value = MagicMock()
            asyncio.run(train_sft(data, config=cfg, backend=backend))

        assert len(backend.fb_datums) == 5
        assert len(backend.optim_lrs) == 3
        assert backend.optim_lrs[0] == pytest.approx(1e-4)
        assert backend.optim_lrs[0] > backend.optim_lrs[1] > backend.optim_lrs[2]
