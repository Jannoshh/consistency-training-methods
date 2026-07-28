"""Contract tests for the vLLM sampler — no vllm installed.

The vllm API surface (LLM.generate / SamplingParams / TokensPrompt / LoRARequest)
is faked; what's under test is OUR side of the contract: request construction,
LoRA hot-reload versioning, base-vs-policy routing, and token/logprob extraction.
Real-engine behaviour is validated on a GPU box (tests there carry @pytest.mark.gpu).
"""

import asyncio
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tinker import types

from ctm.backends.local.engine import HAS_PEFT, LocalBackend
from ctm.backends.local.vllm_sampler import VLLMSampler
from ctm.core.config import LoRAConfig


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeTokensPrompt:
    def __init__(self, prompt_token_ids):
        self.prompt_token_ids = prompt_token_ids


class FakeLoRARequest:
    def __init__(self, name, lora_int_id, path):
        self.name, self.lora_int_id, self.path = name, lora_int_id, path


def fake_api():
    return SimpleNamespace(
        LLM=None, SamplingParams=FakeSamplingParams, TokensPrompt=FakeTokensPrompt, LoRARequest=FakeLoRARequest
    )


class FakeEngine:
    """Records generate() calls; returns two completions with per-token logprobs."""

    def __init__(self, drop_logprob_for_token=None):
        self.calls = []
        self.drop = drop_logprob_for_token
        self.sleep_calls = []
        self.wake_calls = 0

    def sleep(self, level=1):
        self.sleep_calls.append(level)

    def wake_up(self):
        self.wake_calls += 1

    def generate(self, prompts, params, lora_request=None, use_tqdm=False):
        self.calls.append(SimpleNamespace(prompts=prompts, params=params, lora_request=lora_request))

        def lp_dict(token, value):
            if token == self.drop:
                return {}  # sampled token missing its own logprob
            return {token: SimpleNamespace(logprob=value)}

        completions = [
            SimpleNamespace(token_ids=[7, 8], logprobs=[lp_dict(7, -0.5), lp_dict(8, -0.7)]),
            SimpleNamespace(token_ids=[9], logprobs=[lp_dict(9, -1.2)]),
        ]
        return [SimpleNamespace(outputs=completions) for _ in prompts]


def make_sampler(engine=None, *, sleep=False):
    return VLLMSampler(
        model="some/base",
        engine=engine or FakeEngine(),
        api=fake_api(),
        enable_sleep_mode=sleep,
    )


class TestVLLMSampler:
    def test_policy_before_any_snapshot_uses_base(self):
        engine = FakeEngine()
        s = make_sampler(engine)
        s.sample([1, 2], max_tokens=8, temperature=0.7, stop=[], num_samples=2, use_base=False)
        assert engine.calls[0].lora_request is None  # no adapter published yet

    def test_advance_policy_bumps_lora_request_id(self):
        engine = FakeEngine()
        s = make_sampler(engine)
        s.advance_policy("/tmp/adapters/v1")
        s.sample([1], max_tokens=4, temperature=1.0, stop=[], num_samples=1, use_base=False)
        s.advance_policy("/tmp/adapters/v2")
        s.sample([1], max_tokens=4, temperature=1.0, stop=[], num_samples=1, use_base=False)
        first, second = engine.calls[0].lora_request, engine.calls[1].lora_request
        assert (first.lora_int_id, first.path) == (1, "/tmp/adapters/v1")
        assert (second.lora_int_id, second.path) == (2, "/tmp/adapters/v2")
        assert first.name != second.name  # unique id+name defeats vLLM's adapter cache

    def test_base_sampling_never_attaches_adapter(self):
        engine = FakeEngine()
        s = make_sampler(engine)
        s.advance_policy("/tmp/adapters/v1")
        s.sample([1], max_tokens=4, temperature=1.0, stop=[], num_samples=1, use_base=True)
        assert engine.calls[0].lora_request is None

    def test_request_params_and_prompt(self):
        engine = FakeEngine()
        s = make_sampler(engine)
        s.sample([5, 6, 7], max_tokens=32, temperature=0.7, stop=[2, "text-stop", 3], num_samples=4, use_base=False)
        call = engine.calls[0]
        assert call.prompts[0].prompt_token_ids == [5, 6, 7]
        assert call.params.kwargs["n"] == 4
        assert call.params.kwargs["max_tokens"] == 32
        assert call.params.kwargs["temperature"] == 0.7
        assert call.params.kwargs["stop_token_ids"] == [2, 3]  # non-int stops filtered
        assert call.params.kwargs["logprobs"] == 0

    def test_token_and_logprob_extraction(self):
        s = make_sampler()
        seqs = s.sample([1], max_tokens=4, temperature=1.0, stop=[], num_samples=2, use_base=False)
        assert [q.tokens for q in seqs] == [[7, 8], [9]]
        assert seqs[0].logprobs == pytest.approx([-0.5, -0.7])
        assert seqs[1].logprobs == pytest.approx([-1.2])

    def test_batch_uses_one_engine_call_and_preserves_prompt_order(self):
        engine = FakeEngine()
        sampler = make_sampler(engine)

        batches = sampler.sample_batch(
            [[1], [2, 3]],
            max_tokens=4,
            temperature=1.0,
            stop=[],
            num_samples=2,
            use_base=False,
        )

        assert len(engine.calls) == 1
        assert [prompt.prompt_token_ids for prompt in engine.calls[0].prompts] == [[1], [2, 3]]
        assert [[sequence.tokens for sequence in batch] for batch in batches] == [
            [[7, 8], [9]],
            [[7, 8], [9]],
        ]

    def test_missing_sampled_logprob_marks_sequence_logprobless(self):
        s = make_sampler(FakeEngine(drop_logprob_for_token=8))
        seqs = s.sample([1], max_tokens=4, temperature=1.0, stop=[], num_samples=2, use_base=False)
        assert seqs[0].logprobs is None  # token 8's logprob missing → whole seq excluded downstream
        assert seqs[1].logprobs == pytest.approx([-1.2])

    def test_sleep_is_idempotent_and_sampling_wakes_engine(self):
        engine = FakeEngine()
        sampler = make_sampler(engine, sleep=True)
        sampler.sleep()
        sampler.sleep()
        assert engine.sleep_calls == [1]
        assert sampler.sleeping is True

        sampler.sample([1], max_tokens=4, temperature=1.0, stop=[], num_samples=1, use_base=True)
        assert engine.wake_calls == 1
        assert sampler.sleeping is False

        sampler.wake_up()
        assert engine.wake_calls == 1

    def test_shutdown_releases_engine_and_is_idempotent(self):
        engine = FakeEngine()
        engine_ref = weakref.ref(engine)
        sampler = make_sampler(engine)
        del engine
        sampler.shutdown()
        assert sampler.engine is None
        assert engine_ref() is None

        sampler.shutdown()
        assert sampler.engine is None


@pytest.mark.skipif(not HAS_PEFT, reason="peft not installed")
class TestLocalBackendVLLMWiring:
    def _backend(self, engine, *, sleep=False):
        from transformers import GPT2Config, GPT2LMHeadModel

        torch.manual_seed(0)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=64, n_positions=32, n_embd=16, n_layer=1, n_head=1))
        backend = LocalBackend(
            device="cpu",
            use_lora=True,
            model_instance=model,
            sampler="vllm",
            vllm_options={"engine": engine, "api": fake_api(), "enable_sleep_mode": sleep},
        )
        backend.setup(model="tiny-gpt2-test", lora=LoRAConfig(rank=2, seed=0))
        return backend

    def test_setup_defers_engine_boot(self):
        # SFT-family runs call setup() but never request a sampler — no engine
        # boot, no adapter snapshot, no vllm package requirement. No fake
        # engine/api is injected here: a boot inside setup() would hit vllm.
        from transformers import GPT2Config, GPT2LMHeadModel

        torch.manual_seed(0)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=64, n_positions=32, n_embd=16, n_layer=1, n_head=1))
        backend = LocalBackend(device="cpu", use_lora=True, model_instance=model, sampler="vllm")
        backend.setup(model="tiny-gpt2-test", lora=LoRAConfig(rank=2, seed=0))
        assert backend._vllm is None
        assert backend._adapter_scratch is None

    def test_first_sampler_request_boots_and_publishes_adapter(self):
        engine = FakeEngine()
        backend = self._backend(engine)
        backend.policy_sampler("p")
        assert backend._vllm.adapter_version == 1
        adapter_dir = Path(backend._vllm.adapter_dir)
        assert adapter_dir.exists() and any(adapter_dir.iterdir())  # real peft snapshot on disk

    def test_refresh_on_cold_engine_boots_and_publishes_once(self):
        engine = FakeEngine()
        backend = self._backend(engine)
        asyncio.run(backend.refresh_policy_sampler("step1"))
        assert backend._vllm.adapter_version == 1  # lazy boot's publish, no double snapshot

    def test_refresh_snapshots_and_bumps_version(self):
        engine = FakeEngine()
        backend = self._backend(engine)
        backend.policy_sampler("p")  # boot (v1), as the RL loop does at setup
        asyncio.run(backend.refresh_policy_sampler("step1"))
        assert backend._vllm.adapter_version == 2
        assert backend._vllm.adapter_dir.endswith("v2")

    def test_sampling_routes_through_vllm(self):
        engine = FakeEngine()
        backend = self._backend(engine)
        policy = backend.policy_sampler("p")
        seqs = asyncio.run(
            policy.sample(
                types.ModelInput.from_ints(tokens=[1, 2]), max_tokens=4, temperature=1.0, stop=[], num_samples=2
            )
        )
        assert [q.tokens for q in seqs] == [[7, 8], [9]]
        assert engine.calls[-1].lora_request.lora_int_id == 1  # policy = current adapter

        base = backend.base_sampler()
        asyncio.run(
            base.sample(
                types.ModelInput.from_ints(tokens=[1, 2]), max_tokens=4, temperature=1.0, stop=[], num_samples=1
            )
        )
        assert engine.calls[-1].lora_request is None  # base = engine's frozen weights

    def test_backend_sleeps_for_training_and_wakes_for_sampling(self, monkeypatch):
        engine = FakeEngine()
        backend = self._backend(engine, sleep=True)
        policy = backend.policy_sampler("p")
        cache_releases = []
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: cache_releases.append(True))
        backend.device = "cuda"

        backend._sleep_vllm_for_training()
        backend._sleep_vllm_for_training()
        assert engine.sleep_calls == [1]

        asyncio.run(
            policy.sample(
                types.ModelInput.from_ints(tokens=[1, 2]),
                max_tokens=4,
                temperature=1.0,
                stop=[],
                num_samples=1,
            )
        )
        assert engine.wake_calls == 1
        assert engine.calls
        assert cache_releases == [True]

    def test_shutdown_releases_started_vllm(self):
        engine = FakeEngine()
        backend = self._backend(engine)
        backend.policy_sampler("p")
        sampler = backend._vllm

        backend.shutdown()
        assert backend._vllm is None
        assert sampler.engine is None

        backend.shutdown()
        assert backend._vllm is None

    def test_vllm_requires_lora(self):
        backend = LocalBackend(
            device="cpu",
            use_lora=False,
            sampler="vllm",
            model_instance=torch.nn.Linear(2, 2),
            vllm_options={"engine": FakeEngine(), "api": fake_api()},
        )
        with pytest.raises(NotImplementedError):
            backend.setup(model="x", lora=LoRAConfig(rank=2))
