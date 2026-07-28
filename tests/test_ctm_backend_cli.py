"""Tests for the shared --backend CLI plumbing (ctm.backends.cli)."""

import argparse
import asyncio

import pytest
import torch
from tinker import types

from ctm.backends.cli import add_backend_args, build_backend, describe_backend
from ctm.backends.local.engine import LocalBackend
from ctm.backends.tinker import TinkerBackend, TinkerSamplerHandle
from ctm.core.config import LoRAConfig


def parse(argv):
    parser = argparse.ArgumentParser()
    add_backend_args(parser)
    return parser.parse_args(argv)


class TestBackendCLI:
    def test_default_is_tinker_and_builds_tinker_backend(self):
        args = parse([])
        assert args.backend == "tinker"
        backend = build_backend(args)
        assert isinstance(backend, TinkerBackend)
        assert backend.renderer_source == "tinker"
        assert "tinker" in describe_backend(args)

    def test_local_builds_localbackend_with_options(self):
        args = parse(
            [
                "--backend",
                "local",
                "--local-device",
                "cpu",
                "--local-dtype",
                "float32",
                "--local-sampler",
                "hf",
                "--local-gpu-mem-util",
                "0.3",
                "--local-vllm-max-model-len",
                "24576",
                "--local-vllm-max-num-seqs",
                "128",
                "--local-vllm-sleep-during-training",
                "--local-training-microbatch-size",
                "4",
                "--local-gradient-checkpointing",
            ]
        )
        backend = build_backend(args)
        assert isinstance(backend, LocalBackend)
        assert backend.device == "cpu"
        assert backend.dtype == torch.float32
        assert backend.sampler == "hf"
        assert backend.use_lora is True
        assert backend.renderer_source == "hf"
        assert backend.vllm_options == {
            "gpu_memory_utilization": 0.3,
            "max_model_len": 24576,
            "max_num_seqs": 128,
            "enable_sleep_mode": True,
        }
        assert backend.training_microbatch_size == 4
        assert backend.gradient_checkpointing is True
        assert "local" in describe_backend(args) and "hf" in describe_backend(args)

    def test_local_defaults_to_vllm_lora(self):
        args = parse(["--backend", "local", "--local-device", "cpu"])
        backend = build_backend(args)
        assert backend.sampler == "vllm"
        assert backend.dtype == torch.bfloat16
        assert backend.use_lora is True

    @pytest.mark.parametrize(
        "flag",
        [
            "--local-vllm-max-model-len",
            "--local-vllm-max-num-seqs",
            "--local-training-microbatch-size",
        ],
    )
    def test_vllm_bounds_must_be_positive(self, flag):
        with pytest.raises(SystemExit):
            parse([flag, "0"])

    def test_full_finetune_flag(self):
        args = parse(
            [
                "--backend",
                "local",
                "--local-device",
                "cpu",
                "--local-sampler",
                "hf",
                "--local-full-finetune",
                "--local-trainable-modules",
                "self_attn",
            ]
        )
        backend = build_backend(args)
        assert backend.use_lora is False
        assert backend.full_finetune_modules == ["self_attn"]
        assert "full-finetune" in describe_backend(args)

    def test_trainable_modules_require_full_finetune(self):
        args = parse(["--backend", "local", "--local-trainable-modules", "self_attn"])
        with pytest.raises(ValueError, match="requires --local-full-finetune"):
            build_backend(args)


def test_tinker_training_run_records_renderer_metadata(monkeypatch):
    class Service:
        def create_lora_training_client(self, **kwargs):
            self.kwargs = kwargs
            return object()

    service = Service()
    monkeypatch.setattr("ctm.backends.tinker.model_info.get_recommended_renderer_name", lambda _: "unit-renderer")
    TinkerBackend(service).setup(
        model="unit/model",
        lora=LoRAConfig(rank=4, train_mlp=False, train_attn=True, train_unembed=False, seed=9),
    )
    assert service.kwargs["user_metadata"] == {"renderer_name": "unit-renderer"}
    assert {key: service.kwargs[key] for key in ("rank", "train_mlp", "train_attn", "train_unembed", "seed")} == {
        "rank": 4,
        "train_mlp": False,
        "train_attn": True,
        "train_unembed": False,
        "seed": 9,
    }


def test_tinker_policy_handle_scores_only_completion_tokens_with_raw_policy():
    class SamplingClient:
        async def compute_logprobs_async(self, model_input):
            assert model_input.to_ints() == [1, 2, 3, 4]
            return [None, -0.1, -0.2, -0.3]

    scored = asyncio.run(
        TinkerSamplerHandle(SamplingClient()).score_completions(
            [types.ModelInput.from_ints(tokens=[1, 2])],
            [[3, 4]],
        )
    )
    assert scored == [[-0.2, -0.3]]


def test_tinker_async_setup_uses_native_async_constructor_and_resume(monkeypatch):
    events = []

    class PendingLoad:
        async def result_async(self):
            events.append("load_result")

    class TrainingClient:
        async def load_state_with_optimizer_async(self, path):
            events.append(("load_with_optimizer", path))
            return PendingLoad()

    class Service:
        async def create_lora_training_client_async(self, **kwargs):
            events.append("create_async")
            self.kwargs = kwargs
            return TrainingClient()

        def create_lora_training_client(self, **kwargs):  # pragma: no cover - must stay unused
            raise AssertionError("sync constructor called from async setup")

    service = Service()
    monkeypatch.setattr("ctm.backends.tinker.model_info.get_recommended_renderer_name", lambda _: "unit-renderer")
    backend = TinkerBackend(service)
    asyncio.run(
        backend.setup_async(
            model="unit/model",
            lora=LoRAConfig(rank=4),
            resume_from="tinker://unit/weights/1",
            resume_with_optimizer=True,
        )
    )

    assert events == ["create_async", ("load_with_optimizer", "tinker://unit/weights/1"), "load_result"]
    assert service.kwargs["user_metadata"] == {"renderer_name": "unit-renderer"}
