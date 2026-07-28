"""CPU tests for the LocalBackend (torch/transformers engine) with a tiny random GPT-2.

No downloads, no GPU, no peft: the model is randomly initialized from a config,
and LoRA-specific paths are exercised only when peft is installed (skip otherwise).
Uses full fine-tuning (use_lora=False) so the offline suite always covers the
loss/optimizer/sampler/checkpoint machinery.
"""

import asyncio

import pytest
import torch
from tinker import types
from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.data_processing import trajectory_to_data
from tinker_cookbook.rl.types import Trajectory, Transition
from tinker_cookbook.supervised.common import datum_from_model_input_weights

from ctm.backends.local.engine import (
    HAS_PEFT,
    LocalBackend,
    LocalSamplerHandle,
    _lora_target_module_names,
    _lora_target_parameter_names,
)
from ctm.core.config import AdamConfig, LoRAConfig

VOCAB = 128


def tiny_model():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=VOCAB,
            n_positions=64,
            n_embd=32,
            n_layer=2,
            n_head=2,
        )
    )


def make_backend() -> LocalBackend:
    backend = LocalBackend(device="cpu", use_lora=False, model_instance=tiny_model())
    backend.setup(model="tiny-gpt2-test", lora=LoRAConfig(rank=4))
    return backend


def sft_datum(tokens=(5, 6, 7, 8, 9), n_prompt=2):
    # weights align with the FULL token sequence; the helper drops weights[0] when
    # left-shifting targets, so pass len(tokens) weights.
    weights = torch.tensor([0.0] * n_prompt + [1.0] * (len(tokens) - n_prompt))
    return datum_from_model_input_weights(types.ModelInput.from_ints(tokens=list(tokens)), weights)


def rl_datum(backend, prompt=(5, 6), action=(7, 8), advantage=1.0):
    """RL datum whose sampled logprobs are the model's OWN current logprobs → ratio starts at 1."""
    seq = list(prompt) + list(action)
    probe = datum_from_model_input_weights(types.ModelInput.from_ints(tokens=list(seq)), torch.ones(len(seq)))
    with torch.no_grad():
        lp = backend._target_logprobs([probe])[0]
    action_logprobs = lp[len(prompt) - 1 :].tolist()  # logprobs of the action tokens
    transition = Transition(
        ob=types.ModelInput.from_ints(tokens=list(prompt)),
        ac=TokensWithLogprobs(tokens=list(action), maybe_logprobs=action_logprobs),
        reward=0.0,
        episode_done=True,
    )
    traj = Trajectory(transitions=[transition], final_ob=types.ModelInput.from_ints(tokens=[]))
    return trajectory_to_data(traj, traj_advantage=advantage)[0]


async def step(backend, datums, loss_fn, lr=1e-2):
    pending = await backend.submit_forward_backward(datums, loss_fn)
    out = await pending.result()
    opt = await backend.submit_optim_step(learning_rate=lr, adam=AdamConfig(learning_rate=lr))
    await opt.result()
    return out


class TestLocalSFT:
    def test_gradient_checkpointing_is_enabled_during_setup(self):
        model = tiny_model()
        backend = LocalBackend(
            device="cpu",
            use_lora=False,
            model_instance=model,
            gradient_checkpointing=True,
        )
        backend.setup(model="tiny-gpt2-test", lora=LoRAConfig(rank=4))
        assert model.is_gradient_checkpointing
        assert model.config.use_cache is False

    def test_microbatching_preserves_loss_logprobs_and_gradients(self):
        datums = [sft_datum(), sft_datum(tokens=(9, 8, 7, 6, 5))]
        unbounded = make_backend()
        bounded = LocalBackend(
            device="cpu",
            use_lora=False,
            model_instance=tiny_model(),
            training_microbatch_size=1,
        )
        bounded.setup(model="tiny-gpt2-test", lora=LoRAConfig(rank=4))
        for model in (unbounded.model, bounded.model):
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0.0

        async def forward(backend):
            return await (await backend.submit_forward_backward(datums, "cross_entropy")).result()

        unbounded_out = asyncio.run(forward(unbounded))
        bounded_out = asyncio.run(forward(bounded))

        assert bounded_out.metrics["loss"] == pytest.approx(unbounded_out.metrics["loss"], rel=1e-6)
        for bounded_lp, unbounded_lp in zip(bounded_out.logprobs, unbounded_out.logprobs):
            torch.testing.assert_close(bounded_lp, unbounded_lp)
        for bounded_parameter, unbounded_parameter in zip(bounded.model.parameters(), unbounded.model.parameters()):
            if unbounded_parameter.grad is None:
                assert bounded_parameter.grad is None
            else:
                torch.testing.assert_close(bounded_parameter.grad, unbounded_parameter.grad, rtol=1e-5, atol=1e-7)

    def test_cross_entropy_loss_decreases(self):
        backend = make_backend()
        datums = [sft_datum(), sft_datum(tokens=(9, 8, 7, 6, 5))]

        async def run():
            first = await step(backend, datums, "cross_entropy")
            for _ in range(25):
                last = await step(backend, datums, "cross_entropy")
            return first, last

        first, last = asyncio.run(run())
        assert last.metrics["loss"] < first.metrics["loss"] * 0.7
        # per-datum logprob tensors match target lengths
        for out_lp, d in zip(first.logprobs, datums):
            assert out_lp.shape == d.loss_fn_inputs["target_tokens"].to_torch().shape

    def test_gradient_accumulation_two_phase(self):
        backend = make_backend()

        async def run():
            # two forward_backwards accumulate, one optim_step applies + zeroes
            await (await backend.submit_forward_backward([sft_datum()], "cross_entropy")).result()
            await (await backend.submit_forward_backward([sft_datum()], "cross_entropy")).result()
            params = [p for p in backend.model.parameters() if p.requires_grad]
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in params)
            await (await backend.submit_optim_step(learning_rate=1e-3, adam=AdamConfig())).result()
            assert all(p.grad is None for p in params)  # zero_grad(set_to_none=True)

        asyncio.run(run())

    def test_selective_full_finetune_keeps_frozen_base(self):
        model = tiny_model()
        initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
        backend = LocalBackend(
            device="cpu",
            use_lora=False,
            model_instance=model,
            full_finetune_modules=["attn"],
            keep_frozen_base=True,
        )
        backend.setup(model="tiny-gpt2-test", lora=LoRAConfig(rank=4))

        trainable = [name for name, parameter in backend.model.named_parameters() if parameter.requires_grad]
        assert trainable and all("attn" in name for name in trainable)
        assert backend._frozen_base_model is not None
        assert all(not parameter.requires_grad for parameter in backend._frozen_base_model.parameters())
        assert all(torch.equal(initial[name], value) for name, value in backend._frozen_base_model.state_dict().items())

        asyncio.run(step(backend, [sft_datum()], "cross_entropy"))
        assert all(torch.equal(initial[name], value) for name, value in backend._frozen_base_model.state_dict().items())


class TestLocalRL:
    @pytest.mark.parametrize("loss_fn", ["importance_sampling", "ppo"])
    def test_positive_advantage_increases_action_logprob(self, loss_fn):
        backend = make_backend()
        datum = rl_datum(backend, advantage=1.0)
        targets = datum.loss_fn_inputs["target_tokens"].to_torch()
        mask = datum.loss_fn_inputs["mask"].to_torch()

        def action_logprob():
            with torch.no_grad():
                lp = backend._target_logprobs([datum])[0]
            return float((lp * mask).sum())

        before = action_logprob()

        async def run():
            for _ in range(10):
                await step(backend, [datum], loss_fn, lr=5e-3)

        asyncio.run(run())
        assert action_logprob() > before
        assert targets.shape == mask.shape  # sanity: datum structure as expected

    def test_kl_penalty_requires_lora(self):
        backend = make_backend()  # full fine-tune → no frozen base available
        datum = rl_datum(backend)
        with pytest.raises(NotImplementedError):
            asyncio.run(backend.incorporate_kl_penalty([datum], kl_coef=0.1, kl_discount_factor=0.0))
        with pytest.raises(NotImplementedError):
            backend.base_sampler()

    def test_policy_handle_scoring_matches_raw_frozen_base_logits(self):
        model = tiny_model()
        backend = LocalBackend(
            device="cpu",
            use_lora=False,
            model_instance=model,
            keep_frozen_base=True,
        )
        backend.setup(model="tiny-gpt2-test", lora=LoRAConfig(rank=4))
        prompt = types.ModelInput.from_ints(tokens=[5, 6])
        completion = [7, 8]

        scored = asyncio.run(backend.base_sampler().score_completions([prompt], [completion]))[0]
        current_scored = asyncio.run(backend.policy_sampler("current").score_completions([prompt], [completion]))[0]
        frozen = backend._frozen_base_model
        assert frozen is not None
        with torch.no_grad():
            logits = frozen(input_ids=torch.tensor([[5, 6, 7, 8]])).logits
            logprobs = torch.log_softmax(logits.float(), dim=-1)
            expected = [float(logprobs[0, 1, 7]), float(logprobs[0, 2, 8])]
        assert scored == pytest.approx(expected)
        assert current_scored == pytest.approx(expected)


class TestLocalSampler:
    def test_sample_shapes_and_logprobs(self):
        backend = make_backend()
        handle = backend.policy_sampler("test")
        seqs = asyncio.run(
            handle.sample(
                types.ModelInput.from_ints(tokens=[5, 6, 7]),
                max_tokens=6,
                temperature=1.0,
                stop=[],
                num_samples=3,
            )
        )
        assert len(seqs) == 3
        for s in seqs:
            assert 1 <= len(s.tokens) <= 6
            assert len(s.logprobs) == len(s.tokens)
            assert all(lp <= 0.0 for lp in s.logprobs)
            assert all(0 <= t < VOCAB for t in s.tokens)

    def test_stop_token_truncates(self):
        backend = make_backend()
        handle = backend.policy_sampler("test")
        stop = list(range(VOCAB))  # every token is a stop token → length-1 completions
        seqs = asyncio.run(
            handle.sample(
                types.ModelInput.from_ints(tokens=[5, 6]),
                max_tokens=8,
                temperature=1.0,
                stop=stop,
                num_samples=2,
            )
        )
        assert all(len(s.tokens) == 1 for s in seqs)

    def test_refresh_returns_live_policy(self):
        backend = make_backend()
        h = asyncio.run(backend.refresh_policy_sampler("x"))
        assert isinstance(h, type(backend.policy_sampler("x")))

    def test_concurrent_calls_are_coalesced_into_one_backend_batch(self):
        class BatchBackend:
            def __init__(self):
                self.calls = []

            def _sample_batch(self, **kwargs):
                self.calls.append(kwargs)
                return [[] for _ in kwargs["prompt_tokens_batch"]]

        backend = BatchBackend()
        handle = LocalSamplerHandle(backend, use_base=True)

        async def sample_all():
            return await asyncio.gather(
                *(
                    handle.sample(
                        types.ModelInput.from_ints(tokens=[token]),
                        max_tokens=8,
                        temperature=0.7,
                        stop=[0],
                        num_samples=1,
                    )
                    for token in (1, 2, 3, 4)
                )
            )

        assert asyncio.run(sample_all()) == [[], [], [], []]
        assert len(backend.calls) == 1
        assert backend.calls[0]["prompt_tokens_batch"] == [[1], [2], [3], [4]]
        assert backend.calls[0]["use_base"] is True


class TestLocalCheckpoint:
    def test_save_and_resume_full_finetune(self, tmp_path):
        backend = make_backend()
        asyncio.run(step(backend, [sft_datum()], "cross_entropy"))
        paths = asyncio.run(
            backend.save_checkpoint(name="run1_step1", log_dir=tmp_path, loop_state={"step": 1}, kind="both")
        )

        ckpt = tmp_path / "checkpoints" / "run1_step1"
        assert paths["sampler_path"] == f"file://{ckpt.resolve()}"
        assert paths["state_path"] is not None
        assert (ckpt / "weights.pt").exists()
        assert (ckpt / "optimizer.pt").exists()
        assert (ckpt / "manifest.json").exists()

        # A fresh backend (different random init) converges to the saved weights on load.
        torch.manual_seed(123)
        other = LocalBackend(device="cpu", use_lora=False, model_instance=tiny_model())
        other.setup(
            model="tiny-gpt2-test",
            lora=LoRAConfig(rank=4),
            resume_from=paths["sampler_path"],
            resume_with_optimizer=True,
        )
        for p1, p2 in zip(backend.model.state_dict().values(), other.model.state_dict().values()):
            assert torch.equal(p1, p2)

    def test_sampler_kind_skips_optimizer(self, tmp_path):
        backend = make_backend()
        asyncio.run(step(backend, [sft_datum()], "cross_entropy"))
        paths = asyncio.run(backend.save_checkpoint(name="run1", log_dir=tmp_path, loop_state={}, kind="sampler"))
        assert paths["state_path"] is None
        assert not (tmp_path / "checkpoints" / "run1" / "optimizer.pt").exists()


@pytest.mark.skipif(not HAS_PEFT, reason="peft not installed")
class TestLocalLoRA:
    def test_portable_component_flags_select_expected_modules(self):
        model = tiny_model()
        attention = _lora_target_module_names(
            model,
            LoRAConfig(train_mlp=False, train_attn=True, train_unembed=False),
        )
        mlp = _lora_target_module_names(
            model,
            LoRAConfig(train_mlp=True, train_attn=False, train_unembed=False),
        )
        unembed = _lora_target_module_names(
            model,
            LoRAConfig(train_mlp=False, train_attn=False, train_unembed=True),
        )

        assert attention and all("attn" in name for name in attention)
        assert mlp and all("attn" not in name and name != "lm_head" for name in mlp)
        assert unembed == ["lm_head"]

    def test_exact_target_modules_alpha_and_dropout(self):
        model = tiny_model()
        targets = _lora_target_module_names(
            model,
            LoRAConfig(target_modules=["c_attn"], rank=4, alpha=12, dropout=0.15),
        )
        assert targets and all(name.endswith(".c_attn") for name in targets)

        backend = LocalBackend(device="cpu", use_lora=True, model_instance=model)
        backend.setup(
            model="tiny-gpt2-test",
            lora=LoRAConfig(target_modules=["c_attn"], rank=4, alpha=12, dropout=0.15),
        )
        config = backend.model.peft_config["default"]
        assert config.lora_alpha == 12
        assert config.lora_dropout == 0.15

    def test_fused_moe_expert_parameters_are_selected_for_mlp_lora(self):
        class Experts(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_up_proj = torch.nn.Parameter(torch.randn(2, 4, 8))
                self.down_proj = torch.nn.Parameter(torch.randn(2, 4, 4))

        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.experts = Experts()

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = MLP()
                self.self_attn = torch.nn.Linear(4, 4)

        model = torch.nn.ModuleList([Block()])
        config = LoRAConfig(train_mlp=True, train_attn=True, train_unembed=False)

        assert _lora_target_module_names(model, config) == ["0.self_attn"]
        assert _lora_target_parameter_names(model, config) == [
            "0.mlp.experts.gate_up_proj",
            "0.mlp.experts.down_proj",
        ]

    def test_tiny_gpt_oss_wraps_attention_and_fused_expert_parameters(self):
        from transformers import GptOssConfig, GptOssForCausalLM

        model = GptOssForCausalLM(
            GptOssConfig(
                vocab_size=128,
                hidden_size=32,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                num_local_experts=4,
                num_experts_per_tok=2,
                max_position_embeddings=64,
            )
        )
        backend = LocalBackend(device="cpu", use_lora=True, model_instance=model)
        backend.setup(
            model="tiny-gpt-oss",
            lora=LoRAConfig(rank=2, alpha=4, train_mlp=True, train_attn=True, train_unembed=False),
        )
        peft_config = backend.model.peft_config["default"]

        assert any(name.endswith("self_attn.q_proj") for name in peft_config.target_modules)
        assert sorted(peft_config.target_parameters) == [
            "model.layers.0.mlp.experts.down_proj",
            "model.layers.0.mlp.experts.gate_up_proj",
        ]

    def test_lora_wraps_and_trains(self):
        backend = LocalBackend(device="cpu", use_lora=True, model_instance=tiny_model())
        backend.setup(model="tiny-gpt2-test", lora=LoRAConfig(rank=4, seed=0))
        n_trainable = sum(p.numel() for p in backend.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in backend.model.parameters())
        assert 0 < n_trainable < n_total * 0.5  # adapter-only training

        first = asyncio.run(step(backend, [sft_datum()], "cross_entropy"))
        assert first.metrics["loss"] > 0
