"""LocalBackend — in-process torch training + HF sampling for self-hosted GPUs.

Implements the ``TrainingBackend`` protocol on a single node (workstation, a
Vast.ai box, or one Isambard GH200 node): training via ``transformers`` (+ PEFT
LoRA when installed), rollouts via ``model.generate``. The same tinker Datums the
loops already build are consumed directly (they're offline containers), so the
loops don't know which backend they're on.

Notes / current limits (phase 1):
- LoRA requires ``peft`` (``uv pip install peft``); without it use
  ``use_lora=False`` (full fine-tune — no KL-to-base, which needs the frozen
  base via ``disable_adapter``).
- ``submit_*`` executes eagerly and returns an already-resolved pending object —
  the two-phase protocol shape is preserved, the prefetch overlap just buys
  nothing extra in-process.
- Sampling is HF ``generate`` (correct, not fast). A vLLM sampler with LoRA
  hot-reload is the planned fast path for real runs; see
  ``ctm/backends/local/__init__.py``.
- Checkpoints are directories: ``<log_dir>/checkpoints/<name>/`` with adapter or
  full weights, optional ``optimizer.pt``, and a ``manifest.json`` (returned
  paths use the ``file://`` scheme so eval runners can dispatch on it).
"""

import asyncio
import copy
import fnmatch
import json
from collections.abc import Sequence
from contextlib import nullcontext as _nullcontext
from pathlib import Path
from typing import Any

import torch
from tinker_cookbook.rl.metrics import discounted_future_sum_vectorized

from ctm.backends.base import ForwardBackwardOutput, SampledSequence
from ctm.backends.local import losses
from ctm.backends.local.mlp_hooks import MLPHookManager
from ctm.core.config import AdamConfig, LoRAConfig
from ctm.training import consistency_losses

# Internal-consistency loss_fns (ACT / AttCT / MLPCT) — LocalBackend only: they
# need paired forward passes with attentions / hidden states / MLP hooks, which
# the Tinker service API doesn't expose.
CONSISTENCY_LOSS_CLASSES = consistency_losses.CONSISTENCY_LOSS_CLASSES

try:  # optional: LoRA support
    import peft
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model

    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


def _strip_file_scheme(path: str) -> Path:
    return Path(path.removeprefix("file://"))


class _ResolvedPending:
    """Already-computed result behind the two-phase pending interface."""

    def __init__(self, value):
        self._value = value

    async def result(self):
        return self._value


def _name_matches(name: str, selector: str) -> bool:
    """Match a full name by glob, or a plain selector by dotted component."""

    if any(character in selector for character in "*?["):
        return fnmatch.fnmatchcase(name, selector)
    return name == selector or name.endswith(f".{selector}") or selector in name.split(".")


def _lora_target_module_names(model: torch.nn.Module, config: LoRAConfig) -> list[str]:
    """Resolve exact targets or portable component flags to PEFT module names."""

    output = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    components: dict[str, list[str]] = {"mlp": [], "attn": [], "unembed": []}
    for name, module in model.named_modules():
        if not name or not (isinstance(module, torch.nn.Linear) or type(module).__name__ == "Conv1D"):
            continue
        if module is output:
            component = "unembed"
        elif "attn" in name.lower() or "attention" in name.lower():
            component = "attn"
        else:
            component = "mlp"
        components[component].append(name)

    if config.target_modules is not None:
        names = [name for component_names in components.values() for name in component_names]
        selected = [name for name in names if any(_name_matches(name, target) for target in config.target_modules)]
        unmatched = [
            target for target in config.target_modules if not any(_name_matches(name, target) for name in names)
        ]
        if unmatched:
            raise ValueError(f"model {type(model).__name__} has no linear modules matching target_modules={unmatched}")
        return selected

    enabled = {
        "mlp": config.train_mlp,
        "attn": config.train_attn,
        "unembed": config.train_unembed,
    }
    raw_mlp_parameters = _lora_target_parameter_names(model, config)
    missing = [
        component
        for component, selected in enabled.items()
        if selected and not components[component] and not (component == "mlp" and raw_mlp_parameters)
    ]
    if missing:
        raise NotImplementedError(
            f"model {type(model).__name__} exposes no local LoRA modules for selected component(s): {missing}"
        )
    return [name for component, names in components.items() if enabled[component] for name in names]


def _lora_target_parameter_names(model: torch.nn.Module, config: LoRAConfig) -> list[str]:
    """Return fused MoE expert matrices that cannot be targeted as modules.

    GPT-OSS represents its expert projections as three-dimensional Parameters
    rather than ``nn.Linear`` modules. Recent PEFT versions support these
    through ``target_parameters``.
    """

    if config.target_modules is not None or not config.train_mlp:
        return []
    suffixes = (".mlp.experts.gate_up_proj", ".mlp.experts.down_proj")
    return [name for name, _ in model.named_parameters() if name.endswith(suffixes)]


def _configure_full_finetune_parameters(model: torch.nn.Module, selectors: Sequence[str] | None) -> list[str]:
    """Enable either every parameter or the explicitly selected parameter groups."""

    if selectors is None:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return [name for name, _ in model.named_parameters()]
    if not selectors or any(not isinstance(selector, str) or not selector.strip() for selector in selectors):
        raise ValueError("full_finetune_modules must contain non-empty module selectors")

    selected = []
    for name, parameter in model.named_parameters():
        train = any(_name_matches(name, selector) for selector in selectors)
        parameter.requires_grad_(train)
        if train:
            selected.append(name)
    unmatched = [selector for selector in selectors if not any(_name_matches(name, selector) for name in selected)]
    if unmatched:
        raise ValueError(f"model {type(model).__name__} has no parameters matching full_finetune_modules={unmatched}")
    if not selected:
        raise ValueError("full_finetune_modules selected no parameters")
    return selected


class LocalSamplerHandle:
    """Samples from the backend's live model (policy) or its frozen base.

    Concurrent coroutine calls are coalesced into one backend batch. This keeps
    synchronous HF/vLLM generation off the event loop and lets vLLM schedule all
    prompts together without making unsafe concurrent ``LLM.generate`` calls.
    """

    def __init__(self, backend: "LocalBackend", use_base: bool):
        self._backend = backend
        self._use_base = use_base
        self._pending: list[tuple[dict[str, Any], asyncio.Future]] = []
        self._flush_task: asyncio.Task | None = None

    async def sample(
        self, prompt: Any, *, max_tokens: int, temperature: float, stop: Any, num_samples: int
    ) -> list[SampledSequence]:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending.append(
            (
                {
                    "prompt_tokens": list(prompt.to_ints()),
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stop": stop,
                    "num_samples": num_samples,
                },
                future,
            )
        )
        if self._flush_task is None:
            self._flush_task = loop.create_task(self._flush_pending())
        return await future

    async def score_completions(
        self,
        prompts: Sequence[Any],
        completion_tokens: Sequence[Sequence[int]],
    ) -> list[list[float]]:
        """Score supplied tokens with this handle's raw policy distribution."""

        return await asyncio.to_thread(
            self._backend._score_completions,
            prompts,
            completion_tokens,
            use_base=self._use_base,
        )

    async def _flush_pending(self) -> None:
        # Give sibling tasks created by gather() one event-loop turn to enqueue.
        await asyncio.sleep(0)
        pending, self._pending = self._pending, []
        try:
            groups: dict[tuple[Any, ...], list[tuple[dict[str, Any], asyncio.Future]]] = {}
            for request, future in pending:
                stop_ids = tuple(token for token in (request["stop"] or []) if isinstance(token, int))
                key = (
                    request["max_tokens"],
                    float(request["temperature"]),
                    stop_ids,
                    request["num_samples"],
                )
                groups.setdefault(key, []).append((request, future))

            for group in groups.values():
                first = group[0][0]
                try:
                    results = await asyncio.to_thread(
                        self._backend._sample_batch,
                        prompt_tokens_batch=[request["prompt_tokens"] for request, _ in group],
                        max_tokens=first["max_tokens"],
                        temperature=first["temperature"],
                        stop=first["stop"],
                        num_samples=first["num_samples"],
                        use_base=self._use_base,
                    )
                    if len(results) != len(group):
                        raise RuntimeError(
                            f"local sampler returned {len(results)} prompt results for {len(group)} requests"
                        )
                except BaseException as exc:  # noqa: BLE001 -- propagate cancellation/failure to every queued future
                    for _, future in group:
                        if not future.done():
                            future.set_exception(exc)
                else:
                    for result, (_, future) in zip(results, group):
                        if not future.done():
                            future.set_result(result)
        finally:
            self._flush_task = None
            if self._pending:
                self._flush_task = asyncio.get_running_loop().create_task(self._flush_pending())


class LocalBackend:
    """TrainingBackend on local hardware (torch + transformers [+ peft])."""

    renderer_source = "hf"
    # Local sampler handles route to the backend's current in-process model or
    # current vLLM adapter; retaining a handle does not freeze its weights.
    policy_samplers_are_snapshots = False

    def __init__(
        self,
        *,
        device: str | None = None,
        dtype: torch.dtype = torch.float32,
        use_lora: bool = True,
        model_instance: torch.nn.Module | None = None,
        ppo_clip_epsilon: float = losses.PPO_CLIP_EPSILON,
        sampler: str = "hf",
        vllm_options: dict | None = None,
        consistency_loss_options: dict | None = None,
        full_finetune_modules: Sequence[str] | None = None,
        keep_frozen_base: bool = False,
        training_microbatch_size: int | None = None,
        gradient_checkpointing: bool = False,
    ):
        """
        Args:
            device: "cuda" / "cpu" / "mps"; auto-detects cuda when None.
            dtype: model dtype (bfloat16 recommended on GPU).
            use_lora: wrap the model with a PEFT LoRA adapter (requires peft).
            model_instance: pre-built model (e.g. a tiny random model in tests);
                skips ``from_pretrained`` in setup().
            ppo_clip_epsilon: clip range for the "ppo" loss_fn.
            sampler: rollout engine — "hf" (model.generate; correct, slow) or
                "vllm" (in-process vLLM with LoRA hot-reload; production path).
                The vLLM engine boots lazily on the first sampler request, so
                runs that never sample (SFT) pay nothing for it.
            vllm_options: forwarded to VLLMSampler / vllm.LLM
                (gpu_memory_utilization, tensor_parallel_size, max_model_len, ...).
            consistency_loss_options: constructor kwargs for the consistency
                loss_fns (weight, layer_selection, ...); defaults are the
                AttCT-paper settings.
            full_finetune_modules: parameter-name globs or dotted components to
                train when ``use_lora=False``. ``None`` trains every parameter.
            keep_frozen_base: retain an immutable copy of the initial model for
                clean/reference forwards during selective full fine-tuning.
            training_microbatch_size: maximum number of sequences in one
                physical training forward/backward; ``None`` uses the full
                logical batch.
            gradient_checkpointing: recompute transformer blocks during
                backward to reduce peak training memory.
        """
        if sampler not in ("hf", "vllm"):
            raise ValueError(f"Unknown sampler: {sampler!r} (expected 'hf' or 'vllm')")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.use_lora = use_lora
        self.model: torch.nn.Module | None = model_instance
        self.model_name: str | None = None
        self.ppo_clip_epsilon = ppo_clip_epsilon
        self.sampler = sampler
        self.vllm_options = vllm_options or {}
        self.consistency_loss_options = consistency_loss_options or {}
        self.full_finetune_modules = list(full_finetune_modules) if full_finetune_modules is not None else None
        self.keep_frozen_base = keep_frozen_base
        if training_microbatch_size is not None and training_microbatch_size < 1:
            raise ValueError("training_microbatch_size must be positive")
        self.training_microbatch_size = training_microbatch_size
        self.gradient_checkpointing = gradient_checkpointing
        self._vllm = None  # VLLMSampler, booted lazily by _ensure_vllm() when sampler == "vllm"
        self._adapter_scratch: Path | None = None
        self._optimizer: torch.optim.AdamW | None = None
        self._pending_optimizer_state: dict | None = None
        self._consistency_loss_modules: dict[str, consistency_losses.ConsistencyLoss] = {}
        self._mlp_hooks: MLPHookManager | None = None
        self._base_mlp_hooks: MLPHookManager | None = None
        self._frozen_base_model: torch.nn.Module | None = None
        self._gradient_accumulations = 0
        self._trainable_parameter_names: list[str] = []

    # ── lifecycle ────────────────────────────────────────────────────────

    def setup(
        self, *, model: str, lora: LoRAConfig, resume_from: str | None = None, resume_with_optimizer: bool = False
    ) -> None:
        self.model_name = model
        if self.model is None:
            from transformers import AutoModelForCausalLM

            self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=self.dtype)
        if self.gradient_checkpointing:
            if not hasattr(self.model, "gradient_checkpointing_enable"):
                raise NotImplementedError(f"model {type(self.model).__name__} does not support gradient checkpointing")
            self.model.gradient_checkpointing_enable()
            if hasattr(self.model, "config"):
                self.model.config.use_cache = False
        if self.use_lora:
            if self.full_finetune_modules is not None:
                raise ValueError("full_finetune_modules applies only when use_lora=False")
            if self.keep_frozen_base:
                raise ValueError("keep_frozen_base is unnecessary with LoRA; disabling the adapter is the frozen base")
            if not HAS_PEFT:
                raise ImportError(
                    "LocalBackend(use_lora=True) requires peft: `uv pip install peft`, "
                    "or pass use_lora=False for full fine-tuning."
                )
            if lora.seed is not None:
                torch.manual_seed(lora.seed)
            target_modules = _lora_target_module_names(self.model, lora)
            target_parameters = _lora_target_parameter_names(self.model, lora)
            if not target_modules and not target_parameters:
                raise ValueError("LoRA must train at least one of MLP, attention, or unembedding modules")
            peft_cfg = PeftLoraConfig(
                r=lora.rank,
                lora_alpha=lora.resolved_alpha,
                lora_dropout=lora.dropout,
                target_modules=target_modules or [],
                target_parameters=target_parameters or None,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, peft_cfg)
            self._trainable_parameter_names = [
                name for name, parameter in self.model.named_parameters() if parameter.requires_grad
            ]
        else:
            if self.keep_frozen_base:
                self._frozen_base_model = copy.deepcopy(self.model)
                self._frozen_base_model.eval()
                for parameter in self._frozen_base_model.parameters():
                    parameter.requires_grad_(False)
            self._trainable_parameter_names = _configure_full_finetune_parameters(
                self.model, self.full_finetune_modules
            )
        self.model.to(self.device)
        if self._frozen_base_model is not None:
            self._frozen_base_model.to(self.device)
        if resume_from:
            self._load_checkpoint(resume_from, with_optimizer=resume_with_optimizer)
        if self.sampler == "vllm" and not self.use_lora:
            raise NotImplementedError(
                "sampler='vllm' requires use_lora=True: policy refresh works by "
                "hot-reloading the adapter; full-finetune weights cannot be swapped "
                "into a running vLLM engine."
            )
        # The vLLM engine itself boots lazily (_ensure_vllm) on the first sampler
        # request: runs that never sample (SFT) must not pay its GPU memory or
        # require the vllm package at all.

    def _require_model(self) -> torch.nn.Module:
        if self.model is None:
            raise RuntimeError("LocalBackend.setup() must be called before use")
        return self.model

    # ── samplers ─────────────────────────────────────────────────────────

    def _ensure_vllm(self) -> None:
        """Boot the vLLM engine on first use and publish the current adapter."""
        if self.sampler != "vllm" or self._vllm is not None:
            return
        self._require_model()
        import tempfile

        from ctm.backends.local.vllm_sampler import VLLMSampler

        self._adapter_scratch = Path(tempfile.mkdtemp(prefix="ctm-policy-adapter-"))
        self._vllm = VLLMSampler(model=self.model_name, enable_lora=True, **self.vllm_options)
        self._publish_adapter()  # initial policy (fresh or resumed adapter)

    def _sleep_vllm_for_training(self) -> None:
        """Offload a colocated vLLM engine before torch training work."""

        if self._vllm is not None:
            self._vllm.sleep()

    def _wake_vllm_for_sampling(self) -> None:
        """Restore a sleeping vLLM engine immediately before generation."""

        if self._vllm is None or not self._vllm.sleeping:
            return
        # Backward can leave large unused blocks in torch's CUDA allocator.
        # Release them before vLLM restores its weights and KV cache.
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()
        self._vllm.wake_up()

    def shutdown(self) -> None:
        """Release the lazily started vLLM sampler, if any."""
        if self._vllm is None:
            return
        try:
            self._vllm.shutdown()
        finally:
            self._vllm = None

    def policy_sampler(self, name: str) -> LocalSamplerHandle:
        self._require_model()
        self._ensure_vllm()
        return LocalSamplerHandle(self, use_base=False)

    async def refresh_policy_sampler(self, name: str) -> LocalSamplerHandle:
        # HF sampler: the in-process model IS the live policy — refresh is free.
        # vLLM sampler: snapshot the adapter and hot-reload it into the engine.
        # (Cold engine: policy_sampler's lazy boot does the initial publish.)
        if self._vllm is not None:
            self._publish_adapter()
        return self.policy_sampler(name)

    def _publish_adapter(self) -> None:
        """Snapshot the current LoRA adapter and point the vLLM engine at it."""
        assert self._vllm is not None and self._adapter_scratch is not None
        version_dir = self._adapter_scratch / f"v{self._vllm.adapter_version + 1}"
        self._require_model().save_pretrained(str(version_dir))
        self._vllm.advance_policy(str(version_dir))

    def base_sampler(self) -> LocalSamplerHandle:
        self._require_base()
        self._ensure_vllm()
        return LocalSamplerHandle(self, use_base=True)

    def _require_base(self):
        if not ((self.use_lora and HAS_PEFT) or self._frozen_base_model is not None):
            raise NotImplementedError(
                "Base-model access (anchor sampling / KL-to-base / distill) on LocalBackend "
                "requires LoRA or keep_frozen_base=True for full fine-tuning."
            )

    def _base_ctx(self):
        self._require_base()
        if self.use_lora:
            return self._require_model().disable_adapter()  # peft context manager
        return _nullcontext()

    def _model_for(self, *, use_base: bool) -> torch.nn.Module:
        if not use_base:
            return self._require_model()
        self._require_base()
        if self.use_lora:
            return self._require_model()
        assert self._frozen_base_model is not None
        return self._frozen_base_model

    def _sample(
        self,
        *,
        prompt_tokens: list[int],
        max_tokens: int,
        temperature: float,
        stop: Any,
        num_samples: int,
        use_base: bool,
    ) -> list[SampledSequence]:
        """Route sampling to the configured engine (vLLM if set, else HF generate)."""
        return self._sample_batch(
            prompt_tokens_batch=[prompt_tokens],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            num_samples=num_samples,
            use_base=use_base,
        )[0]

    def _sample_batch(
        self,
        *,
        prompt_tokens_batch: Sequence[Sequence[int]],
        max_tokens: int,
        temperature: float,
        stop: Any,
        num_samples: int,
        use_base: bool,
    ) -> list[list[SampledSequence]]:
        """Route a prompt batch to one vLLM/HF generation call."""
        if self._vllm is not None:
            self._wake_vllm_for_sampling()
            return self._vllm.sample_batch(
                [list(tokens) for tokens in prompt_tokens_batch],
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                num_samples=num_samples,
                use_base=use_base,
            )
        return self._generate_batch(
            prompt_tokens_batch=prompt_tokens_batch,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            num_samples=num_samples,
            use_base=use_base,
        )

    def _generate(
        self,
        *,
        prompt_tokens: list[int],
        max_tokens: int,
        temperature: float,
        stop: Any,
        num_samples: int,
        use_base: bool,
    ) -> list[SampledSequence]:
        return self._generate_batch(
            prompt_tokens_batch=[prompt_tokens],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            num_samples=num_samples,
            use_base=use_base,
        )[0]

    def _generate_batch(
        self,
        *,
        prompt_tokens_batch: Sequence[Sequence[int]],
        max_tokens: int,
        temperature: float,
        stop: Any,
        num_samples: int,
        use_base: bool,
    ) -> list[list[SampledSequence]]:
        if not prompt_tokens_batch:
            return []
        model = self._model_for(use_base=use_base)
        was_training = model.training
        model.eval()
        eos_ids = [t for t in (stop or []) if isinstance(t, int)]
        configured_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        if isinstance(configured_eos, int):
            configured_eos = [configured_eos]
        if isinstance(configured_eos, (list, tuple)):
            eos_ids.extend(t for t in configured_eos if isinstance(t, int))
        eos_ids = list(dict.fromkeys(eos_ids)) or None
        max_prompt_len = max(len(tokens) for tokens in prompt_tokens_batch)
        if max_prompt_len == 0:
            raise ValueError("sampling prompts must contain at least one token")
        pad_token_id = eos_ids[0] if eos_ids else 0
        input_ids = torch.full(
            (len(prompt_tokens_batch), max_prompt_len),
            pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for index, tokens in enumerate(prompt_tokens_batch):
            if not tokens:
                raise ValueError("sampling prompts must contain at least one token")
            input_ids[index, -len(tokens) :] = torch.tensor(tokens, dtype=torch.long, device=self.device)
            attention_mask[index, -len(tokens) :] = 1
        try:
            with torch.no_grad():
                ctx = self._base_ctx() if use_base else _nullcontext()
                with ctx:
                    out = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        do_sample=True,
                        temperature=max(temperature, 1e-4),
                        max_new_tokens=max_tokens,
                        num_return_sequences=num_samples,
                        eos_token_id=eos_ids,
                        pad_token_id=pad_token_id,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
        finally:
            if was_training:
                model.train()

        batches: list[list[SampledSequence]] = []
        for prompt_index in range(len(prompt_tokens_batch)):
            sequences: list[SampledSequence] = []
            for sample_index in range(num_samples):
                sequence_index = prompt_index * num_samples + sample_index
                tokens: list[int] = []
                logprobs: list[float] = []
                for step, step_scores in enumerate(out.scores):
                    position = max_prompt_len + step
                    if position >= out.sequences.shape[1]:
                        break
                    token = int(out.sequences[sequence_index, position])
                    logprob = torch.log_softmax(step_scores[sequence_index].float(), dim=-1)[token]
                    tokens.append(token)
                    logprobs.append(float(logprob))
                    if eos_ids and token in eos_ids:
                        break
                sequences.append(SampledSequence(tokens=tokens, logprobs=logprobs))
            batches.append(sequences)
        return batches

    # ── training ─────────────────────────────────────────────────────────

    def _target_logprobs_batch(self, datums: Sequence[Any], use_base: bool = False) -> list[torch.Tensor]:
        """Forward one physical batch and gather each datum's target-token logprobs."""
        model = self._model_for(use_base=use_base)
        token_lists = [d.model_input.to_ints() for d in datums]
        max_len = max(len(t) for t in token_lists)
        input_ids = torch.zeros((len(datums), max_len), dtype=torch.long, device=self.device)
        attention_mask = torch.zeros_like(input_ids)
        for i, toks in enumerate(token_lists):
            input_ids[i, : len(toks)] = torch.tensor(toks, dtype=torch.long)
            attention_mask[i, : len(toks)] = 1
        ctx = self._base_ctx() if use_base else _nullcontext()
        with ctx:
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        out = []
        for i, d in enumerate(datums):
            targets = d.loss_fn_inputs["target_tokens"].to_torch().long().to(self.device)
            n = len(token_lists[i])
            out.append(logprobs[i, :n].gather(1, targets.unsqueeze(1)).squeeze(1))
        return out

    def _microbatches(self, datums: Sequence[Any]) -> list[Sequence[Any]]:
        size = self.training_microbatch_size or len(datums)
        return [datums[start : start + size] for start in range(0, len(datums), size)]

    def _target_logprobs(self, datums: Sequence[Any], use_base: bool = False) -> list[torch.Tensor]:
        """Forward datums in bounded physical batches and preserve their order."""
        return [
            logprobs
            for microbatch in self._microbatches(datums)
            for logprobs in self._target_logprobs_batch(microbatch, use_base=use_base)
        ]

    @staticmethod
    def _loss_denominator(datums: Sequence[Any], loss_fn: str) -> float:
        field = "weights" if loss_fn == "cross_entropy" else "mask"
        return sum(float(d.loss_fn_inputs[field].to_torch().float().sum()) for d in datums)

    async def submit_forward_backward(self, datums: Sequence[Any], loss_fn: str) -> _ResolvedPending:
        self._sleep_vllm_for_training()
        if loss_fn in CONSISTENCY_LOSS_CLASSES:
            return self._consistency_forward_backward(datums, loss_fn)
        model = self._require_model()
        model.train()
        if loss_fn not in ("cross_entropy", "ppo", "importance_sampling"):
            raise ValueError(f"Unknown loss_fn: {loss_fn}")

        total_denominator = self._loss_denominator(datums, loss_fn)
        reported_logprobs: list[torch.Tensor] = []
        reported_loss = 0.0
        for microbatch in self._microbatches(datums):
            target_logprobs = self._target_logprobs_batch(microbatch)
            if loss_fn == "cross_entropy":
                weights = [d.loss_fn_inputs["weights"].to_torch().to(self.device) for d in microbatch]
                microbatch_loss = losses.cross_entropy_loss(target_logprobs, weights)
            else:
                sampled = [d.loss_fn_inputs["logprobs"].to_torch().to(self.device) for d in microbatch]
                advs = [d.loss_fn_inputs["advantages"].to_torch().to(self.device) for d in microbatch]
                masks = [d.loss_fn_inputs["mask"].to_torch().float().to(self.device) for d in microbatch]
                if loss_fn == "ppo":
                    microbatch_loss = losses.ppo_loss(
                        target_logprobs,
                        sampled,
                        advs,
                        masks,
                        clip_epsilon=self.ppo_clip_epsilon,
                    )
                else:
                    microbatch_loss = losses.importance_sampling_loss(target_logprobs, sampled, advs, masks)
            scale = self._loss_denominator(microbatch, loss_fn) / max(total_denominator, 1e-8)
            scaled_loss = microbatch_loss * scale
            scaled_loss.backward()  # accumulate; optim_step applies + zeroes
            reported_logprobs.extend(lp.detach().cpu() for lp in target_logprobs)
            reported_loss += float(scaled_loss.detach())

        self._gradient_accumulations += 1
        return _ResolvedPending(
            ForwardBackwardOutput(
                logprobs=reported_logprobs,
                metrics={"loss": reported_loss},
            )
        )

    def _consistency_loss(self, loss_fn: str) -> consistency_losses.ConsistencyLoss:
        if loss_fn not in self._consistency_loss_modules:
            self._consistency_loss_modules[loss_fn] = consistency_losses.create_consistency_loss(
                loss_fn, self.consistency_loss_options
            )
        return self._consistency_loss_modules[loss_fn]

    def _consistency_forward_backward(self, datums: Sequence[Any], loss_fn: str) -> _ResolvedPending:
        """Paired-pass gradient accumulation for the consistency loss_fns (ACT/AttCT/MLPCT).

        Per datum (batch of 1, mirroring the upstream AttCT pipeline): a
        differentiable pass on the biased prompt, a no-grad reference pass on
        the clean prompt under the disabled adapter (the frozen base), then the
        loss over the aligned window from the datum's loss_fn_inputs. Backward
        runs per datum (mean over the batch), so peak memory holds one graph.
        """
        model = self._require_model()
        self._require_base()  # clean pass needs the frozen base (LoRA adapter disabled)
        model.train()
        loss_module = self._consistency_loss(loss_fn)
        needs_attentions = loss_fn == "attention_consistency"
        needs_hidden = loss_fn == "activation_consistency"

        # sdpa/flash kernels don't materialize attention weights (transformers ≥5
        # returns empty ``attentions`` instead of falling back) — switch to eager.
        if needs_attentions and model.config._attn_implementation != "eager":
            print(
                f"LocalBackend: switching attention from {model.config._attn_implementation!r} to 'eager' for {loss_fn}"
            )
            model.set_attn_implementation("eager")
        base_model = self._model_for(use_base=True)
        if needs_attentions and base_model is not model and base_model.config._attn_implementation != "eager":
            base_model.set_attn_implementation("eager")

        hooks = None
        base_hooks = None
        if loss_module.needs_mlp_hooks:
            if self._mlp_hooks is None:
                self._mlp_hooks = MLPHookManager(model, variant=getattr(loss_module, "variant", "hidden"))
            hooks = self._mlp_hooks.install()
            if base_model is model:
                base_hooks = hooks
            else:
                if self._base_mlp_hooks is None:
                    self._base_mlp_hooks = MLPHookManager(base_model, variant=getattr(loss_module, "variant", "hidden"))
                base_hooks = self._base_mlp_hooks.install()

        def forward(tokens: list[int], use_base: bool):
            input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
            ctx = self._base_ctx() if use_base else _nullcontext()
            active_model = self._model_for(use_base=use_base)
            active_hooks = base_hooks if use_base else hooks
            with ctx:
                outputs = active_model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    output_attentions=needs_attentions,
                    output_hidden_states=needs_hidden,
                )
            mlp_states = None
            if active_hooks is not None:
                mlp_states = active_hooks.get_states()
                active_hooks.clear()
            return outputs, mlp_states

        total_loss = 0.0
        try:
            for d in datums:
                idx = {
                    k: int(d.loss_fn_inputs[k].to_torch()[0])
                    for k in ("start_index", "clean_start_index", "clean_len", "match_len")
                }
                adv_outputs, adv_mlp_states = forward(d.model_input.to_ints(), use_base=False)
                with torch.no_grad():
                    clean_outputs, clean_mlp_states = forward(
                        d.loss_fn_inputs["clean_tokens"].to_torch().long().tolist(), use_base=True
                    )
                out = loss_module(
                    clean_outputs,
                    adv_outputs,
                    **idx,
                    clean_mlp_states=clean_mlp_states,
                    adv_mlp_states=adv_mlp_states,
                )
                (out["loss"] / len(datums)).backward()  # accumulate; frees this datum's graph
                total_loss += float(out["loss"].detach())
        finally:
            if hooks is not None:
                hooks.remove()
            if base_hooks is not None and base_hooks is not hooks:
                base_hooks.remove()

        self._gradient_accumulations += 1
        return _ResolvedPending(ForwardBackwardOutput(logprobs=[], metrics={"loss": total_loss / max(len(datums), 1)}))

    async def submit_optim_step(self, *, learning_rate: float, adam: AdamConfig) -> _ResolvedPending:
        model = self._require_model()
        params = [p for p in model.parameters() if p.requires_grad]
        if self._optimizer is None:
            self._optimizer = torch.optim.AdamW(
                params,
                lr=learning_rate,
                betas=(adam.beta1, adam.beta2),
                eps=adam.eps,
                weight_decay=adam.weight_decay,
            )
            if self._pending_optimizer_state is not None:
                self._optimizer.load_state_dict(self._pending_optimizer_state)
                self._pending_optimizer_state = None
        for group in self._optimizer.param_groups:
            group["lr"] = learning_rate
        if self._gradient_accumulations > 1:
            scale = 1.0 / self._gradient_accumulations
            for parameter in params:
                if parameter.grad is not None:
                    parameter.grad.mul_(scale)
        if adam.grad_clip_norm and adam.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, adam.grad_clip_norm)
        self._optimizer.step()
        self._optimizer.zero_grad(set_to_none=True)
        self._gradient_accumulations = 0
        return _ResolvedPending(None)

    async def incorporate_kl_penalty(
        self, datums: Sequence[Any], *, kl_coef: float, kl_discount_factor: float
    ) -> dict[str, float]:
        """Same math as tinker_cookbook.rl.metrics.incorporate_kl_penalty, with base
        logprobs from a local forward pass under the disabled adapter."""
        import tinker

        self._require_base()
        self._sleep_vllm_for_training()
        with torch.no_grad():
            base_logprobs = self._target_logprobs(datums, use_base=True)

        sampled = [d.loss_fn_inputs["logprobs"].to_torch() for d in datums]
        masks = [d.loss_fn_inputs["mask"].to_torch().float() for d in datums]
        diffs = [(s - b.cpu()) * m for s, b, m in zip(sampled, base_logprobs, masks)]
        total_mask = sum(m.sum() for m in masks)
        avg_diff = sum(d.sum() for d in diffs) / total_mask.clamp(min=1e-8)
        for i, datum in enumerate(datums):
            kl_advantages = kl_coef * masks[i] * (avg_diff - diffs[i])
            if kl_discount_factor > 0:
                kl_advantages = discounted_future_sum_vectorized(kl_advantages, kl_discount_factor)
            datum.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(
                datum.loss_fn_inputs["advantages"].to_torch() + kl_advantages
            )
        return {"kl_policy_base": float(avg_diff)}

    def _score_completions(
        self,
        prompts: Sequence[Any],
        completion_tokens: Sequence[Sequence[int]],
        *,
        use_base: bool,
    ) -> list[list[float]]:
        """Score continuations under the selected raw policy on supplied prompts.

        Logits at prompt position ``R - 1 + t`` predict completion token ``t``.
        This is intentionally independent of the generation temperature used to
        obtain the tokens.
        """

        if len(prompts) != len(completion_tokens):
            raise ValueError(
                "prompts and completion_tokens must have the same length, got "
                f"{len(prompts)} and {len(completion_tokens)}"
            )
        if use_base:
            self._require_base()
        else:
            self._require_model()
        if not prompts:
            return []

        prompt_tokens = [list(prompt.to_ints()) for prompt in prompts]
        continuations = [list(tokens) for tokens in completion_tokens]
        for index, (prompt, completion) in enumerate(zip(prompt_tokens, continuations)):
            if not prompt:
                raise ValueError(f"prompt {index} is empty")
            if not completion:
                raise ValueError(f"completion {index} is empty")

        sequences = [prompt + completion for prompt, completion in zip(prompt_tokens, continuations)]
        max_len = max(len(sequence) for sequence in sequences)
        input_ids = torch.zeros((len(sequences), max_len), dtype=torch.long, device=self.device)
        attention_mask = torch.zeros_like(input_ids)
        for index, sequence in enumerate(sequences):
            input_ids[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=self.device)
            attention_mask[index, : len(sequence)] = 1

        model = self._model_for(use_base=use_base)
        was_training = model.training
        model.eval()
        try:
            ctx = self._base_ctx() if use_base else _nullcontext()
            with ctx, torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                logprobs = torch.log_softmax(logits.float(), dim=-1)
        finally:
            model.train(was_training)

        output: list[list[float]] = []
        for index, (prompt, completion) in enumerate(zip(prompt_tokens, continuations)):
            start = len(prompt) - 1
            positions = logprobs[index, start : start + len(completion)]
            targets = torch.tensor(completion, dtype=torch.long, device=self.device)
            values = positions.gather(1, targets.unsqueeze(1)).squeeze(1)
            output.append(values.detach().cpu().tolist())
        return output

    # ── checkpoints ──────────────────────────────────────────────────────

    async def save_checkpoint(self, *, name: str, log_dir: str | Path, loop_state: dict, kind: str) -> dict:
        model = self._require_model()
        ckpt_dir = Path(log_dir) / "checkpoints" / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if self.use_lora and HAS_PEFT:
            model.save_pretrained(str(ckpt_dir))  # adapter weights only
        else:
            torch.save(model.state_dict(), ckpt_dir / "weights.pt")
        state_saved = False
        if kind in ("state", "both") and self._optimizer is not None:
            torch.save(self._optimizer.state_dict(), ckpt_dir / "optimizer.pt")
            state_saved = True
        (ckpt_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "backend": "local",
                    "model": self.model_name,
                    "lora": self.use_lora,
                    "full_finetune_modules": self.full_finetune_modules,
                    "keep_frozen_base": self.keep_frozen_base,
                    "trainable_parameter_names": self._trainable_parameter_names,
                    "kind": kind,
                    "loop_state": loop_state,
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        uri = f"file://{ckpt_dir.resolve()}"
        return {"sampler_path": uri, "state_path": uri if state_saved else None}

    def _load_checkpoint(self, resume_from: str, with_optimizer: bool) -> None:
        ckpt_dir = _strip_file_scheme(resume_from)
        if self.use_lora and HAS_PEFT:
            state = peft.utils.load_peft_weights(str(ckpt_dir))
            peft.set_peft_model_state_dict(self.model, state)
        else:
            weights = ckpt_dir / "weights.pt"
            self._require_model().load_state_dict(torch.load(weights, map_location=self.device))
        opt_path = ckpt_dir / "optimizer.pt"
        if with_optimizer and opt_path.exists():
            # Optimizer is created lazily at the first optim_step; stage the state.
            self._pending_optimizer_state = torch.load(opt_path, map_location=self.device)
        print(f"LocalBackend: loaded checkpoint from {ckpt_dir} (optimizer: {with_optimizer and opt_path.exists()})")
