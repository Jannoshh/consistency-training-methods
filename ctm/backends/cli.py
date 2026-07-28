"""Shared CLI plumbing for backend selection.

Used by scripts/train_rlct.py and train_bct.py (and any future
training entry point): one flag group, one builder, no per-script duplication.
"""

import argparse

from ctm.backends.base import TrainingBackend

_DTYPES = ("float32", "bfloat16", "float16")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def add_backend_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("backend", "Training compute backend selection")
    g.add_argument(
        "--backend",
        default="tinker",
        choices=["tinker", "local"],
        help="'tinker' (managed service, default) or 'local' (in-process torch/PEFT "
        "on this machine's GPUs — the Isambard/Vast.ai path)",
    )
    g.add_argument("--local-device", default=None, help="LocalBackend device (default: cuda if available, else cpu)")
    g.add_argument("--local-dtype", default="bfloat16", choices=list(_DTYPES), help="LocalBackend model dtype")
    g.add_argument(
        "--local-sampler",
        default="vllm",
        choices=["vllm", "hf"],
        help="Rollout engine for --backend local: 'vllm' (fast, LoRA hot-reload; "
        "production) or 'hf' (model.generate; correct but slow — debugging). "
        "The engine boots lazily on first rollout, so non-sampling (SFT) runs never touch it",
    )
    g.add_argument(
        "--local-gpu-mem-util",
        type=float,
        default=0.45,
        help="vLLM gpu_memory_utilization — leave headroom for the training model " "when colocated on one GPU",
    )
    g.add_argument(
        "--local-vllm-max-model-len",
        type=_positive_int,
        default=None,
        help="Bound vLLM context length instead of inheriting the model maximum. Set this "
        "at least as high as prompt tokens + --max-new-tokens to reduce oversized warmup/capture work",
    )
    g.add_argument(
        "--local-vllm-max-num-seqs",
        type=_positive_int,
        default=None,
        help="Bound vLLM's maximum concurrent sequences. A production-safe starting point is the "
        "largest num_samples in one sampler call",
    )
    g.add_argument(
        "--local-vllm-sleep-during-training",
        action="store_true",
        help="Enable vLLM sleep mode: offload its weights and discard KV cache during local "
        "KL/policy training, then restore them before rollout generation. Recommended when "
        "vLLM and the training model share one GPU",
    )
    g.add_argument(
        "--local-training-microbatch-size",
        type=_positive_int,
        default=None,
        help="Maximum sequences per local model forward/backward. Omit for the legacy all-at-once "
        "batch; use a small value when rollout fan-out would otherwise exhaust GPU memory",
    )
    g.add_argument(
        "--local-gradient-checkpointing",
        action="store_true",
        help="Recompute transformer blocks during backward to reduce local training memory",
    )
    g.add_argument(
        "--local-full-finetune",
        action="store_true",
        help="Disable LoRA and train ordinary model parameters; incompatible with --local-sampler vllm",
    )
    g.add_argument(
        "--local-trainable-modules",
        nargs="+",
        metavar="SELECTOR",
        help="With --local-full-finetune, train only matching parameter groups. Selectors may be "
        "dotted components (self_attn) or full-name globs (*.self_attn.*). Omit to train everything",
    )


def build_backend(
    args: argparse.Namespace,
    *,
    consistency_loss_options: dict | None = None,
    requires_frozen_base: bool = False,
) -> TrainingBackend:
    """Build the concrete backend explicitly selected by the CLI."""

    if args.backend == "tinker":
        from ctm.backends.tinker import TinkerBackend

        return TinkerBackend()
    import torch

    from ctm.backends.local.engine import LocalBackend

    if args.local_trainable_modules and not args.local_full_finetune:
        raise ValueError("--local-trainable-modules requires --local-full-finetune")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.local_dtype]
    vllm_options = {"gpu_memory_utilization": args.local_gpu_mem_util}
    if args.local_vllm_max_model_len is not None:
        vllm_options["max_model_len"] = args.local_vllm_max_model_len
    if args.local_vllm_max_num_seqs is not None:
        vllm_options["max_num_seqs"] = args.local_vllm_max_num_seqs
    if args.local_vllm_sleep_during_training:
        vllm_options["enable_sleep_mode"] = True
    return LocalBackend(
        device=args.local_device,
        dtype=dtype,
        use_lora=not args.local_full_finetune,
        sampler=args.local_sampler,
        vllm_options=vllm_options,
        consistency_loss_options=consistency_loss_options,
        full_finetune_modules=args.local_trainable_modules,
        keep_frozen_base=requires_frozen_base and args.local_full_finetune,
        training_microbatch_size=args.local_training_microbatch_size,
        gradient_checkpointing=args.local_gradient_checkpointing,
    )


def describe_backend(args: argparse.Namespace) -> str:
    """One-line summary for the pre-run config printout."""
    if args.backend == "tinker":
        return "tinker (managed service)"
    return (
        f"local ({args.local_device or 'auto'}, {args.local_dtype}, "
        f"sampler={args.local_sampler}, "
        f"{'full-finetune' if args.local_full_finetune else 'LoRA'}"
        f"{f', modules={args.local_trainable_modules}' if args.local_trainable_modules else ''})"
    )
