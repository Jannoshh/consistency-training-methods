"""SFT-family training (BCT + internal-consistency methods) — backend-agnostic loop.

Methods (``SFTConfig.method``):
    bct    — supervised cross-entropy on messages (any backend)
    act    — activation (residual stream) consistency   ┐ paired variant/reference
    attct  — attention (JSD) consistency                 │ prompts; LocalBackend
    mlpct  — MLP post-activation consistency             ┘ only (needs internals)

Usage:
    from ctm.training.sft import train_sft, SFTConfig

    # Basic usage with defaults (Tinker backend)
    checkpoint = asyncio.run(train_sft(Path("data/train.jsonl")))

    # With custom config
    config = SFTConfig(
        experiment_name="bct_debug",
        run_name="control",
        model="meta-llama/Llama-3.1-8B-Instruct",
        optimizer=AdamConfig(learning_rate=1e-4, lr_schedule="linear"),
        batch_size=128,
        n_epochs=1,
    )
    checkpoint = asyncio.run(train_sft(Path("data/train.jsonl"), config=config))

Training data format (JSONL):
    bct:             {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
    act/attct/mlpct: {"variant_messages": [...], "reference_messages": [...]}
                     (prompt pairs; see ctm/training/consistency_data.py)

"""

import asyncio
import json
import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel
from tqdm import tqdm

from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.utils.lr_scheduling import compute_schedule_lr_multiplier
from tinker_cookbook.utils.ml_log import setup_logging

from ctm.backends.base import TrainingBackend
from ctm.backends.renderers import get_renderer_and_tokenizer
from ctm.backends.tinker import TinkerBackend
from ctm.core.config import AdamConfig, CheckpointConfig, LoRAConfig
from ctm.training.checkpoints import finalize_checkpoint, save_intermediate_checkpoint
from ctm.training.consistency_data import build_consistency_datums
from ctm.training.manifest import write_run_manifest
from ctm.training.run_utils import build_log_dir, get_git_state, get_recommended_lr, warn_if_dirty

# Method to backend operation. Internal-consistency operations are available on
# LocalBackend because they require model activations that remote APIs do not expose.
METHOD_LOSS_FNS = {
    "bct": "cross_entropy",
    "act": "activation_consistency",
    "attct": "attention_consistency",
    "mlpct": "mlp_consistency",
}


@dataclass
class _SubmittedMicrobatch:
    microbatch_index: int
    batch_samples: list[Any]
    batch_data: list[Any]
    current_lr: float
    should_step: bool
    optimizer_step: int
    pending_fwd_bwd: Any
    pending_optim: Any


class SFTConfig(BaseModel):
    """SFT training configuration."""

    experiment_name: str = "sft"
    run_name: str = "default"
    wandb_project: Optional[str] = None
    method: Literal["bct", "act", "attct", "mlpct"] = "bct"
    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    lora: LoRAConfig = LoRAConfig()
    optimizer: AdamConfig = AdamConfig()
    n_epochs: int = 1
    batch_size: int = 128
    gradient_accumulation_steps: int = 1
    checkpoint: CheckpointConfig = CheckpointConfig()
    log_base_dir: str = "logs"
    reference_messages_field: str = "reference_messages"
    variant_messages_field: str = "variant_messages"
    alignment_text_field: Optional[str] = None
    method_config: dict = {}
    # Free-form provenance (setting name, data files, ...) set by the CLI;
    # flows into the run manifest via the config dump — the registry generator reads it.
    run_metadata: dict = {}


def load_samples(file_path: Path) -> list[dict]:
    """
    Load training samples from JSONL file.

    Each line should be a JSON object with a "messages" field containing
    a list of message dicts with "role" and "content" fields.
    """
    samples = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def _mean_nll(logprobs_list, weights_list) -> float:
    """Weighted mean NLL: -sum(logprobs·weights)/sum(weights) over the batch.

    Same computation as tinker_cookbook's ``compute_mean_nll``, but over the
    torch tensors the backend protocol returns (the cookbook version requires
    tinker TensorData).
    """
    total_weighted_logprobs = 0.0
    total_weights = 0.0
    for logprobs, weights in zip(logprobs_list, weights_list):
        total_weighted_logprobs += float(logprobs.double().dot(weights.double()))
        total_weights += float(weights.sum())
    if total_weights == 0:
        return math.nan
    return -total_weighted_logprobs / total_weights


async def train_sft(
    file_path: Path,
    config: Optional[SFTConfig] = None,
    max_samples: Optional[int] = None,
    resume_from: Optional[str] = None,
    resume_with_optimizer: Optional[bool] = None,
    backend: Optional[TrainingBackend] = None,
) -> str:
    """
    Run SFT training from JSONL file.

    Args:
        file_path: Path to JSONL file with {"messages": [...]} format
        config: Training configuration (uses defaults if not provided)
        max_samples: Limit number of samples (None = use all)
        resume_from: Checkpoint path to load weights from before training.
            For Tinker: a state_path (tinker://...weights/...) for full optimizer resume,
            or a sampler_path (tinker://...sampler_weights/...) for weights-only.
        resume_with_optimizer: Explicitly choose optimizer-state restore. None (default)
            infers from the URI; pass True/False to override the fragile URI heuristic
            (e.g. for non-standard/local checkpoint paths).
        backend: Training backend; defaults to TinkerBackend().

    Returns:
        Final checkpoint path
    """
    cfg = config or SFTConfig()
    loss_fn = METHOD_LOSS_FNS[cfg.method]
    if cfg.method != "bct" and (backend is None or isinstance(backend, TinkerBackend)):
        raise ValueError(
            f"method={cfg.method!r} trains on internal activations (paired forward passes with "
            "attentions/hidden states) — the Tinker service API doesn't expose them. Use the local backend."
        )
    backend = backend if backend is not None else TinkerBackend()

    # Build log directory: logs/{experiment_name}/{run_name}/
    log_dir = Path(build_log_dir(cfg.log_base_dir, cfg.experiment_name, cfg.run_name))
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging (writes to files + WandB with experiment_name as project, run_name as name)
    logger = setup_logging(
        log_dir=str(log_dir),
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.run_name,
        config=cfg.model_dump(),
    )

    # Log git state for reproducibility
    git_state = get_git_state()
    warn_if_dirty(git_state)
    logger.log_hparams({"git": git_state})

    # Load training data
    samples = load_samples(file_path)
    if max_samples and max_samples < len(samples):
        samples = samples[:max_samples]

    renderer, tokenizer = get_renderer_and_tokenizer(cfg.model, source=backend.renderer_source)

    # Consistency methods train on paired variant/reference prompts: build the datums
    # once up front (dropping unalignable rows) and shuffle datums per epoch.
    # BCT keeps building datums per batch from messages.
    train_items: list = samples
    if cfg.method != "bct":
        train_items, _ = build_consistency_datums(
            tokenizer,
            samples,
            reference_field=cfg.reference_messages_field,
            variant_field=cfg.variant_messages_field,
            alignment_text_field=cfg.alignment_text_field,
        )
        if not train_items:
            raise ValueError(
                f"No usable consistency pairs in {file_path} — rows need "
                f"{cfg.reference_messages_field}/{cfg.variant_messages_field} with the reference user "
                "content contained verbatim in the variant prompt."
            )

    n_samples = len(train_items)
    # Ceiling division: the batch loop below is range(0, n_samples, batch_size), which
    # yields a partial final batch when n_samples % batch_size != 0. Flooring undercounts
    # total_steps, so the LR schedule hits 0 on that batch and goes NEGATIVE on multi-epoch
    # runs (gradient ascent), and gives total_steps=0 (a hard ConfigurationError from the
    # scheduler) whenever n_samples < batch_size. Match the actual loop with ceil.
    if cfg.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if cfg.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    microbatches_per_epoch = (n_samples + cfg.batch_size - 1) // cfg.batch_size
    steps_per_epoch = (microbatches_per_epoch + cfg.gradient_accumulation_steps - 1) // cfg.gradient_accumulation_steps
    total_steps = steps_per_epoch * cfg.n_epochs
    if total_steps == 0:
        raise ValueError(
            f"No training steps: {n_samples} samples, batch_size={cfg.batch_size}, "
            f"n_epochs={cfg.n_epochs}. Add data or lower batch_size."
        )

    # Reproducibility: seed the per-epoch shuffle below. LoRA init is seeded separately by
    # the SDK via lora.seed; without this, "same --seed" runs still differ in data order.
    if cfg.lora.seed is not None:
        random.seed(cfg.lora.seed)

    # Determine learning rate: use configured value or get recommended LR for model
    base_lr: float = (
        cfg.optimizer.learning_rate if cfg.optimizer.learning_rate is not None else get_recommended_lr(cfg.model)
    )

    print(
        f"SFT Training ({cfg.method}): {n_samples} samples, microbatch={cfg.batch_size}, "
        f"grad_accum={cfg.gradient_accumulation_steps}, "
        f"{total_steps} steps, lr={base_lr:.2e}"
    )
    logger.log_hparams({"n_samples": n_samples, "total_steps": total_steps, "file": str(file_path), "base_lr": base_lr})

    write_run_manifest(
        log_dir,
        kind="sft",
        model=cfg.model,
        backend=backend,
        config_dump=cfg.model_dump(),
        extra={"data_file": str(file_path), "n_samples": n_samples},
    )

    # Initialize the backend (resume handling: explicit resume_with_optimizer wins;
    # otherwise infer from the URI — state paths contain "/weights/", sampler paths
    # "/sampler_weights/". The URI heuristic is a fragile fallback; callers can override.)
    with_opt = False
    if resume_from:
        if resume_with_optimizer is None:
            with_opt = "/weights/" in resume_from and "/sampler_weights/" not in resume_from
        else:
            with_opt = resume_with_optimizer
        print(f"Resuming from: {resume_from} (optimizer state: {with_opt})")
    setup_kwargs = {
        "model": cfg.model,
        "lora": cfg.lora,
        "resume_from": resume_from,
        "resume_with_optimizer": with_opt,
    }
    setup_async = getattr(backend, "setup_async", None)
    if setup_async is not None:
        await setup_async(**setup_kwargs)
    else:
        backend.setup(**setup_kwargs)
    if resume_from:
        logger.log_hparams({"resume_from": resume_from, "resume_with_optimizer": with_opt})

    checkpoint_paths: list[str] = []
    global_step = 0
    global_microbatch = 0
    submitted_optimizer_steps = 0
    submit_ahead = min(1, max(0, int(getattr(backend, "training_submit_ahead", 0))))
    metric_key = "nll" if cfg.method == "bct" else "loss"

    # Training loop
    for epoch in range(cfg.n_epochs):
        # Shuffle samples each epoch
        epoch_samples = list(train_items)
        random.shuffle(epoch_samples)
        epoch_loss = 0.0
        n_examples = 0

        batch_starts = list(range(0, n_samples, cfg.batch_size))
        pbar = tqdm(total=len(batch_starts), desc=f"Epoch {epoch+1}")
        pending: deque[_SubmittedMicrobatch] = deque()

        async def _submit_microbatch(
            microbatch_index: int, batch_start: int, *, epoch_samples=epoch_samples, batch_starts=batch_starts
        ) -> _SubmittedMicrobatch:
            nonlocal submitted_optimizer_steps
            batch_samples = epoch_samples[batch_start : batch_start + cfg.batch_size]

            if cfg.method == "bct":
                # Create datums with proper token shifting for next-token prediction
                batch_data = []
                for sample in batch_samples:
                    tokens, weights = renderer.build_supervised_example(sample["messages"])
                    batch_data.append(datum_from_model_input_weights(tokens, weights))
            else:
                batch_data = batch_samples  # already paired consistency datums

            # Compute LR with schedule
            # max(0.0, ...) is belt-and-suspenders against a negative multiplier; the
            # ceil-division total_steps above already keeps step < total_steps.
            lr_mult = max(
                0.0,
                compute_schedule_lr_multiplier(
                    lr_schedule=cfg.optimizer.lr_schedule,
                    step=submitted_optimizer_steps,
                    total_steps=total_steps,
                ),
            )
            current_lr = base_lr * lr_mult

            should_step = (microbatch_index + 1) % cfg.gradient_accumulation_steps == 0 or microbatch_index + 1 == len(
                batch_starts
            )

            # The backend accumulates gradients across forward/backward calls.
            # On the final microbatch in a group, submit the optimizer step
            # immediately behind the forward pass so remote backends preserve
            # their two-phase overlap.
            pending_fwd_bwd = await backend.submit_forward_backward(batch_data, loss_fn=loss_fn)
            pending_optim = None
            if should_step:
                pending_optim = await backend.submit_optim_step(learning_rate=current_lr, adam=cfg.optimizer)
                submitted_optimizer_steps += 1

            return _SubmittedMicrobatch(
                microbatch_index=microbatch_index,
                batch_samples=batch_samples,
                batch_data=batch_data,
                current_lr=current_lr,
                should_step=should_step,
                optimizer_step=submitted_optimizer_steps,
                pending_fwd_bwd=pending_fwd_bwd,
                pending_optim=pending_optim,
            )

        def _checkpoint_due(record: _SubmittedMicrobatch) -> bool:
            every = cfg.checkpoint.save_every_n_steps
            if not record.should_step or not every or record.optimizer_step % every != 0:
                return False
            return total_steps - record.optimizer_step > cfg.checkpoint.skip_near_final_steps

        async def _finish_microbatch(record: _SubmittedMicrobatch, *, epoch=epoch, pbar=pbar) -> None:
            nonlocal epoch_loss, n_examples, global_microbatch, global_step

            # Await results
            fwd_bwd_output = await record.pending_fwd_bwd.result()
            if record.pending_optim is not None:
                await record.pending_optim.result()

            if cfg.method == "bct":
                # Compute proper per-token NLL
                weights = [d.loss_fn_inputs["weights"].to_torch() for d in record.batch_data]
                batch_metric = _mean_nll(fwd_bwd_output.logprobs, weights)
            else:
                batch_metric = fwd_bwd_output.metrics["loss"]

            # Weight each batch's (token-pooled) metric by its sample count so the epoch
            # mean isn't skewed by an unequal final/remainder batch.
            epoch_loss += batch_metric * len(record.batch_samples)
            n_examples += len(record.batch_samples)
            global_microbatch += 1
            if record.should_step:
                global_step += 1

            pbar.set_postfix(
                {
                    metric_key: f"{batch_metric:.4f}",
                    "lr": f"{record.current_lr:.2e}",
                    "step": global_step,
                }
            )
            logger.log_metrics(
                {
                    f"train/{metric_key}": batch_metric,
                    "train/lr": record.current_lr,
                    "train/optimizer_step": global_step,
                },
                step=global_microbatch,
            )

            # Intermediate checkpoint (skip if near final to avoid duplicates)
            if record.should_step:
                await save_intermediate_checkpoint(
                    backend,
                    experiment_name=cfg.experiment_name,
                    run_name=cfg.run_name,
                    checkpoint_cfg=cfg.checkpoint,
                    global_step=global_step,
                    total_steps=total_steps,
                    epoch=epoch,
                    log_dir=log_dir,
                    checkpoint_paths=checkpoint_paths,
                    logger=logger,
                )

            pbar.update(1)

        for microbatch_index, batch_start in enumerate(batch_starts):
            # Do not queue work past a checkpoint: it must capture precisely the
            # requested optimizer state before subsequent mutations are submitted.
            if pending and _checkpoint_due(pending[-1]):
                while pending:
                    await _finish_microbatch(pending.popleft())

            pending.append(await _submit_microbatch(microbatch_index, batch_start))
            if len(pending) >= 1 + submit_ahead:
                await _finish_microbatch(pending.popleft())

        while pending:
            await _finish_microbatch(pending.popleft())
        pbar.close()

        # Epoch summary
        if n_examples > 0:
            avg_loss = epoch_loss / n_examples
            print(f"Epoch {epoch+1} avg {metric_key}: {avg_loss:.4f}")
            logger.log_metrics({f"train/epoch_{metric_key}": avg_loss, "train/epoch": epoch + 1}, step=global_step)

    # Final checkpoint (no step suffix)
    final_path = await finalize_checkpoint(
        backend,
        experiment_name=cfg.experiment_name,
        run_name=cfg.run_name,
        n_epochs=cfg.n_epochs,
        save_state=cfg.checkpoint.save_state,
        global_step=global_step,
        log_dir=log_dir,
        checkpoint_paths=checkpoint_paths,
        logger=logger,
    )

    return final_path


def train_sft_sync(
    file_path: Path,
    config: Optional[SFTConfig] = None,
    max_samples: Optional[int] = None,
    resume_from: Optional[str] = None,
    resume_with_optimizer: Optional[bool] = None,
    backend: Optional[TrainingBackend] = None,
) -> str:
    """Synchronous wrapper for train_sft."""
    return asyncio.run(train_sft(file_path, config, max_samples, resume_from, resume_with_optimizer, backend))
