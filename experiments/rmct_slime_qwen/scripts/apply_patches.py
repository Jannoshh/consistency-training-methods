"""Idempotent local patches for the Miles image (run by setup_pod.sh).

Each patch targets a bug we hit in production; remove entries as upstream
fixes land (check the referenced symptoms first).

P1  Megatron dist-ckpt save stores the live ``args`` Namespace, whose
    ``rank`` field is per-process — the common-state validation then rejects
    every save at DP>1 ("Rank N common state dict differs ... Mismatched
    keys: ('args',)"). Fix: store a rank-normalized copy.

P2  (diagnostic, optional) When the common-state validation flags a
    Namespace mismatch, log WHICH attributes differ — turns an opaque
    save failure into a one-line diagnosis.
"""

import sys

CHECKPOINTING = "/root/Megatron-LM/megatron/training/checkpointing.py"
VALIDATION = "/root/Megatron-LM/megatron/core/dist_checkpointing/validation.py"


def patch_save_args_rank() -> str:
    s = open(CHECKPOINTING).read()
    if "RMCT patch: args.rank" in s:
        return "P1 already applied"
    old = "    state_dict['args'] = args\n"
    if s.count(old) != 1:
        raise SystemExit(f"P1 anchor not found or ambiguous in {CHECKPOINTING} — upstream changed, re-derive the patch")
    new = (
        "    # RMCT patch: args.rank is per-process; storing it verbatim makes the\n"
        "    # dist-ckpt common-state validation reject the save at DP>1 (ranks\n"
        "    # disagree on the stored Namespace). Store a rank-normalized copy.\n"
        "    from argparse import Namespace as _NS\n"
        "    state_dict['args'] = _NS(**{**vars(args), 'rank': 0})\n"
    )
    open(CHECKPOINTING, "w").write(s.replace(old, new, 1))
    return "P1 applied"


def patch_validation_diagnostics() -> str:
    s = open(VALIDATION).read()
    if "RMCT-DIAG" in s:
        return "P2 already applied"
    anchor = """        if only_in_rank0 or only_in_current_rank or mismatch:
            logger.warning("""
    if anchor not in s:
        return "P2 anchor not found (upstream changed) — skipping diagnostic patch"
    inject = """        if only_in_rank0 or only_in_current_rank or mismatch:
            for key_tuple, t0, t1 in mismatch:
                a = rank0_state_dict
                b = current_rank_state_dict
                for k in key_tuple:
                    a = a[k] if not hasattr(a, k) else getattr(a, k)
                    b = b[k] if not hasattr(b, k) else getattr(b, k)
                if hasattr(a, "__dict__") and hasattr(b, "__dict__"):
                    diffs = {f: (getattr(a, f, None), getattr(b, f, None))
                             for f in set(vars(a)) | set(vars(b))
                             if getattr(a, f, None) != getattr(b, f, None)}
                    logger.warning(f"RMCT-DIAG rank {rank} Namespace diff at {key_tuple}: {diffs}")
            logger.warning("""
    open(VALIDATION, "w").write(s.replace(anchor, inject, 1))
    return "P2 applied"


BRIDGE_HELPERS = "/root/miles/miles/backends/megatron_utils/bridge_lora_helpers.py"


def patch_bridge_lora_load_weights() -> str:
    """P3: bridge-built LoRA models start with load_weights=False and the later
    checkpoint load never populates the language side (dense Qwen3.5 VL-nested
    names) — every language weight stays zero => uniform logits, dead LoRA
    training. Let megatron-bridge load the HF weights itself at construction
    (verified: logits become real, top-token sanity passes)."""
    s = open(BRIDGE_HELPERS).read()
    if "load_weights=True)  # RMCT" in s:
        return "P3 already applied"
    old = "    provider = bridge.to_megatron_provider(load_weights=False)"
    if old not in s:
        raise SystemExit(f"P3 anchor not found in {BRIDGE_HELPERS} — upstream changed, re-derive")
    s = s.replace(old, "    provider = bridge.to_megatron_provider(load_weights=True)  # RMCT P3", 1)
    open(BRIDGE_HELPERS, "w").write(s)
    return "P3 applied"


LOSSES = "/root/miles/miles/backends/training_utils/loss_hub/losses.py"


def patch_logprob_probe() -> str:
    """P4 (diagnostic, env-gated RMCT_LOGPROB_PROBE): print mask-independent
    per-position |train - rollout| logprob profile — localizes garbage
    (uniform vs positional/pad-boundary) without a debugger on the actor."""
    s = open(LOSSES).read()
    if "RMCT-PROBE" in s:
        return "P4 already applied"
    anchor = """        rollout_log_probs = torch.cat(batch["rollout_log_probs"], dim=0)
        abs_diff = (train_scored_log_probs - rollout_log_probs).abs()"""
    if anchor not in s:
        return "P4 anchor not found — upstream changed, re-derive"
    probe = anchor + """
        import os as _os
        if _os.environ.get("RMCT_LOGPROB_PROBE") and torch.distributed.get_rank() == 0:
            _t = train_scored_log_probs.detach().float()
            _r = rollout_log_probs.detach().float()
            _d = (_t - _r).abs()
            _n = _d.numel()
            print("RMCT-PROBE train head/mid/tail:",
                  [round(v, 3) for v in _t[:5].tolist()],
                  [round(v, 3) for v in _t[_n // 2 : _n // 2 + 5].tolist()],
                  [round(v, 3) for v in _t[-5:].tolist()], flush=True)
            print("RMCT-PROBE |diff| head/mid/tail mean:",
                  round(_d[: _n // 3].mean().item(), 3),
                  round(_d[_n // 3 : 2 * _n // 3].mean().item(), 3),
                  round(_d[2 * _n // 3 :].mean().item(), 3), flush=True)"""
    open(LOSSES, "w").write(s.replace(anchor, probe, 1))
    return "P4 applied"


MODEL_PY = "/root/miles/miles/backends/megatron_utils/model.py"
ACTOR_PY = "/root/miles/miles/backends/megatron_utils/actor.py"


def patch_skip_destructive_load() -> str:
    """P5: Miles defaults ``args.load`` to the base torch_dist conversion,
    which carries ``latest_checkpointed_iteration.txt`` and therefore looks
    resumable — but loading it into the BRIDGE-built LoRA model cannot be
    mapped and silently ZEROES the language weights (word embedding first),
    turning every logit uniform. Skip the load unless it is a genuine LoRA
    run checkpoint (has ``iter_*/adapter*`` dirs)."""
    s = open(MODEL_PY).read()
    if "RMCT P5" in s:
        return "P5 already applied"
    old = """    load_dir = getattr(args, "load", None)
    # --load may be unset: setup_model_and_optimizer already asserted pretrained_checkpoint covers it.
    if load_dir is None or _has_loadable_ckpt(load_dir):"""
    if old not in s:
        raise SystemExit(f"P5 anchor not found in {MODEL_PY} — upstream changed, re-derive")
    new = """    load_dir = getattr(args, "load", None)

    def _rmct_is_lora_resume(_d):
        if _d is None:
            return False
        from pathlib import Path as _P
        return any(_P(_d).glob("iter_*/adapter*"))

    # RMCT P5: see experiments/rmct_slime_qwen/upstream_issues.md — loading the
    # base torch_dist conversion into a bridge-built LoRA model zeroes the
    # language weights. Only load genuine LoRA-resume checkpoints here.
    if (
        getattr(args, "lora_rank", 0) > 0
        and getattr(args, "megatron_to_hf_mode", None) == "bridge"
        and not _rmct_is_lora_resume(load_dir)
    ):
        logger.warning("RMCT P5: skipping fallback load; bridge LoRA model already carries HF weights")
        iteration = 0
        # RMCT P8: --lora-adapter-path is normally consumed inside
        # load_checkpoint, which this branch skips — load it directly.
        # Adapter saves have no latest_checkpointed_iteration.txt and the
        # generic loader rejects their format anyway ("unknown checkpoint
        # format"), so this is the ONLY working LoRA resume path.
        _ap = getattr(args, "lora_adapter_path", None)
        if _ap:
            from .lora_utils import load_lora_adapter as _rmct_load_adapter
            _loaded, _it = _rmct_load_adapter(
                model, _ap, optimizer=optimizer, opt_param_scheduler=opt_param_scheduler
            )
            if _loaded:
                if _it is None and optimizer is not None:
                    # no training_state file: refresh fp32 masters from the
                    # freshly written adapter params, or the first step()
                    # restores the pre-load init values
                    optimizer.reload_model_params()
                iteration = _it if _it is not None else 0
                logger.warning(f"RMCT P8: resumed LoRA adapter from {_ap} at iteration {iteration}")
            else:
                logger.warning(f"RMCT P8: could not load --lora-adapter-path={_ap}; fresh adapter")
    # --load may be unset: setup_model_and_optimizer already asserted pretrained_checkpoint covers it.
    elif load_dir is None or _has_loadable_ckpt(load_dir):"""
    s = s.replace(old, new, 1)
    open(MODEL_PY, "w").write(s)
    return "P5 applied"


def patch_ref_from_snapshot() -> str:
    """P6: for bridge LoRA runs the KL reference IS the frozen base the model
    already carries — loading ``args.ref_load`` (torch_dist) into the bridge
    model zeroes it (same mapping failure as P5). Snapshot instead."""
    s = open(ACTOR_PY).read()
    if "RMCT P6" in s:
        return "P6 already applied"
    old = """        if with_ref:
            self.load_other_checkpoint("ref", args.ref_load)"""
    if old not in s:
        raise SystemExit(f"P6 anchor not found in {ACTOR_PY} — upstream changed, re-derive")
    new = """        if with_ref:
            if getattr(args, "lora_rank", 0) > 0 and getattr(args, "megatron_to_hf_mode", None) == "bridge":
                # RMCT P6: ref = the frozen base already in the bridge model.
                self.weights_backuper.backup("ref")
            else:
                self.load_other_checkpoint("ref", args.ref_load)"""
    s = s.replace(old, new, 1)
    open(ACTOR_PY, "w").write(s)
    return "P6 applied"


BRIDGE_HELPERS_P7_ANCHOR = "    model = provider.provide_distributed_model(wrap_with_ddp=True, ddp_config=ddp_config)\n    return model"


def patch_embedding_repair() -> str:
    """P7 (belt-and-braces): if any language embedding is zero after build,
    repair it directly from the HF safetensors. With P5/P6 in place this
    should never fire; it exists to catch new zero-load paths loudly."""
    s = open(BRIDGE_HELPERS).read()
    if "RMCT P7" in s:
        return "P7 already applied"
    if BRIDGE_HELPERS_P7_ANCHOR not in s:
        return "P7 anchor not found (upstream changed) — skipping"
    new = """    model = provider.provide_distributed_model(wrap_with_ddp=True, ddp_config=ddp_config)

    # RMCT P7: belt-and-braces repair of zero-loaded embeddings (see
    # upstream_issues.md). Should not fire when P5/P6 are active.
    import torch as _t
    from safetensors import safe_open as _so
    import json as _j, os as _o
    _core = model[0]
    while hasattr(_core, "module"):
        _core = _core.module
    for _n, _p in _core.named_parameters():
        if "word_embeddings" in _n and "vision" not in _n and "mtp" not in _n and _p.float().norm().item() == 0.0:
            _idx = _j.load(open(_o.path.join(args.hf_checkpoint, "model.safetensors.index.json")))
            _hf_name = "model.language_model.embed_tokens.weight"
            _file = _o.path.join(args.hf_checkpoint, _idx["weight_map"][_hf_name])
            with _so(_file, framework="pt", device="cpu") as _f:
                _w = _f.get_tensor(_hf_name)
            with _t.no_grad():
                _p.data[: _w.shape[0]].copy_(_w.to(_p.dtype))
            logger.warning(f"RMCT P7: repaired zero embedding {_n} from HF ({tuple(_w.shape)})")
    return model"""
    s = s.replace(BRIDGE_HELPERS_P7_ANCHOR, new, 1)
    open(BRIDGE_HELPERS, "w").write(s)
    return "P7 applied"


if __name__ == "__main__":
    results = [
        patch_save_args_rank(),
        patch_validation_diagnostics(),
        patch_bridge_lora_load_weights(),
        patch_logprob_probe(),
        patch_skip_destructive_load(),
        patch_ref_from_snapshot(),
        patch_embedding_repair(),
    ]
    print("; ".join(results))
    sys.exit(0)
