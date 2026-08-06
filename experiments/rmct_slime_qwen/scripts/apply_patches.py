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


if __name__ == "__main__":
    results = [patch_save_args_rank(), patch_validation_diagnostics()]
    print("; ".join(results))
    sys.exit(0)
