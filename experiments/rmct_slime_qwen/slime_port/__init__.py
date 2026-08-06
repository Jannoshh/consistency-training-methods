"""RMCT math ported for the slime (SGLang + Megatron) fast path.

``rewards.py`` and ``advantages.py`` are vendored verbatim from the CTM
repository at commit 58c889c354c03d21fc17ae5ef04d6892b43d890b (branch
training-profiling-rebase); only the ``Rollout`` import is redirected to the
local ``types`` module. ``pipeline.py`` reproduces the batch-level advantage
construction of ``ctm.training.rl.RLTrainer._build_training_batch`` without
tinker datum construction. Gate A (tests/test_reward_parity.py) checks both
against the originals on identical inputs.
"""

CTM_SOURCE_COMMIT = "58c889c354c03d21fc17ae5ef04d6892b43d890b"
