#!/usr/bin/env python3
"""Dry-resolve the VDCT hydra overlay against a verl checkout — no GPU, no Ray.

The Phase 2 acceptance's "one dry-resolved config prints the full run plan":
composes ``recipe/vdct/config/vdct_trainer.yaml`` on top of verl's
``ppo_trainer`` defaults (exactly what ``main_vdct`` does at launch), applies
any extra hydra overrides from the command line, checks the keys and
invariants the recipe depends on, and prints the resolved run plan. Run it as
a preflight before submitting a pod job — a typo'd override or a key that
verl renamed fails here instead of after the first rollout.

The contract lives in the recipe, not here: the ``vdct.*`` key list derives
from ``VDCTConfig``'s fields, and the invariant checks are
``vdct_schema.vdct_config_problems`` — the same function ``main_vdct`` and
``RayVDCTTrainer`` enforce at launch.

Needs ``hydra-core`` and ``omegaconf`` (present in any verl environment).

Usage:
    python experiments/vdct_verl/scripts/resolve_config.py \
        --verl-dir /workspace/verl \
        data.train_batch_size=64 vdct.lambda_log_score=0.3
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

_RECIPE_ROOT = Path(__file__).resolve().parents[1]
if str(_RECIPE_ROOT) not in sys.path:
    sys.path.insert(0, str(_RECIPE_ROOT))

from recipe.vdct.vdct_schema import ROWS_PER_DATAPOINT, VDCTConfig, vdct_config_problems

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "recipe" / "vdct" / "config"
_MISSING = "\0missing\0"  # OmegaConf.select default must be a config-representable value

# Keys the recipe reads at runtime; a rename on either side must fail here.
# The vdct block derives from VDCTConfig so a new knob is covered automatically.
REQUIRED_KEYS = [
    *(f"vdct.{spec.name}" for spec in fields(VDCTConfig)),
    "vdct.kl_coef",
    "vdct.kl_logprob_source",
    "vdct.parse_sum_tolerance",
    "algorithm.use_kl_in_reward",
    "actor_rollout_ref.actor.use_kl_loss",
    "actor_rollout_ref.rollout.n",
    "actor_rollout_ref.rollout.temperature",
    "actor_rollout_ref.rollout.prompt_length",
    "actor_rollout_ref.rollout.response_length",
    "actor_rollout_ref.rollout.agent.agent_loop_config_path",
    "data.prompt_key",
    "data.return_raw_chat",
    "data.train_batch_size",
    "data.max_prompt_length",
    "data.max_response_length",
    "trainer.use_v1",
    "trainer.rollout_data_dir",
]

# The run-plan summary printed for approval.
PLAN_KEYS = REQUIRED_KEYS + [
    "actor_rollout_ref.model.path",
    "actor_rollout_ref.rollout.top_p",
    "actor_rollout_ref.rollout.top_k",
    "data.train_files",
    "trainer.total_epochs",
    "trainer.experiment_name",
    "trainer.default_local_dir",
    "trainer.n_gpus_per_node",
    "trainer.nnodes",
]


def resolve(verl_dir: Path, overrides: list[str]):
    searchpath = f"file://{verl_dir.resolve()}/verl/trainer/config"
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        return compose(
            config_name="vdct_trainer",
            overrides=[f"hydra.searchpath=[{searchpath}]", *overrides],
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verl-dir", required=True, type=Path, help="verl checkout (pinned SHA)")
    parser.add_argument("--full", action="store_true", help="print the entire resolved config, not just the plan")
    parser.add_argument("overrides", nargs="*", help="extra hydra overrides, exactly as passed to main_vdct")
    args = parser.parse_args(argv)

    if not (args.verl_dir / "verl" / "trainer" / "config" / "ppo_trainer.yaml").exists():
        print(f"error: {args.verl_dir} does not look like a verl checkout", file=sys.stderr)
        return 2

    config = resolve(args.verl_dir, args.overrides)

    missing = [key for key in REQUIRED_KEYS if OmegaConf.select(config, key, default=_MISSING) == _MISSING]
    if missing:
        print("error: resolved config is missing keys the recipe reads:", file=sys.stderr)
        for key in missing:
            print(f"  {key}", file=sys.stderr)
        return 1

    problems = vdct_config_problems(config)
    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        return 1

    if args.full:
        print(OmegaConf.to_yaml(config, resolve=False))
    print("resolved run plan (vdct_trainer over verl ppo_trainer):")
    for key in PLAN_KEYS:
        print(f"  {key} = {OmegaConf.select(config, key)}")
    rows = int(config.data.train_batch_size)
    n = int(config.actor_rollout_ref.rollout.n)
    datapoints = rows // ROWS_PER_DATAPOINT
    print(f"  -> {datapoints} datapoints/step, {rows * n} rollouts/step, {datapoints * 2 * n} carrying gradient")
    kl = float(config.vdct.kl_coef)
    print(f"  -> reference forward {'ON (KL vs frozen base)' if kl != 0.0 else 'OFF (vdct.kl_coef=0)'}")
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
