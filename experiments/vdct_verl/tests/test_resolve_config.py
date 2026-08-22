"""Tests for the config-preflight tool.

The compose test needs ``hydra-core`` and a verl checkout — both optional in
the CPU test environment, so it skips cleanly when either is absent (on a pod
they are always present).
"""

import importlib.util

import pytest
from vdct_test_helpers import checkout_or_none, load_script

resolve_config = load_script("resolve_config") if importlib.util.find_spec("hydra") else None
VERL_DIR = checkout_or_none("VERL_DIR", "/home/user/volcengine/verl", "verl/trainer/config/ppo_trainer.yaml")

pytestmark = pytest.mark.skipif(
    resolve_config is None or VERL_DIR is None,
    reason="needs hydra-core and a verl checkout (VERL_DIR)",
)


def test_resolves_and_prints_the_run_plan(capsys):
    assert resolve_config.main(["--verl-dir", str(VERL_DIR), "data.train_batch_size=64"]) == 0
    out = capsys.readouterr().out
    assert "ok" in out
    assert "16 datapoints/step, 512 rollouts/step, 256 carrying gradient" in out
    assert "vdct.consistency_weight = 1.0" in out
    # The overlay must pin the prompt ceiling; verl's 512 default truncates.
    assert "data.max_prompt_length = 4096" in out


def test_rejects_v1_trainer(capsys):
    assert resolve_config.main(["--verl-dir", str(VERL_DIR), "trainer.use_v1=True"]) == 1
    assert "use_v1" in capsys.readouterr().err


def test_rejects_reenabled_verl_kl(capsys):
    assert resolve_config.main(["--verl-dir", str(VERL_DIR), "actor_rollout_ref.actor.use_kl_loss=True"]) == 1
    assert "KL" in capsys.readouterr().err


def test_rejects_misaligned_batch_size(capsys):
    assert resolve_config.main(["--verl-dir", str(VERL_DIR), "data.train_batch_size=66"]) == 1
    assert "4 x datapoints_per_step" in capsys.readouterr().err
