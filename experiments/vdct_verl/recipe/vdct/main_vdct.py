"""VDCT entry point: ``python -m recipe.vdct.main_vdct <hydra overrides>``.

Built on the RMCT runner (see ``recipe.rmct.main_rmct``'s docstring for the
V0/V1 split and why ``BaseTaskRunner.run`` is replicated rather than
subclassed): ``VDCTTaskRunner`` inherits ``RMCTTaskRunner``'s non-LoRA
role-mapping fix — the verl-version-specific re-keying lives in one place —
and replaces only ``run()`` (different trainer class) plus the
reference-policy decision: VDCT runs the reference forward only when
``vdct.kl_coef`` is nonzero, since the forward exists solely for the KL term
(``RayVDCTTrainer.__init__`` applies the same condition).
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import run_ppo
from verl.trainer.main_ppo_v0 import BaseTaskRunner
from verl.trainer.ppo.utils import create_rl_dataset, create_rl_sampler, need_critic
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device

from . import vdct_core

# Importing vdct_core installs the recipe.rmct sys.path entry; the assert is
# an import-sorter barrier keeping the RMCT-runner import below the bootstrap.
assert vdct_core
from recipe.rmct.main_rmct import RMCTTaskRunner

from .vdct_schema import vdct_config_problems


def _uses_reference_policy(config) -> bool:
    """The reference forward exists only for the VDCT KL term."""
    return float(config.vdct.get("kl_coef", 0.0)) != 0.0


class VDCTTaskRunner(RMCTTaskRunner):
    """RMCTTaskRunner with the VDCT trainer and a conditional reference policy."""

    def add_actor_rollout_worker(self, config):
        if _uses_reference_policy(config):
            # Inherit RMCT's re-keying to Role.ActorRolloutRef (needed because
            # both of verl's own KL switches are off while the trainer still
            # forces the reference forward).
            return super().add_actor_rollout_worker(config)
        # No reference forward: verl's stock registration is already right.
        return BaseTaskRunner.add_actor_rollout_worker(self, config)

    def run(self, config):
        from pprint import pprint

        print(f"VDCTTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_resource_pool(config)
        self.add_teacher_model_resource_pool(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=_uses_reference_policy(config),
            use_critic=need_critic(config),
        )

        from verl.utils.config import omega_conf_to_dataclass
        from verl.workers.config import HFModelConfig

        model_config: HFModelConfig = omega_conf_to_dataclass(config.actor_rollout_ref.model)
        tokenizer = model_config.tokenizer
        processor = model_config.processor

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        from .vdct_trainer import RayVDCTTrainer

        trainer = RayVDCTTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()


@hydra.main(config_path="config", config_name="vdct_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    problems = vdct_config_problems(config)
    if problems:
        raise ValueError("; ".join(problems))
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(VDCTTaskRunner))


if __name__ == "__main__":
    main()
