"""VDCT entry point: ``python -m recipe.vdct.main_vdct <hydra overrides>``.

Structural copy of ``recipe.rmct.main_rmct`` (see that module's docstring for
the V0/V1 split and why ``BaseTaskRunner.run`` is replicated rather than
subclassed): built on ``verl.trainer.main_ppo_v0.BaseTaskRunner``, pins
``trainer.use_v1=False``, forces ``use_reference_policy=True`` through
``validate_config``, and carries the non-LoRA role-mapping fix — which VDCT
exercises by default, since the runs are full-parameter (no LoRA, so no
``ref_in_actor``; the reference forward runs in a separate worker).
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


class VDCTTaskRunner(BaseTaskRunner):
    """TaskRunner that builds a ``RayVDCTTrainer``."""

    def add_actor_rollout_worker(self, config):
        # BaseTaskRunner picks the fused ActorRolloutRef role only when
        # need_reference_policy(config) is true, which reads verl's two KL
        # switches — both off for VDCT (the trainer owns the KL term). Without
        # LoRA (ref_in_actor) that would register the worker under
        # Role.ActorRollout while init_workers, seeing our forced
        # use_reference_policy, asserts on Role.ActorRolloutRef. Re-key it.
        cls, wg_cls = super().add_actor_rollout_worker(config)
        from verl.trainer.ppo.ray_trainer import Role

        if Role.ActorRollout in self.role_worker_mapping:
            lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
            if lora_rank <= 0:
                lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
            ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
            if not ref_in_actor:
                self.role_worker_mapping[Role.ActorRolloutRef] = self.role_worker_mapping.pop(Role.ActorRollout)
                self.mapping[Role.ActorRolloutRef] = self.mapping.pop(Role.ActorRollout)
        return cls, wg_cls

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
            # VDCT always needs the reference forward, even though both of
            # verl's own KL switches are off. See RayVDCTTrainer.__init__.
            use_reference_policy=True,
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
    if config.trainer.get("use_v1", True):
        raise ValueError(
            "The VDCT recipe overrides the V0 RayPPOTrainer._update_actor; set trainer.use_v1=False "
            "(config/vdct_trainer.yaml pins it). Porting to the V1 TransferQueue trainer is separate work."
        )
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(VDCTTaskRunner))


if __name__ == "__main__":
    main()
