"""RMCT run configuration for the slime fast path.

Loaded from the JSON file named by the ``RMCT_CONFIG`` environment variable
(slime's argparse surface is not extended; one env var keeps the coupling
minimal and the config file lands in the run's provenance record).
"""

import json
import os
from dataclasses import dataclass, field

from .pipeline import AdvantageConfig


@dataclass
class RMCTConfig:
    # Frozen mcq-bias training artifacts (native JSONL, exact paths).
    data_paths: list[str] = field(default_factory=list)
    n_datapoints: int = 64
    # Rollout populations, from the experiment's rate_matching block.
    n_ref_rollouts: int = 128
    n_train_rollouts: int = 128
    batch_size: int = 4  # datapoints per training step
    n_epochs: int = 1
    # Sampling — pinned explicitly; model-card defaults (presence_penalty etc.)
    # must NOT leak in. Rate-affecting: any change is an experiment change.
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_new_tokens: int = 20480
    # Advantage construction (see pipeline.AdvantageConfig).
    advantage: AdvantageConfig = field(default_factory=AdvantageConfig)
    # KL penalty vs the frozen base, tinker semantics (centered, per-token).
    kl_coef: float = 0.05
    # Parse handling: "discard" only for now ("resample" is a later port).
    unparsed_handling: str = "discard"
    # Where step_*.jsonl.zst rollout records land.
    rollout_dir: str = "rollouts"
    seed: int = 42
    # control condition: perturbation 1 uses the unbiased messages too.
    control: bool = False

    @classmethod
    def load(cls) -> "RMCTConfig":
        path = os.environ["RMCT_CONFIG"]
        with open(path) as f:
            raw = json.load(f)
        raw = {k: v for k, v in raw.items() if not k.startswith("_")}  # drop _comment etc.
        adv = AdvantageConfig(**raw.pop("advantage", {}))
        config = cls(advantage=adv, **raw)
        if not config.data_paths:
            raise ValueError("RMCT_CONFIG must set data_paths")
        if config.unparsed_handling != "discard":
            raise NotImplementedError("only unparsed_handling='discard' is ported")
        return config
