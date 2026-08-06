"""Miles post-init hook (``--custom-megatron-init-path``): install RMCT advantages.

Miles (radixark/miles, the slime fork with LoRA support) dropped slime's
``--custom-advantage-function-path``. This hook — invoked inside each Megatron
trainer process after initialization — rebinds the actor module's
``compute_advantages_and_returns`` to the parity-tested RMCT implementation in
``slime_port.rmct_advantage`` (same ``(args, rollout_data)`` signature, sets
``advantages``/``returns`` in place).

This is an explicit, single-point patch rather than a fork: the substitution
is declared on the command line, and it fails loudly if Miles renames the
target. If Miles regains a native advantage hook, delete this module and pass
the function directly.
"""

import logging

logger = logging.getLogger(__name__)


def megatron_init(*_args, **_kwargs) -> None:
    from miles.backends.megatron_utils import actor as miles_actor

    from .rmct_advantage import compute_advantages

    if not hasattr(miles_actor, "compute_advantages_and_returns"):
        raise RuntimeError(
            "miles.backends.megatron_utils.actor no longer exposes compute_advantages_and_returns; "
            "the RMCT advantage substitution point moved — inspect the Miles version and update miles_init.py"
        )
    miles_actor.compute_advantages_and_returns = compute_advantages
    logger.info("RMCT: replaced compute_advantages_and_returns with slime_port.rmct_advantage.compute_advantages")
