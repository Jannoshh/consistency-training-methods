"""RMCT as an external verl recipe (verl @ 2b0fe51).

Deliberately empty of imports: ``rmct_core`` must be importable (and testable)
without verl installed, which an eager ``from .rmct_trainer import ...`` here
would break. The pieces are imported where they are used —
``main_rmct`` imports the trainer, and the ``rmct`` agent loop is registered
through ``config/agent_loop.yaml`` (``actor_rollout_ref.rollout.agent.
agent_loop_config_path``), which needs no import side effect in the
``AgentLoopWorker`` Ray actor.

``recipe`` itself is left as an implicit namespace package so that this
directory and verl's own ``recipe/`` can coexist on ``PYTHONPATH``.
"""
