"""Framework compat: the RL runtime is slime upstream or its Miles fork (D10).

Both expose identical utility APIs under different package names. Import the
handful the RMCT port needs from whichever is installed — Miles preferred
(run tier), slime as fallback (the Phase 3 dev tier images).
"""

from importlib import import_module

for _pkg in ("miles", "slime"):
    try:
        import_module(_pkg)
        FRAMEWORK = _pkg
        break
    except ImportError:
        continue
else:
    raise ImportError("neither 'miles' nor 'slime' is importable")

load_tokenizer = import_module(f"{FRAMEWORK}.utils.processing_utils").load_tokenizer
post = import_module(f"{FRAMEWORK}.utils.http_utils").post
run = import_module(f"{FRAMEWORK}.utils.async_utils").run
RolloutFnTrainOutput = import_module(f"{FRAMEWORK}.rollout.base_types").RolloutFnTrainOutput
Sample = import_module(f"{FRAMEWORK}.utils.types").Sample
