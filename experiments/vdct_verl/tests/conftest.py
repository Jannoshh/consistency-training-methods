"""Puts the recipe root on ``sys.path`` before any test module imports
``recipe.vdct.*`` — the one copy of the bootstrap all five test files used to
carry individually."""

import sys
from pathlib import Path

_VDCT_VERL = Path(__file__).resolve().parents[1]
if str(_VDCT_VERL) not in sys.path:
    sys.path.insert(0, str(_VDCT_VERL))
