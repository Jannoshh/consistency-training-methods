"""Minimal rollout container for the slime port.

Field-compatible subset of ``ctm.core.types.Rollout`` (same defaults). The
reward/advantage math reads only ``trait_value``, ``perturbation_idx``, and
``parsed_successfully``; the remaining fields keep parity with the original
loop's bookkeeping (rate denominators, skip reasons).
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class Rollout:
    """A single rollout (sampled response)."""

    tokens: list[int]
    logprobs: list[float]
    text: str
    trait_value: float | None
    perturbation_idx: int
    parsed_successfully: bool = True
    answer_parsed: bool = True
    has_logprobs: bool = True
    grader_evaluated: bool = True
    grader_failed: bool = False
    prompt: Any | None = None
