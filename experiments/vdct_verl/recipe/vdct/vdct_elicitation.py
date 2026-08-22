"""VDCT elicitation instruction and strict distribution parser.

Single source of truth for the verbalized-distribution surface format: the
dataset builder appends :func:`elicitation_instruction` to distribution-row
prompts, and the agent loop parses responses with
:func:`parse_option_distribution`. The format decision and its literature
grounding are in ``notes/elicitation_scheme.md``.

Pure string processing — no verl, no torch, CPU-testable.
"""

from __future__ import annotations

import re

DISTRIBUTION_OPEN = "<distribution>"
DISTRIBUTION_CLOSE = "</distribution>"

# Tolerance on the stated total: unit-probability totals must land in
# 1 ± SUM_TOLERANCE, percentage totals in 100 * (1 ± SUM_TOLERANCE).
SUM_TOLERANCE = 0.05
# Post-parse floor applied after renormalization (renormalize -> floor ->
# renormalize). Matches the reward-side epsilon default.
PROBABILITY_FLOOR = 1e-3

_LINE_RE = re.compile(r"^\s*([A-Za-z0-9]+)\s*[:=]\s*([0-9]*\.?[0-9]+)\s*(%?)\s*$")


def elicitation_instruction(option_labels: list[str]) -> str:
    """The instruction appended to a distribution-row prompt.

    Asks for CoT first (always-CoT is decided upstream), then the strict
    block. Wording says "your probability distribution over the options" —
    the model outputs its distribution; no "report" framing.
    """
    labels = ", ".join(option_labels)
    example = "\n".join(f"{label}: <probability>" for label in option_labels)
    return (
        "\n\nAfter thinking through the question step by step, output your "
        "probability distribution over the answer options — the probability "
        "that you would give each option as your final answer. End your "
        "response with exactly one block of this form, covering every option "
        f"({labels}) exactly once, with probabilities that sum to 1:\n"
        f"{DISTRIBUTION_OPEN}\n{example}\n{DISTRIBUTION_CLOSE}"
    )


def _last_block(text: str) -> str | None:
    """The contents of the last well-formed distribution block, or None."""
    start = text.rfind(DISTRIBUTION_OPEN)
    if start == -1:
        return None
    end = text.find(DISTRIBUTION_CLOSE, start)
    if end == -1:
        return None
    return text[start + len(DISTRIBUTION_OPEN) : end]


def parse_option_distribution(
    text: str,
    option_labels: list[str],
    sum_tolerance: float = SUM_TOLERANCE,
    floor: float = PROBABILITY_FLOOR,
) -> list[float] | None:
    """Parse the stated distribution, dense in ``option_labels`` order.

    Strict by design (format compliance is itself trained): returns None
    unless the LAST ``<distribution>`` block contains every option label
    exactly once and nothing else, each line ``LABEL: value`` with a
    non-negative number. Values may be unit probabilities (total ~1) or
    percentages (total ~100, ``%`` suffixes allowed); the scale is decided
    from the total, within ``sum_tolerance`` relative tolerance. On success
    the vector is renormalized to sum 1, floored at ``floor``, and
    renormalized once more, so every entry is strictly positive and the
    result is a proper distribution.
    """
    if not option_labels:
        raise ValueError("option_labels must be non-empty")
    block = _last_block(text)
    if block is None:
        return None

    values: dict[str, float] = {}
    saw_percent_sign = False
    for line in block.splitlines():
        if not line.strip():
            continue
        match = _LINE_RE.match(line)
        if match is None:
            return None
        label, raw_value, percent_sign = match.group(1), match.group(2), match.group(3)
        if label not in option_labels:
            return None
        if label in values:
            return None
        values[label] = float(raw_value)
        saw_percent_sign = saw_percent_sign or bool(percent_sign)
    if set(values) != set(option_labels):
        return None

    total = sum(values.values())
    if abs(total - 1.0) <= sum_tolerance:
        scale = 1.0
        if saw_percent_sign:
            return None  # "%": the numbers claim percent but total ~1 — ambiguous
    elif abs(total - 100.0) <= 100.0 * sum_tolerance:
        scale = 100.0
    else:
        return None

    dense = [values[label] / scale for label in option_labels]
    if any(v < 0.0 for v in dense):
        return None
    if any(v > 1.0 + sum_tolerance for v in dense):
        return None

    normalized = [v / (total / scale) for v in dense]
    floored = [max(v, floor) for v in normalized]
    floored_total = sum(floored)
    return [v / floored_total for v in floored]
