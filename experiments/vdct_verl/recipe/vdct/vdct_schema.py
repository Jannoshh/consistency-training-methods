"""Shared VDCT row vocabulary, JSONL reading, and converter provenance helpers.

Single owner of the strings that tie the dataset parquet, the agent loop, the
trainer math, and the diagnostics together. Deliberately a zero-dependency
leaf module — importing it must NOT trigger ``vdct_core``'s
``rmct_verl``/``slime_port`` bootstrap, so the dataset builder can run in an
environment where only this experiment directory is synced. ``vdct_core``
re-exports the vocabulary for its callers.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable
from pathlib import Path

REFERENCE_VARIANT = "reference"
TRAINING_VARIANT = "training"
DISTRIBUTION_KIND = "distribution"
ANSWER_KIND = "answer"


def read_jsonl_rows(path: str | Path, required: Iterable[str] = ()) -> list[dict]:
    """Read non-blank JSONL rows, requiring ``required`` keys on each.

    One shared implementation for the builder, the converter, and the
    diagnostics, so malformed files fail with the same ``path:line`` context
    everywhere.
    """
    path = Path(path)
    required = set(required)
    rows: list[dict] = []
    with path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = required - row.keys()
            if missing:
                raise ValueError(f"{path}:{line_no}: row missing {sorted(missing)}")
            rows.append(row)
    return rows


def git_head_sha(path: str | Path) -> str | None:
    """HEAD commit of a checkout, for converter provenance; None if unavailable."""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None
