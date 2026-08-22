"""Shared VDCT contracts: row vocabulary, pair-artifact schema, config
invariants, and converter helpers.

Single owner of the strings and rules that tie the dataset parquet, the pair
converters, the agent loop, the trainer, the preflight, and the diagnostics
together. Deliberately a light leaf module — importing it must NOT trigger
``vdct_core``'s ``rmct_verl``/``slime_port`` bootstrap (its own imports are
stdlib-only; :func:`publish_pair_artifact` imports ``ctm.artifacts`` lazily),
so the dataset builder and the preflight can run in an environment where only
this experiment directory is synced. ``vdct_core`` re-exports the vocabulary
and :class:`VDCTConfig` for its callers.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass, fields
from pathlib import Path

REFERENCE_VARIANT = "reference"
TRAINING_VARIANT = "training"
DISTRIBUTION_KIND = "distribution"
ANSWER_KIND = "answer"

# One datapoint = every (variant, kind) cell, each a dataset row. The batch
# arithmetic everywhere (builder prints, preflight checks) derives from this.
ROWS_PER_DATAPOINT = 2 * 2  # (reference, training) x (distribution, answer)

# The native paired-prompt artifact: what the converters publish and the
# dataset builders consume.
PAIR_ARTIFACT_SCHEMA = "vdct.paired_prompts"
PAIR_ARTIFACT_SCHEMA_VERSION = 1
PAIR_ROW_REQUIRED = frozenset({"question_id", "unbiased_messages", "biased_messages", "biased_option", "option_labels"})


@dataclass
class VDCTConfig:
    """The ``vdct`` hydra block's math-relevant fields.

    The dataclass defaults are the single owner of the fallback values;
    :meth:`from_mapping` derives the config generically so a new knob is one
    field here plus its yaml entry.
    """

    lambda_log_score: float = 1.0
    consistency_weight: float = 1.0  # 0.0 = the proper-scoring-only ablation arm
    epsilon: float = 1e-3
    normalization: str = "per_item"  # per_item | pooled, slime_port semantics

    @classmethod
    def from_mapping(cls, mapping) -> VDCTConfig:
        """Build from a hydra block (or any mapping with ``.get``), coercing
        each value to its field default's type."""
        return cls(**{f.name: type(f.default)(mapping.get(f.name, f.default)) for f in fields(cls)})


def vdct_config_problems(config) -> list[str]:
    """Launch invariants the recipe depends on — one owner, three callers
    (``main_vdct`` at launch, ``RayVDCTTrainer.__init__``, and the
    ``resolve_config`` preflight). ``config`` is the full resolved trainer
    config (OmegaConf or an equivalent attribute/``.get`` structure)."""
    problems = []
    if config.trainer.get("use_v1", True):
        problems.append(
            "trainer.use_v1 must be False: the recipe overrides the V0 RayPPOTrainer "
            "(config/vdct_trainer.yaml pins it; the V1 TransferQueue port is separate work)"
        )
    if config.algorithm.get("use_kl_in_reward", False) or config.actor_rollout_ref.actor.use_kl_loss:
        problems.append(
            "VDCT owns the KL term; set algorithm.use_kl_in_reward=False and "
            "actor_rollout_ref.actor.use_kl_loss=False (vdct.kl_coef controls the VDCT KL)"
        )
    kl_source = config.vdct.get("kl_logprob_source", "old_log_probs")
    if kl_source not in ("old_log_probs", "rollout_log_probs"):
        problems.append(f"vdct.kl_logprob_source must be old_log_probs or rollout_log_probs, got {kl_source!r}")
    rows = int(config.data.train_batch_size)
    if rows % ROWS_PER_DATAPOINT != 0:
        problems.append(
            f"data.train_batch_size={rows} is not {ROWS_PER_DATAPOINT} x datapoints_per_step " "(rows, not datapoints)"
        )
    return problems


def read_jsonl_rows(path: str | Path, required: Iterable[str] = ()) -> list[dict]:
    """Read non-blank JSONL rows, requiring ``required`` keys on each.

    One shared implementation for the builders, the converters, and the
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


def load_module_from_path(name: str, path: str | Path):
    """Import a single module by file path.

    The converters use this for stdlib-only submodules of packages whose
    ``__init__`` pulls heavy dependencies (mcq-bias imports inspect_ai) and
    for unpackaged checkouts whose top-level names would collide (the AttCT
    monorepo's ``data``).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_pair_row(row: dict) -> dict:
    """Producer-side check of the native paired-prompt contract — the same
    key set the dataset builders require on read."""
    missing = PAIR_ROW_REQUIRED - row.keys()
    if missing:
        raise ValueError(f"paired-prompt row missing {sorted(missing)}")
    if row["biased_option"] not in list(row["option_labels"]):
        raise ValueError(f"biased_option {row['biased_option']!r} not in option_labels {row['option_labels']}")
    return row


def publish_pair_artifact(output: str | Path, pairs: list[dict], *, provenance: dict, force: bool) -> Path:
    """Publish a verified paired-prompt JSONL/manifest pair; the shared tail
    of every converter.

    Refuses to overwrite unless ``force``; with ``force`` the old pair is
    unlinked before the new write (not atomic — a failure mid-write leaves
    neither, which the immutable-artifact layer prefers over a half-replaced
    pair). Rows are validated against :data:`PAIR_ROW_REQUIRED` at write
    time, so a converter emitting a bad row fails here, not in the parquet
    builder.
    """
    from ctm.artifacts import artifact_manifest_path, write_verified_jsonl_artifact

    output = Path(output)
    manifest_path = artifact_manifest_path(output)
    if (output.exists() or manifest_path.exists()) and not force:
        raise SystemExit(f"{output} exists; pass --force to overwrite")
    if force:
        output.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
    write_verified_jsonl_artifact(
        output,
        pairs,
        artifact_schema=PAIR_ARTIFACT_SCHEMA,
        schema_version=PAIR_ARTIFACT_SCHEMA_VERSION,
        provenance=provenance,
        row_validator=validate_pair_row,
        nonempty=True,
    )
    print(f"wrote {len(pairs)} pairs -> {output}")
    print(f"  manifest -> {manifest_path}")
    return manifest_path


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
