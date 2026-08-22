#!/usr/bin/env python3
"""Convert a paired-prompt JSONL plus frozen anchors into the VDCT verl parquet.

Input 1 — the paired-prompt artifact, in either supported schema
(``--input-format``):

- ``native`` (default): native mcq-bias rows with at least ``question_id``,
  ``unbiased_messages``, ``biased_messages``, ``biased_option`` and
  ``option_labels`` (the parser contract);
- ``prompt_pairs``: the shared ``ctm.prompt_pairs`` schema produced by
  ``python -m ctm_data.adapters.mcq_bias.materialize --output-format
  prompt_pairs`` — e.g. the sycophancy_bct training-pairs artifact the
  irpan_2510_27062 reproduction consumes. Fields are mapped onto the native
  view (``source_id``→``question_id``, ``reference_messages``→
  ``unbiased_messages``, ``variant_messages``→``biased_messages``,
  ``metadata.valid_labels``→``option_labels``, etc.).

Input 2 — the frozen initial-anchor artifact (``--anchors``), one JSON object
per line: ``question_id``, ``option_labels``, ``q_ref_initial`` (the base
model's mean stated distribution on the REFERENCE prompt, precomputed once in
Phase 1 step 7). Coverage must be 100%; option orders must match.

Output: one parquet with FOUR rows per datapoint sharing a ``group_id`` —
``variant ∈ {reference, training}`` × ``kind ∈ {distribution, answer}``:

    (reference, distribution)  unbiased prompt + elicitation instruction; gradient
    (training,  distribution)  biased prompt + elicitation instruction; gradient
    (reference, answer)        unbiased prompt, unmodified; outcome samples only
    (training,  answer)        biased prompt, unmodified; outcome samples only

With ``--control`` every row uses ``unbiased_messages`` (repo convention:
reference prompt on both variants, everything else identical).

Batch-size relationship
-----------------------
verl repeats each dataset row ``actor_rollout_ref.rollout.n`` times, uniform
across rows, so with the plan's standard-GRPO shape:

    actor_rollout_ref.rollout.n = 8
    data.train_batch_size       = 4 * datapoints_per_step   (rows, not datapoints)

16 datapoints/step -> train_batch_size=64 -> 512 rollouts/step, of which the
256 distribution rollouts carry gradient; M = 8 answer samples per side falls
out of the uniform rollout.n.

Usage
-----
    uv run --no-sync python experiments/vdct_verl/scripts/make_vdct_dataset.py \
        --input pairs.jsonl --anchors anchors.jsonl \
        --output /workspace/vdct/data/vdct_rows.parquet
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd

_RECIPE_ROOT = Path(__file__).resolve().parents[1]
if str(_RECIPE_ROOT) not in sys.path:
    sys.path.insert(0, str(_RECIPE_ROOT))

from recipe.vdct.vdct_elicitation import elicitation_instruction

REQUIRED = {"question_id", "unbiased_messages", "biased_messages", "biased_option", "option_labels"}
PAIR_REQUIRED = {"source_id", "reference_messages", "variant_messages", "metadata"}
PAIR_METADATA_REQUIRED = {"biased_option", "valid_labels"}
ANCHOR_REQUIRED = {"question_id", "option_labels", "q_ref_initial"}
INPUT_FORMATS = ("native", "prompt_pairs")
REFERENCE_VARIANT = "reference"
TRAINING_VARIANT = "training"
DISTRIBUTION_KIND = "distribution"
ANSWER_KIND = "answer"


def _read_jsonl(path: Path, required: set[str]) -> list[dict]:
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


def _pair_row_to_native(row: dict, path: Path, line_no: int) -> dict:
    """Map one ``ctm.prompt_pairs`` row onto the native mcq-bias view."""
    metadata = row["metadata"]
    missing = PAIR_METADATA_REQUIRED - metadata.keys()
    if missing:
        raise ValueError(f"{path}:{line_no}: prompt_pairs metadata missing {sorted(missing)}")
    return {
        "question_id": str(row["source_id"]),
        "unbiased_messages": row["reference_messages"],
        "biased_messages": row["variant_messages"],
        "biased_option": metadata["biased_option"],
        "option_labels": metadata["valid_labels"],
        "ground_truth": metadata.get("correct_label", ""),
        "source_dataset": row.get("source", ""),
        "bias_type": metadata.get("bias_type", ""),
    }


def load_rows(path: Path, n_datapoints: int | None, input_format: str = "native") -> list[dict]:
    if input_format == "native":
        rows = _read_jsonl(path, REQUIRED)
    elif input_format == "prompt_pairs":
        rows = [
            _pair_row_to_native(row, path, line_no) for line_no, row in enumerate(_read_jsonl(path, PAIR_REQUIRED), 1)
        ]
    else:
        raise ValueError(f"unknown input format {input_format!r}; expected one of {INPUT_FORMATS}")
    if n_datapoints is not None:
        if len(rows) < n_datapoints:
            raise ValueError(f"need {n_datapoints} datapoints, {path} has {len(rows)}")
        rows = rows[:n_datapoints]
    return rows


def load_anchors(path: Path) -> dict[str, dict]:
    anchors: dict[str, dict] = {}
    for row in _read_jsonl(path, ANCHOR_REQUIRED):
        question_id = str(row["question_id"])
        if question_id in anchors:
            raise ValueError(f"{path}: duplicate anchor for question_id {question_id!r}")
        anchors[question_id] = row
    return anchors


def resolve_anchor(row: dict, anchors: dict[str, dict]) -> list[float]:
    """The frozen ``q_ref_initial`` for one datapoint, validated against it."""
    question_id = str(row["question_id"])
    anchor = anchors.get(question_id)
    if anchor is None:
        raise ValueError(f"no anchor for question_id {question_id!r}; anchor coverage must be 100%")
    if list(anchor["option_labels"]) != list(row["option_labels"]):
        raise ValueError(
            f"anchor option_labels mismatch for {question_id!r}: "
            f"{anchor['option_labels']} vs {row['option_labels']}"
        )
    q = [float(v) for v in anchor["q_ref_initial"]]
    if len(q) != len(row["option_labels"]):
        raise ValueError(f"anchor length mismatch for {question_id!r}")
    if any(v < 0.0 for v in q) or abs(sum(q) - 1.0) > 1e-6:
        raise ValueError(f"anchor for {question_id!r} is not a distribution: {q}")
    return q


def with_elicitation(messages: list[dict], option_labels: list[str]) -> list[dict]:
    """Append the elicitation instruction to the last user message."""
    messages = copy.deepcopy(messages)
    for message in reversed(messages):
        if message.get("role") == "user":
            message["content"] = str(message["content"]) + elicitation_instruction(option_labels)
            return messages
    raise ValueError("prompt has no user message to carry the elicitation instruction")


def build_records(rows: list[dict], anchors: dict[str, dict], data_source: str, control: bool) -> list[dict]:
    records: list[dict] = []
    for index, row in enumerate(rows):
        group_id = str(row["question_id"])
        option_labels = [str(label) for label in row["option_labels"]]
        q_ref_initial = resolve_anchor(row, anchors)
        for variant in (REFERENCE_VARIANT, TRAINING_VARIANT):
            if variant == REFERENCE_VARIANT or control:
                base_messages = row["unbiased_messages"]
            else:
                base_messages = row["biased_messages"]
            for kind in (DISTRIBUTION_KIND, ANSWER_KIND):
                if kind == DISTRIBUTION_KIND:
                    messages = with_elicitation(base_messages, option_labels)
                else:
                    messages = base_messages
                records.append(
                    {
                        "data_source": data_source,
                        "prompt": messages,
                        "agent_name": "vdct",
                        "group_id": group_id,
                        "variant": variant,
                        "kind": kind,
                        "biased_option": row["biased_option"],
                        "option_labels": option_labels,
                        "q_ref_initial": q_ref_initial,
                        "question_id": row["question_id"],
                        "ground_truth": row.get("ground_truth", ""),
                        "source_dataset": row.get("source_dataset", ""),
                        "bias_type": row.get("bias_type", ""),
                        "ability": "vdct",
                        "reward_model": {"style": "rule", "ground_truth": row.get("ground_truth", "")},
                        "extra_info": {"index": index, "split": "train", "variant": variant, "kind": kind},
                    }
                )
    return records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="paired-prompt JSONL")
    parser.add_argument(
        "--input-format",
        choices=INPUT_FORMATS,
        default="native",
        help="native mcq-bias rows or the shared ctm.prompt_pairs schema",
    )
    parser.add_argument("--anchors", required=True, type=Path, help="frozen q_ref_initial JSONL (Phase 1 step 7)")
    parser.add_argument("--output", required=True, type=Path, help="destination .parquet")
    parser.add_argument("--n-datapoints", type=int, default=None, help="take the first N rows (frozen selection)")
    parser.add_argument("--data-source", default="vdct", help="verl data_source tag")
    parser.add_argument("--control", action="store_true", help="all rows use the unbiased prompt (control arm)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing output")
    args = parser.parse_args(argv)

    if args.output.exists() and not args.force:
        raise SystemExit(f"{args.output} exists; pass --force to overwrite")

    rows = load_rows(args.input, args.n_datapoints, args.input_format)
    anchors = load_anchors(args.anchors)
    records = build_records(rows, anchors, args.data_source, args.control)
    assert len(records) == 4 * len(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(args.output, index=False)

    n_datapoints = len(rows)
    print(f"wrote {len(records)} rows ({n_datapoints} datapoints x 2 variants x 2 kinds) -> {args.output}")
    print("  data.train_batch_size = 4 * datapoints_per_step  (e.g. 64 for 16 datapoints/step)")
    print("  actor_rollout_ref.rollout.n = 8 (uniform; also fixes M = 8 answer samples per side)")
    print(f"  one epoch = {4 * n_datapoints} rows")


if __name__ == "__main__":
    main()
