#!/usr/bin/env python3
"""Convert a paired-prompt JSONL into the VDCT verl parquet.

Input 1 — the paired-prompt artifact, in either supported schema
(``--input-format``):

- ``native`` (default): native mcq-bias rows with at least ``question_id``,
  ``unbiased_messages``, ``biased_messages``, ``biased_option`` and
  ``option_labels`` (the parser contract);
- ``prompt_pairs``: a verified ``ctm.prompt_pairs`` artifact (JSONL +
  manifest sidecar) produced by ``python -m
  ctm_data.adapters.mcq_bias.materialize --output-format prompt_pairs`` —
  e.g. the sycophancy_bct training-pairs artifact the irpan_2510_27062
  reproduction consumes. Loaded through
  ``ctm.settings.pairs.load_pair_artifact``, so the manifest's schema
  version, row count, and content hash are enforced (the repo's
  frozen-artifact rule); fields are then mapped onto the native view
  (``source_id``→``question_id``, ``reference_messages``→
  ``unbiased_messages``, ``variant_messages``→``biased_messages``,
  ``metadata.valid_labels``→``option_labels``, etc.).

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
        --input pairs.jsonl --output /workspace/vdct/data/vdct_rows.parquet
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import pandas as pd

_RECIPE_ROOT = Path(__file__).resolve().parents[1]
if str(_RECIPE_ROOT) not in sys.path:
    sys.path.insert(0, str(_RECIPE_ROOT))

from recipe.vdct.vdct_elicitation import elicitation_instruction
from recipe.vdct.vdct_schema import (
    ANSWER_KIND,
    DISTRIBUTION_KIND,
    PAIR_ROW_REQUIRED,
    REFERENCE_VARIANT,
    ROWS_PER_DATAPOINT,
    TRAINING_VARIANT,
    read_jsonl_rows,
)

PAIR_METADATA_REQUIRED = {"biased_option", "valid_labels"}
INPUT_FORMATS = ("native", "prompt_pairs")


def _pair_row_to_native(row: dict, path: Path, line_no: int) -> dict:
    """Map one ``ctm.prompt_pairs`` row onto the native mcq-bias view."""
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        raise TypeError(f"{path}:{line_no}: prompt_pairs row has no metadata mapping")
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
        rows = read_jsonl_rows(path, PAIR_ROW_REQUIRED)
        if n_datapoints is not None:
            if len(rows) < n_datapoints:
                raise ValueError(f"need {n_datapoints} datapoints, {path} has {len(rows)}")
            rows = rows[:n_datapoints]
        return rows
    if input_format == "prompt_pairs":
        # Manifest-verified load (schema version, row count, content hash,
        # canonical pair shape, unique pair_id) — the repo's frozen-artifact
        # rule. Prefix selection happens inside the loader.
        from ctm.settings.pairs import load_pair_artifact

        pair_rows, _manifest = load_pair_artifact(path, n_datapoints=n_datapoints)
        return [_pair_row_to_native(row, path, line_no) for line_no, row in enumerate(pair_rows, 1)]
    raise ValueError(f"unknown input format {input_format!r}; expected one of {INPUT_FORMATS}")


def with_elicitation(messages: list[dict], option_labels: list[str]) -> list[dict]:
    """Append the elicitation instruction to the last user message."""
    messages = copy.deepcopy(messages)
    for message in reversed(messages):
        if message.get("role") == "user":
            message["content"] = str(message["content"]) + elicitation_instruction(option_labels)
            return messages
    raise ValueError("prompt has no user message to carry the elicitation instruction")


def build_records(rows: list[dict], data_source: str, control: bool) -> list[dict]:
    records: list[dict] = []
    for index, row in enumerate(rows):
        group_id = str(row["question_id"])
        option_labels = [str(label) for label in row["option_labels"]]
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
    parser.add_argument("--output", required=True, type=Path, help="destination .parquet")
    parser.add_argument("--n-datapoints", type=int, default=None, help="take the first N rows (frozen selection)")
    parser.add_argument("--data-source", default="vdct", help="verl data_source tag")
    parser.add_argument("--control", action="store_true", help="all rows use the unbiased prompt (control arm)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing output")
    args = parser.parse_args(argv)

    if args.output.exists() and not args.force:
        raise SystemExit(f"{args.output} exists; pass --force to overwrite")

    rows = load_rows(args.input, args.n_datapoints, args.input_format)
    records = build_records(rows, args.data_source, args.control)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(args.output, index=False)

    print(f"wrote {len(records)} rows ({len(rows)} datapoints x 2 variants x 2 kinds) -> {args.output}")
    print(f"  data.train_batch_size = {ROWS_PER_DATAPOINT} * datapoints_per_step  (rows, not datapoints)")
    print("  actor_rollout_ref.rollout.n = 8 (uniform; also fixes M = 8 answer samples per side)")
    print(f"  one epoch = {ROWS_PER_DATAPOINT * len(rows)} rows")


if __name__ == "__main__":
    main()
