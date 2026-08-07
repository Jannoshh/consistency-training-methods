#!/usr/bin/env python3
"""Convert an RMCT paired-prompt JSONL into the verl parquet the recipe expects.

Input: the frozen paired-prompt artifact, one JSON object per line with at
least ``question_id``, ``unbiased_messages``, ``biased_messages`` and
``biased_option`` — e.g.
``experiments/rmct_slime_qwen/data/run_9b/wrong-argument-pairs-64.jsonl``.

Output: one parquet with TWO rows per datapoint, sharing a ``group_id``:

    variant="reference" -> prompt = unbiased_messages   (supplies p_ref)
    variant="training"  -> prompt = biased_messages     (supplies p_hat, carries the gradient)

With ``--control`` the training row also uses ``unbiased_messages``, matching
``slime_port.rmct_rollout``'s control condition.

Columns
-------
prompt          list[{role, content}]  — RLHFDataset builds ``raw_prompt`` from it
                (``data.prompt_key=prompt``, ``data.return_raw_chat=True``)
agent_name      "rmct"                 — routes the row to the RMCT agent loop
group_id        str                    — ties the two variants of one datapoint
variant         "reference" | "training"
biased_option   str                    — the trait classifier's target
question_id, ground_truth, source_dataset, bias_type — provenance passthrough
data_source, reward_model, extra_info  — verl's standard columns; the RMCT
                agent loop returns reward_score=0.0 so no reward manager runs.

Batch-size relationship
-----------------------
verl repeats each DATASET row ``actor_rollout_ref.rollout.n`` times, so with the
run_9b numbers (n_ref_rollouts = n_train_rollouts = 128):

    actor_rollout_ref.rollout.n = 128
    data.train_batch_size       = 2 * datapoints_per_step        (rows, not datapoints)

For the run_9b batch of 4 datapoints per step that is
``train_batch_size=8`` and 8*128 = 1024 sampled rollouts per step, of which the
512 training-variant rollouts carry gradient. ``rollout.n`` is necessarily the
SAME for both variants, so this port cannot express n_ref != n_train — asserted
below against the config values you pass in.

Usage
-----
    uv run --no-sync python experiments/rmct_verl/scripts/make_rmct_dataset.py \
        --input experiments/rmct_slime_qwen/data/run_9b/wrong-argument-pairs-64.jsonl \
        --output /workspace/rmct/data/run_9b/rmct_pairs.parquet \
        --n-datapoints 64
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REQUIRED = {"question_id", "unbiased_messages", "biased_messages", "biased_option"}
REFERENCE_VARIANT = "reference"
TRAINING_VARIANT = "training"


def load_rows(path: Path, n_datapoints: int | None) -> list[dict]:
    rows: list[dict] = []
    with path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = REQUIRED - row.keys()
            if missing:
                raise ValueError(f"{path}:{line_no}: row missing {sorted(missing)}")
            rows.append(row)
    if n_datapoints is not None:
        if len(rows) < n_datapoints:
            raise ValueError(f"need {n_datapoints} datapoints, {path} has {len(rows)}")
        rows = rows[:n_datapoints]
    return rows


def build_records(rows: list[dict], data_source: str, control: bool) -> list[dict]:
    records: list[dict] = []
    for index, row in enumerate(rows):
        group_id = str(row["question_id"])
        for variant in (REFERENCE_VARIANT, TRAINING_VARIANT):
            if variant == REFERENCE_VARIANT or control:
                messages = row["unbiased_messages"]
            else:
                messages = row["biased_messages"]
            records.append(
                {
                    "data_source": data_source,
                    "prompt": messages,
                    "agent_name": "rmct",
                    "group_id": group_id,
                    "variant": variant,
                    "biased_option": row["biased_option"],
                    "question_id": row["question_id"],
                    "ground_truth": row.get("ground_truth", ""),
                    "source_dataset": row.get("source_dataset", ""),
                    "bias_type": row.get("bias_type", ""),
                    "ability": "rmct",
                    "reward_model": {"style": "rule", "ground_truth": row.get("ground_truth", "")},
                    "extra_info": {"index": index, "split": "train", "variant": variant},
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="paired-prompt JSONL")
    parser.add_argument("--output", required=True, type=Path, help="destination .parquet")
    parser.add_argument("--n-datapoints", type=int, default=None, help="take the first N rows (frozen selection)")
    parser.add_argument("--data-source", default="rmct", help="verl data_source tag")
    parser.add_argument("--control", action="store_true", help="training rows use the unbiased prompt too")
    parser.add_argument(
        "--n-ref-rollouts", type=int, default=None, help="cross-check against --n-train-rollouts and rollout.n"
    )
    parser.add_argument("--n-train-rollouts", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="overwrite an existing output")
    args = parser.parse_args()

    if (
        args.n_ref_rollouts is not None
        and args.n_train_rollouts is not None
        and args.n_ref_rollouts != args.n_train_rollouts
    ):
        raise SystemExit(
            f"verl samples rollout.n completions per row, so n_ref_rollouts ({args.n_ref_rollouts}) must "
            f"equal n_train_rollouts ({args.n_train_rollouts}) on this stack."
        )

    if args.output.exists() and not args.force:
        raise SystemExit(f"{args.output} exists; pass --force to overwrite")

    rows = load_rows(args.input, args.n_datapoints)
    records = build_records(rows, args.data_source, args.control)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(args.output, index=False)

    n_datapoints = len(rows)
    print(f"wrote {len(records)} rows ({n_datapoints} datapoints x 2 variants) -> {args.output}")
    print("  data.train_batch_size = 2 * datapoints_per_step  (e.g. 8 for 4 datapoints/step)")
    print("  actor_rollout_ref.rollout.n = n_ref_rollouts = n_train_rollouts")
    print(f"  one epoch = {2 * n_datapoints} rows")


if __name__ == "__main__":
    main()
