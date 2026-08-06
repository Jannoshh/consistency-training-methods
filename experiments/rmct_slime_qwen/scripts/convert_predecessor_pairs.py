"""Convert predecessor-repo RMCT training rows to the slime-port schema.

The paper RMCT runs trained on ``dataset_dumps/test/distractor_argument_g4/``
rows from the predecessor repo (old cot-transparency schema). This converter
reshapes them — deterministically, no LLM calls, gemma-generated arguments
byte-identical — into the fields ``slime_port.rmct_rollout`` requires, and
replicates the paper loader's datapoint selection (``train_rl.py::
load_datapoints``: the first ``per_combo`` rows of EACH source file, in file
order).

Field mapping (old → new):
    unbiased_question      → unbiased_messages   (verbatim message list)
    biased_question        → biased_messages     (verbatim message list)
    original_question_hash → question_id
    biased_option          → biased_option       (verbatim)
    ground_truth           → ground_truth        (verbatim)
    original_dataset       → source_dataset
    bias_name              → bias_type           (recorded, not consumed)

Frozen-artifact rules: refuses to overwrite the output pair; the manifest
records source paths, SHA-256, row counts, and the mapping above.

Usage:
    python convert_predecessor_pairs.py --per-file 32 \
        --out data/run_9b/wrong-argument-pairs-64.jsonl \
        SRC1.jsonl SRC2.jsonl
"""

import argparse
import hashlib
import json
from pathlib import Path

REQUIRED_SOURCE_FIELDS = {
    "unbiased_question",
    "biased_question",
    "original_question_hash",
    "biased_option",
    "ground_truth",
    "original_dataset",
    "bias_name",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def convert_row(row: dict) -> dict:
    missing = REQUIRED_SOURCE_FIELDS - row.keys()
    if missing:
        raise ValueError(f"source row missing {sorted(missing)}")
    return {
        "question_id": row["original_question_hash"],
        "unbiased_messages": row["unbiased_question"],
        "biased_messages": row["biased_question"],
        "biased_option": row["biased_option"],
        "ground_truth": row["ground_truth"],
        "source_dataset": row["original_dataset"],
        "bias_type": row["bias_name"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--per-file", type=int, required=True, help="rows taken from the head of each source file")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.out.with_suffix(args.out.suffix + ".manifest.json")
    for path in (args.out, manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    converted: list[dict] = []
    source_records = []
    for source in args.sources:
        taken = skipped_empty = 0
        with open(source) as f:
            for line in f:
                if taken >= args.per_file:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                # Source files contain rows whose argument generation failed
                # (empty biased_question — 324/2400 in logiqa). The paper
                # loader fed them anyway (dead items: no biased measurements);
                # they are not executable here, so take the first per_file
                # rows with BOTH prompts non-empty (deviation D12).
                if not row["biased_question"] or not row["unbiased_question"]:
                    skipped_empty += 1
                    continue
                converted.append(convert_row(row))
                taken += 1
        if taken < args.per_file:
            raise ValueError(f"{source}: only {taken} rows, need {args.per_file}")
        source_records.append(
            {"path": str(source), "sha256": sha256(source), "rows_taken": taken, "empty_rows_skipped": skipped_empty}
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for row in converted:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "schema": "rmct_slime_wrong_argument_pairs.v1",
        "selection": f"first {args.per_file} rows of each source, in file order (train_rl.py load_datapoints semantics)",
        "sources": source_records,
        "n_rows": len(converted),
        "output_sha256": sha256(args.out),
        "field_mapping": {
            "unbiased_question": "unbiased_messages",
            "biased_question": "biased_messages",
            "original_question_hash": "question_id",
            "biased_option": "biased_option",
            "ground_truth": "ground_truth",
            "original_dataset": "source_dataset",
            "bias_name": "bias_type",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(converted)} rows to {args.out}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
