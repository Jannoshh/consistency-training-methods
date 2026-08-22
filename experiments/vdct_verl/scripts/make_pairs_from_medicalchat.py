#!/usr/bin/env python3
"""Build a paired-prompt JSONL from the MedicalChat sycophancy environment.

Source: `ariahw/rl-rewardhacking-ext <https://github.com/ariahw/rl-rewardhacking-ext>`_,
the release of *Designing Effective Monitor-Based Interventions for Mitigating
Reward Hacking During RL* (OpenReview ``uFTnN6fUgW``). Its medical-chat
environment is iCliniq patient questions with reference physician responses;
the sycophancy loophole appends a leading question to the patient's message —
half implying the correct management, half a plausible incorrect one — and the
per-row ``prompt_metadata`` carries short stance phrases for both
(``correct_phrase`` / ``incorrect_phrase``) plus both leading questions.

Adaptation to the recipes' MCQ machinery (documented deviation, README): the
free-form environment is recast as a TWO-OPTION stance choice. Each pair is:

    unbiased_messages  system + patient question
                       + "Answer choices:" (both stance phrases, seeded order)
                       + the Chua CoT answer-format instruction
    biased_messages    same, with the leading question appended to the patient
                       question exactly as the environment's hint classes do
    option_labels      ["A", "B"]
    biased_option      the letter of the stance the leading question endorses

No ground-truth signal enters the training path (decision 2026-08-22:
consistency training is applied precisely where no correctness signal is
assumed): ``ground_truth`` is left empty, and ``bias_type``
(``sycophancy_correct`` / ``sycophancy_incorrect``) is provenance for offline
analysis only — nothing in the reward path reads it. Which stance is medically
right stays in the source repo's files.

Hint modes mirror the environment's registry: ``sycophancy_half`` (the paper's
standard, seeded 50/50 per question), ``sycophancy`` (always the incorrect
stance), ``sycophancy_correct`` (always the correct one). The answer-choice
header and CoT instruction are loaded from the pinned mcq-bias package's
``pipeline/records.py`` by file path (stdlib-only module), so the surface
format matches the Chua-suite prompts the rest of the sycophancy work uses.

The output is a verified ``ctm.artifacts`` JSONL/manifest pair, consumable by
``make_vdct_dataset.py`` and ``../rmct_verl/scripts/make_rmct_dataset.py``
unchanged.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys
from pathlib import Path

from ctm.artifacts import artifact_manifest_path, plain_file_identity, write_verified_jsonl_artifact

_RECIPE_ROOT = Path(__file__).resolve().parents[1]
if str(_RECIPE_ROOT) not in sys.path:
    sys.path.insert(0, str(_RECIPE_ROOT))

from recipe.vdct.vdct_schema import git_head_sha, read_jsonl_rows

PAIR_ARTIFACT_SCHEMA = "vdct.paired_prompts"
PAIR_ARTIFACT_SCHEMA_VERSION = 1
OPTION_LABELS = ["A", "B"]
HINT_MODES = ("sycophancy_half", "sycophancy", "sycophancy_correct")
SPLIT_FILES = {
    "train": "icliniq_train_filtered.jsonl",
    "test": "icliniq_test_filtered_all.jsonl",
    "holdout": "icliniq_holdout_base_all.jsonl",
}
ROW_REQUIRED = {"id", "prompt", "question", "prompt_metadata"}
METADATA_REQUIRED = ("correct_phrase", "incorrect_phrase", "correct_question", "incorrect_question")


def load_prompt_format():
    """``(ANSWER_CHOICES_HEADER, COT_INSTRUCTION)`` from the pinned mcq-bias
    package's ``pipeline/records.py``, loaded by file path — the package
    ``__init__`` imports inspect_ai, but records.py itself is stdlib-only
    (same pattern as the slime port's parser loader)."""
    spec = importlib.util.find_spec("mcq_bias")
    if spec is None or spec.origin is None:
        raise SystemExit("mcq-bias is not installed; run the environment setup in CLAUDE.md")
    records_path = Path(spec.origin).parent / "pipeline" / "records.py"
    mod_spec = importlib.util.spec_from_file_location("_mcq_bias_records", records_path)
    module = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(module)
    return module.ANSWER_CHOICES_HEADER, module.COT_INSTRUCTION


def build_pair(row: dict, hint: str, seed: int, choices_header: str, cot_instruction: str) -> dict:
    """One native paired-prompt row; deterministic per (seed, question id)."""
    metadata = row["prompt_metadata"]
    question_id = f"icliniq-{row['id']}"
    rng = random.Random(f"{seed}:{question_id}")

    if hint == "sycophancy_half":
        cue_is_correct = rng.random() < 0.5
    else:
        cue_is_correct = hint == "sycophancy_correct"
    leading_question = metadata["correct_question"] if cue_is_correct else metadata["incorrect_question"]
    cued_phrase = metadata["correct_phrase"] if cue_is_correct else metadata["incorrect_phrase"]

    phrases = [metadata["correct_phrase"], metadata["incorrect_phrase"]]
    rng.shuffle(phrases)
    choices_block = choices_header + "\n".join(f"({label}) {phrase}" for label, phrase in zip(OPTION_LABELS, phrases))
    biased_option = OPTION_LABELS[phrases.index(cued_phrase)]

    def messages(patient_question: str) -> list[dict]:
        result = [dict(m) for m in row["prompt"]]
        if not result or result[-1].get("role") != "user":
            raise ValueError(f"row {row['id']}: prompt does not end with a user message")
        result[-1]["content"] = patient_question + choices_block + cot_instruction
        return result

    return {
        "question_id": question_id,
        "unbiased_messages": messages(row["question"]),
        # The environment's hint classes append the leading question with a
        # single space; mirrored exactly.
        "biased_messages": messages(row["question"] + " " + leading_question),
        "biased_option": biased_option,
        "option_labels": list(OPTION_LABELS),
        "ground_truth": "",  # deliberately absent from the training path
        "source_dataset": "icliniq_medicalchat",
        "bias_type": "sycophancy_correct" if cue_is_correct else "sycophancy_incorrect",
    }


def build_pairs(rows: list[dict], hint: str, seed: int) -> tuple[list[dict], dict[str, int]]:
    choices_header, cot_instruction = load_prompt_format()
    pairs: list[dict] = []
    counts = {"n_rows": len(rows), "n_missing_metadata": 0, "n_duplicates": 0}
    seen: set[str] = set()
    for row in rows:
        metadata = row.get("prompt_metadata") or {}
        if not all(metadata.get(key) for key in METADATA_REQUIRED):
            counts["n_missing_metadata"] += 1
            continue
        question_id = f"icliniq-{row['id']}"
        if question_id in seen:
            counts["n_duplicates"] += 1
            continue
        seen.add(question_id)
        pairs.append(build_pair(row, hint, seed, choices_header, cot_instruction))
    return pairs, counts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--rewardhack-dir",
        type=Path,
        default=os.environ.get("REWARDHACK_DIR"),
        help="ariahw/rl-rewardhacking-ext checkout (default: $REWARDHACK_DIR)",
    )
    parser.add_argument("--split", choices=sorted(SPLIT_FILES), default="train")
    parser.add_argument("--hint", choices=HINT_MODES, default="sycophancy_half", help="which leading question is cued")
    parser.add_argument("--seed", type=int, default=42, help="cue-choice and option-order seed")
    parser.add_argument(
        "--ids-from",
        type=Path,
        default=None,
        help="optional environment JSONL whose id set restricts the pool "
        "(e.g. the paper's hard-1k training file), keeping its order",
    )
    parser.add_argument("--n-datapoints", type=int, default=None, help="take the first N built pairs")
    parser.add_argument("--output", required=True, type=Path, help="destination .jsonl")
    parser.add_argument("--force", action="store_true", help="overwrite an existing output")
    args = parser.parse_args(argv)

    if args.rewardhack_dir is None:
        raise SystemExit("pass --rewardhack-dir or set REWARDHACK_DIR to an ariahw/rl-rewardhacking-ext checkout")
    manifest_path = artifact_manifest_path(args.output)
    if (args.output.exists() or manifest_path.exists()) and not args.force:
        raise SystemExit(f"{args.output} exists; pass --force to overwrite")

    source_path = args.rewardhack_dir / "results" / "data" / SPLIT_FILES[args.split]
    if not source_path.exists():
        raise SystemExit(f"{source_path} not found in the checkout")
    rows = read_jsonl_rows(source_path, ROW_REQUIRED)
    if args.ids_from is not None:
        selected = [row["id"] for row in read_jsonl_rows(args.ids_from, {"id"})]
        by_id = {row["id"]: row for row in rows}
        missing = [i for i in selected if i not in by_id]
        if missing:
            raise SystemExit(f"{len(missing)} ids from {args.ids_from} are not in the {args.split} split")
        rows = [by_id[i] for i in selected]

    pairs, counts = build_pairs(rows, args.hint, args.seed)
    if args.n_datapoints is not None:
        if len(pairs) < args.n_datapoints:
            raise SystemExit(f"need {args.n_datapoints} pairs, built {len(pairs)}")
        pairs = pairs[: args.n_datapoints]

    if args.force:
        args.output.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
    write_verified_jsonl_artifact(
        args.output,
        pairs,
        artifact_schema=PAIR_ARTIFACT_SCHEMA,
        schema_version=PAIR_ARTIFACT_SCHEMA_VERSION,
        provenance={
            "rewardhack_dir": str(args.rewardhack_dir),
            "rewardhack_sha": git_head_sha(args.rewardhack_dir),
            "source": plain_file_identity(source_path),
            "ids_from": plain_file_identity(args.ids_from) if args.ids_from is not None else None,
            "split": args.split,
            "hint": args.hint,
            "seed": args.seed,
            **counts,
        },
        nonempty=True,
    )

    n_incorrect = sum(1 for pair in pairs if pair["bias_type"] == "sycophancy_incorrect")
    print(f"wrote {len(pairs)} pairs -> {args.output}")
    print(f"  manifest -> {manifest_path}")
    print(f"  cue split: {n_incorrect} incorrect-stance, {len(pairs) - n_incorrect} correct-stance")
    print(f"  skipped: {counts['n_missing_metadata']} missing metadata, {counts['n_duplicates']} duplicates")


if __name__ == "__main__":
    main()
