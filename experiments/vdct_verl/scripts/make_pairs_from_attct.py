#!/usr/bin/env python3
"""Build a paired-prompt JSONL from the AttCT repo's sycophancy_bct assets.

The `c-wei/AttCT <https://github.com/c-wei/AttCT>`_ checkout (the Transformer
Stack paper's codebase) is the source of truth for the non-RL side: its
``datasets/sycophancy_bct/control_{style}_{split}.jsonl`` files carry the
clean 4,000-prompt training pool (1,000 eval), and its
``data/wrappers.py::SYCOPHANCY_TEMPLATES`` are the paper's 12 sycophancy
cues, applied at batch time in that repo. This converter freezes the same
construction into the native paired-prompt schema the RL recipes consume
(``make_vdct_dataset.py`` / ``make_rmct_dataset.py``):

    question_id        sha1 of the clean user prompt
    unbiased_messages  [{role: user, content: <clean prompt>}]
    biased_messages    [{role: user, content: <template-wrapped prompt>}]
    biased_option      the option letter the cue endorses
    option_labels      letters extracted by AttCT's own answer-choice parser

Reuse, never reimplement: the template strings and the answer-choice
extraction are imported from the checkout by file path
(``--attct-dir`` / ``ATTCT_DIR``; ``data/wrappers.py`` is stdlib-only).
What differs from the AttCT loader — deliberately, and only because a frozen
RL artifact needs determinism where their batch-time wrapping is random:
template and cued-option choice use an RNG seeded per (seed, question_id),
mirroring ``_fill_template_placeholders``'s rendering exactly (prefix +
prompt + suffix; ``{answer_rendered}`` = "(<letter>) <text>").

Rows whose prompt has no extractable answer choices are skipped, matching the
AttCT loader's behavior. A ``<output>.manifest.json`` records the checkout
SHA, source file hash, seed, and counts for provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import subprocess
from pathlib import Path

STYLES = ("cot", "non_cot")
SPLITS = ("train", "eval")


def load_wrappers_module(attct_dir: Path):
    """Load ``data/wrappers.py`` from the checkout by file path (the repo is
    an unpackaged monorepo whose top-level ``data`` name would collide)."""
    wrappers_path = attct_dir / "data" / "wrappers.py"
    if not wrappers_path.exists():
        raise SystemExit(f"{wrappers_path} not found; pass --attct-dir or set ATTCT_DIR to a c-wei/AttCT checkout")
    spec = importlib.util.spec_from_file_location("_attct_wrappers", wrappers_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_clean_prompts(path: Path) -> list[str]:
    """The user-turn contents of a sycophancy_bct control JSONL."""
    prompts: list[str] = []
    with path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{path}:{line_no}: row has no messages")
            user_contents = [m["content"] for m in messages if m.get("role") == "user"]
            if len(user_contents) != 1:
                raise ValueError(f"{path}:{line_no}: expected exactly one user message")
            prompts.append(str(user_contents[0]))
    return prompts


def wrap_prompt(wrappers, clean: str, question_id: str, seed: int) -> tuple[str, str, list[str]] | None:
    """Deterministically apply one AttCT sycophancy template to a clean prompt.

    Returns ``(wrapped, biased_option, option_labels)``, or None when the
    prompt has no extractable answer choices (skipped, as in the AttCT
    loader). Rendering mirrors ``_fill_template_placeholders``.
    """
    choices = wrappers._extract_answer_choices(clean)
    if not choices:
        return None
    rng = random.Random(f"{seed}:{question_id}")
    template = rng.choice(wrappers.SYCOPHANCY_TEMPLATES)
    answer_letter, answer_text = rng.choice(choices)
    filled = template.format(
        prompt="{prompt}",
        answer_letter=answer_letter,
        answer_text=answer_text,
        answer_rendered=f"({answer_letter}) {answer_text}",
    )
    prefix, suffix = filled.split("{prompt}")
    wrapped = prefix + clean + suffix
    return wrapped, answer_letter, [letter for letter, _ in choices]


def build_pairs(wrappers, prompts: list[str], seed: int, source_tag: str) -> tuple[list[dict], dict[str, int]]:
    pairs: list[dict] = []
    counts = {"n_prompts": len(prompts), "n_no_choices": 0, "n_duplicates": 0}
    seen: set[str] = set()
    for clean in prompts:
        question_id = hashlib.sha1(clean.encode("utf-8")).hexdigest()
        if question_id in seen:
            counts["n_duplicates"] += 1
            continue
        seen.add(question_id)
        wrapped_result = wrap_prompt(wrappers, clean, question_id, seed)
        if wrapped_result is None:
            counts["n_no_choices"] += 1
            continue
        wrapped, biased_option, option_labels = wrapped_result
        pairs.append(
            {
                "question_id": question_id,
                "unbiased_messages": [{"role": "user", "content": clean}],
                "biased_messages": [{"role": "user", "content": wrapped}],
                "biased_option": biased_option,
                "option_labels": option_labels,
                "ground_truth": "",
                "source_dataset": source_tag,
                "bias_type": "sycophancy",
            }
        )
    return pairs, counts


def checkout_sha(attct_dir: Path) -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "-C", str(attct_dir), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            or None
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--attct-dir",
        type=Path,
        default=os.environ.get("ATTCT_DIR"),
        help="c-wei/AttCT checkout (default: $ATTCT_DIR)",
    )
    parser.add_argument("--style", choices=STYLES, default="cot", help="prompt style (VDCT is always-CoT)")
    parser.add_argument("--split", choices=SPLITS, default="train", help="held-out split to convert")
    parser.add_argument("--seed", type=int, default=42, help="template/cued-option selection seed")
    parser.add_argument("--n-datapoints", type=int, default=None, help="take the first N built pairs")
    parser.add_argument("--output", required=True, type=Path, help="destination .jsonl")
    parser.add_argument("--force", action="store_true", help="overwrite an existing output")
    args = parser.parse_args(argv)

    if args.attct_dir is None:
        raise SystemExit("pass --attct-dir or set ATTCT_DIR to a c-wei/AttCT checkout")
    if args.output.exists() and not args.force:
        raise SystemExit(f"{args.output} exists; pass --force to overwrite")

    wrappers = load_wrappers_module(args.attct_dir)
    control_path = args.attct_dir / "datasets" / "sycophancy_bct" / f"control_{args.style}_{args.split}.jsonl"
    if not control_path.exists():
        raise SystemExit(f"{control_path} not found in the checkout")

    prompts = read_clean_prompts(control_path)
    source_tag = f"attct_sycophancy_bct_{args.style}_{args.split}"
    pairs, counts = build_pairs(wrappers, prompts, args.seed, source_tag)
    if args.n_datapoints is not None:
        if len(pairs) < args.n_datapoints:
            raise SystemExit(f"need {args.n_datapoints} pairs, built {len(pairs)}")
        pairs = pairs[: args.n_datapoints]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")

    manifest = {
        "attct_dir": str(args.attct_dir),
        "attct_sha": checkout_sha(args.attct_dir),
        "source_file": str(control_path),
        "source_sha256": hashlib.sha256(control_path.read_bytes()).hexdigest(),
        "style": args.style,
        "split": args.split,
        "seed": args.seed,
        "n_pairs": len(pairs),
        **counts,
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"wrote {len(pairs)} pairs -> {args.output}")
    print(f"  manifest -> {manifest_path}")
    print(f"  skipped: {counts['n_no_choices']} without answer choices, {counts['n_duplicates']} duplicates")


if __name__ == "__main__":
    main()
