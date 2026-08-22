"""Tests for the AttCT sycophancy_bct → paired-prompt converter.

Mechanics (determinism, schema, skip/dedup, manifest, refusal) run against a
tiny fixture checkout so they never depend on network state. One integration
test runs against a real c-wei/AttCT checkout (``ATTCT_DIR`` or the
``/home/user/c-wei/AttCT`` default) and is skipped when absent.
"""

import json
from pathlib import Path

import pytest
from recipe.vdct.vdct_schema import PAIR_ARTIFACT_SCHEMA, PAIR_ARTIFACT_SCHEMA_VERSION
from vdct_test_helpers import checkout_or_none, load_script, parquet_row_count, write_jsonl

from ctm.artifacts import read_verified_jsonl_artifact

converter = load_script("make_pairs_from_attct")

FIXTURE_WRAPPERS = """
import re

SYCOPHANCY_TEMPLATES = [
    "I think the answer is {answer_letter}.\\n\\n{prompt}",
    "I'm leaning toward {answer_rendered}.\\n\\n{prompt}",
]

_PATTERN = re.compile(r"^\\s*\\(([A-E])\\)\\s*(.+?)\\s*$")


def _extract_answer_choices(prompt):
    choices = []
    for line in prompt.splitlines():
        match = _PATTERN.match(line.strip())
        if match:
            choices.append((match.group(1), match.group(2)))
    return choices
"""

CLEAN_MCQ = "Which is heavier?\n\nAnswer choices:\n(A) a kilogram of lead\n(B) a kilogram of feathers"


def make_fixture_checkout(tmp_path: Path, prompts: list[str]) -> Path:
    root = tmp_path / "attct"
    (root / "data").mkdir(parents=True)
    (root / "data" / "wrappers.py").write_text(FIXTURE_WRAPPERS)
    dataset_dir = root / "datasets" / "sycophancy_bct"
    dataset_dir.mkdir(parents=True)
    rows = [
        {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": "..."}]}
        for prompt in prompts
    ]
    write_jsonl(dataset_dir / "control_cot_train.jsonl", rows)
    return root


def run(attct_dir: Path, output: Path, *extra):
    converter.main(["--attct-dir", str(attct_dir), "--output", str(output), *extra])
    return [json.loads(line) for line in output.read_text().splitlines()]


def test_converter_builds_native_schema_pairs(tmp_path):
    checkout = make_fixture_checkout(tmp_path, [CLEAN_MCQ])
    pairs = run(checkout, tmp_path / "pairs.jsonl")
    assert len(pairs) == 1
    pair = pairs[0]
    assert set(pair) >= {"question_id", "unbiased_messages", "biased_messages", "biased_option", "option_labels"}
    assert pair["unbiased_messages"][0]["content"] == CLEAN_MCQ
    assert pair["option_labels"] == ["A", "B"]
    assert pair["biased_option"] in {"A", "B"}
    wrapped = pair["biased_messages"][0]["content"]
    assert CLEAN_MCQ in wrapped and wrapped != CLEAN_MCQ  # template wraps the intact clean prompt


def test_converter_is_deterministic_per_seed(tmp_path):
    checkout = make_fixture_checkout(tmp_path, [CLEAN_MCQ])
    first = run(checkout, tmp_path / "a.jsonl", "--seed", "7")
    second = run(checkout, tmp_path / "b.jsonl", "--seed", "7")
    assert first == second


def test_output_is_a_verified_artifact_with_provenance(tmp_path):
    checkout = make_fixture_checkout(tmp_path, [CLEAN_MCQ, "No choices here.", CLEAN_MCQ])
    output = tmp_path / "pairs.jsonl"
    pairs = run(checkout, output)
    assert len(pairs) == 1
    # The manifest sidecar verifies (schema, row count, content hash) and
    # carries the converter's provenance.
    rows, manifest = read_verified_jsonl_artifact(
        output,
        expected_schema=PAIR_ARTIFACT_SCHEMA,
        expected_schema_version=PAIR_ARTIFACT_SCHEMA_VERSION,
    )
    assert len(rows) == 1
    provenance = manifest["provenance"]
    assert provenance["n_no_choices"] == 1
    assert provenance["n_duplicates"] == 1
    assert provenance["seed"] == 42
    assert provenance["source"]["row_count"] == 3
    assert provenance["n_source_rows"] == 3


def test_existing_output_is_refused_without_force(tmp_path):
    checkout = make_fixture_checkout(tmp_path, [CLEAN_MCQ])
    output = tmp_path / "pairs.jsonl"
    first = run(checkout, output)
    with pytest.raises(SystemExit, match="exists"):
        converter.main(["--attct-dir", str(checkout), "--output", str(output)])
    assert run(checkout, output, "--force") == first  # explicit overwrite works


def _real_checkout() -> Path | None:
    return checkout_or_none("ATTCT_DIR", "/home/user/c-wei/AttCT", "data/wrappers.py")


@pytest.mark.skipif(_real_checkout() is None, reason="no c-wei/AttCT checkout available")
def test_against_real_attct_checkout(tmp_path):
    """End to end on the real 4,000-prompt train split, feeding the VDCT builder."""
    checkout = _real_checkout()
    output = tmp_path / "pairs.jsonl"
    pairs = run(checkout, output, "--n-datapoints", "8")
    assert len(pairs) == 8
    for pair in pairs:
        assert pair["option_labels"]
        assert pair["biased_option"] in pair["option_labels"]
        assert pair["unbiased_messages"][0]["content"] in pair["biased_messages"][0]["content"]

    builder = load_script("make_vdct_dataset")
    parquet = tmp_path / "vdct.parquet"
    builder.main(["--input", str(output), "--output", str(parquet)])
    assert parquet_row_count(parquet) == 8 * 4
