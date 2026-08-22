"""Tests for the AttCT sycophancy_bct → paired-prompt converter.

Mechanics (determinism, schema, skip/dedup, manifest, refusal) run against a
tiny fixture checkout so they never depend on network state. One integration
test runs against a real c-wei/AttCT checkout (``ATTCT_DIR`` or the
``/home/user/c-wei/AttCT`` default) and is skipped when absent.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

_VDCT_VERL = Path(__file__).resolve().parents[1]
if str(_VDCT_VERL) not in sys.path:
    sys.path.insert(0, str(_VDCT_VERL))

_SPEC = importlib.util.spec_from_file_location(
    "make_pairs_from_attct", _VDCT_VERL / "scripts" / "make_pairs_from_attct.py"
)
converter = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(converter)

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
    (dataset_dir / "control_cot_train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
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
    third = run(checkout, tmp_path / "c.jsonl", "--seed", "8")
    assert first == second
    assert first != third or first[0]["biased_messages"] == third[0]["biased_messages"]


def test_rows_without_choices_are_skipped_and_duplicates_dropped(tmp_path):
    checkout = make_fixture_checkout(tmp_path, [CLEAN_MCQ, "No choices here.", CLEAN_MCQ])
    output = tmp_path / "pairs.jsonl"
    pairs = run(checkout, output)
    assert len(pairs) == 1
    manifest = json.loads((tmp_path / "pairs.jsonl.manifest.json").read_text())
    assert manifest["n_no_choices"] == 1
    assert manifest["n_duplicates"] == 1
    assert manifest["n_pairs"] == 1
    assert manifest["seed"] == 42


def test_existing_output_is_refused_without_force(tmp_path):
    checkout = make_fixture_checkout(tmp_path, [CLEAN_MCQ])
    output = tmp_path / "pairs.jsonl"
    run(checkout, output)
    with pytest.raises(SystemExit, match="exists"):
        converter.main(["--attct-dir", str(checkout), "--output", str(output)])


def _real_checkout() -> Path | None:
    candidate = os.environ.get("ATTCT_DIR", "/home/user/c-wei/AttCT")
    path = Path(candidate)
    return path if (path / "data" / "wrappers.py").exists() else None


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

    builder_spec = importlib.util.spec_from_file_location(
        "make_vdct_dataset", _VDCT_VERL / "scripts" / "make_vdct_dataset.py"
    )
    builder = importlib.util.module_from_spec(builder_spec)
    builder_spec.loader.exec_module(builder)
    parquet = tmp_path / "vdct.parquet"
    builder.main(["--input", str(output), "--output", str(parquet)])
    import pandas as pd

    assert len(pd.read_parquet(parquet)) == 32
