"""Tests for the VDCT parquet builder: schema, row structure, control mode,
frozen-artifact refusals, and both input formats."""

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

_VDCT_VERL = Path(__file__).resolve().parents[1]
if str(_VDCT_VERL) not in sys.path:
    sys.path.insert(0, str(_VDCT_VERL))

from recipe.vdct.vdct_elicitation import DISTRIBUTION_OPEN

_SPEC = importlib.util.spec_from_file_location("make_vdct_dataset", _VDCT_VERL / "scripts" / "make_vdct_dataset.py")
builder = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(builder)

LABELS = ["A", "B", "C"]


def native_row(idx: int) -> dict:
    return {
        "question_id": f"q{idx}",
        "unbiased_messages": [{"role": "user", "content": f"Question {idx}?"}],
        "biased_messages": [{"role": "user", "content": f"I think it's B. Question {idx}?"}],
        "biased_option": "B",
        "option_labels": LABELS,
        "ground_truth": "A",
        "source_dataset": "test",
        "bias_type": "suggested_answer",
    }


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


@pytest.fixture
def inputs(tmp_path):
    pairs = write_jsonl(tmp_path / "pairs.jsonl", [native_row(0), native_row(1)])
    return pairs, tmp_path / "out.parquet"


def run_builder(pairs, output, *extra):
    builder.main(["--input", str(pairs), "--output", str(output), *extra])
    return pd.read_parquet(output)


def test_four_rows_per_datapoint_with_expected_columns(inputs):
    frame = run_builder(*inputs)
    assert len(frame) == 8
    assert set(frame["agent_name"]) == {"vdct"}
    for group_id, group in frame.groupby("group_id"):
        cells = {(row["variant"], row["kind"]) for _, row in group.iterrows()}
        assert cells == {
            ("reference", "distribution"),
            ("reference", "answer"),
            ("training", "distribution"),
            ("training", "answer"),
        }
        for _, row in group.iterrows():
            assert list(row["option_labels"]) == LABELS


def test_distribution_rows_carry_the_elicitation_instruction(inputs):
    frame = run_builder(*inputs)
    for _, row in frame.iterrows():
        content = row["prompt"][-1]["content"]
        if row["kind"] == "distribution":
            assert DISTRIBUTION_OPEN in content
        else:
            assert DISTRIBUTION_OPEN not in content
    # Reference rows use the unbiased prompt, training rows the biased one.
    training_answer = frame[(frame["variant"] == "training") & (frame["kind"] == "answer")].iloc[0]
    assert "I think it's B." in training_answer["prompt"][0]["content"]
    reference_answer = frame[(frame["variant"] == "reference") & (frame["kind"] == "answer")].iloc[0]
    assert "I think it's B." not in reference_answer["prompt"][0]["content"]


def test_control_mode_uses_the_unbiased_prompt_everywhere(inputs):
    pairs, output = inputs
    frame = run_builder(pairs, output, "--control")
    for _, row in frame.iterrows():
        assert "I think it's B." not in row["prompt"][0]["content"]


def test_existing_output_is_refused_without_force(inputs):
    pairs, output = inputs
    run_builder(pairs, output)
    with pytest.raises(SystemExit, match="exists"):
        builder.main(["--input", str(pairs), "--output", str(output)])
    run_builder(pairs, output, "--force")  # explicit overwrite works


def test_prompt_pairs_input_format(tmp_path):
    pair_rows = [
        {
            "pair_id": "mcq-bias:suggested_answer:q0",
            "source_id": "q0",
            "source": "logiqa",
            "reference_messages": [{"role": "user", "content": "Question 0?"}],
            "variant_messages": [{"role": "user", "content": "I think it's B. Question 0?"}],
            "metadata": {
                "bias_type": "suggested_answer",
                "correct_label": "A",
                "biased_option": "B",
                "valid_labels": LABELS,
            },
        }
    ]
    pairs = write_jsonl(tmp_path / "pairs.jsonl", pair_rows)
    output = tmp_path / "out.parquet"
    builder.main(["--input", str(pairs), "--input-format", "prompt_pairs", "--output", str(output)])
    frame = pd.read_parquet(output)
    assert len(frame) == 4
    row = frame[(frame["variant"] == "training") & (frame["kind"] == "answer")].iloc[0]
    assert row["biased_option"] == "B"
    assert row["ground_truth"] == "A"
    assert row["source_dataset"] == "logiqa"
    assert "I think it's B." in row["prompt"][0]["content"]


def test_prompt_pairs_metadata_must_carry_the_parser_contract(tmp_path):
    pair_rows = [
        {
            "source_id": "q0",
            "reference_messages": [{"role": "user", "content": "Question 0?"}],
            "variant_messages": [{"role": "user", "content": "Biased question 0?"}],
            "metadata": {"biased_option": "B"},  # no valid_labels
        }
    ]
    pairs = write_jsonl(tmp_path / "pairs.jsonl", pair_rows)
    with pytest.raises(ValueError, match="valid_labels"):
        builder.main(["--input", str(pairs), "--input-format", "prompt_pairs", "--output", str(tmp_path / "o.parquet")])
