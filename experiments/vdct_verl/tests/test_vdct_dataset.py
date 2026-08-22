"""Tests for the VDCT parquet builder: schema, row structure, control mode,
frozen-artifact refusals, and both input formats."""

import pandas as pd
import pytest
from recipe.vdct.vdct_elicitation import DISTRIBUTION_OPEN
from vdct_test_helpers import load_script, write_jsonl

from ctm.artifacts import write_verified_jsonl_artifact
from ctm.settings.pairs import PAIR_ARTIFACT_SCHEMA, PAIR_ARTIFACT_SCHEMA_VERSION

builder = load_script("make_vdct_dataset")

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


def pair_row(idx: int, metadata: dict | None = None) -> dict:
    return {
        "pair_id": f"mcq-bias:suggested_answer:q{idx}",
        "source_id": f"q{idx}",
        "source": "logiqa",
        "reference_messages": [{"role": "user", "content": f"Question {idx}?"}],
        "variant_messages": [{"role": "user", "content": f"I think it's B. Question {idx}?"}],
        "metadata": (
            metadata
            if metadata is not None
            else {
                "bias_type": "suggested_answer",
                "correct_label": "A",
                "biased_option": "B",
                "valid_labels": LABELS,
            }
        ),
    }


def write_pair_artifact(path, rows):
    """prompt_pairs inputs must be verified artifacts (JSONL + manifest)."""
    write_verified_jsonl_artifact(
        path,
        rows,
        artifact_schema=PAIR_ARTIFACT_SCHEMA,
        schema_version=PAIR_ARTIFACT_SCHEMA_VERSION,
        provenance={"test": True},
    )
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
    pairs = write_pair_artifact(tmp_path / "pairs.jsonl", [pair_row(0)])
    output = tmp_path / "out.parquet"
    builder.main(["--input", str(pairs), "--input-format", "prompt_pairs", "--output", str(output)])
    frame = pd.read_parquet(output)
    assert len(frame) == 4
    row = frame[(frame["variant"] == "training") & (frame["kind"] == "answer")].iloc[0]
    assert row["biased_option"] == "B"
    assert row["ground_truth"] == "A"
    assert row["source_dataset"] == "logiqa"
    assert "I think it's B." in row["prompt"][0]["content"]


def test_prompt_pairs_requires_a_verified_manifest(tmp_path):
    # A bare JSONL without its manifest sidecar is not a frozen artifact.
    pairs = write_jsonl(tmp_path / "pairs.jsonl", [pair_row(0)])
    with pytest.raises(Exception, match="manifest"):
        builder.main(["--input", str(pairs), "--input-format", "prompt_pairs", "--output", str(tmp_path / "o.parquet")])


def test_prompt_pairs_metadata_must_carry_the_parser_contract(tmp_path):
    pairs = write_pair_artifact(tmp_path / "pairs.jsonl", [pair_row(0, metadata={"biased_option": "B"})])
    with pytest.raises(ValueError, match="valid_labels"):
        builder.main(["--input", str(pairs), "--input-format", "prompt_pairs", "--output", str(tmp_path / "o.parquet")])
