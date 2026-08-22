"""Tests for the strict verbalized-distribution parser, including malformed cases."""

import pytest
from recipe.vdct.vdct_elicitation import (
    DISTRIBUTION_CLOSE,
    DISTRIBUTION_OPEN,
    elicitation_instruction,
    parse_option_distribution,
)

LABELS = ["A", "B", "C", "D"]


def block(body: str) -> str:
    return f"Some chain of thought.\n{DISTRIBUTION_OPEN}\n{body}\n{DISTRIBUTION_CLOSE}\n"


def test_instruction_names_every_label_and_the_block():
    text = elicitation_instruction(LABELS)
    assert DISTRIBUTION_OPEN in text and DISTRIBUTION_CLOSE in text
    for label in LABELS:
        assert f"{label}: <probability>" in text
    assert "probability distribution over the answer options" in text
    assert "report" not in text.lower()


def test_parse_unit_probabilities():
    parsed = parse_option_distribution(block("A: 0.62\nB: 0.21\nC: 0.09\nD: 0.08"), LABELS)
    assert parsed == pytest.approx([0.62, 0.21, 0.09, 0.08], abs=1e-9)


def test_parse_percentages_with_and_without_sign():
    with_sign = parse_option_distribution(block("A: 62%\nB: 21%\nC: 9%\nD: 8%"), LABELS)
    without = parse_option_distribution(block("A: 62\nB: 21\nC: 9\nD: 8"), LABELS)
    assert with_sign == pytest.approx([0.62, 0.21, 0.09, 0.08], abs=1e-9)
    assert without == pytest.approx([0.62, 0.21, 0.09, 0.08], abs=1e-9)


def test_parse_renormalizes_within_tolerance():
    parsed = parse_option_distribution(block("A: 0.5\nB: 0.28\nC: 0.13\nD: 0.13"), LABELS)  # sums to 1.04
    assert parsed == pytest.approx([0.5 / 1.04, 0.28 / 1.04, 0.13 / 1.04, 0.13 / 1.04], abs=1e-9)
    assert sum(parsed) == pytest.approx(1.0, abs=1e-12)


def test_parse_floors_zero_entries():
    parsed = parse_option_distribution(block("A: 1.0\nB: 0\nC: 0\nD: 0"), LABELS)
    assert all(v > 0 for v in parsed)
    assert sum(parsed) == pytest.approx(1.0, abs=1e-12)
    assert parsed[1] == pytest.approx(1e-3 / (1.0 + 3e-3), abs=1e-9)


def test_last_block_wins():
    text = f"I'll answer in {DISTRIBUTION_OPEN}A: 1{DISTRIBUTION_CLOSE} format... thinking...\n" + block(
        "A: 0.1\nB: 0.2\nC: 0.3\nD: 0.4"
    )
    assert parse_option_distribution(text, LABELS) == pytest.approx([0.1, 0.2, 0.3, 0.4], abs=1e-9)


def test_separator_and_whitespace_tolerance():
    parsed = parse_option_distribution(block("  A = 0.25 \n B : 0.25\nC:0.25\n  D: 0.25  "), LABELS)
    assert parsed == pytest.approx([0.25] * 4, abs=1e-9)


@pytest.mark.parametrize(
    "body",
    [
        "A: 0.5\nB: 0.3\nC: 0.2",  # missing D
        "A: 0.4\nB: 0.3\nC: 0.2\nD: 0.05\nE: 0.05",  # unknown label
        "A: 0.5\nA: 0.5\nB: 0\nC: 0\nD: 0",  # duplicate label
        "A: 0.5\nB: 0.2\nC: 0.05\nD: 0.05",  # sums to 0.8
        "A: 30\nB: 30\nC: 30\nD: 30",  # sums to 120
        "A: 0.7%\nB: 0.1%\nC: 0.1%\nD: 0.1%",  # % signs but unit-scale total: ambiguous
        "A: high\nB: low\nC: 0\nD: 0",  # non-numeric
        "",  # empty block
    ],
)
def test_malformed_blocks_are_rejected(body):
    assert parse_option_distribution(block(body), LABELS) is None


def test_missing_or_unclosed_block_is_rejected():
    assert parse_option_distribution("The answer is A.", LABELS) is None
    assert parse_option_distribution(f"{DISTRIBUTION_OPEN}\nA: 1\nB: 0\nC: 0\nD: 0", LABELS) is None


def test_sum_tolerance_is_configurable():
    body = "A: 0.5\nB: 0.2\nC: 0.1\nD: 0.05"  # sums to 0.85
    assert parse_option_distribution(block(body), LABELS) is None
    parsed = parse_option_distribution(block(body), LABELS, sum_tolerance=0.2)
    assert parsed == pytest.approx([v / 0.85 for v in (0.5, 0.2, 0.1, 0.05)], abs=1e-9)


def test_empty_labels_are_an_error():
    with pytest.raises(ValueError, match="option_labels"):
        parse_option_distribution(block("A: 1"), [])
