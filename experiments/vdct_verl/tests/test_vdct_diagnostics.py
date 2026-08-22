"""Tests for the diagnostics executable on constructed rollout rows."""

import json
import math

import pytest
from vdct_test_helpers import load_script, write_jsonl

diagnostics = load_script("vdct_diagnostics")

LABELS = ["A", "B"]


def dist_rollout(group, variant, dist, parse_ok=True, biased="B"):
    return {
        "group_id": group,
        "variant": variant,
        "kind": "distribution",
        "parse_ok": parse_ok,
        "option_labels": LABELS,
        "biased_option": biased,
        "option_distribution": dist,
    }


def answer_rollout(group, variant, index, parse_ok=True, biased="B"):
    return {
        "group_id": group,
        "variant": variant,
        "kind": "answer",
        "parse_ok": parse_ok,
        "option_labels": LABELS,
        "biased_option": biased,
        "answer_index": index,
    }


def test_perfectly_calibrated_side_has_zero_ece_and_tv():
    rows = (
        [dist_rollout("g0", "reference", [0.75, 0.25]) for _ in range(4)]
        + [answer_rollout("g0", "reference", 0) for _ in range(3)]
        + [answer_rollout("g0", "reference", 1)]
    )  # stated mean [0.75, 0.25] == empirical [3/4, 1/4]
    report = diagnostics.build_report(rows)
    assert report["calibration"]["ece"] == pytest.approx(0.0, abs=1e-12)
    assert report["calibration"]["tv_stated_vs_empirical_mean"] == pytest.approx(0.0, abs=1e-12)
    assert report["calibration"]["n_sides_scored"] == 1


def test_miscalibrated_side_has_positive_ece():
    rows = [dist_rollout("g0", "reference", [1.0, 0.0])] + [
        answer_rollout("g0", "reference", i % 2) for i in range(4)
    ]  # stated one-hot vs empirical 50/50
    report = diagnostics.build_report(rows)
    assert report["calibration"]["ece"] == pytest.approx(0.5, abs=1e-12)
    assert report["calibration"]["tv_stated_vs_empirical_mean"] == pytest.approx(0.5, abs=1e-12)


def test_entropy_report_per_variant():
    rows = [
        dist_rollout("g0", "reference", [0.5, 0.5]),
        dist_rollout("g0", "training", [1.0, 0.0]),
    ]
    report = diagnostics.build_report(rows)
    assert report["entropy"]["reference"]["mean_entropy"] == pytest.approx(math.log(2), abs=1e-12)
    assert report["entropy"]["training"]["mean_entropy"] == pytest.approx(0.0, abs=1e-12)


def test_cue_invariance_shift_on_the_cued_option():
    rows = [
        dist_rollout("g0", "reference", [0.7, 0.3]),
        dist_rollout("g0", "training", [0.3, 0.7]),  # mass moved onto cued option B
    ]
    report = diagnostics.build_report(rows)["cue_invariance"]
    assert report["n_groups"] == 1
    assert report["tv_cued_vs_reference_mean"] == pytest.approx(0.4, abs=1e-12)
    assert report["biased_option_shift_share_mean"] == pytest.approx(0.5, abs=1e-12)
    assert report["largest_shift_on_cued_frac"] in (0.0, 1.0)  # tie between A and B


def test_sides_without_answers_are_skipped_in_calibration():
    rows = [dist_rollout("g0", "reference", [0.6, 0.4])]
    report = diagnostics.build_report(rows)
    assert report["calibration"]["n_sides_scored"] == 0
    assert report["calibration"]["n_sides_skipped"] == 1


def test_unparsed_rollouts_are_ignored():
    rows = (
        [dist_rollout("g0", "reference", [0.6, 0.4])]
        + [dist_rollout("g0", "reference", None, parse_ok=False)]
        + [answer_rollout("g0", "reference", 0)]
        + [answer_rollout("g0", "reference", None, parse_ok=False)]
    )
    report = diagnostics.build_report(rows)
    assert report["calibration"]["n_sides_scored"] == 1
    # mean stated = [0.6, 0.4] (one parsed dist), empirical = [1.0, 0.0] (one parsed answer)
    assert report["calibration"]["tv_stated_vs_empirical_mean"] == pytest.approx(0.4, abs=1e-12)


def test_main_reads_files_and_writes_report(tmp_path):
    rows = [
        dist_rollout("g0", "reference", [0.5, 0.5]),
        answer_rollout("g0", "reference", 0),
    ]
    input_path = write_jsonl(tmp_path / "rollouts.jsonl", rows)
    output_path = tmp_path / "report.json"
    diagnostics.main(["--input", str(input_path), "--output", str(output_path)])
    report = json.loads(output_path.read_text())
    assert report["n_rows"] == 2
    assert report["calibration"]["n_sides_scored"] == 1


def test_missing_required_field_is_an_error(tmp_path):
    input_path = tmp_path / "rollouts.jsonl"
    input_path.write_text(json.dumps({"group_id": "g0", "variant": "reference"}) + "\n")
    with pytest.raises(ValueError, match="missing"):
        diagnostics.main(["--input", str(input_path)])
