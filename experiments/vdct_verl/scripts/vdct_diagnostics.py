#!/usr/bin/env python3
"""VDCT distribution diagnostics: calibration, entropy, cue-invariance.

Consumes per-rollout JSONL rows from either source:

- training rollout dumps written by the VDCT trainer
  (``trainer.rollout_data_dir``; the trainer threads the parsed fields in), or
- audit-set generation files produced in Phase 1/5 with the same field names.

Required fields per row: ``group_id``, ``variant``, ``kind``, ``parse_ok``,
``option_labels``; plus ``option_distribution`` on parsed distribution rows
and ``answer_index`` on parsed answer rows. ``biased_option`` enables the
cue-invariance breakdown.

Report (JSON to ``--output``, summary to stdout):

- **calibration** — per (group, variant) the mean parsed stated distribution
  vs the empirical answer distribution from the same side's parsed answer
  rows: expected calibration error over per-option (stated, empirical) pairs
  in 10 bins, plus the mean total-variation distance. Sides lacking either
  parsed distributions or parsed answers are skipped and counted.
- **entropy** — mean entropy of parsed stated distributions, per variant
  (the hedging monitor).
- **cue_invariance** — per group with both sides present:
  TV(mean cued stated, mean reference stated), the share of the shifted mass
  on the cued (biased) option, and how often the largest single shift is the
  cued option — the full-distribution coverage rate matching cannot see.

CPU-only; tested in ``tests/test_vdct_diagnostics.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_RECIPE_ROOT = Path(__file__).resolve().parents[1]
if str(_RECIPE_ROOT) not in sys.path:
    sys.path.insert(0, str(_RECIPE_ROOT))

from recipe.vdct.vdct_core import (
    ANSWER_KIND,
    DISTRIBUTION_KIND,
    REFERENCE_VARIANT,
    TRAINING_VARIANT,
    entropy,
    mean_distribution,
    total_variation,
)

N_BINS = 10


def read_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with path.open() as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                for key in ("group_id", "variant", "kind", "parse_ok", "option_labels"):
                    if key not in row:
                        raise ValueError(f"{path}:{line_no}: row missing {key!r}")
                rows.append(row)
    return rows


def collect_sides(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """Group parsed rollouts into (group_id, variant) sides."""
    sides: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"dists": [], "answer_indices": [], "option_labels": None, "biased_option": None}
    )
    for row in rows:
        key = (str(row["group_id"]), str(row["variant"]))
        side = sides[key]
        if side["option_labels"] is None:
            side["option_labels"] = [str(v) for v in row["option_labels"]]
        if row.get("biased_option"):
            side["biased_option"] = str(row["biased_option"])
        if not row["parse_ok"]:
            continue
        if row["kind"] == DISTRIBUTION_KIND and row.get("option_distribution") is not None:
            side["dists"].append(tuple(float(v) for v in row["option_distribution"]))
        elif row["kind"] == ANSWER_KIND and row.get("answer_index") is not None:
            side["answer_indices"].append(int(row["answer_index"]))
    return dict(sides)


def expected_calibration_error(pairs: list[tuple[float, float]], n_bins: int = N_BINS) -> float:
    """ECE over (stated probability, empirical frequency) pairs, equal-width bins."""
    if not pairs:
        raise ValueError("no calibration pairs")
    bins: list[list[tuple[float, float]]] = [[] for _ in range(n_bins)]
    for stated, empirical in pairs:
        index = min(int(stated * n_bins), n_bins - 1)
        bins[index].append((stated, empirical))
    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        mean_stated = sum(p for p, _ in bucket) / len(bucket)
        mean_empirical = sum(e for _, e in bucket) / len(bucket)
        ece += (len(bucket) / len(pairs)) * abs(mean_stated - mean_empirical)
    return ece


def calibration_report(sides: dict[tuple[str, str], dict]) -> dict:
    pairs: list[tuple[float, float]] = []
    tv_values: list[float] = []
    skipped = 0
    for side in sides.values():
        if not side["dists"] or not side["answer_indices"]:
            skipped += 1
            continue
        stated = mean_distribution(side["dists"])
        n_options = len(stated)
        counts = [0] * n_options
        for index in side["answer_indices"]:
            counts[index] += 1
        empirical = [c / len(side["answer_indices"]) for c in counts]
        pairs.extend(zip(stated, empirical))
        tv_values.append(total_variation(stated, empirical))
    if not pairs:
        return {"n_sides": 0, "n_sides_skipped": skipped}
    return {
        "n_sides": len(tv_values),
        "n_sides_skipped": skipped,
        "ece": expected_calibration_error(pairs),
        "tv_stated_vs_empirical_mean": sum(tv_values) / len(tv_values),
    }


def entropy_report(sides: dict[tuple[str, str], dict]) -> dict:
    per_variant: dict[str, list[float]] = defaultdict(list)
    for (_, variant), side in sides.items():
        per_variant[variant].extend(entropy(d) for d in side["dists"])
    return {
        variant: {"mean_entropy": sum(values) / len(values), "n_distributions": len(values)}
        for variant, values in per_variant.items()
        if values
    }


def cue_invariance_report(sides: dict[tuple[str, str], dict]) -> dict:
    tv_values: list[float] = []
    biased_shares: list[float] = []
    n_largest_shift_on_cued = 0
    n_with_cue_info = 0
    for group_id in {g for g, _ in sides}:
        reference = sides.get((group_id, REFERENCE_VARIANT))
        training = sides.get((group_id, TRAINING_VARIANT))
        if not reference or not training or not reference["dists"] or not training["dists"]:
            continue
        mean_ref = mean_distribution(reference["dists"])
        mean_train = mean_distribution(training["dists"])
        tv_values.append(total_variation(mean_train, mean_ref))
        labels = training["option_labels"]
        biased = training["biased_option"]
        if biased is None or biased not in labels:
            continue
        n_with_cue_info += 1
        deltas = [abs(t - r) for t, r in zip(mean_train, mean_ref)]
        total_shift = sum(deltas)
        biased_idx = labels.index(biased)
        if total_shift > 0.0:
            biased_shares.append(deltas[biased_idx] / total_shift)
            if max(range(len(deltas)), key=deltas.__getitem__) == biased_idx:
                n_largest_shift_on_cued += 1
    if not tv_values:
        return {"n_groups": 0}
    report = {"n_groups": len(tv_values), "tv_cued_vs_reference_mean": sum(tv_values) / len(tv_values)}
    if biased_shares:
        report.update(
            {
                "n_groups_with_cue_info": n_with_cue_info,
                "biased_option_shift_share_mean": sum(biased_shares) / len(biased_shares),
                "largest_shift_on_cued_frac": n_largest_shift_on_cued / len(biased_shares),
            }
        )
    return report


def build_report(rows: list[dict]) -> dict:
    sides = collect_sides(rows)
    return {
        "n_rows": len(rows),
        "n_sides": len(sides),
        "calibration": calibration_report(sides),
        "entropy": entropy_report(sides),
        "cue_invariance": cue_invariance_report(sides),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, nargs="+", help="rollout dump / audit JSONL files")
    parser.add_argument("--output", type=Path, default=None, help="write the JSON report here")
    args = parser.parse_args(argv)

    report = build_report(read_rows(args.input))
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
