"""Unit tests for the GPU-free parts of the slime rollout module:
data loading, deterministic batching, classification, and the ctm-schema
rollout writer round-trip (read back by the Gate A replay reader)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slime_port.pipeline import AdvantageConfig
from slime_port.rmct_config import RMCTConfig
from slime_port.rmct_rollout import _batch_for_step, _classify, _load_datapoints, _perturbation_messages
from slime_port.rollout_writer import RolloutWriter

DEV_DATA = Path(__file__).resolve().parents[1] / "data" / "dev_smoke" / "suggested-answer-pairs.jsonl"

pytestmark = pytest.mark.skipif(not DEV_DATA.exists(), reason="dev smoke dataset not built")


def _config(**overrides):
    defaults = {
        "data_paths": [str(DEV_DATA)],
        "n_datapoints": 8,
        "batch_size": 4,
        "advantage": AdvantageConfig(),
    }
    defaults.update(overrides)
    return RMCTConfig(**defaults)


def test_load_and_batching_deterministic():
    config = _config()
    datapoints = _load_datapoints(config)
    assert len(datapoints) == 8
    b0 = _batch_for_step(config, datapoints, 0)
    b1 = _batch_for_step(config, datapoints, 1)
    assert _batch_for_step(config, datapoints, 0) == b0  # deterministic
    epoch_indices = sorted(idx for idx, _ in b0 + b1)
    assert epoch_indices == list(range(8))  # one epoch covers each datapoint once
    b2 = _batch_for_step(config, datapoints, 2)  # next epoch, reshuffled
    assert len(b2) == 4


def test_perturbations_and_control():
    config = _config()
    dp = _load_datapoints(config)[0]
    assert _perturbation_messages(config, dp, 0) == dp["unbiased_messages"]
    assert _perturbation_messages(config, dp, 1) == dp["biased_messages"]
    control = _config(control=True)
    assert _perturbation_messages(control, dp, 1) == dp["unbiased_messages"]


def test_vendored_matches_bias_identical():
    from mcq_bias.scorers import matches_bias
    from slime_port.rmct_rollout import _matches_bias

    for answer, option in [("A", "A"), ("B", "A"), ("A", "NOT A"), ("B", "NOT A"), ("A", "")]:
        assert _matches_bias(answer, option) == matches_bias(answer, option)


def test_classifier_matches_ctm_adapter():
    from ctm_data.adapters.mcq_bias.setting import trait_classifier

    dp = _load_datapoints(_config())[0]
    for text in [
        f"Some reasoning. The answer is: ({dp['biased_option']})",
        "Some reasoning. The answer is: (A)",
        "no parseable answer here",
    ]:
        trait, parsed = _classify(text, dp)
        expected = trait_classifier(text, dp, [])
        assert trait == expected
        from mcq_bias.parsers import parse_answer

        assert parsed == (parse_answer(text) is not None)


def test_rollout_writer_round_trip(tmp_path):
    writer = RolloutWriter(str(tmp_path))
    records = [
        {
            "step": 1,
            "epoch": 0,
            "datapoint_idx": 0,
            "perturbation_idx": 1,
            "role": "train",
            "sample_source": "policy",
            "prompt_text": "p",
            "prompt_context": {},
            "completion_text": "c",
            "trait_value": 1.0,
            "parsed_successfully": True,
            "grader_failed": False,
            "reward": -0.1,
            "advantage": 0.5,
            "skipped_from_training": False,
            "skip_reason": None,
            "p_hat": 0.5,
            "p_ref": 0.25,
            "p_ref_init": None,
        },
        {
            "step": 1,
            "epoch": 0,
            "datapoint_idx": 0,
            "perturbation_idx": 0,
            "role": "rate",
            "sample_source": "policy",
            "prompt_text": "p",
            "prompt_context": {},
            "completion_text": "c2",
            "trait_value": 0.0,
            "parsed_successfully": False,
            "grader_failed": False,
            "reward": None,
            "advantage": None,
            "skipped_from_training": True,
            "skip_reason": "answer_parse_failure",
            "p_hat": None,
            "p_ref": 0.25,
            "p_ref_init": None,
        },
    ]
    writer.write_step(1, records)

    # Readable by the ctm analysis reader (schema contract) and Gate A reader.
    from ctm.evals.analysis.rollouts import iter_rollouts, load_index

    loaded = list(iter_rollouts(str(tmp_path)))
    assert len(loaded) == 2
    assert loaded[0].advantage == 0.5
    index = load_index(str(tmp_path))
    assert index[0]["n_train"] == 1

    with pytest.raises(ValueError):
        writer.write_step(2, [dict(records[1], skip_reason=None)])

    # Kill/resume regeneration: rewriting a step supersedes the old attempt —
    # never overwrites it — and replay (index "steps") sees only the new one.
    regenerated = [dict(records[0], completion_text="c-resume", advantage=0.75)]
    writer.write_step(1, regenerated)
    assert (tmp_path / "step_000001.jsonl.zst.superseded-1").exists()
    loaded = list(iter_rollouts(str(tmp_path)))
    assert len(loaded) == 1
    assert loaded[0].advantage == 0.75
    index = load_index(str(tmp_path))
    assert [e["step"] for e in index] == [1]
    raw_index = json.loads((tmp_path / "index.json").read_text())
    assert raw_index["superseded"][0]["file"] == "step_000001.jsonl.zst.superseded-1"

    # A fresh writer on the same dir (restart) keeps the supersede behavior.
    writer2 = RolloutWriter(str(tmp_path))
    writer2.write_step(1, records)
    assert (tmp_path / "step_000001.jsonl.zst.superseded-2").exists()
    assert len(list(iter_rollouts(str(tmp_path)))) == 2


def test_config_loader_env(tmp_path, monkeypatch):
    config_path = tmp_path / "rmct.json"
    config_path.write_text(
        json.dumps(
            {
                "_comment": "ignored",
                "data_paths": [str(DEV_DATA)],
                "n_datapoints": 4,
                "kl_coef": 0.05,
                "advantage": {"advantage_estimator": "grpo_normalized", "anchor_weight": 0.0},
            }
        )
    )
    monkeypatch.setenv("RMCT_CONFIG", str(config_path))
    config = RMCTConfig.load()
    assert config.kl_coef == 0.05
    assert config.temperature == 1.0 and config.top_p == 1.0  # pinned defaults
