"""ctm-schema rollout persistence for slime runs.

Writes the same ``rollouts/step_*.jsonl.zst`` + ``index.json`` layout as
``ctm.training.rollout_log.RolloutLogger`` (ctm @ 58c889c) so slime runs keep
the repository's provenance contract and feed the same Gate A replay test.
Every sampled response is recorded; skipped records must carry a skip_reason.
"""

import json
from pathlib import Path

import zstandard

INDEX_NAME = "index.json"


def step_filename(step: int) -> str:
    return f"step_{step:06d}.jsonl.zst"


class RolloutWriter:
    def __init__(self, rollout_dir: str):
        self.dir = Path(rollout_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.dir / INDEX_NAME
        if self._index_path.exists():
            self._index = json.loads(self._index_path.read_text())
        else:
            self._index = {"steps": []}

    def write_step(self, step: int, records: list[dict]) -> None:
        """Persist one step. Never overwrites (frozen-artifact rule); a step
        regenerated after a kill/resume supersedes the old attempt — the prior
        file is renamed ``*.superseded-<n>`` and its index entry marked, so
        both attempts stay on disk but replay sees only the trained one.
        Skipped records lacking a reason are rejected (same guard as ctm)."""
        path = self.dir / step_filename(step)
        if path.exists():
            n = 1
            while (superseded := path.with_name(f"{path.name}.superseded-{n}")).exists():
                n += 1
            path.rename(superseded)
            # Move the old attempt's entry out of "steps" (the canonical
            # trained sequence ctm's iter_rollouts replays) into "superseded".
            keep, old = [], []
            for entry in self._index["steps"]:
                (old if entry["step"] == step else keep).append(entry)
            for entry in old:
                entry["file"] = superseded.name
            self._index["steps"] = keep
            self._index.setdefault("superseded", []).extend(old)
        for record in records:
            if record.get("skipped_from_training") and not record.get("skip_reason"):
                raise ValueError("skipped rollout record lacks skip_reason")
        payload = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
        path.write_bytes(zstandard.ZstdCompressor().compress(payload.encode()))

        n_train = sum(1 for r in records if r["role"] == "train" and not r["skipped_from_training"])
        n_anchor = sum(1 for r in records if r["role"] == "anchor" and not r["skipped_from_training"])
        p_refs = [r["p_ref"] for r in records if r.get("p_ref") is not None]
        p_hats = [r["p_hat"] for r in records if r.get("p_hat") is not None]
        traits = [r["trait_value"] for r in records if r.get("trait_value") is not None]
        self._index["steps"].append(
            {
                "step": step,
                "file": path.name,
                "n_records": len(records),
                "n_train": n_train,
                "n_anchor": n_anchor,
                "n_skipped": sum(1 for r in records if r["skipped_from_training"]),
                "p_ref_mean": sum(p_refs) / len(p_refs) if p_refs else None,
                "p_hat_mean": sum(p_hats) / len(p_hats) if p_hats else None,
                "trait_mean": sum(traits) / len(traits) if traits else None,
            }
        )
        self._index_path.write_text(json.dumps(self._index, indent=1))
