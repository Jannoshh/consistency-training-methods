# Project instructions

This repository succeeds the
[published implementation](https://github.com/sohaibimran7/cot-transparency).
Treat the published repository as a frozen paper artifact. Do not add
compatibility shims. Implement required functionality against the current
interfaces or leave it in the published repository.

## Training and evaluation approval

- Show the exact commands and resolved parameters before every training or
  evaluation run.
- Present all commands in a multi-stage run before execution.
- Do not execute a paid or remote run until the user explicitly approves it.

## Repository scope

CTM supports modular consistency-training research for sycophancy, jailbreak
robustness, and evaluation awareness. RLCT consumes an explicitly imported
training `Setting` on Tinker or `LocalBackend`. BCT, ACT, AttCT, and MLPCT are
file-driven. Evaluation remains independent of the training setting.

## Environment

Use `uv` for environment management and command execution. `requirements.txt`
is the single dependency source of truth.

```bash
uv venv
uv pip install -r requirements.txt   # includes the mcq-bias git pin
uv pip install -e . --no-deps
```

Store API keys in the gitignored `.env` file. Relevant names include
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `TINKER_API_KEY`,
and `WANDB_API_KEY`. A W&B key does not enable logging by itself; an experiment
must also set `wandb_project`. No dataset artifacts are committed here.

## Where things live

- `ctm/settings/base.py` — generic `Setting` protocol and explicit
  `module:callable` construction; `runtime.py` validates the setting-to-trainer handoff and
  records artifact identities; `families.py` is the strict fixed-K family
  schema.
- `ctm_data/adapters/` — concrete mcq-bias, WildJailbreak, and EvalAwareBench
  adapters and source builders. `ctm/` must never import this directory or its
  benchmark dependencies.
- `ctm/training/refusal/` — refusal trait used only by relevant training
  settings. Official benchmark evaluation remains upstream.
- `ctm/evals/runner.py` + `tinker_model.py` — task-factory Inspect runner
  and a validated wrapper around tinker-cookbook's checkpoint adapter.
- `python -m ctm_data.adapters.wildjailbreak.builder` and
  `python -m ctm_data.adapters.eval_awareness.builder` — training-artifact
  builders for explicitly supplied source JSONL files.
- `scripts/train_rlct.py` — generic `--setting-factory`, `--setting-config`, and
  `--load-config`.
- `scripts/train_bct.py` — file-driven SFT/internal-consistency training,
  `--method bct|act|attct|mlpct`.
- `scripts/run_evals.py` — runs `module:task_factory` without a benchmark
  registry or setting coupling.
- `scripts/run_experiment.py` — minimal YAML command pipeline; training and
  evaluation entries are independent.
- Decision 2026-08-22: sycophancy consistency-training pools and the
  three-way sycophancy evaluation (held-out-bias BRR, bias-on-MMLU BRR,
  Anthropic model-written-evals sycophancy rate) are sourced from a
  [c-wei/AttCT](https://github.com/c-wei/AttCT) checkout rather than
  generated in-tree; `experiments/vdct_verl/scripts/make_pairs_from_attct.py`
  freezes its `sycophancy_bct` assets into the native paired-prompt schema
  for the RL recipes, and evaluation runs through that repo's `run_evals.py`.
  The MedicalChat sycophancy environment
  ([ariahw/rl-rewardhacking-ext](https://github.com/ariahw/rl-rewardhacking-ext))
  is a second sycophancy pool via
  `experiments/vdct_verl/scripts/make_pairs_from_medicalchat.py`; its
  converter carries no correctness signal into training (consistency
  training assumes none), and its judged evaluation runs upstream.
  The RL implementations (RMCT/VDCT on verl) stay in this repository.
- ACT/AttCT/MLPCT losses (`ctm/training/consistency_losses.py`, local execution
  in `ctm/backends/local/engine.py`, and paired-datum construction in
  `ctm/training/consistency_data.py`)
  are vendored from [AttCT](https://github.com/c-wei/AttCT) at commit `79527cf`.
  The code is copied rather than installed because that repository is an
  unpackaged, unlicensed, fast-moving research monorepo
  (top-level `losses`/`data` modules, hard deps on vllm/bitsandbytes). To pick
  up upstream changes, diff against the noted SHA and port deliberately.
- Sycophancy eval tasks/scorers can come directly from the pinned **mcq-bias package**
  (co-develop with `uv pip install -e /path/to/mcq-bias --no-deps`). Its
  training rows are native `mcq_bias` JSONL files selected explicitly in an
  experiment config; there is no CTM split policy, dump fallback, or runtime
  prompt conversion.

## Frozen artifact rules

- Treat every JSONL and its manifest as an inseparable pair. Loaders verify the
  schema version, row count, and content SHA-256. Builders refuse to overwrite
  either path; changed source revision, split, seed, prompt style, or fixed-K
  choice always gets a new output path.
- Sycophancy training accepts exact native mcq-bias files through `data_paths`.
  CTM neither reserves an evaluation prefix nor infers evaluation from them.
- Jailbreak materialization freezes fixed-K training families from exactly the
  supplied WildJailbreak rows. Shipped completions are never used. Evaluation
  datasets and judges come from the chosen upstream task/CLI.
- The refusal training grader defaults to temperature 0, a 32,768-token
  ceiling, bounded retries and concurrency, and strict XML parsing. Training
  manifests record its resolved generation configuration and rubric hash.
- EvalAwareBench artifacts retain their source revision and CC-BY-NC-4.0
  license provenance. Evaluation is configured separately through the chosen
  benchmark task and judge.
- Family-based runs persist the exact selected source-ID selection and K for run
  provenance. That record does not restrict later evaluations.
- RL runs persist every sampled response by default. Each record states whether
  it was skipped from training and records the reason when applicable.

## Known limitations

1. `train_bct.py` consumes completed response rows; this repository does not
   generate self-target BCT response data.
2. Inspect logs can be analyzed individually, but the repository does not yet
   provide cross-checkpoint aggregation or plotting commands.
3. The evaluation runner can load LocalBackend LoRA checkpoints directly with
   `--local-checkpoint`. Full-weight LocalBackend checkpoints are not supported
   by this evaluation bridge.

## Tests and formatting

```bash
uv run python -m pytest
uv run python -m pytest tests
uv run black --check .
uv run ruff check .
```
