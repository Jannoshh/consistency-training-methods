"""Standalone SGLang throughput bench for the RMCT rollout shape.

Drives a bare SGLang server (no ray, no Megatron, no relaunch cost) with the
exact request pattern of one RMCT generation: per datapoint, K=2 perturbation
prompts (neutral + biased) x n rollouts each, sampled at the science settings.
Measures what the optimization phase needs:

    gen_wall_s          wall-clock for the whole generation
    output_tok_per_s    aggregate decode throughput
    completion p50/p99  completion-length distribution (budget sizing)
    prefix_cache_rate   cached_tokens / prompt_tokens (radix under fan-out)
    parse_rate          answer-parse success (sanity: truncation biases p_hat)

Usage (server already running, e.g.):
    python3 -m sglang.launch_server --model-path /workspace/models/Qwen3.5-9B \
        --port 30000 --mem-fraction-static 0.85 &
    RMCT_CONFIG=/workspace/rmct/configs/run_9b.json \
    python3 bench_sglang.py --url http://127.0.0.1:30000 \
        --datapoints 4 --rollouts 128 [--max-new-tokens 20480] [--label mtp-on]

Results append as JSONL to --out (default bench_results.jsonl) with the label,
so sweeps accumulate into one comparable file.
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slime_port.rmct_config import RMCTConfig  # noqa: E402
from slime_port.rmct_rollout import _load_parse_answer  # noqa: E402


def build_prompts(config: RMCTConfig, tokenizer, n_datapoints: int) -> list[dict]:
    """One entry per (datapoint, perturbation): tokenized chat prompt."""
    rows = []
    with open(config.data_paths[0]) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= n_datapoints:
                break
    prompts = []
    for idx, row in enumerate(rows):
        for pert, key in ((0, "unbiased_messages"), (1, "biased_messages")):
            ids = tokenizer.apply_chat_template(row[key], add_generation_prompt=True, tokenize=True)
            if hasattr(ids, "input_ids"):
                ids = ids.input_ids
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            prompts.append({"datapoint": idx, "pert": pert, "input_ids": [int(t) for t in ids], "row": row})
    return prompts


async def generate(session, url, input_ids, sampling, semaphore):
    async with semaphore:
        payload = {"input_ids": input_ids, "sampling_params": sampling}
        async with session.post(f"{url}/generate", json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


async def run_bench(args, config: RMCTConfig) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path or config_model_path(args.url), trust_remote_code=True)
    parse_answer = _load_parse_answer()
    prompts = build_prompts(config, tokenizer, args.datapoints)
    sampling = {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "max_new_tokens": args.max_new_tokens or config.max_new_tokens,
    }
    semaphore = asyncio.Semaphore(args.concurrency)
    t0 = time.perf_counter()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
        tasks = [
            generate(session, args.url, p["input_ids"], sampling, semaphore)
            for p in prompts
            for _ in range(args.rollouts)
        ]
        results = await asyncio.gather(*tasks)
    wall = time.perf_counter() - t0

    lens, cached, prompt_toks, parsed, truncated = [], 0, 0, 0, 0
    for r in results:
        meta = r["meta_info"]
        lens.append(meta["completion_tokens"])
        cached += meta.get("cached_tokens", 0)
        prompt_toks += meta.get("prompt_tokens", 0)
        if meta.get("finish_reason", {}).get("type") == "length":
            truncated += 1
        elif parse_answer(r["text"]) is not None:
            parsed += 1
    n = len(results)
    out = {
        "label": args.label,
        "n_requests": n,
        "gen_wall_s": round(wall, 2),
        "output_tok_per_s": round(sum(lens) / wall, 1),
        "completion_p50": int(statistics.median(lens)),
        "completion_p99": int(sorted(lens)[int(0.99 * (n - 1))]),
        "prefix_cache_rate": round(cached / max(prompt_toks, 1), 4),
        "parse_rate": round(parsed / n, 4),
        "truncation_rate": round(truncated / n, 4),
        "concurrency": args.concurrency,
        "sampling": sampling,
        "datapoints": args.datapoints,
        "rollouts_per_prompt": args.rollouts,
    }
    return out


def config_model_path(url: str) -> str:
    import urllib.request

    with urllib.request.urlopen(f"{url}/get_model_info") as resp:
        return json.load(resp)["model_path"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--datapoints", type=int, default=4)
    parser.add_argument("--rollouts", type=int, default=128, help="per perturbation prompt")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="override config")
    parser.add_argument("--concurrency", type=int, default=512)
    parser.add_argument("--label", required=True, help="config label for this measurement")
    parser.add_argument("--model-path", default=None, help="tokenizer path; default: ask the server")
    parser.add_argument("--out", default="bench_results.jsonl")
    args = parser.parse_args()

    config = RMCTConfig.load()
    result = asyncio.run(run_bench(args, config))
    print(json.dumps(result, indent=2))
    with open(args.out, "a") as f:
        f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
