"""Minimal probe: does Miles' bridge-built LoRA model actually load weights?

Answers the uniform-logits question in ~2 min without ray/SGLang: builds the
model exactly like Miles' LoRA path (`_setup_lora_model_via_bridge`), loads
the requested checkpoint the way the actor does, then prints
- per-module weight norms (zero/xavier-scale norms = nothing loaded), and
- logits for a short prompt (spread ≈ 0 = uniform bug reproduced).

Usage (single rank):
    torchrun --nproc-per-node 1 probe_bridge_lora_load.py \
        --hf-checkpoint /root/models/Qwen3.5-4B \
        --load /root/models/Qwen3.5-4B  [--skip-load]
plus the usual Miles LoRA args (lora-rank etc.) appended by the wrapper below.
"""

import sys

import torch


def main() -> None:
    from miles.utils.arguments import parse_args

    argv_extra = [
        "--train-backend",
        "megatron",
        "--lora-rank",
        "8",
        "--lora-alpha",
        "16",
        "--target-modules",
        "language_model.decoder.layers.*.mlp.linear_fc1,language_model.decoder.layers.*.mlp.linear_fc2",
        "--megatron-to-hf-mode",
        "bridge",
        "--qkv-format",
        "bshd",
        "--micro-batch-size",
        "1",
        "--rollout-num-gpus",
        "1",
        "--trust-remote-code",
    ]
    sys.argv += argv_extra
    args = parse_args()

    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(0)

    from miles.backends.megatron_utils.initialize import init

    init(args)

    from miles.backends.megatron_utils.bridge_lora_helpers import _setup_lora_model_via_bridge

    model = _setup_lora_model_via_bridge(args)
    module = model[0]

    if not getattr(args, "probe_skip_load", False):
        from miles.backends.megatron_utils.checkpoint import load_checkpoint

        load_checkpoint(model, None, None, {}, False)

    named = dict(module.named_parameters())
    print(f"PROBE {len(named)} params")
    for pattern in ["embed", "layers.0.", "layers.3.", "output_layer", "lm_head", "final"]:
        hits = [(n, p) for n, p in named.items() if pattern in n][:3]
        for n, p in hits:
            print(f"PROBE norm {p.float().norm().item():14.4f}  {tuple(p.shape)}  {n}")

    module.eval()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    ids = tok("The capital of France is", return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        pos = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        mask = None
        out = module(ids, pos, mask)
    logits = out[0, -1].float()
    print(
        f"PROBE logits: std={logits.std().item():.4f} max-min={(logits.max() - logits.min()).item():.4f} "
        f"top-token={tok.decode(logits.argmax().item())!r}"
    )
    print("PROBE done: std~0 => uniform-logits bug; healthy models have std >> 1")


if __name__ == "__main__":
    main()
