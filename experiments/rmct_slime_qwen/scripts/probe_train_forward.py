"""Standalone repro of the actor's TRAINING forward (~90s, no ray/SGLang).

Builds the bridge LoRA model exactly like the actor (P3/P5 applied), then runs
the same forward the training loss uses — Miles' forward_only path over a bshd
batch — and prints per-token logprobs of a fixed sentence. Healthy ≈ the
values SGLang/HF give; broken = constant -log(vocab).

Run:  torchrun --nproc-per-node 1 probe_train_forward.py <model args...> \
        --hf-checkpoint ... --ref-load ... <misc miles required args>
"""

import sys

import torch


def main() -> None:
    from miles.utils.arguments import parse_args

    sys.argv += [
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
    args = parse_args()

    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(0)

    from miles.backends.megatron_utils.initialize import init

    init(args)

    from miles.backends.megatron_utils.bridge_lora_helpers import _setup_lora_model_via_bridge

    model = _setup_lora_model_via_bridge(args)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    text = "The capital of France is Paris. The capital of Germany is Berlin."
    ids = tok(text, return_tensors="pt").input_ids[0].cuda()
    n = ids.numel()

    # Mimic the training batch layout: bshd, one sequence, loss over all
    # positions but the first.
    rollout_data = {
        "tokens": [ids],
        "unconcat_tokens": [ids],
        "loss_masks": [torch.ones(n - 1, dtype=torch.int, device=ids.device)],
        "total_lengths": [n],
        "response_lengths": [n - 1],
        "rewards": [0.0],
        "sample_indices": [0],
    }

    from miles.backends.training_utils.data import get_data_iterator
    from miles.backends.megatron_utils.model import forward_only

    try:
        data_iterator, num_microbatches = get_data_iterator(args, model, rollout_data)
        out = forward_only(args, model, data_iterator, num_microbatches, store_prefix="")
        lp = out["log_probs"][0].float()
        print("PROBE-TRAINFWD logprobs head:", [round(v, 3) for v in lp[:8].tolist()], flush=True)
        print(
            f"PROBE-TRAINFWD mean={lp.mean().item():.3f} std={lp.std().item():.3f} "
            f"(uniform bug would be all ~{-torch.log(torch.tensor(float(args.vocab_size))).item():.3f})",
            flush=True,
        )
    except Exception as e:  # print full context — this script exists to iterate fast
        import traceback

        traceback.print_exc()
        print("PROBE-TRAINFWD failed:", e, flush=True)


if __name__ == "__main__":
    main()
