"""Generate slime/Megatron MODEL_ARGS for a Qwen3.5 dense checkpoint from its
HF config.json, instead of hand-copying numbers. Cross-checked against slime's
scripts/models/qwen3.5-{4B,9B}.sh (same flag set; only sizes differ).

Usage: python derive_model_args.py /path/to/Qwen3.5-2B > qwen3.5-2B.sh
"""

import json
import sys
from pathlib import Path


def main() -> None:
    model_dir = Path(sys.argv[1])
    config = json.loads((model_dir / "config.json").read_text())
    text = config.get("text_config", config)
    if config.get("model_type") not in ("qwen3_5", "qwen3_next") and text is config:
        raise SystemExit(f"unexpected model_type {config.get('model_type')}")
    if text.get("num_experts"):
        raise SystemExit("MoE variant — this deriver only handles dense Qwen3.5")

    n_heads = text["num_attention_heads"]
    n_kv = text["num_key_value_heads"]
    head_dim = text.get("head_dim") or text["hidden_size"] // n_heads
    rope = text.get("rope_parameters") or {}
    rope_theta = text.get("rope_theta") or rope.get("rope_theta")
    rotary_percent = text.get("partial_rotary_factor") or rope.get("partial_rotary_factor") or 0.25
    if rope_theta is None:
        raise SystemExit("no rope_theta in config.json (checked top level and rope_parameters)")
    untie = not (text.get("tie_word_embeddings") or config.get("tie_word_embeddings", False))
    lines = [
        "MODEL_ARGS=(",
        '   --spec "slime_plugins.models.qwen3_5" "get_qwen3_5_spec"',
        "",
        "   --disable-bias-linear",
        "   --qk-layernorm",
        "   --group-query-attention",
        f"   --num-attention-heads {n_heads}",
        f"   --num-query-groups {n_kv}",
        f"   --kv-channels {head_dim}",
        f"   --num-layers {text['num_hidden_layers']}",
        f"   --hidden-size {text['hidden_size']}",
        f"   --ffn-hidden-size {text['intermediate_size']}",
        "   --use-gated-attention",
        "",
        "   --normalization RMSNorm",
        "   --apply-layernorm-1p",
        "   --position-embedding-type rope",
        f"   --norm-epsilon {text['rms_norm_eps']}",
        f"   --rotary-percent {rotary_percent}",
        "   --swiglu",
        *(["   --untie-embeddings-and-output-weights"] if untie else []),
        f"   --vocab-size {text['vocab_size']}",
        "",
        f"   --rotary-base {int(rope_theta)}",
        "",
        "   # Qwen3.5 specific",
        "   --attention-output-gate",
        ")",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
