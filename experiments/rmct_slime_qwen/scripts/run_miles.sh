#!/bin/bash
# RMCT on Miles (radixark/miles — slime fork with LoRA). One launcher for all
# run-tier work (D10):
#   MODEL_NAME=Qwen3.5-2B|Qwen3.5-9B   (default 9B)
#   PHASE=rollout|train|full           (default full; same semantics as dev script)
#   LORA=1                             paper adapter config r8/alpha16/dropout0
#   NUM_GPUS=1..4                      (default 4; use 1 for the 2B parity tier)
#   NUM_ROLLOUT=N                      generations (default 16 = one epoch at batch 4)
#   TP=1|2                             tensor parallel (default 2 for 9B, 1 otherwise)
#
# Miles differences from the slime dev script, all deliberate:
#   - no --custom-advantage-function-path: Miles removed the hook; the RMCT
#     advantage fn is installed by --custom-megatron-init-path
#     slime_port.miles_init.megatron_init (explicit rebind, D10).
#   - no --rollout-global-dataset: default-on in Miles.
#   - LoRA args are native Miles.

set -ex
export PYTHONUNBUFFERED=1

RMCT_DIR=${RMCT_DIR:-/workspace/rmct}
MODEL_NAME=${MODEL_NAME:-Qwen3.5-9B}
MODEL_DIR=${MODEL_DIR:-/root/models/${MODEL_NAME}}   # local NVMe; /workspace is network-fs
RUN_DIR=${RUN_DIR:?set RUN_DIR explicitly (fresh dir per experiment)}
export RMCT_CONFIG=${RMCT_CONFIG:?set RMCT_CONFIG explicitly}
PHASE=${PHASE:-full}
NUM_GPUS=${NUM_GPUS:-4}
NUM_ROLLOUT=${NUM_ROLLOUT:-16}
if [ -z "${TP:-}" ]; then [ "${MODEL_NAME}" = "Qwen3.5-9B" ] && TP=2 || TP=1; fi
if [ -z "${SLIME_DIR:-}" ]; then
    for candidate in /root/miles /opt/miles /root/slime; do
        [ -f "${candidate}/train.py" ] && SLIME_DIR="${candidate}" && break
    done
fi
SLIME_DIR=${SLIME_DIR:?no framework dir with train.py found}

pkill -9 sglang 2>/dev/null || true
sleep 2
ray stop --force 2>/dev/null || true
sleep 2

SHIPPED_ARGS="${SLIME_DIR}/scripts/models/$(echo "${MODEL_NAME}" | sed 's/^Q/q/').sh"
if [ -f "${SHIPPED_ARGS}" ]; then
    source "${SHIPPED_ARGS}"
else
    python3 "${RMCT_DIR}/scripts/derive_model_args.py" "${MODEL_DIR}" > "/tmp/${MODEL_NAME}-args.sh"
    source "/tmp/${MODEL_NAME}-args.sh"
fi

# Resume-aware load (see run_dev_2b.sh: base conversion carries iteration=1,
# so fresh runs must pin --start-rollout-id 0).
if [ -n "${LOAD_DIR_OVERRIDE:-}" ]; then
   # e.g. the HF checkpoint dir: bridge-built LoRA models cannot load the
   # spec-converted torch_dist (parameter structure differs — silent zero
   # load, uniform logits); bridge mode supports loading HF directly.
   LOAD_DIR="${LOAD_DIR_OVERRIDE}"
   START_ARGS=(--start-rollout-id 0)
elif [ -f "${RUN_DIR}/checkpoints/latest_checkpointed_iteration.txt" ]; then
   LOAD_DIR="${RUN_DIR}/checkpoints"
   START_ARGS=()
else
   LOAD_DIR="${MODEL_DIR}_torch_dist"
   START_ARGS=(--start-rollout-id 0)
fi

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}"
   --ref-load "${MODEL_DIR}_torch_dist"     # frozen base = KL reference
   "${START_ARGS[@]}"
)
if [ "${LORA:-0}" = "1" ] && [ ! -f "${RUN_DIR}/checkpoints/latest_checkpointed_iteration.txt" ] && [ -z "${LOAD_DIR_OVERRIDE:-}" ]; then
   # Fresh LoRA runs: NO --load. The bridge-built model self-loads HF weights
   # at construction (patch P3); the Megatron checkpoint-load path cannot map
   # this model and silently zeroes the language side. --load returns for
   # resume (adapter checkpoints).
   :
else
   CKPT_ARGS+=(--load "${LOAD_DIR}")
fi
if [ "${NOSAVE:-0}" != "1" ]; then
   # NOSAVE=1 for benchmark/diagnostic runs: skips checkpointing entirely.
   CKPT_ARGS+=(--save "${RUN_DIR}/checkpoints" --save-interval "${SAVE_INTERVAL:-8}")
   # EVAL_CKPT=1: weights-only checkpoints (no optimizer/rng state). LoRA
   # runs already save only the ~90MB adapter; for full-param this cuts a
   # 47GB (4B) / ~100GB (9B) save to just the weights — right for the
   # paper's checkpoint_every=8 EVALUATION checkpoints. NOT resumable:
   # pair with a rare full save when crash-resume matters.
   if [ "${EVAL_CKPT:-0}" = "1" ]; then
      CKPT_ARGS+=(--no-save-optim --no-save-rng)
   fi
fi

LORA_ARGS=()
if [ "${LORA:-0}" = "1" ]; then
   # Qwen3.5 LoRA constraints, from Miles' own run_qwen3_5_35b_a3b_lora.py:
   #   - weight sync only works through bridge mode (raw mode exports zero chunks)
   #   - target modules must be explicit wildcards under decoder.layers.* (keeps
   #     adapters off MTP/vision, which have no export mapping)
   #   - megatron-core GatedDeltaNet rejects packed sequences -> bshd
   # Coverage = paper's attention+MLP (r8/alpha16/dropout0), translated to the
   # hybrid arch: attention linear_qkv/linear_proj, DeltaNet in_proj/out_proj,
   # dense MLP linear_fc1/linear_fc2. No unembed, matching the paper.
   L="language_model.decoder.layers.*"
   # MLP-only (D11): SGLang's LoRA memory pool assumes one buffer shape per
   # module type across layers; Qwen3.5's hybrid attention/GDN layers violate
   # that (fused qkv 6144 vs GDN in_proj 10240 rows) — adapters on those
   # modules load misaligned and garble generation (verified with a zero
   # adapter repro). MLP shapes are uniform, so MLP-only serves correctly.
   # Paper coverage was attn+mlp; revisit when SGLang fixes hybrid LoRA.
   DEFAULT_TARGETS="${L}.mlp.linear_fc1,${L}.mlp.linear_fc2"
   LORA_ARGS=(
      --lora-rank 8
      --lora-alpha 16
      --lora-dropout 0.0
      --target-modules "${TARGET_MODULES:-${DEFAULT_TARGETS}}"
      --megatron-to-hf-mode bridge
      # Miles' 35B-A3B example mandates bshd for GDN+LoRA, but full-param
      # trains correctly under thd on dense Qwen3.5 — QKV_FORMAT=thd tests
      # whether LoRA can avoid the broken bshd actor-forward path entirely.
      --qkv-format "${QKV_FORMAT:-bshd}"
      # Enables skip_base_sync: the frozen base is NEVER pushed to SGLang
      # (it already serves the pristine HF checkpoint; a CPU backup restores
      # it across colocate sleep/wake). Critical here because Miles' mbridge
      # base export for dense Qwen3.5 is broken (garbles weights) — with this
      # flag only adapters flow through export_adapter_weights.
      --lora-base-cpu-backup
   )
   # LoRA resume: Miles' adapter-only saves (iter_*/adapter/) are NOT loadable
   # via --load — the generic Megatron loader raises "unknown checkpoint
   # format" (and without latest_checkpointed_iteration.txt, which adapter
   # saves never write, --load silently starts fresh). The intended path is
   # --lora-adapter-path pointing at the adapter dir; it restores adapter
   # weights AND optimizer/scheduler state from training_state_rank*.pt.
   # Requires apply_patches.py P8 (upstream consumes the flag inside the
   # load_checkpoint branch that P5 must skip). If the resume run's total
   # iteration count differs from the saved one (e.g. resuming at a reduced
   # shape), add EXTRA_ARGS=--override-opt-param-scheduler.
   if [ -n "${LORA_ADAPTER_PATH:-}" ]; then
      LORA_ARGS+=(--lora-adapter-path "${LORA_ADAPTER_PATH}")
   fi
fi

RMCT_ARGS=(
   --rollout-function-path slime_port.rmct_rollout.generate_rollout
   --custom-megatron-init-path slime_port.miles_init.megatron_init
   --advantage-estimator grpo               # bypassed by the installed RMCT fn
   --disable-rewards-normalization          # sample.reward IS the advantage
   --loss-type policy_loss                  # paper loss: ppo
   --eps-clip 0.2
   --calculate-per-token-loss               # ctm's global-token-mean reduction (D6)
   # KL lives in the RMCT advantage fn; slime KL paths pinned to zero, but
   # --use-kl-loss forces ref-model loading (needed for ref_log_probs).
   --kl-coef 0.0
   --use-kl-loss
   --kl-loss-coef 0.0
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA:-${RMCT_DIR}/data/run_9b/wrong-argument-pairs-64.jsonl}"
   --input-key unbiased_messages
   --apply-chat-template
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH:-4}"
   --n-samples-per-prompt "${N_SAMPLES:-128}"
   --rollout-max-response-len "${MAX_RESPONSE_LEN:-20480}"
   --rollout-temperature 1.0
   # The rollout fn emits exactly rollout_batch x n_train samples per
   # generation (skipped rollouts carry zero loss masks), so this is one
   # optimizer step per generation and divisible by any DP size.
   --global-batch-size "$(( ${ROLLOUT_BATCH:-4} * ${N_SAMPLES:-128} ))"
)

PERF_ARGS=(
   --tensor-model-parallel-size "${TP}"
   --pipeline-model-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
)
if { [ "${LORA:-0}" = "1" ] && [ "${QKV_FORMAT:-bshd}" = "bshd" ]; } || [ "${BSHD:-0}" = "1" ]; then
   # bshd forbids dynamic batching.
   PERF_ARGS+=(--micro-batch-size "${MICRO_BATCH:-1}")
else
   PERF_ARGS+=(--use-dynamic-batch-size --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-24576}")
fi

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR:-1e-5}"                       # D7: operator-approved 1e-5 constant
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.95
   --use-distributed-optimizer
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1          # gen_tp>1 garbles Qwen3.5-dense
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION:-0.5}"
   --sglang-mamba-radix-cache-strategy extra_buffer
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)
# COLOCATE=0: disaggregated actor/rollout GPUs (no sleep/wake offload cycle).
if [ "${COLOCATE:-1}" = "1" ]; then
   MISC_ARGS+=(--colocate)
   ACTOR_GPUS="${NUM_GPUS}"
   ROLLOUT_GPUS="${NUM_GPUS}"
else
   ACTOR_GPUS="${ACTOR_GPUS:-$(( NUM_GPUS / 2 ))}"
   ROLLOUT_GPUS="${ROLLOUT_GPUS:-$(( NUM_GPUS - ACTOR_GPUS ))}"
fi

# Escape hatch for one-off experiment flags (space-separated).
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"

case "${PHASE}" in
  rollout) DEBUG_ARGS=(--debug-rollout-only --save-debug-rollout-data "${RUN_DIR}/debug/rollout_{rollout_id}.pt") ;;
  train)   DEBUG_ARGS=(--debug-train-only --load-debug-rollout-data "${RUN_DIR}/debug/rollout_{rollout_id}.pt") ;;
  full)    DEBUG_ARGS=() ;;
  *) echo "unknown PHASE=${PHASE}"; exit 1 ;;
esac

export MASTER_ADDR=127.0.0.1
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats

MEGATRON_DIR=${MEGATRON_DIR:-/root/Megatron-LM}
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_DIR}/:${RMCT_DIR}\",
    \"RMCT_CONFIG\": \"${RMCT_CONFIG}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 "${SLIME_DIR}/train.py" \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${ACTOR_GPUS}" \
   --rollout-num-gpus "${ROLLOUT_GPUS}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${LORA_ARGS[@]}" \
   "${RMCT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${DEBUG_ARGS[@]}" \
   "${EXTRA[@]}"
