#!/bin/bash
# Dev-tier RMCT bring-up: Qwen3.5-2B, 1 GPU, synchronous loop (Phase 3).
# Run inside the slime container on the dev pod. Prerequisites:
#   /workspace/rmct/                  this experiment dir (rsynced from the repo)
#   /workspace/models/Qwen3.5-2B      HF checkpoint (huggingface-cli download)
#   /workspace/models/Qwen3.5-2B_torch_dist   from tools/convert_hf_to_torch_dist.py
#   RMCT_CONFIG=/workspace/rmct/configs/dev_smoke.json (or override)
#
# Stage selection (Phase 3 steps): PHASE=rollout|train|full (default full)
#   rollout — --debug-rollout-only: exercises the RMCT rollout fn + classifier
#   train   — --debug-train-only with --load-debug-rollout-data
#   full    — 5-step synchronous loop

set -ex
export PYTHONUNBUFFERED=1

RMCT_DIR=${RMCT_DIR:-/workspace/rmct}
MODEL_DIR=${MODEL_DIR:-/workspace/models/Qwen3.5-2B}
RUN_DIR=${RUN_DIR:-/workspace/runs/dev_smoke}
export RMCT_CONFIG=${RMCT_CONFIG:-${RMCT_DIR}/configs/dev_smoke.json}
PHASE=${PHASE:-full}
SLIME_DIR=${SLIME_DIR:-/root/slime}

pkill -9 sglang 2>/dev/null || true
sleep 2
ray stop --force 2>/dev/null || true
sleep 2

# Megatron model args derived from the checkpoint's own config.json.
python3 "${RMCT_DIR}/scripts/derive_model_args.py" "${MODEL_DIR}" > /tmp/qwen3.5-2B-args.sh
source /tmp/qwen3.5-2B-args.sh

# Resume: slime derives start_rollout_id from the checkpoint loaded via
# --load. A fresh run loads the base conversion (whose iteration metadata is
# 1, so pin --start-rollout-id 0 or the first generation is silently
# skipped); a restart loads the run's own save dir and continues.
if [ -f "${RUN_DIR}/checkpoints/latest_checkpointed_iteration.txt" ]; then
   LOAD_DIR="${RUN_DIR}/checkpoints"
   START_ARGS=()
else
   LOAD_DIR="${MODEL_DIR}_torch_dist"
   START_ARGS=(--start-rollout-id 0)
fi

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}"
   --ref-load "${MODEL_DIR}_torch_dist"     # frozen base = KL reference
   --load "${LOAD_DIR}"
   "${START_ARGS[@]}"
   --save "${RUN_DIR}/checkpoints"
   --save-interval 2                        # exercise checkpointing in the smoke
)

# RMCT replaces the generic GRPO rollout/advantage path entirely.
RMCT_ARGS=(
   --rollout-function-path slime_port.rmct_rollout.generate_rollout
   --custom-advantage-function-path slime_port.rmct_advantage.compute_advantages
   --advantage-estimator grpo               # bypassed by the custom function
   --disable-rewards-normalization          # sample.reward IS the advantage — hands off
   --loss-type policy_loss                  # PPO clip, matching ctm loss_fn: ppo
   --eps-clip 0.2
   # ctm's ppo_loss is a global token mean over the step (Σ mask·surrogate / Σ mask).
   # slime's default is sum-of-sample-means / global_batch_size, which with our
   # --global-batch-size 1 is an unnormalized SUM over samples. Per-token mode
   # reproduces ctm's reduction exactly (deviation D6, closed).
   --calculate-per-token-loss
   # KL lives inside the custom advantage fn (tinker semantics, kl_coef from
   # RMCT_CONFIG). slime's own KL contributions stay at zero — but
   # --use-kl-loss with coef 0 is required: slime only loads the ref model
   # (and computes ref_log_probs) when kl_coef != 0 or use_kl_loss is set.
   --kl-coef 0.0
   --use-kl-loss
   --kl-loss-coef 0.0
)

ROLLOUT_ARGS=(
   # data flows through RMCT_CONFIG; --prompt-data only feeds slime's (unused)
   # buffer plumbing.
   --prompt-data "${RMCT_DIR}/data/dev_smoke/suggested-answer-pairs.jsonl"
   --input-key unbiased_messages
   --apply-chat-template
   --rollout-global-dataset
   --num-rollout 5                          # 5 full steps for the smoke loop
   --rollout-batch-size 4                   # datapoints per step (matches config batch_size)
   --n-samples-per-prompt 8                 # matches n_train_rollouts (dev smoke)
   --rollout-max-response-len 1024          # matches config max_new_tokens
   --rollout-temperature 1.0
   # One rollout group per generation (all samples share rollout_id), so one
   # optimizer step per generation regardless of the parse-dependent sample
   # count — matching ctm's step structure.
   --global-batch-size 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.95
   --use-distributed-optimizer
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1          # gen_tp>1 garbles Qwen3.5-dense (sglang#19393)
   --sglang-mem-fraction-static 0.5         # colocated with training on 1 GPU
   --sglang-mamba-radix-cache-strategy extra_buffer
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --colocate
)

case "${PHASE}" in
  rollout) DEBUG_ARGS=(--debug-rollout-only --save-debug-rollout-data "${RUN_DIR}/debug/rollout_{rollout_id}.pt") ;;
  train)   DEBUG_ARGS=(--debug-train-only --load-debug-rollout-data "${RUN_DIR}/debug/rollout_{rollout_id}.pt") ;;
  full)    DEBUG_ARGS=() ;;
  *) echo "unknown PHASE=${PHASE}"; exit 1 ;;
esac

export MASTER_ADDR=127.0.0.1
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus 1 --disable-usage-stats

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${RMCT_DIR}\",
    \"RMCT_CONFIG\": \"${RMCT_CONFIG}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 "${SLIME_DIR}/train.py" \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 1 \
   --rollout-num-gpus 1 \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${RMCT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${DEBUG_ARGS[@]}"
