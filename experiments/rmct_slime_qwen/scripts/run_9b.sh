#!/bin/bash
# Run-tier RMCT science run: Qwen3.5-9B full-parameter, 4 GPUs (Phase 4).
# DO NOT launch without operator approval — this is a paid science run.
#
# Prerequisites on the pod:
#   /workspace/rmct/                          this experiment dir (sync_to_pod.sh)
#   /workspace/models/Qwen3.5-9B              HF checkpoint (hf download)
#   /workspace/models/Qwen3.5-9B_torch_dist   tools/convert_hf_to_torch_dist.py
#   /workspace/rmct/data/run_9b/wrong-argument-pairs.jsonl  frozen training artifact
#   RMCT_CONFIG=/workspace/rmct/configs/run_9b.json
#
# Topology (starting hypothesis per AGENT_PROMPT Phase 4; validate against
# measured memory before the long run): TP2 (NVLink), PP1, DP=2, colocated
# SGLang engines at gen_tp=1 (sglang#19393: gen_tp>1 garbles Qwen3.5-dense).

set -ex
export PYTHONUNBUFFERED=1

RMCT_DIR=${RMCT_DIR:-/workspace/rmct}
MODEL_DIR=${MODEL_DIR:-/workspace/models/Qwen3.5-9B}
RUN_DIR=${RUN_DIR:-/workspace/runs/run_9b}
export RMCT_CONFIG=${RMCT_CONFIG:-${RMCT_DIR}/configs/run_9b.json}
NUM_GPUS=${NUM_GPUS:-4}
SLIME_DIR=${SLIME_DIR:-/root/slime}

pkill -9 sglang 2>/dev/null || true
sleep 2
ray stop --force 2>/dev/null || true
sleep 2

python3 "${RMCT_DIR}/scripts/derive_model_args.py" "${MODEL_DIR}" > /tmp/qwen3.5-9B-args.sh
source /tmp/qwen3.5-9B-args.sh

# Resume: slime derives start_rollout_id from the checkpoint loaded via
# --load. Fresh runs load the base conversion (iteration metadata 1 — pin
# --start-rollout-id 0 or generation 0 is silently skipped); restarts load
# the run's own save dir and continue where they left off.
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
   --save-interval 8                        # paper checkpoint_every: 8
)

RMCT_ARGS=(
   --rollout-function-path slime_port.rmct_rollout.generate_rollout
   --custom-advantage-function-path slime_port.rmct_advantage.compute_advantages
   --advantage-estimator grpo               # bypassed by the custom function
   --disable-rewards-normalization
   --loss-type policy_loss                  # paper loss: ppo
   --eps-clip 0.2
   --calculate-per-token-loss               # match ctm's global-token-mean reduction (D6)
   # KL lives in the custom advantage fn (tinker semantics, kl_coef 0.05 from
   # RMCT_CONFIG). --use-kl-loss with coef 0 only forces ref-model loading.
   --kl-coef 0.0
   --use-kl-loss
   --kl-loss-coef 0.0
)

ROLLOUT_ARGS=(
   --prompt-data "${RMCT_DIR}/data/run_9b/wrong-argument-pairs-64.jsonl"
   --input-key unbiased_messages
   --apply-chat-template
   --rollout-global-dataset
   --num-rollout 16                         # 64 datapoints / batch 4 = 16 steps/epoch
   --rollout-batch-size 4                   # paper batch_size: 4
   --n-samples-per-prompt 128               # paper training rollouts: 128
   --rollout-max-response-len 20480         # paper max_new_tokens — never cut
   --rollout-temperature 1.0
   --global-batch-size 1                    # one optimizer step per generation
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --pipeline-model-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 24576
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5                                # D7: paper LoRA 2.86e-4 / ~30 ≈ full-param 1e-5
   --lr-decay-style constant                # paper learning_rate_schedule: constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.95
   --use-distributed-optimizer
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1          # gen_tp>1 garbles Qwen3.5-dense
   --sglang-mem-fraction-static 0.5
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

export MASTER_ADDR=127.0.0.1
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats

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
   --actor-num-gpus-per-node "${NUM_GPUS}" \
   --rollout-num-gpus "${NUM_GPUS}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${RMCT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"
