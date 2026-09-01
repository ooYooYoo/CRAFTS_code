#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 {build|sft|infer} TEACHER_MODEL_PATH STUDENT_MODEL_PATH [GPU_ID]"
  exit 2
fi

STAGE=$1
TEACHER_MODEL_PATH=$2
STUDENT_MODEL_PATH=$3
GPU_ID=${4:-0}

DATASET=Instruments

SKILL_DIR="artifacts/skills/${DATASET}"
MODEL_DIR="artifacts/models"
ADAPTER_DIR="artifacts/skill_student/qwen3_4b_${DATASET}_lora"
DISTILL_DATA="${SKILL_DIR}/distill/query_user_skill_distill_train.jsonl"

COMMON_ARGS=(
  --dataset "$DATASET"
  --data_path data/clean
  --model_type agent
  --model_name ARS
  --agent true

  --tool_name SASRec
  --tool_type sequential
  --config_file_path tool_conf/default/SASRec.yaml

  # Student / Recommendation Agent
  --local_model_path "$STUDENT_MODEL_PATH"

  # Teacher LLM
  --teacher_model_path "$TEACHER_MODEL_PATH"

  --enable_memskill
  --skill_bank_path "$SKILL_DIR"
  --save_dir "$MODEL_DIR"

  --gpu_id "$GPU_ID"
  --seed 2018
  --topK 1

  --agent_workers 1
  --max_epoch 1

  --user_skill_top_k 2
  --functional_skill_top_k 2

  --skill_update_interval 5
  --max_skill_cases_per_round 5
  --initial_user_skill_budget 200
  --user_skill_cluster_size 6
  --skill_similarity_threshold 0.35
  --skill_update_rounds 1
)

case "$STAGE" in
  build)
    # Qwen3-8B:
    #   - Skill Bank initialization
    #   - Skill Bank evolution
    #   - Teacher trajectory generation
    #
    # Qwen3-4B:
    #   - Recommendation-agent interactions used during training
    python main.py \
      "${COMMON_ARGS[@]}" \
      --load_recommender false \
      --student_lora_path none
    ;;

  sft)
    # Trajectory-supervised LoRA SFT of the Qwen3-4B student.
    if [[ ! -f "$DISTILL_DATA" ]]; then
      echo "Distillation data not found: $DISTILL_DATA" >&2
      echo "Run the build stage first." >&2
      exit 1
    fi

    python train_skill_distill_lora.py \
      --model_path "$STUDENT_MODEL_PATH" \
      --data_path "$DISTILL_DATA" \
      --output_dir "$ADAPTER_DIR"
    ;;

  infer)
    # Frozen Skill Bank + trained retriever + Qwen3-4B student LoRA.
    if [[ ! -d "$ADAPTER_DIR" ]]; then
      echo "Student LoRA adapter not found: $ADAPTER_DIR" >&2
      echo "Run the sft stage first." >&2
      exit 1
    fi

    python main.py \
      "${COMMON_ARGS[@]}" \
      --load_recommender true \
      --student_lora_path "$ADAPTER_DIR" \
      --retriever_checkpoint "${SKILL_DIR}/retriever/projector.pt" \
      --eval_output_dir "artifacts/evaluation/${DATASET}" \
      --test_only
    ;;

  *)
    echo "Unsupported stage: $STAGE" >&2
    exit 2
    ;;
esac
