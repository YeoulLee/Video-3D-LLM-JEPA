#!/bin/bash

# Evaluation for the JEPA-only LoRA SQA3D run. The use_jepa_only flag is persisted
# in the trained checkpoint's config.json, so the model rebuild during inference
# automatically takes the JEPA-only path.

export python3WARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

# Uncomment to make CUDA errors synchronous (slow; only for debugging illegal-memory-access).
# export CUDA_LAUNCH_BLOCKING=1

# Usage: sh scripts/3d/eval/eval_sqa3d_jepa_only.sh <ckpt_dir_under_./ckpt> <frame_sampling_strategy> <max_frame_num> [zero]
# Example: sh scripts/3d/eval/eval_sqa3d_jepa_only.sh llavanext-qwen-video3dllm-sqa3d-lora-jepaonly uniform 32
# Visual ablation: pass "zero" as 4th arg to replace JEPA features with zeros.
#   sh ... llavanext-qwen-video3dllm-sqa3d-lora-jepaonly uniform 32 zero
# This writes to a separate "<run>_zero.jsonl" so it doesn't collide with the normal run.

BASE_MODEL="data/models/LLaVA-Video-7B-Qwen2"
CKPT="./ckpt/$1"
JEPA_FEATURE_FOLDER="data/3d-jepa-features"

if [ "$4" = "zero" ]; then
    ANWSER_FILE="results/sqa3d/$1_zero.jsonl"
    ZERO_FLAG="--zero_jepa_features"
    echo "[ablation] zero_jepa_features=True; writing to $ANWSER_FILE"
else
    ANWSER_FILE="results/sqa3d/$1.jsonl"
    ZERO_FLAG=""
fi

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 llava/eval/model_sqa3d.py \
    --model-path $BASE_MODEL \
    --lora-path $CKPT \
    --video-folder ./data \
    --embodiedscan-folder data/embodiedscan \
    --jepa-feature-folder $JEPA_FEATURE_FOLDER \
    --n_gpu 8 \
    --question-file data/processed/sqa3d_test_llava_style.json \
    --conv-mode qwen_1_5 \
    --answer-file $ANWSER_FILE \
    --frame_sampling_strategy $2 \
    --max_frame_num $3 \
    $ZERO_FLAG

python llava/eval/eval_sqa3d.py --input-file $ANWSER_FILE
