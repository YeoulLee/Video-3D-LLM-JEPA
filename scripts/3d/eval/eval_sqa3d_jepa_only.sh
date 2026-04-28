#!/bin/bash

# Evaluation for the JEPA-only LoRA SQA3D run. The use_jepa_only flag is persisted
# in the trained checkpoint's config.json, so the model rebuild during inference
# automatically takes the JEPA-only path.

export python3WARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

# Uncomment to make CUDA errors synchronous (slow; only for debugging illegal-memory-access).
# export CUDA_LAUNCH_BLOCKING=1

# Usage: sh scripts/3d/eval/eval_sqa3d_jepa_only.sh <ckpt_dir_under_./ckpt> <frame_sampling_strategy> <max_frame_num>
# Example: sh scripts/3d/eval/eval_sqa3d_jepa_only.sh llavanext-qwen-video3dllm-sqa3d-lora-jepaonly uniform 32

BASE_MODEL="data/models/LLaVA-Video-7B-Qwen2"
CKPT="./ckpt/$1"
ANWSER_FILE="results/sqa3d/$1.jsonl"
JEPA_FEATURE_FOLDER="data/3d-jepa-features"

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
    --max_frame_num $3

python llava/eval/eval_sqa3d.py --input-file $ANWSER_FILE
