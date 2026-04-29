#!/bin/bash

# JEPA-only LoRA SQA3D training: SigLIP vision tower is bypassed; visual tokens
# come solely from pre-extracted 3D-JEPA features via jepa_projector (256 -> hidden).

# Set up the data folder
IMAGE_FOLDER="data"
VIDEO_FOLDER="data"
DATA_YAML="scripts/3d/train/sqa3d.yaml"  # Update this path to your SQA3D training yaml
JEPA_FEATURE_FOLDER="data/3d-jepa-features"

############### Prepare Envs #################
# python3 -m pip install flash-attn --no-build-isolation
alias python=python3
############### Show Envs ####################

nvidia-smi

################ SQA3D LoRA JEPA-only Training ################

LLM_VERSION="Qwen/Qwen2-7B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

PROMPT_VERSION="qwen_1_5"
MID_RUN_NAME="llavanext-qwen-video3dllm-sqa3d-lora-jepaonly-r64"
PREV_STAGE_CHECKPOINT="data/models/LLaVA-Video-7B-Qwen2"

NUM_GPUS=8
BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=$((BATCH_SIZE/NUM_GPUS))

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nnodes=1 --nproc_per_node="${NUM_GPUS}" --master_port 43001 \
    llava/train/train_3d.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path $DATA_YAML \
    --image_folder $IMAGE_FOLDER \
    --video_folder $VIDEO_FOLDER \
    --embodiedscan_folder data/embodiedscan/ \
    --jepa_feature_folder $JEPA_FEATURE_FOLDER \
    --use_jepa_only True \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --mm_tunable_parts="mm_language_model" \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir ./ckpt/$MID_RUN_NAME \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --mm_newline_position grid \
    --add_spatial_instruction True \
    --force_sample True \
    --mm_spatial_pool_stride 2 \
    --world_position_embedding_type avg-sin3d \
    --object_feature_type patch14-pe \
    --ground_head_type infonce \
    --group_by_task_length True \
    --frame_sampling_strategy uniform \
    --frames_upbound 32 \
    > "./ckpt/${MID_RUN_NAME}.log" 2>&1
exit 0;
