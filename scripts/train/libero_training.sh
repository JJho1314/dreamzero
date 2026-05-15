#!/bin/bash
# DreamZero LIBERO Training Script (Wan2.2-TI2V-5B backbone, 2 cameras, delta-EEF action)
#
# Pretrained weights expected at:
#   $WAN22_CKPT_DIR: Wan2.2_VAE.pth, models_t5_umt5-xxl-enc-bf16.pth, diffusion safetensors
#   $IMAGE_ENCODER_DIR: models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
#   $TOKENIZER_DIR: umt5-xxl tokenizer files

export HYDRA_FULL_ERROR=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DREAMZERO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$DREAMZERO_ROOT"

# ============ USER CONFIGURATION ============
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
NUM_GPUS=${NUM_GPUS:-8}

LIBERO_GOAL=${LIBERO_GOAL:-"/data/user/jhe724/workspace/data/libero_fastwam/libero_goal_no_noops_lerobot"}
LIBERO_OBJECT=${LIBERO_OBJECT:-"/data/user/jhe724/workspace/data/libero_fastwam/libero_object_no_noops_lerobot"}
LIBERO_SPATIAL=${LIBERO_SPATIAL:-"/data/user/jhe724/workspace/data/libero_fastwam/libero_spatial_no_noops_lerobot"}
LIBERO_10=${LIBERO_10:-"/data/user/jhe724/workspace/data/libero_fastwam/libero_10_no_noops_lerobot"}

OUTPUT_DIR=${OUTPUT_DIR:-"$DREAMZERO_ROOT/checkpoints/dreamzero_libero_wan22_full"}

WAN22_CKPT_DIR=${WAN22_CKPT_DIR:-"/data/user/jhe724/workspace/weights/TI2V_5B"}
IMAGE_ENCODER_DIR=${IMAGE_ENCODER_DIR:-"/data/user/jhe724/workspace/weights/Wan2.1-FLF2V-14B-720P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"$WAN22_CKPT_DIR/google/umt5-xxl"}

export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_BASE_URL=${WANDB_BASE_URL:-"http://10.12.1.245:8080"}
WANDB_PROJECT=${WANDB_PROJECT:-dreamzero-libero}

PER_DEVICE_BS=${PER_DEVICE_BS:-${PER_DEVICE_TRAIN_BATCH_SIZE:-1}}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}

if ! [[ "$PER_DEVICE_BS" =~ ^[0-9]+$ ]] || [ "$PER_DEVICE_BS" -lt 1 ]; then
    echo "ERROR: PER_DEVICE_BS must be a positive integer, got: $PER_DEVICE_BS"
    exit 1
fi
if ! [[ "$GLOBAL_BATCH_SIZE" =~ ^[0-9]+$ ]] || [ "$GLOBAL_BATCH_SIZE" -lt 1 ]; then
    echo "ERROR: GLOBAL_BATCH_SIZE must be a positive integer, got: $GLOBAL_BATCH_SIZE"
    exit 1
fi
# =============================================

for f in \
    "$WAN22_CKPT_DIR/Wan2.2_VAE.pth" \
    "$WAN22_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
    "$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    "$TOKENIZER_DIR/spiece.model" ; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing required file: $f"
        exit 1
    fi
done

for d in "$LIBERO_GOAL" "$LIBERO_OBJECT" "$LIBERO_SPATIAL" "$LIBERO_10"; do
    if [ ! -d "$d" ]; then
        echo "ERROR: Dataset not found: $d"
        exit 1
    fi
    if [ ! -f "$d/meta/embodiment.json" ]; then
        echo "ERROR: $d/meta/embodiment.json missing; run scripts/data/convert_lerobot_to_gear.py first."
        exit 1
    fi
done

EXPERIMENT_PY="$DREAMZERO_ROOT/groot/vla/experiment/experiment.py"

PYTHON_BIN="${PYTHON_BIN:-/data/user/jhe724/.conda/envs/dreamzero/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN=$(command -v python3)
fi

TRAIN_ARCHITECTURE=${TRAIN_ARCHITECTURE:-full}
SAVE_LORA_ONLY=${SAVE_LORA_ONLY:-false}
TUNE_PROJECTOR=${TUNE_PROJECTOR:-true}
TUNE_DIFFUSION_MODEL=${TUNE_DIFFUSION_MODEL:-true}
SAVE_STEPS=${SAVE_STEPS:-5000}
SAVE_STRATEGY=${SAVE_STRATEGY:-steps}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
DATALOADER_PIN_MEMORY=${DATALOADER_PIN_MEMORY:-true}
DATASET_SHARD_SAMPLING_RATE=${DATASET_SHARD_SAMPLING_RATE:-0.5}
USE_GRADIENT_CHECKPOINTING=${USE_GRADIENT_CHECKPOINTING:-true}
NUM_FRAMES=${NUM_FRAMES:-33}
NUM_VIEWS=${NUM_VIEWS:-2}
DEEPSPEED_CFG=${DEEPSPEED_CFG:-zero2}
ACTION_HEAD_CONFIG=${ACTION_HEAD_CONFIG:-wan_flow_matching_action_tf_wan22}
TRANSFORM_CONFIG=${TRANSFORM_CONFIG:-dreamzero_cotrain_libero}

USE_FRAME_CACHE=${USE_FRAME_CACHE:-true}
if [ -z "${FRAME_CACHE_SIZE:-}" ]; then
    if [ "$ACTION_HEAD_CONFIG" = "wan_flow_matching_action_tf_wan22_224" ]; then
        FRAME_CACHE_SIZE=224
    else
        FRAME_CACHE_SIZE=160
    fi
fi
if [ "$USE_FRAME_CACHE" = "true" ]; then
    FRAME_CACHE_ROOT=${FRAME_CACHE_ROOT:-${DREAMZERO_FRAME_CACHE_ROOT:-/data/user/jhe724/workspace/data/libero_fastwam_frame_cache_${FRAME_CACHE_SIZE}}}
else
    FRAME_CACHE_ROOT=${FRAME_CACHE_ROOT:-${DREAMZERO_FRAME_CACHE_ROOT:-}}
fi
FRAME_CACHE_SOURCE_ROOT=${FRAME_CACHE_SOURCE_ROOT:-${DREAMZERO_FRAME_CACHE_SOURCE_ROOT:-/data/user/jhe724/workspace/data/libero_fastwam}}
if [ -n "$FRAME_CACHE_ROOT" ]; then
    IMAGE_RESOLUTION_WIDTH=${IMAGE_RESOLUTION_WIDTH:-${FRAME_CACHE_SIZE}}
    IMAGE_RESOLUTION_HEIGHT=${IMAGE_RESOLUTION_HEIGHT:-${FRAME_CACHE_SIZE}}
else
    IMAGE_RESOLUTION_WIDTH=${IMAGE_RESOLUTION_WIDTH:-224}
    IMAGE_RESOLUTION_HEIGHT=${IMAGE_RESOLUTION_HEIGHT:-224}
fi
if [ -n "$FRAME_CACHE_ROOT" ]; then
    if [ ! -d "$FRAME_CACHE_ROOT" ]; then
        echo "ERROR: Frame cache not found: $FRAME_CACHE_ROOT"
        echo "       Run scripts/data/precompute_fastwam_frame_cache.sbatch first, or set USE_FRAME_CACHE=false."
        exit 1
    fi
    export DREAMZERO_FRAME_CACHE_ROOT="$FRAME_CACHE_ROOT"
    export DREAMZERO_FRAME_CACHE_SOURCE_ROOT="$FRAME_CACHE_SOURCE_ROOT"
    export DREAMZERO_DISABLE_VIDEO_CROP=${DREAMZERO_DISABLE_VIDEO_CROP:-true}
    export DREAMZERO_DISABLE_VIDEO_RESIZE=${DREAMZERO_DISABLE_VIDEO_RESIZE:-true}
fi

echo "Using NUM_GPUS=$NUM_GPUS PER_DEVICE_BS=$PER_DEVICE_BS GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE MAX_STEPS=${MAX_STEPS:-30000} TRAIN_ARCHITECTURE=$TRAIN_ARCHITECTURE SAVE_LORA_ONLY=$SAVE_LORA_ONLY TUNE_DIFFUSION_MODEL=$TUNE_DIFFUSION_MODEL USE_GRADIENT_CHECKPOINTING=$USE_GRADIENT_CHECKPOINTING SAVE_STRATEGY=$SAVE_STRATEGY SAVE_STEPS=$SAVE_STEPS DATALOADER_NUM_WORKERS=$DATALOADER_NUM_WORKERS DATASET_SHARD_SAMPLING_RATE=$DATASET_SHARD_SAMPLING_RATE NUM_FRAMES=$NUM_FRAMES DEEPSPEED_CFG=$DEEPSPEED_CFG ACTION_HEAD_CONFIG=$ACTION_HEAD_CONFIG TRANSFORM_CONFIG=$TRANSFORM_CONFIG IMAGE_RESOLUTION=${IMAGE_RESOLUTION_HEIGHT}x${IMAGE_RESOLUTION_WIDTH} FRAME_CACHE_ROOT=${FRAME_CACHE_ROOT:-none}"

"$PYTHON_BIN" -m torch.distributed.run --nproc_per_node "$NUM_GPUS" --standalone "$EXPERIMENT_PY" \
    report_to=wandb \
    data=dreamzero/libero_relative \
    wandb_project=${WANDB_PROJECT} \
    train_architecture=${TRAIN_ARCHITECTURE} \
    image_resolution_width=${IMAGE_RESOLUTION_WIDTH} \
    image_resolution_height=${IMAGE_RESOLUTION_HEIGHT} \
    num_frames=${NUM_FRAMES} \
    action_horizon=24 \
    num_views=${NUM_VIEWS} \
    model=dreamzero/vla \
    model/dreamzero/action_head=${ACTION_HEAD_CONFIG} \
    action_head_cfg.config.tune_projector=${TUNE_PROJECTOR} \
    action_head_cfg.config.tune_diffusion_model=${TUNE_DIFFUSION_MODEL} \
    action_head_cfg.config.use_gradient_checkpointing=${USE_GRADIENT_CHECKPOINTING} \
    model/dreamzero/transform=${TRANSFORM_CONFIG} \
    dataset_shard_sampling_rate=${DATASET_SHARD_SAMPLING_RATE} \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    training_args.deepspeed="groot/vla/configs/deepspeed/${DEEPSPEED_CFG}.json" \
    save_steps=${SAVE_STEPS} \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=${PER_DEVICE_BS} \
    global_batch_size=${GLOBAL_BATCH_SIZE} \
    max_steps=${MAX_STEPS:-30000} \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=${DATALOADER_PIN_MEMORY} \
    dataloader_num_workers=${DATALOADER_NUM_WORKERS} \
    save_lora_only=${SAVE_LORA_ONLY} \
    max_chunk_size=4 \
    save_strategy=${SAVE_STRATEGY} \
    libero_data_root_goal=$LIBERO_GOAL \
    libero_data_root_object=$LIBERO_OBJECT \
    libero_data_root_spatial=$LIBERO_SPATIAL \
    libero_data_root_10=$LIBERO_10 \
    dit_version=$WAN22_CKPT_DIR \
    text_encoder_pretrained_path=$WAN22_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN22_CKPT_DIR/Wan2.2_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR
