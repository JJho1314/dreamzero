#!/bin/bash
# =====================================================================
# DreamZero-LIBERO full fine-tuning on HPC3 (8 GPUs, single node)
# =====================================================================
#SBATCH --job-name=dz_libero_full
#SBATCH --account=d_haoangli_team
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=7-00:00:00
#SBATCH --output=/data/user/jhe724/workspace/dreamzero/logs/dz_libero_full_%j.log
#SBATCH --error=/data/user/jhe724/workspace/dreamzero/logs/dz_libero_full_%j.err

set -euo pipefail

ROOT=/data/user/jhe724/workspace/dreamzero
mkdir -p "${ROOT}/logs"
cd "${ROOT}"

echo "=========================================="
echo "JobID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Date: $(date)"
echo "=========================================="
nvidia-smi -L

module purge
module load anaconda3 cuda/12.4 nccl/2.25 || true

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/data/user/jhe724/.cache/huggingface

export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_NVLS_ENABLE=0
export NCCL_TIMEOUT=10000
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_BASE_URL=${WANDB_BASE_URL:-"http://10.12.1.245:8080"}
export WANDB_PROJECT=${WANDB_PROJECT:-dreamzero-libero}
export WANDB_DIR=${WANDB_DIR:-${ROOT}/logs}

export NUM_GPUS=${NUM_GPUS:-8}
export TRAIN_ARCHITECTURE=${TRAIN_ARCHITECTURE:-full}
export SAVE_LORA_ONLY=${SAVE_LORA_ONLY:-false}
export TUNE_PROJECTOR=${TUNE_PROJECTOR:-true}
export TUNE_DIFFUSION_MODEL=${TUNE_DIFFUSION_MODEL:-true}
export USE_GRADIENT_CHECKPOINTING=${USE_GRADIENT_CHECKPOINTING:-true}

export PER_DEVICE_BS=${PER_DEVICE_BS:-4}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
export MAX_STEPS=${MAX_STEPS:-30000}
export MAX_CHUNK_SIZE=${MAX_CHUNK_SIZE:-1}
export NUM_FRAMES=${NUM_FRAMES:-9}
export DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-1}
export DATALOADER_PIN_MEMORY=${DATALOADER_PIN_MEMORY:-true}
export DATASET_SHARD_SAMPLING_RATE=${DATASET_SHARD_SAMPLING_RATE:-0.5}
export ACTION_HEAD_CONFIG=${ACTION_HEAD_CONFIG:-wan_flow_matching_action_tf_wan22_224}
export TRANSFORM_CONFIG=${TRANSFORM_CONFIG:-dreamzero_cotrain_libero}
export DECOUPLE_VIDEO_ACTION_NOISE=${DECOUPLE_VIDEO_ACTION_NOISE:-true}
export DECOUPLE_INFERENCE_NOISE=${DECOUPLE_INFERENCE_NOISE:-false}

RUN_ID=${RUN_ID:-dreamzero_libero_wan22_full_${SLURM_JOB_ID}}
export OUTPUT_DIR=${OUTPUT_DIR:-${ROOT}/checkpoints/${RUN_ID}}

echo "=========================================="
echo "RUN_ID: ${RUN_ID}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "NUM_GPUS: ${NUM_GPUS}"
echo "PER_DEVICE_BS: ${PER_DEVICE_BS}"
echo "GLOBAL_BATCH_SIZE: ${GLOBAL_BATCH_SIZE}"
echo "MAX_STEPS: ${MAX_STEPS}"
echo "MAX_CHUNK_SIZE: ${MAX_CHUNK_SIZE}"
echo "NUM_FRAMES: ${NUM_FRAMES}"
echo "TRAIN_ARCHITECTURE: ${TRAIN_ARCHITECTURE}"
echo "USE_GRADIENT_CHECKPOINTING: ${USE_GRADIENT_CHECKPOINTING}"
echo "DATALOADER_NUM_WORKERS: ${DATALOADER_NUM_WORKERS}"
echo "ACTION_HEAD_CONFIG: ${ACTION_HEAD_CONFIG}"
echo "TRANSFORM_CONFIG: ${TRANSFORM_CONFIG}"
echo "DECOUPLE_VIDEO_ACTION_NOISE: ${DECOUPLE_VIDEO_ACTION_NOISE}"
echo "DECOUPLE_INFERENCE_NOISE: ${DECOUPLE_INFERENCE_NOISE}"
echo "=========================================="

bash scripts/train/libero_training.sh
