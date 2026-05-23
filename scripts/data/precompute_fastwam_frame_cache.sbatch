#!/bin/bash
#SBATCH --job-name=dz_fcache
#SBATCH --account=d_haoangli_team
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/data/user/jhe724/workspace/dreamzero/logs/dz_fcache_%j.log
#SBATCH --error=/data/user/jhe724/workspace/dreamzero/logs/dz_fcache_%j.err

set -euo pipefail

cd /data/user/jhe724/workspace/dreamzero
mkdir -p logs

SOURCE_ROOT=${SOURCE_ROOT:-/data/user/jhe724/workspace/data/libero_fastwam}
CACHE_ROOT=${CACHE_ROOT:-/data/user/jhe724/workspace/data/libero_fastwam_frame_cache_224}
SIZE=${SIZE:-224}
WORKERS=${WORKERS:-12}
PYTHON_BIN=${PYTHON_BIN:-/data/user/jhe724/.conda/envs/dreamzero/bin/python}

echo "SOURCE_ROOT=$SOURCE_ROOT"
echo "CACHE_ROOT=$CACHE_ROOT"
echo "SIZE=$SIZE WORKERS=$WORKERS"

"$PYTHON_BIN" scripts/data/precompute_frame_cache.py \
    --source-root "$SOURCE_ROOT" \
    --cache-root "$CACHE_ROOT" \
    --size "$SIZE" \
    --workers "$WORKERS"
