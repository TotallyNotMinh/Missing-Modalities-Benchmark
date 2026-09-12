#!/usr/bin/env bash
set -e

# ==============================================================================
# mmFormer Baseline Training Pipeline for BraTS 2020
# Architecture: Option A (36.65M parameters, basic_dims=8)
# Target GPU  : Physical GPU 2 (CUDA_VISIBLE_DEVICES=2)
# ==============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# 1. Python binary detection
PYTHON_BIN="${CONDA_PREFIX}/bin/python"
if [ ! -f "${PYTHON_BIN}" ]; then
    PYTHON_BIN="python"
fi

# 2. Select physical GPU (Default to GPU 2 for student14@ict20)
TARGET_GPU="${CUDA_VISIBLE_DEVICES:-2}"
export CUDA_VISIBLE_DEVICES="${TARGET_GPU}"

# 3. Check for smoke-test mode
SMOKE_TEST_FLAG=""
if [[ "$1" == "--smoke-test" || "$1" == "-s" ]]; then
    SMOKE_TEST_FLAG="--smoke_test"
    echo "================================================================="
    echo " Running mmFormer Pre-Flight Smoke Test (1 Epoch, 3 Batches)"
    echo " GPU ID       : ${TARGET_GPU}"
    echo "================================================================="
else
    echo "================================================================="
    echo " Launching Official mmFormer 1000-Epoch Training Run"
    echo " GPU ID       : ${TARGET_GPU}"
    echo " Batch Size   : 1"
    echo " Learning Rate: 2e-4"
    echo " Checkpoints  : checkpoints/mmformer"
    echo " Log File     : results/logs/mmformer_training.log"
    echo "================================================================="
fi

mkdir -p results/logs checkpoints/mmformer

"${PYTHON_BIN}" scripts/train_mmformer.py \
    --device "cuda:0" \
    --epochs 1000 \
    --batch_size 1 \
    --lr 2e-4 \
    --weight_decay 1e-4 \
    --val_interval 20 \
    --num_workers 4 \
    --save_dir checkpoints/mmformer \
    ${SMOKE_TEST_FLAG} 2>&1 | tee -a results/logs/mmformer_training.log
