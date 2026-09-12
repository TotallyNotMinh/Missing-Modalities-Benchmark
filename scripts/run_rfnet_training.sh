#!/usr/bin/env bash
set -e

# ==============================================================================
# RFNet Baseline Training Pipeline for BraTS 2020
# Architecture: RFNet (8.98M parameters, basic_dims=16)
# Standardized patch size: 128x128x128
# Target GPU  : Physical GPU 2 (CUDA_VISIBLE_DEVICES=2)
# ==============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# 1. Python binary detection (auto-detects missing-modalities environment)
if [ -n "${CONDA_PREFIX}" ] && [[ "${CONDA_PREFIX}" == *"missing-modalities"* ]] && [ -f "${CONDA_PREFIX}/bin/python" ]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
elif [ -f "${HOME}/miniconda3/envs/missing-modalities/bin/python" ]; then
    PYTHON_BIN="${HOME}/miniconda3/envs/missing-modalities/bin/python"
elif [ -f "/storage/student14/minhdn/miniconda3/envs/missing-modalities/bin/python" ]; then
    PYTHON_BIN="/storage/student14/minhdn/miniconda3/envs/missing-modalities/bin/python"
elif [ -n "${CONDA_PREFIX}" ] && [ -f "${CONDA_PREFIX}/bin/python" ]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
    PYTHON_BIN="python"
fi

# 2. Argument parsing: --gpu/-g, --smoke-test/-s, and passthrough flags
TARGET_GPU="${CUDA_VISIBLE_DEVICES:-2}"
SMOKE_TEST_FLAG=""
PASSTHROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu|-g)
            TARGET_GPU="$2"
            shift 2
            ;;
        --smoke-test|-s)
            SMOKE_TEST_FLAG="--smoke_test"
            shift
            ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

export CUDA_VISIBLE_DEVICES="${TARGET_GPU}"

if [[ -n "${SMOKE_TEST_FLAG}" ]]; then
    echo "================================================================="
    echo " Running RFNet Pre-Flight Smoke Test (1 Epoch, 2 Batches)"
    echo " Target GPU ID : ${TARGET_GPU} (CUDA_VISIBLE_DEVICES=${TARGET_GPU})"
    echo "================================================================="
else
    echo "================================================================="
    echo " Launching Official RFNet 300-Epoch Training Run"
    echo " Target GPU ID : ${TARGET_GPU} (CUDA_VISIBLE_DEVICES=${TARGET_GPU})"
    echo " Batch Size    : 1"
    echo " Patch Size    : 128x128x128"
    echo " Learning Rate : 1e-4"
    echo " Checkpoints   : checkpoints/rfnet"
    echo " Log File      : results/logs/rfnet_training.log"
    echo "================================================================="
fi

mkdir -p results/logs checkpoints/rfnet

"${PYTHON_BIN}" scripts/train_rfnet.py \
    --device "cuda:0" \
    --epochs 300 \
    --batch_size 1 \
    --lr 1e-4 \
    --patch_size 128 128 128 \
    --val_interval 5 \
    --num_workers 4 \
    --save_dir checkpoints/rfnet \
    ${SMOKE_TEST_FLAG} \
    "${PASSTHROUGH_ARGS[@]}" 2>&1 | tee -a results/logs/rfnet_training.log
