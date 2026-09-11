#!/usr/bin/env bash
set -e

# ==============================================================================
# End-to-End Official nnU-Net v2 Pipeline for BraTS 2020 Oracle
# ==============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# Activate python environment if needed
PYTHON_BIN="${CONDA_PREFIX}/bin/python"
if [ ! -f "${PYTHON_BIN}" ]; then
    PYTHON_BIN="python"
fi

# 1. Export standard nnU-Net v2 environment variables
export nnUNet_raw="${PROJECT_ROOT}/data/nnunet_raw"
export nnUNet_preprocessed="${PROJECT_ROOT}/data/nnunet_preprocessed"
export nnUNet_results="${PROJECT_ROOT}/checkpoints/nnUNet_results"

echo "================================================================="
echo " Official nnU-Net v2 Oracle Execution Pipeline"
echo " nnUNet_raw          : ${nnUNet_raw}"
echo " nnUNet_preprocessed : ${nnUNet_preprocessed}"
echo " nnUNet_results      : ${nnUNet_results}"
echo "================================================================="

# 2. Prepare raw dataset structure if not already prepared
if [ ! -f "${nnUNet_raw}/Dataset001_BraTS2020/dataset.json" ]; then
    echo "[Step 1/4] Preparing raw dataset into nnUNet_raw..."
    "${PYTHON_BIN}" scripts/prepare_nnunet_raw.py
else
    echo "[Step 1/4] nnUNet_raw/Dataset001_BraTS2020 already prepared."
fi

# 3. Preprocess for 3d_fullres using frozen plans (prevents silent plan mutation)
PLANS_FILE="${nnUNet_preprocessed}/Dataset001_BraTS2020/nnUNetPlans.json"
if [ -f "${PLANS_FILE}" ]; then
    echo "[Step 2/4] Preprocessing using authoritative frozen plans (${PLANS_FILE})..."
    nnUNetv2_preprocess -d 001 -c 3d_fullres
else
    echo "[Step 2/4] Authoritative plans not found. Running initial planning and preprocessing..."
    nnUNetv2_plan_and_preprocess -d 001 -c 3d_fullres --verify_dataset_integrity
fi

# 4. Inject benchmark splits so Fold 0 matches benchmark train/val splits
echo "[Step 3/4] Aligning Fold 0 splits with benchmark data/splits/splits.json..."
"${PYTHON_BIN}" scripts/generate_nnunet_splits.py

# 5. Train Fold 0 on target GPU
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
echo "[Step 4/4] Starting training on GPU ${GPU_ID} (Fold 0, 3d_fullres, 1000 epochs)..."
CUDA_VISIBLE_DEVICES="${GPU_ID}" nnUNetv2_train 001 3d_fullres 0

echo "================================================================="
echo " nnU-Net v2 Training Complete!"
echo " Results stored in: ${nnUNet_results}/Dataset001_BraTS2020"
echo "================================================================="

# 6. Link trained weights to checkpoints/oracle_nnunet for benchmark adapter
TRAINED_DIR="${nnUNet_results}/Dataset001_BraTS2020/nnUNetTrainer__nnUNetPlans__3d_fullres"
if [ -d "${TRAINED_DIR}" ]; then
    echo "[Post-Training] Linking ${TRAINED_DIR} to checkpoints/oracle_nnunet..."
    mkdir -p "${PROJECT_ROOT}/checkpoints"
    rm -rf "${PROJECT_ROOT}/checkpoints/oracle_nnunet"
    ln -s "${TRAINED_DIR}" "${PROJECT_ROOT}/checkpoints/oracle_nnunet"
    echo "[Post-Training] Done! Benchmark nnUNetAdapter is ready to evaluate."
fi
