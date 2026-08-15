"""
Offline preprocessing pipeline for BraTS 2020.

Run this script once before any model training:
    python -m src.data.preprocess --raw_dir data/raw/brats2020 --out_dir data/processed

Steps applied per patient, per modality:
    1. N4 Bias Field Correction (SimpleITK)
    2. Skull-Stripping (zero-out voxels outside BraTS brain mask)
    3. Per-modality Z-Score Normalization (mean/std over non-zero brain voxels)
    4. Spatial Resampling to isotropic 1mm³ spacing (if not already)

Outputs are saved as .nii.gz files in:
    <out_dir>/<patient_id>/<patient_id>_<suffix>.nii.gz
"""

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from .scenarios import MODALITY_SUFFIXES
SEG_SUFFIX = "seg"
TARGET_SPACING = (1.0, 1.0, 1.0)  # mm³ isotropic


def n4_bias_correction(
    sitk_image: sitk.Image,
    mask_image: sitk.Image = None
) -> sitk.Image:
    """Applies N4 ITK bias field correction with optional brain mask for speed and accuracy."""
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
    if mask_image is not None:
        return corrector.Execute(sitk_image, mask_image)
    return corrector.Execute(sitk_image)


def resample_to_spacing(
    sitk_image: sitk.Image,
    target_spacing: tuple = TARGET_SPACING,
    is_label: bool = False,
) -> sitk.Image:
    """Resamples a SimpleITK image to target voxel spacing."""
    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()

    scale = [orig / tgt for orig, tgt in zip(original_spacing, target_spacing)]
    new_size = [int(round(sz * sc)) for sz, sc in zip(original_size, scale)]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(sitk_image)


def zscore_normalize(volume: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
    """Z-score normalizes a single modality over non-zero brain voxels."""
    non_zero = volume[brain_mask > 0]
    if len(non_zero) == 0 or non_zero.std() == 0:
        return volume
    mean = non_zero.mean()
    std = non_zero.std()
    normalized = (volume - mean) / (std + 1e-8)
    # Zero out non-brain voxels
    normalized[brain_mask == 0] = 0.0
    return normalized.astype(np.float32)


def preprocess_patient(
    patient_dir: Path,
    out_dir: Path,
    patient_id: str,
) -> None:
    """Full preprocessing pipeline for a single patient."""
    out_patient_dir = out_dir / patient_id
    out_patient_dir.mkdir(parents=True, exist_ok=True)

    seg_path = patient_dir / f"{patient_id}_{SEG_SUFFIX}.nii.gz"

    # Process each modality — brain mask is derived per-modality AFTER resampling
    # to guarantee shape consistency when volumes are not already at 1mm³.
    for i, suffix in enumerate(MODALITY_SUFFIXES):
        in_path = patient_dir / f"{patient_id}_{suffix}.nii.gz"
        if not in_path.exists():
            raise FileNotFoundError(f"Missing file: {in_path}")

        # Step 1: Read as SimpleITK float32
        sitk_img = sitk.ReadImage(str(in_path), sitk.sitkFloat32)

        # Step 2: N4 Bias Field Correction (uses non-zero brain mask to avoid background distortion)
        mask_sitk = sitk.Cast(sitk_img > 0, sitk.sitkUInt8)
        sitk_img = n4_bias_correction(sitk_img, mask_image=mask_sitk)

        # Step 3: Resample to target spacing
        current_spacing = sitk_img.GetSpacing()
        if not all(abs(c - t) < 0.01 for c, t in zip(current_spacing, TARGET_SPACING)):
            sitk_img = resample_to_spacing(sitk_img, TARGET_SPACING, is_label=False)

        # Step 4: Build brain mask from *resampled* T1 (first modality, index 0)
        # Re-derive from the resampled T1 so that mask shape always matches arr.
        arr = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
        if i == 0:  # T1 — derive the brain mask at resampled resolution
            brain_mask = (arr > 0).astype(np.uint8)

        # Step 5: Z-Score Normalization (uses brain_mask from resampled T1)
        arr = zscore_normalize(arr, brain_mask)

        # Step 6: Save to output
        out_img = sitk.GetImageFromArray(arr)
        out_img.CopyInformation(sitk_img)
        out_path = out_patient_dir / f"{patient_id}_{suffix}.nii.gz"
        sitk.WriteImage(out_img, str(out_path))

    # Resample and save segmentation mask (nearest neighbour)
    seg_sitk = sitk.ReadImage(str(seg_path), sitk.sitkUInt8)
    seg_sitk = resample_to_spacing(seg_sitk, TARGET_SPACING, is_label=True)
    out_seg_path = out_patient_dir / f"{patient_id}_{SEG_SUFFIX}.nii.gz"
    sitk.WriteImage(seg_sitk, str(out_seg_path))


def run_preprocessing(raw_dir: str, out_dir: str) -> None:
    raw_path = Path(raw_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted([p for p in raw_path.iterdir() if p.is_dir()])
    print(f"[Preprocessing] Found {len(patient_dirs)} patients in {raw_path}")

    for patient_dir in tqdm(patient_dirs, desc="Preprocessing patients"):
        patient_id = patient_dir.name
        out_patient_dir = out_path / patient_id
        if out_patient_dir.exists() and any(out_patient_dir.iterdir()):
            continue  # Already preprocessed, skip
        try:
            preprocess_patient(patient_dir, out_path, patient_id)
        except Exception as e:
            print(f"[Preprocessing] ERROR on {patient_id}: {e}")

    print(f"[Preprocessing] Done. Outputs in {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BraTS 2020 Offline Preprocessing")
    parser.add_argument("--raw_dir", default="data/raw/brats2020")
    parser.add_argument("--out_dir", default="data/processed")
    args = parser.parse_args()
    run_preprocessing(args.raw_dir, args.out_dir)
