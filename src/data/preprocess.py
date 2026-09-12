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

try:
    from .scenarios import MODALITY_SUFFIXES
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.data.scenarios import MODALITY_SUFFIXES

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


def _resolve_raw_path(patient_dir: Path, patient_id: str, suffix: str) -> Path:
    """Finds either .nii.gz or .nii file for a given patient modality/mask."""
    gz_path = patient_dir / f"{patient_id}_{suffix}.nii.gz"
    if gz_path.exists():
        return gz_path
    nii_path = patient_dir / f"{patient_id}_{suffix}.nii"
    if nii_path.exists():
        return nii_path

    # Fallback for segmentation mask with non-standard naming (known BraTS 2020 anomaly for case 355)
    if suffix == "seg":
        seg_candidates = sorted([
            f for f in patient_dir.glob("*[sS]eg*.nii*")
            if not f.name.startswith(".")
        ])
        if seg_candidates:
            return seg_candidates[0]

    # Fallback for modality with non-standard naming
    mod_candidates = sorted([
        f for f in patient_dir.glob(f"*{suffix}*.nii*")
        if not f.name.startswith(".")
    ])
    if mod_candidates:
        return mod_candidates[0]

    raise FileNotFoundError(
        f"Missing file for {patient_id} ({suffix}): neither .nii.gz nor .nii found in {patient_dir}"
    )


def preprocess_patient(
    patient_dir: Path,
    out_dir: Path,
    patient_id: str,
    skip_n4: bool = True,
) -> None:
    """Full preprocessing pipeline for a single patient."""
    out_patient_dir = out_dir / patient_id
    out_patient_dir.mkdir(parents=True, exist_ok=True)

    seg_path = _resolve_raw_path(patient_dir, patient_id, SEG_SUFFIX)

    # Pass 1: Read all modalities and resample if needed
    modality_imgs = []
    modality_arrays = []
    for suffix in MODALITY_SUFFIXES:
        in_path = _resolve_raw_path(patient_dir, patient_id, suffix)
        sitk_img = sitk.ReadImage(str(in_path), sitk.sitkFloat32)

        # Step 2: N4 Bias Field Correction (optional; skipped by default for BraTS which is already standardized)
        if not skip_n4:
            mask_sitk = sitk.Cast(sitk_img > 0, sitk.sitkUInt8)
            sitk_img = n4_bias_correction(sitk_img, mask_image=mask_sitk)

        # Step 3: Resample to target spacing
        current_spacing = sitk_img.GetSpacing()
        if not all(abs(c - t) < 0.01 for c, t in zip(current_spacing, TARGET_SPACING)):
            sitk_img = resample_to_spacing(sitk_img, TARGET_SPACING, is_label=False)

        arr = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
        modality_imgs.append(sitk_img)
        modality_arrays.append(arr)

    # Union brain mask across all 4 modalities: prevents zeroing out valid tissue or lesions
    union_mask = np.zeros_like(modality_arrays[0], dtype=bool)
    for arr in modality_arrays:
        union_mask |= (arr > 0)
    union_mask = union_mask.astype(np.uint8)

    # Step 5: Normalize and save each modality
    for i, suffix in enumerate(MODALITY_SUFFIXES):
        arr = zscore_normalize(modality_arrays[i], union_mask)
        out_img = sitk.GetImageFromArray(arr)
        out_img.CopyInformation(modality_imgs[i])
        out_path = out_patient_dir / f"{patient_id}_{suffix}.nii.gz"
        sitk.WriteImage(out_img, str(out_path))

    # Resample and save segmentation mask (nearest neighbour)
    seg_sitk = sitk.ReadImage(str(seg_path), sitk.sitkUInt8)
    seg_sitk = resample_to_spacing(seg_sitk, TARGET_SPACING, is_label=True)
    out_seg_path = out_patient_dir / f"{patient_id}_{SEG_SUFFIX}.nii.gz"
    sitk.WriteImage(seg_sitk, str(out_seg_path))


def _preprocess_worker(args_tuple):
    patient_dir, out_path, patient_id, skip_n4 = args_tuple
    out_patient_dir = out_path / patient_id
    expected_files = [
        out_patient_dir / f"{patient_id}_{suffix}.nii.gz"
        for suffix in (*MODALITY_SUFFIXES, SEG_SUFFIX)
    ]
    if out_patient_dir.exists() and all(f.exists() for f in expected_files):
        return patient_id, True, None  # Already preprocessed
    try:
        preprocess_patient(patient_dir, out_path, patient_id, skip_n4=skip_n4)
        return patient_id, True, None
    except Exception as e:
        return patient_id, False, str(e)


def run_preprocessing(raw_dir: str, out_dir: str, num_workers: int = 4, skip_n4: bool = True) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    raw_path = Path(raw_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted([p for p in raw_path.iterdir() if p.is_dir()])
    print(f"[Preprocessing] Found {len(patient_dirs)} patients in {raw_path} (workers={num_workers}, skip_n4={skip_n4})")

    tasks = [(p, out_path, p.name, skip_n4) for p in patient_dirs]

    if num_workers <= 1:
        for task in tqdm(tasks, desc="Preprocessing patients"):
            pid, ok, err = _preprocess_worker(task)
            if not ok:
                print(f"[Preprocessing] ERROR on {pid}: {err}")
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_preprocess_worker, t): t[2] for t in tasks}
            for fut in tqdm(as_completed(futures), total=len(tasks), desc="Preprocessing patients"):
                pid, ok, err = fut.result()
                if not ok:
                    print(f"[Preprocessing] ERROR on {pid}: {err}")

    print(f"[Preprocessing] Done. Outputs in {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BraTS 2020 Offline Preprocessing")
    parser.add_argument("--raw_dir", default="data/raw/brats2020")
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel worker processes")
    parser.add_argument("--skip_n4", action="store_true", default=True, help="Skip N4 bias correction (BraTS is already skull-stripped/normalized)")
    parser.add_argument("--run_n4", dest="skip_n4", action="store_false", help="Run N4 bias correction")
    args = parser.parse_args()
    run_preprocessing(args.raw_dir, args.out_dir, num_workers=args.num_workers, skip_n4=args.skip_n4)
