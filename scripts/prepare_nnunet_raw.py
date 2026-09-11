#!/usr/bin/env python3
"""
Formats raw BraTS 2020 dataset into official nnU-Net v2 raw structure:
    nnUNet_raw/Dataset001_BraTS2020/
        ├── imagesTr/
        │   ├── BraTS20_Training_001_0000.nii (T1 symlink)
        │   ├── BraTS20_Training_001_0001.nii (T1ce symlink)
        │   ├── BraTS20_Training_001_0002.nii (T2 symlink)
        │   └── BraTS20_Training_001_0003.nii (FLAIR symlink)
        ├── labelsTr/
        │   └── BraTS20_Training_001.nii (Continuous region-remapped mask)
        └── dataset.json

Usage:
    python scripts/prepare_nnunet_raw.py
"""

import argparse
import multiprocessing
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json


def _convert_single_mask(task: Tuple[Path, Path]) -> None:
    """Reads raw BraTS mask and maps to nnU-Net continuous convention."""
    in_path, out_path = task
    if out_path.exists():
        return

    img = sitk.ReadImage(str(in_path))
    arr = sitk.GetArrayFromImage(img)

    # Convert BraTS labels:
    # 0 -> 0: Background
    # 2 -> 1: Peritumoral edema (ED)
    # 1 -> 2: Necrotic core / non-enhancing tumor (NCR/NET)
    # 4 -> 3: Enhancing tumor (ET)
    seg_new = np.zeros_like(arr, dtype=np.uint8)
    seg_new[arr == 2] = 1
    seg_new[arr == 1] = 2
    seg_new[arr == 4] = 3

    out_img = sitk.GetImageFromArray(seg_new)
    out_img.CopyInformation(img)
    sitk.WriteImage(out_img, str(out_path))


def prepare_dataset(raw_dir: Path, out_dir: Path, num_processes: int = 8) -> None:
    images_tr = out_dir / "imagesTr"
    labels_tr = out_dir / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted([p for p in raw_dir.iterdir() if p.is_dir() and p.name.startswith("BraTS")])
    if not patient_dirs:
        raise FileNotFoundError(f"No patient folders found in {raw_dir}")

    print(f"[nnU-Net Prep] Found {len(patient_dirs)} patients in {raw_dir}")
    print(f"[nnU-Net Prep] Destination: {out_dir}")

    # 1. Symlink 4 image modalities
    print("[nnU-Net Prep] Creating symlinks for image modalities...")
    modality_channel_map = {
        "t1": "0000",
        "t1ce": "0001",
        "t2": "0002",
        "flair": "0003",
    }

    mask_tasks: List[Tuple[Path, Path]] = []
    file_ending = ".nii"

    for p in patient_dirs:
        case_id = p.name
        for mod, chan_idx in modality_channel_map.items():
            # Check for .nii or .nii.gz
            src_nii = p / f"{case_id}_{mod}.nii"
            src_gz = p / f"{case_id}_{mod}.nii.gz"
            if src_nii.exists():
                src_path = src_nii.resolve()
                dst_path = images_tr / f"{case_id}_{chan_idx}.nii"
                file_ending = ".nii"
            elif src_gz.exists():
                src_path = src_gz.resolve()
                dst_path = images_tr / f"{case_id}_{chan_idx}.nii.gz"
                file_ending = ".nii.gz"
            else:
                raise FileNotFoundError(f"Missing {mod} for patient {case_id}")

            if not dst_path.exists():
                os.symlink(src_path, dst_path)

        # Mask path
        src_mask_nii = p / f"{case_id}_seg.nii"
        src_mask_gz = p / f"{case_id}_seg.nii.gz"
        src_mask = src_mask_nii if src_mask_nii.exists() else src_mask_gz
        if not src_mask.exists():
            raise FileNotFoundError(f"Missing seg for patient {case_id}")
        dst_mask = labels_tr / f"{case_id}{file_ending}"
        mask_tasks.append((src_mask.resolve(), dst_mask))

    # 2. Convert and write segmentation masks in parallel
    print(f"[nnU-Net Prep] Converting {len(mask_tasks)} segmentation masks ({num_processes} workers)...")
    if num_processes > 1:
        with multiprocessing.Pool(processes=num_processes) as pool:
            list(tqdm(pool.imap_unordered(_convert_single_mask, mask_tasks), total=len(mask_tasks), desc="Converting masks"))
    else:
        for t in tqdm(mask_tasks, desc="Converting masks"):
            _convert_single_mask(t)

    # 3. Generate dataset.json
    print("[nnU-Net Prep] Generating dataset.json with BraTS overlapping regions...")
    generate_dataset_json(
        output_folder=str(out_dir),
        channel_names={0: "T1", 1: "T1ce", 2: "T2", 3: "Flair"},
        labels={
            "background": 0,
            "whole tumor": (1, 2, 3),
            "tumor core": (2, 3),
            "enhancing tumor": (3,),
        },
        num_training_cases=len(patient_dirs),
        file_ending=file_ending,
        regions_class_order=(1, 2, 3),
        reference="BraTS 2020 Benchmark / Isensee et al. Nature Methods 2021",
        license="BraTS 2020",
        release="1.0",
        description="BraTS 2020 Dataset formatted for official nnU-Net v2 Oracle training",
    )
    print(f"[nnU-Net Prep] Dataset001_BraTS2020 ready at: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare BraTS 2020 for official nnU-Net v2")
    parser.add_argument("--raw_dir", default="data/raw/brats2020", help="Path to raw BraTS cases")
    parser.add_argument("--out_dir", default="data/nnunet_raw/Dataset001_BraTS2020", help="Path to nnUNet_raw destination")
    parser.add_argument("--num_processes", type=int, default=8, help="Parallel worker processes for mask conversion")
    args = parser.parse_args()

    prepare_dataset(
        raw_dir=Path(args.raw_dir),
        out_dir=Path(args.out_dir),
        num_processes=args.num_processes,
    )


if __name__ == "__main__":
    main()
