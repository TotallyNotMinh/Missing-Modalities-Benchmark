import torch
from pathlib import Path
from typing import Callable, Dict, List, Optional
import nibabel as nib
import numpy as np


# BraTS 2020 file suffix for each modality channel (must match channel order in scenarios.py)
# Index 0=T1, 1=T1ce, 2=T2, 3=FLAIR
MODALITY_SUFFIXES = ["t1", "t1ce", "t2", "flair"]
SEG_SUFFIX = "seg"


class BraTSDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for the preprocessed BraTS 2020 dataset.

    Loads 4 MRI modality volumes (T1, T1ce, T2, FLAIR) and the
    multi-class segmentation mask for each patient.

    Expects the following directory structure in `data_dir`:

        <data_dir>/
          <patient_id>/
            <patient_id>_t1.nii.gz
            <patient_id>_t1ce.nii.gz
            <patient_id>_t2.nii.gz
            <patient_id>_flair.nii.gz
            <patient_id>_seg.nii.gz

    Returns a dict per patient:
        {
            'patient_id' : str,
            'modalities' : torch.FloatTensor (4, H, W, D)  — all 4 channels stacked,
            'mask'       : torch.LongTensor  (1, H, W, D)  — segmentation label map,
            'spacing'    : tuple — voxel spacing from NIfTI header,
        }
    """

    def __init__(
        self,
        data_dir: str,
        patient_ids: List[str],
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            data_dir: Root directory of preprocessed BraTS data.
            patient_ids: List of patient IDs to include.
            transform: Optional MONAI/torchvision transform applied to each sample dict.
        """
        self.data_dir = Path(data_dir)
        self.patient_ids = patient_ids
        self.transform = transform

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Dict:
        patient_id = self.patient_ids[idx]
        patient_dir = self.data_dir / patient_id

        # Load all 4 modality volumes and stack into (4, H, W, D)
        modality_arrays = []
        spacing: tuple = (1.0, 1.0, 1.0)  # default; overwritten from first modality header
        for i, suffix in enumerate(MODALITY_SUFFIXES):
            path = patient_dir / f"{patient_id}_{suffix}.nii.gz"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing modality file: {path}. "
                    "Ensure preprocessing has been run."
                )
            img = nib.load(str(path))
            if i == 0:  # capture spacing from T1 (first modality) explicitly
                spacing = tuple(img.header.get_zooms()[:3])
            modality_arrays.append(img.get_fdata(dtype=np.float32))

        modalities = torch.from_numpy(np.stack(modality_arrays, axis=0))  # (4, H, W, D)

        # Load segmentation mask (rounded to exact int64 to avoid float truncation noise like 3.999 -> 3)
        seg_path = patient_dir / f"{patient_id}_{SEG_SUFFIX}.nii.gz"
        if not seg_path.exists():
            raise FileNotFoundError(f"Missing segmentation file: {seg_path}")
        seg_img = nib.load(str(seg_path))
        seg_arr = np.round(seg_img.get_fdata()).astype(np.int64)
        mask = torch.from_numpy(seg_arr).unsqueeze(0)  # (1, H, W, D)

        sample = {
            "patient_id": patient_id,
            "modalities": modalities,
            "mask": mask,
            "spacing": spacing,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
