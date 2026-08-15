import numpy as np
from typing import List, Sequence
import torch
from monai.transforms import (
    Compose,
    SpatialPadd,
    RandSpatialCropd,
    RandFlipd,
    RandRotated,
    RandZoomd,
    Rand3DElasticd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandAdjustContrastd,
    RandSimulateLowResolutiond,
    CenterSpatialCropd,
    EnsureTyped,
)

# Keys used in the sample dict from BraTSDataset
MODALITY_KEY = "modalities"
MASK_KEY = "mask"
ALL_KEYS = [MODALITY_KEY, MASK_KEY]


def get_synthesis_train_transforms(
    patch_size: Sequence[int] = (128, 128, 128),
) -> Compose:
    """
    Synthesis Models Augmentation Protocol (Controlled Variable).
    
    Rationale & Design Principles:
      1. Anatomical Correspondence: Modality synthesis requires preserving exact
         geometric alignment between sequences. Spatial transforms are applied
         jointly across all 4 MRI sequences.
      2. Conservative Geometric Perturbations: Restricted to sagittal flipping
         (L-R) and subtle in-plane axial rotation (±10°) to preserve physiological
         tissue geometry without unrealistic spatial distortions.
      3. Conservative Intensity Perturbations: Subtle contrast scaling (0.9-1.1)
         and low-amplitude Gaussian noise (std=0.02) to prevent artificial
         intensity distribution drift.
      4. Explicit Exclusions: Elastic deformation, multi-axis flipping (A-P/S-I),
         and mixup/cutmix are strictly excluded to avoid non-anatomical warping
         and synthetic artifacts in cross-modal mapping.
      5. Patch Sampling: Random 3D spatial patch crop (128x128x128) during training.
    """
    return Compose([
        # 0. Spatial Padding (guarantee volume >= patch_size)
        SpatialPadd(keys=ALL_KEYS, spatial_size=patch_size, mode="constant"),
        # 1. Random 3D patch crop (train-time spatial sampling)
        RandSpatialCropd(keys=ALL_KEYS, roi_size=patch_size, random_size=False),
        # 2. Sagittal flip (L-R only, p = 0.5)
        RandFlipd(keys=ALL_KEYS, prob=0.5, spatial_axis=0),
        # 3. In-plane axial rotation (±10°, p = 0.5)
        RandRotated(
            keys=ALL_KEYS,
            range_z=10.0 * (np.pi / 180.0),
            range_x=0.0,
            range_y=0.0,
            prob=0.5,
            mode=["bilinear", "nearest"],
            padding_mode="zeros",
        ),
        # 4. Subtle contrast/brightness jitter (scale 0.9 - 1.1, p = 0.3)
        RandAdjustContrastd(keys=[MODALITY_KEY], prob=0.3, gamma=(0.9, 1.1)),
        # 5. Additive Gaussian noise (std = 0.02 on [0, 1] normalized intensities, p = 0.2)
        RandGaussianNoised(keys=[MODALITY_KEY], prob=0.2, mean=0.0, std=0.02),
        # 6. Type Enforcement
        EnsureTyped(keys=[MODALITY_KEY], dtype="float32"),
        EnsureTyped(keys=[MASK_KEY], dtype="int64"),
    ])


def get_segmentation_train_transforms(
    patch_size: Sequence[int] = (128, 128, 128),
) -> Compose:
    """
    Downstream Segmentation Models Augmentation Protocol (nnU-Net-derived Standard).
    
    Rationale & Design Principles:
      1. Oracle Consistency: Matches the established nnU-Net augmentation distribution
         used to pretrain state-of-the-art segmentation networks.
      2. Invariant Semantic Representation: Heavy geometric (3D rotation ±30°,
         3-axis mirroring, 3D elastic deformation, random scaling) and aggressive
         intensity regularizations (contrast 0.75-1.25, gamma inversion, simulated
         low resolution) force the segmenter to be robust against scanner variation
         and synthetic synthesis imperfections.
      3. Patch Sampling: Random 3D spatial patch crop (128x128x128) during training.
    """
    return Compose([
        # 0. Ensure spatial dimensions >= patch_size
        SpatialPadd(keys=ALL_KEYS, spatial_size=patch_size, mode="constant"),
        # 1. Random 3D patch crop (train-time spatial sampling)
        RandSpatialCropd(keys=ALL_KEYS, roi_size=patch_size, random_size=False),
        # 2. Random 3D rotation (±30° per axis, prob = 0.20)
        RandRotated(
            keys=ALL_KEYS,
            range_x=30.0 * (np.pi / 180.0),
            range_y=30.0 * (np.pi / 180.0),
            range_z=30.0 * (np.pi / 180.0),
            prob=0.20,
            mode=["bilinear", "nearest"],
            padding_mode="zeros",
        ),
        # 3. Random scaling (0.7 - 1.4, prob = 0.20)
        RandZoomd(
            keys=ALL_KEYS,
            min_zoom=0.7,
            max_zoom=1.4,
            prob=0.20,
            mode=["bilinear", "nearest"],
            padding_mode="constant",
        ),
        # 4. 3D Elastic deformation (nnU-Net-derived moderate displacement field, prob = 0.15)
        Rand3DElasticd(
            keys=ALL_KEYS,
            sigma_range=(5, 8),
            magnitude_range=(50, 150),
            prob=0.15,
            mode=["bilinear", "nearest"],
            padding_mode="zeros",
        ),
        # 5. Mirroring/flip (3-axis spatial mirroring per nnU-Net default, prob = 0.50 per axis)
        RandFlipd(keys=ALL_KEYS, prob=0.50, spatial_axis=0),
        RandFlipd(keys=ALL_KEYS, prob=0.50, spatial_axis=1),
        RandFlipd(keys=ALL_KEYS, prob=0.50, spatial_axis=2),
        # 6. Gaussian noise (std = 0.10 on [0, 1] normalized intensities, prob = 0.20)
        RandGaussianNoised(keys=[MODALITY_KEY], prob=0.20, mean=0.0, std=0.10),
        # 7. Gaussian blur (sigma = 0.5 - 1.0, prob = 0.20)
        RandGaussianSmoothd(
            keys=[MODALITY_KEY],
            sigma_x=(0.5, 1.0),
            sigma_y=(0.5, 1.0),
            sigma_z=(0.5, 1.0),
            prob=0.20,
        ),
        # 8. Brightness multiplier (scale 0.75 - 1.25, prob = 0.15)
        RandScaleIntensityd(keys=[MODALITY_KEY], factors=0.25, prob=0.15),
        # 9. Contrast range (gamma = 0.75 - 1.25, prob = 0.15)
        RandAdjustContrastd(keys=[MODALITY_KEY], gamma=(0.75, 1.25), invert_image=False, prob=0.15),
        # 10. Gamma correction with intensity inversion (gamma = 0.7 - 1.5, prob = 0.10)
        RandAdjustContrastd(keys=[MODALITY_KEY], gamma=(0.7, 1.5), invert_image=True, prob=0.10),
        # 11. Gamma correction without inversion (gamma = 0.7 - 1.5, prob = 0.30)
        RandAdjustContrastd(keys=[MODALITY_KEY], gamma=(0.7, 1.5), invert_image=False, prob=0.30),
        # 12. Simulated low resolution (downsampling scale 0.5 - 1.0, prob = 0.20)
        RandSimulateLowResolutiond(keys=[MODALITY_KEY], zoom_range=(0.5, 1.0), prob=0.20),
        # 13. Type Enforcement
        EnsureTyped(keys=[MODALITY_KEY], dtype="float32"),
        EnsureTyped(keys=[MASK_KEY], dtype="int64"),
    ])


def get_val_transforms(
    patch_size: Sequence[int] = (128, 128, 128),
) -> Compose:
    """
    Deterministic Validation and Test Transform Pipeline.
    
    Rationale:
      Zero stochastic augmentations. Evaluates strictly on deterministic
      center crops (128x128x128) to guarantee exact reproducibility of
      evaluation metrics (Dice, HD95, PSNR, SSIM) across all models.
    """
    return Compose([
        SpatialPadd(keys=ALL_KEYS, spatial_size=patch_size, mode="constant"),
        CenterSpatialCropd(keys=ALL_KEYS, roi_size=patch_size),
        EnsureTyped(keys=[MODALITY_KEY], dtype="float32"),
        EnsureTyped(keys=[MASK_KEY], dtype="int64"),
    ])


# Aliases for backward compatibility
get_train_transforms = get_segmentation_train_transforms
get_test_transforms = get_val_transforms
