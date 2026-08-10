from typing import List
from monai.transforms import (
    Compose,
    SpatialPadd,
    RandSpatialCropd,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGaussianNoised,
    CenterSpatialCropd,
    EnsureTyped,
)

# Keys used in the sample dict from BraTSDataset
MODALITY_KEY = "modalities"
MASK_KEY = "mask"
ALL_KEYS = [MODALITY_KEY, MASK_KEY]


def get_train_transforms(
    patch_size: List[int] = [128, 128, 128],
    prob: float = 0.5,
) -> Compose:
    """
    Returns the training augmentation pipeline.

    Augmentations are applied identically to both modalities and mask tensors
    using MONAI's dictionary-based transforms (the same random state is shared).

    Args:
        patch_size: 3D spatial crop size.
        prob: Probability for each random augmentation.

    Returns:
        MONAI Compose transform.
    """
    return Compose([
        # 0. Ensure spatial dimensions are at least patch_size before cropping
        SpatialPadd(keys=ALL_KEYS, spatial_size=patch_size),
        # 1. Random spatial crop to patch_size
        RandSpatialCropd(
            keys=ALL_KEYS,
            roi_size=patch_size,
            random_size=False,
        ),
        # 2. Random flips along each axis
        RandFlipd(keys=ALL_KEYS, prob=prob, spatial_axis=0),
        RandFlipd(keys=ALL_KEYS, prob=prob, spatial_axis=1),
        RandFlipd(keys=ALL_KEYS, prob=prob, spatial_axis=2),
        # 3. Random 90° rotations on spatial H-W plane (spatial_axes=(0, 1) in MONAI dict transforms)
        RandRotate90d(keys=ALL_KEYS, prob=prob, max_k=3, spatial_axes=(0, 1)),
        # 4. Mild intensity augmentations (modality channels only)
        RandScaleIntensityd(keys=[MODALITY_KEY], factors=0.1, prob=prob),
        RandShiftIntensityd(keys=[MODALITY_KEY], offsets=0.1, prob=prob),
        RandGaussianNoised(keys=[MODALITY_KEY], prob=prob / 2, mean=0.0, std=0.01),
        # 5. Ensure correct tensor types
        EnsureTyped(keys=[MODALITY_KEY], dtype="float32"),
        EnsureTyped(keys=[MASK_KEY], dtype="int64"),
    ])


def get_val_transforms(
    patch_size: List[int] = [128, 128, 128],
) -> Compose:
    """
    Returns the deterministic validation/test transform pipeline.
    No random augmentation — only center crop.

    Args:
        patch_size: 3D center crop size.

    Returns:
        MONAI Compose transform.
    """
    return Compose([
        SpatialPadd(
            keys=ALL_KEYS,
            spatial_size=patch_size,
        ),
        CenterSpatialCropd(
            keys=ALL_KEYS,
            roi_size=patch_size,
        ),
        EnsureTyped(keys=[MODALITY_KEY], dtype="float32"),
        EnsureTyped(keys=[MASK_KEY], dtype="int64"),
    ])
