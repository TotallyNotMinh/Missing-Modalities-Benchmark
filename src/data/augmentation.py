import numpy as np
from typing import List, Sequence, Dict, Any, Callable, Optional
import torch

try:
    from monai.transforms import (
        Compose as MonaiCompose,
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
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False

# Keys used in the sample dict from BraTSDataset
MODALITY_KEY = "modalities"
MASK_KEY = "mask"
ALL_KEYS = [MODALITY_KEY, MASK_KEY]


# ---------------------------------------------------------------------------
# Pure PyTorch / NumPy Fallback Transforms (when MONAI is not installed)
# ---------------------------------------------------------------------------

class PyTorchCompose:
    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = transforms

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for t in self.transforms:
            sample = t(sample)
        return sample


class PyTorchSpatialPad:
    def __init__(self, keys: Sequence[str], spatial_size: Sequence[int]):
        self.keys = keys
        self.spatial_size = tuple(spatial_size)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for k in self.keys:
            if k in sample:
                val = sample[k]  # (C, H, W, D)
                pad_w = [max(0, s - val.shape[i + 1]) for i, s in enumerate(self.spatial_size)]
                if any(p > 0 for p in pad_w):
                    # Pad (left, right, top, bottom, front, back)
                    pad_tuple = (0, pad_w[2], 0, pad_w[1], 0, pad_w[0])
                    sample[k] = torch.nn.functional.pad(val, pad_tuple, mode="constant", value=0)
        return sample


class PyTorchSpatialCrop:
    def __init__(self, keys: Sequence[str], roi_size: Sequence[int], mode: str = "random"):
        self.keys = keys
        self.roi_size = tuple(roi_size)
        self.mode = mode

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        first_tensor = sample[self.keys[0]]
        spatial_shape = first_tensor.shape[1:]  # (H, W, D)
        starts = []
        for d_size, r_size in zip(spatial_shape, self.roi_size):
            if self.mode == "center":
                starts.append(max(0, (d_size - r_size) // 2))
            else:
                max_start = max(0, d_size - r_size)
                starts.append(np.random.randint(0, max_start + 1) if max_start > 0 else 0)

        for k in self.keys:
            if k in sample:
                s0, s1, s2 = starts
                r0, r1, r2 = self.roi_size
                sample[k] = sample[k][:, s0:s0 + r0, s1:s1 + r1, s2:s2 + r2]
        return sample


class PyTorchRandFlip:
    def __init__(self, keys: Sequence[str], prob: float = 0.5, spatial_axis: int = 0):
        self.keys = keys
        self.prob = prob
        self.spatial_axis = spatial_axis  # 0->dim 1, 1->dim 2, 2->dim 3

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if np.random.rand() < self.prob:
            dim = self.spatial_axis + 1
            for k in self.keys:
                if k in sample:
                    sample[k] = torch.flip(sample[k], dims=[dim])
        return sample


class PyTorchRandNoise:
    def __init__(self, keys: Sequence[str], prob: float = 0.2, std: float = 0.02):
        self.keys = keys
        self.prob = prob
        self.std = std

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if np.random.rand() < self.prob:
            for k in self.keys:
                if k in sample:
                    noise = torch.randn_like(sample[k]) * self.std
                    sample[k] = sample[k] + noise
        return sample


class PyTorchEnsureTyped:
    def __init__(self, keys: Sequence[str], dtype: str):
        self.keys = keys
        self.dtype = getattr(torch, dtype) if hasattr(torch, dtype) else torch.float32

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for k in self.keys:
            if k in sample:
                if isinstance(sample[k], np.ndarray):
                    sample[k] = torch.from_numpy(sample[k])
                sample[k] = sample[k].to(self.dtype)
        return sample


# ---------------------------------------------------------------------------
# Synthesis Training Transforms (Conservative Policy)
# ---------------------------------------------------------------------------

def get_synthesis_train_transforms(
    patch_size: Sequence[int] = (128, 128, 128),
    cfg: Optional[dict] = None,
) -> Any:
    """
    Synthesis Models Augmentation Protocol (Controlled Variable).
    
    Rationale & Design Principles:
      1. Anatomical Correspondence: Modality synthesis requires preserving exact
         geometric alignment between sequences. Spatial transforms are applied
         jointly across all 4 MRI sequences.
      2. Conservative Geometric Perturbations: Restricted to sagittal flipping
         (L-R) and subtle in-plane axial rotation (+-10 deg) to preserve physiological
         tissue geometry without unrealistic spatial distortions.
      3. Conservative Intensity Perturbations: Subtle contrast scaling (0.9-1.1)
         and low-amplitude Gaussian noise (std=0.02) to prevent artificial
         intensity distribution drift.
      4. Explicit Exclusions: Elastic deformation, multi-axis flipping (A-P/S-I),
         and mixup/cutmix are strictly excluded to avoid non-anatomical warping
         and synthetic artifacts in cross-modal mapping.
      5. Patch Sampling: Random 3D spatial patch crop (128x128x128) during training.
    """
    if cfg is not None:
        syn_cfg = cfg.get("augmentation_synthesis", {})
        spatial = syn_cfg.get("spatial", {})
        intensity = syn_cfg.get("intensity", {})
        flip_prob = spatial.get("flip", {}).get("prob", 0.5)
        rot_range_deg = spatial.get("rotation", {}).get("range_deg", [-10, 10])
        rot_prob = spatial.get("rotation", {}).get("prob", 0.5)
        gamma_range = intensity.get("gamma_jitter", {}).get("range", [0.9, 1.1])
        gamma_prob = intensity.get("gamma_jitter", {}).get("prob", 0.3)
        noise_std_range = intensity.get("gaussian_noise", {}).get("sigma_range", [0.01, 0.03])
        noise_prob = intensity.get("gaussian_noise", {}).get("prob", 0.2)
        noise_std = (noise_std_range[0] + noise_std_range[1]) / 2 if isinstance(noise_std_range, (list, tuple)) else noise_std_range
    else:
        flip_prob = 0.5
        rot_range_deg = [-10, 10]
        rot_prob = 0.5
        gamma_range = (0.9, 1.1)
        gamma_prob = 0.3
        noise_std = 0.02
        noise_prob = 0.2

    rot_rad = max(abs(rot_range_deg[0]), abs(rot_range_deg[1])) * (np.pi / 180.0)

    if HAS_MONAI:
        return MonaiCompose([
            SpatialPadd(keys=ALL_KEYS, spatial_size=patch_size, mode="constant"),
            RandSpatialCropd(keys=ALL_KEYS, roi_size=patch_size, random_size=False),
            RandFlipd(keys=ALL_KEYS, prob=flip_prob, spatial_axis=0),
            RandRotated(
                keys=ALL_KEYS,
                range_z=rot_rad,
                range_x=0.0,
                range_y=0.0,
                prob=rot_prob,
                mode=["bilinear", "nearest"],
                padding_mode="zeros",
            ),
            RandAdjustContrastd(keys=[MODALITY_KEY], prob=gamma_prob, gamma=gamma_range),
            RandGaussianNoised(keys=[MODALITY_KEY], prob=noise_prob, mean=0.0, std=noise_std),
            EnsureTyped(keys=[MODALITY_KEY], dtype="float32"),
            EnsureTyped(keys=[MASK_KEY], dtype="int64"),
        ])
    else:
        return PyTorchCompose([
            PyTorchSpatialPad(keys=ALL_KEYS, spatial_size=patch_size),
            PyTorchSpatialCrop(keys=ALL_KEYS, roi_size=patch_size, mode="random"),
            PyTorchRandFlip(keys=ALL_KEYS, prob=flip_prob, spatial_axis=0),
            PyTorchRandNoise(keys=[MODALITY_KEY], prob=noise_prob, std=noise_std),
            PyTorchEnsureTyped(keys=[MODALITY_KEY], dtype="float32"),
            PyTorchEnsureTyped(keys=[MASK_KEY], dtype="int64"),
        ])


# ---------------------------------------------------------------------------
# Downstream Segmentation Training Transforms (nnU-Net Standard)
# ---------------------------------------------------------------------------

def get_segmentation_train_transforms(
    patch_size: Sequence[int] = (128, 128, 128),
    cfg: Optional[dict] = None,
) -> Any:
    """
    Downstream Segmentation Models Augmentation Protocol (nnU-Net-derived Standard).
    
    Rationale & Design Principles:
      1. Oracle Consistency: Matches the established nnU-Net augmentation distribution
         used to pretrain state-of-the-art segmentation networks.
      2. Invariant Semantic Representation: Heavy geometric (3D rotation +-30 deg,
         3-axis mirroring, 3D elastic deformation, random scaling) and aggressive
         intensity regularizations (contrast 0.75-1.25, gamma inversion, simulated
         low resolution) force the segmenter to be robust against scanner variation
         and synthetic synthesis imperfections.
      3. Patch Sampling: Random 3D spatial patch crop (128x128x128) during training.
    """
    if cfg is not None:
        seg_cfg = cfg.get("augmentation_segmentation", {})
        spatial = seg_cfg.get("spatial", {})
        intensity = seg_cfg.get("intensity", {})
        
        rot_range_deg = spatial.get("rotation", {}).get("range_deg", [-30, 30])
        rot_prob = spatial.get("rotation", {}).get("prob", 0.20)
        
        zoom_range = spatial.get("scaling", {}).get("range", [0.7, 1.4])
        zoom_prob = spatial.get("scaling", {}).get("prob", 0.20)
        
        elastic_sigma = spatial.get("elastic", {}).get("sigma_range", [5, 8])
        elastic_mag = spatial.get("elastic", {}).get("magnitude_range", [50, 150])
        elastic_prob = spatial.get("elastic", {}).get("prob", 0.15)
        
        flip_prob = spatial.get("flip", {}).get("prob", 0.50)
        
        noise_std_range = intensity.get("gaussian_noise", {}).get("sigma_range", [0.0, 0.20])
        noise_prob = intensity.get("gaussian_noise", {}).get("prob", 0.20)
        noise_std = (noise_std_range[0] + noise_std_range[1]) / 2 if isinstance(noise_std_range, (list, tuple)) else noise_std_range
        
        smooth_sigma = intensity.get("gaussian_blur", {}).get("sigma_range", [0.5, 1.0])
        smooth_prob = intensity.get("gaussian_blur", {}).get("prob", 0.20)
        
        scale_factors = intensity.get("brightness", {}).get("multiplier_range", [0.75, 1.25])
        scale_factor_val = (scale_factors[1] - 1.0) if scale_factors else 0.25
        scale_prob = intensity.get("brightness", {}).get("prob", 0.15)
        
        contrast_range = intensity.get("contrast", {}).get("range", [0.75, 1.25])
        contrast_prob = intensity.get("contrast", {}).get("prob", 0.15)
        
        gamma_inv_prob = intensity.get("gamma", {}).get("invert_prob", 0.10)
        gamma_range = intensity.get("gamma", {}).get("range", [0.7, 1.5])
        gamma_prob = intensity.get("gamma", {}).get("prob", 0.30)
        
        low_res_zoom = intensity.get("simulate_low_res", {}).get("zoom_range", [0.5, 1.0])
        low_res_prob = intensity.get("simulate_low_res", {}).get("prob", 0.20)
    else:
        rot_range_deg = [-30, 30]
        rot_prob = 0.20
        zoom_range = (0.7, 1.4)
        zoom_prob = 0.20
        elastic_sigma = (5, 8)
        elastic_mag = (50, 150)
        elastic_prob = 0.15
        flip_prob = 0.50
        noise_std = 0.10
        noise_prob = 0.20
        smooth_sigma = (0.5, 1.0)
        smooth_prob = 0.20
        scale_factor_val = 0.25
        scale_prob = 0.15
        contrast_range = (0.75, 1.25)
        contrast_prob = 0.15
        gamma_inv_prob = 0.10
        gamma_range = (0.7, 1.5)
        gamma_prob = 0.30
        low_res_zoom = (0.5, 1.0)
        low_res_prob = 0.20

    rot_rad = max(abs(rot_range_deg[0]), abs(rot_range_deg[1])) * (np.pi / 180.0)

    if HAS_MONAI:
        return MonaiCompose([
            SpatialPadd(keys=ALL_KEYS, spatial_size=patch_size, mode="constant"),
            RandSpatialCropd(keys=ALL_KEYS, roi_size=patch_size, random_size=False),
            RandRotated(
                keys=ALL_KEYS,
                range_x=rot_rad,
                range_y=rot_rad,
                range_z=rot_rad,
                prob=rot_prob,
                mode=["bilinear", "nearest"],
                padding_mode="zeros",
            ),
            RandZoomd(
                keys=ALL_KEYS,
                min_zoom=zoom_range[0],
                max_zoom=zoom_range[1],
                prob=zoom_prob,
                mode=["bilinear", "nearest"],
                padding_mode="constant",
            ),
            Rand3DElasticd(
                keys=ALL_KEYS,
                sigma_range=elastic_sigma,
                magnitude_range=elastic_mag,
                prob=elastic_prob,
                mode=["bilinear", "nearest"],
                padding_mode="zeros",
            ),
            RandFlipd(keys=ALL_KEYS, prob=flip_prob, spatial_axis=0),
            RandFlipd(keys=ALL_KEYS, prob=flip_prob, spatial_axis=1),
            RandFlipd(keys=ALL_KEYS, prob=flip_prob, spatial_axis=2),
            RandGaussianNoised(keys=[MODALITY_KEY], prob=noise_prob, mean=0.0, std=noise_std),
            RandGaussianSmoothd(
                keys=[MODALITY_KEY],
                sigma_x=smooth_sigma,
                sigma_y=smooth_sigma,
                sigma_z=smooth_sigma,
                prob=smooth_prob,
            ),
            RandScaleIntensityd(keys=[MODALITY_KEY], factors=scale_factor_val, prob=scale_prob),
            RandAdjustContrastd(keys=[MODALITY_KEY], gamma=contrast_range, invert_image=False, prob=contrast_prob),
            RandAdjustContrastd(keys=[MODALITY_KEY], gamma=gamma_range, invert_image=True, prob=gamma_inv_prob),
            RandAdjustContrastd(keys=[MODALITY_KEY], gamma=gamma_range, invert_image=False, prob=gamma_prob),
            RandSimulateLowResolutiond(keys=[MODALITY_KEY], zoom_range=low_res_zoom, prob=low_res_prob),
            EnsureTyped(keys=[MODALITY_KEY], dtype="float32"),
            EnsureTyped(keys=[MASK_KEY], dtype="int64"),
        ])
    else:
        return PyTorchCompose([
            PyTorchSpatialPad(keys=ALL_KEYS, spatial_size=patch_size),
            PyTorchSpatialCrop(keys=ALL_KEYS, roi_size=patch_size, mode="random"),
            PyTorchRandFlip(keys=ALL_KEYS, prob=flip_prob, spatial_axis=0),
            PyTorchRandFlip(keys=ALL_KEYS, prob=flip_prob, spatial_axis=1),
            PyTorchRandFlip(keys=ALL_KEYS, prob=flip_prob, spatial_axis=2),
            PyTorchRandNoise(keys=[MODALITY_KEY], prob=noise_prob, std=noise_std),
            PyTorchEnsureTyped(keys=[MODALITY_KEY], dtype="float32"),
            PyTorchEnsureTyped(keys=[MASK_KEY], dtype="int64"),
        ])


# ---------------------------------------------------------------------------
# Validation / Test Deterministic Transforms
# ---------------------------------------------------------------------------

def get_val_transforms(
    patch_size: Sequence[int] = (128, 128, 128),
    cfg: Optional[dict] = None,
) -> Any:
    """
    Deterministic Validation and Test Transform Pipeline.
    
    Rationale:
      Zero stochastic augmentations. Evaluates strictly on deterministic
      center crops (128x128x128) to guarantee exact reproducibility of
      evaluation metrics (Dice, HD95, PSNR, SSIM) across all models.
    """
    if HAS_MONAI:
        return MonaiCompose([
            SpatialPadd(keys=ALL_KEYS, spatial_size=patch_size, mode="constant"),
            CenterSpatialCropd(keys=ALL_KEYS, roi_size=patch_size),
            EnsureTyped(keys=[MODALITY_KEY], dtype="float32"),
            EnsureTyped(keys=[MASK_KEY], dtype="int64"),
        ])
    else:
        return PyTorchCompose([
            PyTorchSpatialPad(keys=ALL_KEYS, spatial_size=patch_size),
            PyTorchSpatialCrop(keys=ALL_KEYS, roi_size=patch_size, mode="center"),
            PyTorchEnsureTyped(keys=[MODALITY_KEY], dtype="float32"),
            PyTorchEnsureTyped(keys=[MASK_KEY], dtype="int64"),
        ])


# Aliases for backward compatibility
get_train_transforms = get_segmentation_train_transforms
get_test_transforms = get_val_transforms
