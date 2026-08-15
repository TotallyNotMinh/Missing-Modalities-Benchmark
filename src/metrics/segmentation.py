import numpy as np
import torch
from typing import Dict, Union, Tuple, List
from scipy.spatial import cKDTree
from scipy.ndimage import binary_erosion


def convert_brats_labels_to_subregions(mask: Union[np.ndarray, torch.Tensor]) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
    """
    Converts standard BraTS multi-class segmentation labels into clinical subregions:
    - WT (Whole Tumor): labels 1, 2, 4
    - TC (Tumor Core): labels 1, 4
    - ET (Enhancing Tumor): label 4
    """
    if isinstance(mask, np.ndarray):
        is_numpy = True
        mask_tensor = torch.from_numpy(mask)
    else:
        is_numpy = False
        mask_tensor = mask

    # Remove singleton channel dimension if present (e.g., from (1, H, W, D) to (H, W, D))
    if mask_tensor.dim() == 4 and mask_tensor.shape[0] == 1:
        mask_tensor = mask_tensor.squeeze(0)

    wt = (mask_tensor == 1) | (mask_tensor == 2) | (mask_tensor == 4)
    tc = (mask_tensor == 1) | (mask_tensor == 4)
    et = (mask_tensor == 4)

    stacked = torch.stack([wt, tc, et], dim=0).float()

    if is_numpy:
        return {
            "WT": wt.numpy().astype(np.uint8),
            "TC": tc.numpy().astype(np.uint8),
            "ET": et.numpy().astype(np.uint8),
            "stacked": stacked.numpy()
        }
    return {
        "WT": wt.float(),
        "TC": tc.float(),
        "ET": et.float(),
        "stacked": stacked
    }


def _dice_numpy(pred: np.ndarray, target: np.ndarray) -> float:
    """Computes binary Dice coefficient between two binary numpy masks."""
    pred_bin = pred > 0
    target_bin = target > 0
    intersection = np.logical_and(pred_bin, target_bin).sum()
    denom = pred_bin.sum() + target_bin.sum()
    if denom == 0:
        return 1.0  # Both empty -> perfect agreement by convention
    return float(2.0 * intersection / denom)


def _hd95_numpy(
    pred: np.ndarray,
    target: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> float:
    """
    Computes exact 95th-percentile Hausdorff Distance (HD95) in millimeters
    using surface voxels extracted via binary erosion.
    
    Uses KDTree nearest-neighbor queries from surface points of pred to target
    and target to pred, then computes max(P95(d(P->T)), P95(d(T->P))).
    """
    pred_bin = pred > 0
    target_bin = target > 0

    if not pred_bin.any() and not target_bin.any():
        return 0.0  # Both empty -> perfect agreement by convention
    if not pred_bin.any() or not target_bin.any():
        return float(np.nan)  # One empty -> undefined boundary distance

    # Extract surface voxels via binary erosion
    pred_surface = np.logical_xor(pred_bin, binary_erosion(pred_bin))
    target_surface = np.logical_xor(target_bin, binary_erosion(target_bin))
    
    # Fallback: if erosion eliminates everything (single-voxel regions), use all voxels
    if not pred_surface.any():
        pred_surface = pred_bin
    if not target_surface.any():
        target_surface = target_bin

    pred_pts = np.argwhere(pred_surface)
    target_pts = np.argwhere(target_surface)

    # Scale coordinates by physical voxel spacing (mm)
    spacing_arr = np.array(voxel_spacing, dtype=np.float64)
    pred_pts_mm = pred_pts * spacing_arr
    target_pts_mm = target_pts * spacing_arr

    # Forward distances: pred -> target
    tree_target = cKDTree(target_pts_mm)
    d_fwd, _ = tree_target.query(pred_pts_mm)

    # Backward distances: target -> pred
    tree_pred = cKDTree(pred_pts_mm)
    d_bwd, _ = tree_pred.query(target_pts_mm)

    hd95 = max(np.percentile(d_fwd, 95), np.percentile(d_bwd, 95))
    return float(hd95)


def compute_segmentation_metrics(
    target_mask: Union[np.ndarray, torch.Tensor],
    pred_mask: Union[np.ndarray, torch.Tensor],
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    include_background: bool = False
) -> Dict[str, float]:
    """
    Computes 3D segmentation metrics (Dice and 95% Hausdorff Distance) across subregions (WT, TC, ET).

    Args:
        target_mask: Ground truth multi-class label map (1, 2, 4).
        pred_mask: Predicted multi-class label map.
        voxel_spacing: Physical voxel dimensions in mm (default 1.0, 1.0, 1.0).
        include_background: Whether to include background in MONAI evaluation.

    Returns:
        Dict with Dice_WT, Dice_TC, Dice_ET, Dice_Mean, HD95_WT, HD95_TC, HD95_ET, HD95_Mean.
    """
    if isinstance(target_mask, torch.Tensor):
        target_mask = target_mask.detach().cpu().numpy()
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()

    sub_target = convert_brats_labels_to_subregions(target_mask)
    sub_pred = convert_brats_labels_to_subregions(pred_mask)

    dice_scores = [
        _dice_numpy(sub_pred["WT"], sub_target["WT"]),
        _dice_numpy(sub_pred["TC"], sub_target["TC"]),
        _dice_numpy(sub_pred["ET"], sub_target["ET"])
    ]
    hd95_scores = [
        _hd95_numpy(sub_pred["WT"], sub_target["WT"], voxel_spacing=voxel_spacing),
        _hd95_numpy(sub_pred["TC"], sub_target["TC"], voxel_spacing=voxel_spacing),
        _hd95_numpy(sub_pred["ET"], sub_target["ET"], voxel_spacing=voxel_spacing)
    ]

    valid_hd95 = [s for s in hd95_scores if not np.isnan(s)]
    hd95_mean = float(np.mean(valid_hd95)) if len(valid_hd95) > 0 else float("nan")

    return {
        "Dice_WT": float(dice_scores[0]),
        "Dice_TC": float(dice_scores[1]),
        "Dice_ET": float(dice_scores[2]),
        "Dice_Mean": float(np.nanmean(dice_scores)),
        "HD95_WT": float(hd95_scores[0]),
        "HD95_TC": float(hd95_scores[1]),
        "HD95_ET": float(hd95_scores[2]),
        "HD95_Mean": hd95_mean,
    }
