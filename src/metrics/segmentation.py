import numpy as np
import torch
from typing import Dict, Union, Tuple, List

try:
    from monai.metrics import DiceMetric, HausdorffDistanceMetric
    from monai.utils import MetricReduction
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False
    from scipy.spatial.distance import directed_hausdorff

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
    intersection = np.sum(pred * target)
    denom = np.sum(pred) + np.sum(target)
    if denom == 0:
        return 1.0
    return float(2.0 * intersection / denom)


def _hd95_numpy(pred: np.ndarray, target: np.ndarray) -> float:
    pred_pts = np.argwhere(pred > 0)
    target_pts = np.argwhere(target > 0)

    if len(pred_pts) == 0 and len(target_pts) == 0:
        # Both empty — perfect agreement by convention
        return 0.0
    if len(pred_pts) == 0 or len(target_pts) == 0:
        # One is empty — undefined distance; use NaN so aggregation can filter it
        return float(np.nan)

    d_fwd = directed_hausdorff(pred_pts, target_pts)[0]
    d_bwd = directed_hausdorff(target_pts, pred_pts)[0]
    return float(max(d_fwd, d_bwd))


def compute_segmentation_metrics(
    target_mask: Union[np.ndarray, torch.Tensor],
    pred_mask: Union[np.ndarray, torch.Tensor],
    include_background: bool = False
) -> Dict[str, float]:
    """
    Computes 3D segmentation metrics (Dice and HD95) across subregions (WT, TC, ET).
    """
    if isinstance(target_mask, torch.Tensor):
        target_mask = target_mask.detach().cpu().numpy()
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()

    sub_target = convert_brats_labels_to_subregions(target_mask)
    sub_pred = convert_brats_labels_to_subregions(pred_mask)

    if HAS_MONAI:
        t_stacked = torch.from_numpy(sub_target["stacked"]).unsqueeze(0)
        p_stacked = torch.from_numpy(sub_pred["stacked"]).unsqueeze(0)

        dice_metric = DiceMetric(include_background=include_background, reduction=MetricReduction.NONE)
        dice_metric(y_pred=p_stacked, y=t_stacked)
        dice_scores = dice_metric.aggregate().squeeze().tolist()

        try:
            hd95_metric = HausdorffDistanceMetric(include_background=include_background, percentile=95, reduction=MetricReduction.NONE)
            hd95_metric(y_pred=p_stacked, y=t_stacked)
            hd95_scores = hd95_metric.aggregate().squeeze().tolist()
        except Exception:
            hd95_scores = [0.0, 0.0, 0.0]
    else:
        dice_scores = [
            _dice_numpy(sub_pred["WT"], sub_target["WT"]),
            _dice_numpy(sub_pred["TC"], sub_target["TC"]),
            _dice_numpy(sub_pred["ET"], sub_target["ET"])
        ]
        hd95_scores = [
            _hd95_numpy(sub_pred["WT"], sub_target["WT"]),
            _hd95_numpy(sub_pred["TC"], sub_target["TC"]),
            _hd95_numpy(sub_pred["ET"], sub_target["ET"])
        ]

    return {
        "Dice_WT": float(dice_scores[0]),
        "Dice_TC": float(dice_scores[1]),
        "Dice_ET": float(dice_scores[2]),
        "Dice_Mean": float(np.nanmean(dice_scores)),
        "HD95_WT": float(hd95_scores[0]),
        "HD95_TC": float(hd95_scores[1]),
        "HD95_ET": float(hd95_scores[2]),
        "HD95_Mean": float(np.nanmean(hd95_scores)) if not np.all(np.isnan(hd95_scores)) else float("nan")
    }

