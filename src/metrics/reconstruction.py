import numpy as np
import torch
from typing import Dict, Union, Optional
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim


def compute_psnr(
    target: np.ndarray,
    pred: np.ndarray,
    data_range: Optional[float] = None,
    mask: Optional[np.ndarray] = None
) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) between ground truth real modality 
    and synthesized modality.
    
    Args:
        target: Real ground truth volume.
        pred: Synthesized volume.
        data_range: Dynamic range of image. If None, computed from target (or target within mask).
        mask: Optional binary foreground brain mask. If provided, metrics are computed over foreground.
    """
    if mask is not None:
        target_eval = target[mask > 0]
        pred_eval = pred[mask > 0]
    else:
        target_eval = target
        pred_eval = pred

    if len(target_eval) == 0:
        return 0.0

    if data_range is None:
        data_range = float(target_eval.max() - target_eval.min())
        if data_range == 0:
            data_range = 1.0

    mse = np.mean((target_eval - pred_eval) ** 2)
    if mse == 0:
        return float("inf")

    return float(10 * np.log10((data_range ** 2) / mse))


def compute_ssim(
    target: np.ndarray,
    pred: np.ndarray,
    data_range: Optional[float] = None,
    slice_wise: bool = True,
    mask: Optional[np.ndarray] = None
) -> float:
    """
    Computes Structural Similarity Index (SSIM) between target and pred.
    
    Args:
        target: Real ground truth volume.
        pred: Synthesized volume.
        data_range: Dynamic range. If None, computed from target.
        slice_wise: If True, computes 2D SSIM per axial slice and averages non-empty slices.
        mask: Optional binary mask.
    """
    if data_range is None:
        data_range = float(target.max() - target.min())
        if data_range == 0:
            data_range = 1.0

    if slice_wise:
        # Depth is typically the last dimension (H, W, D) or first dimension (D, H, W)
        slice_axis = -1 if target.shape[-1] <= min(target.shape[:-1]) or target.shape[-1] > 10 else 0
        num_slices = target.shape[slice_axis]
        
        ssims = []
        for i in range(num_slices):
            t_slice = target[..., i] if slice_axis == -1 else target[i, ...]
            p_slice = pred[..., i] if slice_axis == -1 else pred[i, ...]
            
            if mask is not None:
                m_slice = mask[..., i] if slice_axis == -1 else mask[i, ...]
                if m_slice.sum() == 0:
                    continue  # Skip slices with no brain tissue
            
            # Skip uniform / completely empty slices
            if t_slice.max() - t_slice.min() > 1e-5:
                val = skimage_ssim(t_slice, p_slice, data_range=data_range)
                ssims.append(val)
        return float(np.mean(ssims)) if len(ssims) > 0 else 1.0
    else:
        return float(skimage_ssim(target, pred, data_range=data_range))


def compute_mae(
    target: np.ndarray,
    pred: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> float:
    """Computes Mean Absolute Error (MAE)."""
    if mask is not None:
        target_eval = target[mask > 0]
        pred_eval = pred[mask > 0]
    else:
        target_eval = target
        pred_eval = pred

    if len(target_eval) == 0:
        return 0.0
    return float(np.mean(np.abs(target_eval - pred_eval)))


def compute_mse(
    target: np.ndarray,
    pred: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> float:
    """Computes Mean Squared Error (MSE)."""
    if mask is not None:
        target_eval = target[mask > 0]
        pred_eval = pred[mask > 0]
    else:
        target_eval = target
        pred_eval = pred

    if len(target_eval) == 0:
        return 0.0
    return float(np.mean((target_eval - pred_eval) ** 2))


def compute_reconstruction_metrics(
    target: Union[np.ndarray, torch.Tensor],
    pred: Union[np.ndarray, torch.Tensor],
    data_range: Optional[float] = None,
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None
) -> Dict[str, float]:
    """
    Convenience wrapper to compute all pixel/voxel-level reconstruction metrics at once.
    """
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    target = np.squeeze(target)
    pred = np.squeeze(pred)
    if mask is not None:
        mask = np.squeeze(mask)

    return {
        "PSNR": compute_psnr(target, pred, data_range=data_range, mask=mask),
        "SSIM": compute_ssim(target, pred, data_range=data_range, slice_wise=True, mask=mask),
        "MAE": compute_mae(target, pred, mask=mask),
        "MSE": compute_mse(target, pred, mask=mask)
    }
