import numpy as np
import torch
from typing import Dict, Union
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim

def compute_psnr(
    target: np.ndarray,
    pred: np.ndarray,
    data_range: float = None
) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) between ground truth real modality 
    and synthesized modality.
    """
    if data_range is None:
        data_range = target.max() - target.min()
        if data_range == 0:
            data_range = 1.0
            
    return float(skimage_psnr(target, pred, data_range=data_range))


def compute_ssim(
    target: np.ndarray,
    pred: np.ndarray,
    data_range: float = None,
    slice_wise: bool = True
) -> float:
    """
    Computes Structural Similarity Index (SSIM) between target and pred.
    """
    if data_range is None:
        data_range = target.max() - target.min()
        if data_range == 0:
            data_range = 1.0

    if slice_wise:
        # Determine if input is a 3D volume (H, W, D) or a 2D batch (B, H, W).
        # We assume typical medical volumes have depth as the last dimension.
        # If ndim is 3 and the last dimension is likely depth, we slice along -1.
        # If it's a 2D batch, we slice along 0.
        slice_axis = -1 if target.shape[-1] > 10 else 0
        
        ssims = []
        for i in range(target.shape[slice_axis]):
            t_slice = target[..., i] if slice_axis == -1 else target[i, ...]
            p_slice = pred[..., i] if slice_axis == -1 else pred[i, ...]
            if t_slice.max() - t_slice.min() > 1e-5:
                val = skimage_ssim(t_slice, p_slice, data_range=data_range)
                ssims.append(val)
        return float(np.mean(ssims)) if len(ssims) > 0 else 1.0
    else:
        return float(skimage_ssim(target, pred, data_range=data_range))


def compute_mae(target: np.ndarray, pred: np.ndarray) -> float:
    """Computes Mean Absolute Error (MAE)."""
    return float(np.mean(np.abs(target - pred)))


def compute_mse(target: np.ndarray, pred: np.ndarray) -> float:
    """Computes Mean Squared Error (MSE)."""
    return float(np.mean((target - pred) ** 2))


def compute_reconstruction_metrics(
    target: Union[np.ndarray, torch.Tensor],
    pred: Union[np.ndarray, torch.Tensor],
    data_range: float = None
) -> Dict[str, float]:
    """
    Convenience wrapper to compute all pixel/voxel-level reconstruction metrics at once.
    """
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
        
    target = np.squeeze(target)
    pred = np.squeeze(pred)

    return {
        "PSNR": compute_psnr(target, pred, data_range=data_range),
        "SSIM": compute_ssim(target, pred, data_range=data_range, slice_wise=True),
        "MAE": compute_mae(target, pred),
        "MSE": compute_mse(target, pred)
    }
