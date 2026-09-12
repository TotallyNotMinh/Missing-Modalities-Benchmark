import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference

from contextlib import contextmanager

from src.metrics.segmentation import compute_segmentation_metrics
from src.utils.pipeline_utils import Config, load_config

RFNET_ROOT = Path(__file__).resolve().parent.parent.parent / "externals" / "rfnet"


@contextmanager
def _scoped_import(root_path: Path):
    """Context manager isolating external model imports from global namespace collisions."""
    str_path = str(root_path.resolve())
    old_sys_path = list(sys.path)
    saved_modules = {}
    for mod_name in ("layers", "models", "rfnet"):
        if mod_name in sys.modules:
            saved_modules[mod_name] = sys.modules.pop(mod_name)
    try:
        sys.path.insert(0, str_path)
        yield
    finally:
        sys.path = old_sys_path
        for mod_name in ("layers", "models", "rfnet"):
            sys.modules.pop(mod_name, None)
            if mod_name in saved_modules:
                sys.modules[mod_name] = saved_modules[mod_name]


class RFNetAdapter:
    """
    Adapter for Region-Aware Fusion Network (RFNet) downstream missing-modality segmentation evaluator.

    Handles:
      1. Canonical 4-channel BraTS input ordering: (T1, T1ce, T2, FLAIR).
      2. Automated permutation to RFNet's internal ordering: (FLAIR, T1ce, T1, T2).
      3. Missing-modality mask construction: supports Scenario IDs ('S1'-'S4'),
         explicit boolean masks, or automatic non-zero channel detection.
      4. Sliding-window 3D volumetric inference via MONAI (default patch size 80x80x80).
      5. Label re-mapping to standard BraTS convention:
           0: Background
           1: Necrotic / Non-enhancing tumor (NCR/NET)
           2: Peritumoral edema (ED)
           4: Enhancing tumor (ET)
      6. Subregion metric computation (WT, TC, ET Dice and HD95).
    """

    # Mapping from RFNet class index (0, 1, 2, 3) to standard BraTS label (0, 1, 2, 4)
    # 0: BG, 1: NCR/NET, 2: ED, 3: ET -> mapped to BraTS 0, 1, 2, 4
    CLASS_TO_BRATS_LABEL = np.array([0, 1, 2, 4], dtype=np.uint8)

    # Permutation from benchmark ordering [T1 (0), T1ce (1), T2 (2), FLAIR (3)]
    # to RFNet ordering [FLAIR (3), T1ce (1), T1 (0), T2 (2)]
    BENCHMARK_TO_RFNET_INDICES = [3, 1, 0, 2]

    SCENARIO_MASKS = {
        "S1": [True, True, True, False],                   # Missing FLAIR
        "S2": [True, False, True, True],                   # Missing T1ce
        "S3": [False, True, True, True],                   # Missing T1
        "S4": [True, True, False, True],                   # Missing T2
        "FULL": [True, True, True, True],                  # All 4 modalities
        "SINGLE_MISSING_FLAIR": [True, True, True, False], # Alias for S1
        "SINGLE_MISSING_T1CE": [True, False, True, True],  # Alias for S2
        "TWO_MISSING": [True, False, True, False],         # Missing T1ce and FLAIR
        "THREE_MISSING": [True, False, False, False],      # Missing T1ce, T2, and FLAIR
    }

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        patch_size: Tuple[int, int, int] = (80, 80, 80),
        num_classes: int = 4,
        network: Optional[nn.Module] = None,
    ):
        """
        Args:
            weights_path: Path to checkpoint (.pth/.pt) trained weights.
            device: Device to run inference on ('cuda', 'cpu', or torch.device).
            patch_size: 3D patch ROI size for sliding-window evaluation (default: (80, 80, 80)).
            num_classes: Number of output classes (default: 4 for BG, NCR, ED, ET).
            network: Optional pre-instantiated PyTorch nn.Module. If None, builds RFNet Model.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        else:
            self.device = device

        self.patch_size = tuple(patch_size)
        self.num_classes = num_classes
        self.num_input_channels = 4
        self.weights_path = Path(weights_path) if weights_path is not None else None

        if network is not None:
            self.network = network.to(self.device)
            if self.weights_path is not None and self.weights_path.is_file():
                self._load_state_dict(self.weights_path)
                print(f"[RFNetAdapter] Loaded weights into supplied network from '{self.weights_path}' on {self.device}.")
            else:
                print(f"[RFNetAdapter] Using supplied custom PyTorch network on {self.device}.")
            if hasattr(self.network, "is_training"):
                self.network.is_training = False
            self.network.eval()
        else:
            self.network = self._build_model()
            if self.weights_path is not None and self.weights_path.is_file():
                self._load_state_dict(self.weights_path)
                print(f"[RFNetAdapter] Loaded weights from '{self.weights_path}' on {self.device}.")
            elif self.weights_path is not None and not self.weights_path.exists():
                print(f"[RFNetAdapter] Initialized RFNet Model on {self.device} (weights path '{self.weights_path}' does not exist yet).")
            else:
                print(f"[RFNetAdapter] Initialized RFNet Model on {self.device} with fresh weights.")
            self.network.to(self.device)
            if hasattr(self.network, "is_training"):
                self.network.is_training = False
            self.network.eval()

    def _build_model(self) -> nn.Module:
        """Dynamically imports and constructs the RFNet Model architecture."""
        try:
            with _scoped_import(RFNET_ROOT):
                import models
                model = models.Model(num_cls=self.num_classes)
                return model
        except Exception as e:
            raise ImportError(
                f"[RFNetAdapter] Could not import or build RFNet Model from {RFNET_ROOT}: {e}"
            )

    def _load_state_dict(self, checkpoint_path: Path) -> None:
        """Loads weights into self.network, handling DataParallel / DDP prefixes."""
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and "network_weights" in ckpt:
            state_dict = ckpt["network_weights"]
        elif isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
        elif isinstance(ckpt, dict):
            state_dict = ckpt
        else:
            raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}")

        # Strip 'module.' prefix if present
        clean_state_dict = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }
        missing_keys, unexpected_keys = self.network.load_state_dict(clean_state_dict, strict=False)
        if missing_keys:
            print(f"[RFNetAdapter] Warning: {len(missing_keys)} missing keys during checkpoint load (sample: {missing_keys[:3]})")
        if unexpected_keys:
            print(f"[RFNetAdapter] Warning: {len(unexpected_keys)} unexpected keys during checkpoint load (sample: {unexpected_keys[:3]})")

    @classmethod
    def from_config(
        cls,
        cfg: Optional[Config] = None,
        weights_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ) -> "RFNetAdapter":
        """Factory constructor instantiating adapter from project Config."""
        if cfg is None:
            cfg = load_config()

        target_weights = (
            weights_path
            or cfg.paths.get("rfnet_weights", None)
            or cfg.paths.get("segmentation_weights", None)
        )
        target_device = device or cfg.get("device", "cuda")
        target_patch = tuple(cfg.patch.get("size", (80, 80, 80)))
        model_cfg = cfg.get("model", {})
        target_classes = model_cfg.get("num_classes", 4)

        return cls(
            weights_path=target_weights,
            device=target_device,
            patch_size=target_patch,
            num_classes=target_classes,
        )

    def _prepare_input_tensor(
        self,
        input_data: Union[np.ndarray, torch.Tensor, Dict[str, Any]],
        mask: Optional[Union[str, Sequence[bool], np.ndarray, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, bool, Optional[Union[str, List[bool], torch.Tensor]]]:
        """
        Converts input data into a 5D PyTorch Tensor (B, 4, H, W, D) in benchmark order.

        Returns:
            tensor: (B, 4, H, W, D) on self.device
            was_4d: Boolean indicating if input was originally 4D
            dict_mask: Mask or scenario identifier if extracted from input dict
        """
        dict_mask = None
        if isinstance(input_data, dict):
            if "modalities" in input_data:
                tensor = input_data["modalities"]
            elif "inputs" in input_data:
                tensor = input_data["inputs"]
            else:
                raise KeyError(
                    f"Expected 'modalities' or 'inputs' in dictionary, got {list(input_data.keys())}"
                )
            dict_mask = input_data.get("mask", input_data.get("scenario", None))
        elif isinstance(input_data, np.ndarray):
            tensor = torch.from_numpy(input_data)
        elif isinstance(input_data, torch.Tensor):
            tensor = input_data
        else:
            raise TypeError(f"Unsupported input type: {type(input_data)}")

        tensor = tensor.float()

        if tensor.dim() == 4:
            was_4d = True
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() == 5:
            was_4d = False
        else:
            raise ValueError(
                f"Expected 4D (4, H, W, D) or 5D (B, 4, H, W, D) tensor, got {tensor.shape}"
            )

        active_mask = mask if mask is not None else dict_mask
        if active_mask is not None and isinstance(active_mask, str) and active_mask.upper() in self.SCENARIO_MASKS:
            # Automatically reconstruct (B, 4, H, W, D) by placing available channels at their true positions
            scenario_key = active_mask.upper()
            target_mask_bools = self.SCENARIO_MASKS[scenario_key]
            avail_indices = [idx for idx, b in enumerate(target_mask_bools) if b]
            if tensor.shape[1] == len(avail_indices) and tensor.shape[1] < self.num_input_channels:
                reconstructed = torch.zeros((tensor.shape[0], 4, *tensor.shape[2:]), dtype=tensor.dtype, device=tensor.device)
                for in_i, out_i in enumerate(avail_indices):
                    reconstructed[:, out_i] = tensor[:, in_i]
                tensor = reconstructed
        elif tensor.shape[1] != self.num_input_channels:
            raise ValueError(
                f"Expected {self.num_input_channels} input channels (T1, T1ce, T2, FLAIR), got {tensor.shape[1]}"
            )

        return tensor.to(self.device), was_4d, dict_mask

    def _resolve_mask(
        self,
        x_bench: torch.Tensor,
        mask: Optional[Union[str, Sequence[bool], np.ndarray, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Resolves the missing-modality mask into a boolean Tensor of shape (B, 4)
        aligned with RFNet's internal modality order: [FLAIR, T1ce, T1, T2].

        Args:
            x_bench: (B, 4, H, W, D) tensor in benchmark order [T1, T1ce, T2, FLAIR].
            mask: Optional mask representation (e.g. 'S1', [T, T, T, F], or tensor).

        Returns:
            torch.BoolTensor of shape (B, 4) on self.device.
        """
        B = x_bench.shape[0]

        if isinstance(mask, str):
            scenario_key = mask.upper()
            if scenario_key in self.SCENARIO_MASKS:
                bench_bools = self.SCENARIO_MASKS[scenario_key]
            else:
                raise ValueError(f"Unknown scenario '{mask}'. Expected one of {list(self.SCENARIO_MASKS.keys())}")
            # Map benchmark order [T1, T1ce, T2, FLAIR] -> RFNet [FLAIR, T1ce, T1, T2]
            rf_bools = [bench_bools[i] for i in self.BENCHMARK_TO_RFNET_INDICES]
            return torch.tensor([rf_bools] * B, dtype=torch.bool, device=self.device)

        elif mask is not None:
            if isinstance(mask, (list, tuple)):
                mask_t = torch.tensor(mask, dtype=torch.bool, device=self.device)
            elif isinstance(mask, np.ndarray):
                mask_t = torch.from_numpy(mask).to(device=self.device, dtype=torch.bool)
            elif isinstance(mask, torch.Tensor):
                mask_t = mask.to(device=self.device, dtype=torch.bool)
            else:
                raise TypeError(f"Unsupported mask type: {type(mask)}")

            if mask_t.dim() == 1:
                if mask_t.shape[0] != 4:
                    raise ValueError(f"Expected 4-element mask, got length {mask_t.shape[0]}")
                mask_t = mask_t.unsqueeze(0).repeat(B, 1)
            elif mask_t.dim() == 2:
                if mask_t.shape[1] != 4:
                    raise ValueError(f"Expected mask of shape (B, 4), got {mask_t.shape}")
                if mask_t.shape[0] != B:
                    mask_t = mask_t.repeat(B, 1)
            else:
                raise ValueError(f"Mask tensor must be 1D or 2D, got shape {mask_t.shape}")

            # Reorder from benchmark order [T1, T1ce, T2, FLAIR] to RFNet order [FLAIR, T1ce, T1, T2]
            return mask_t[:, self.BENCHMARK_TO_RFNET_INDICES]

        else:
            # Auto-detect missing modalities: any channel that is all zeros is marked False
            bench_mask = torch.ones((B, 4), dtype=torch.bool, device=self.device)
            for b in range(B):
                for c in range(4):
                    channel_energy = x_bench[b, c].abs().sum()
                    if channel_energy < 1e-7:
                        bench_mask[b, c] = False

            # If all modalities were detected as zero (pathological case), fallback to all True
            all_zero = (bench_mask.sum(dim=1) == 0)
            bench_mask[all_zero] = True

            # Permute to RFNet order
            return bench_mask[:, self.BENCHMARK_TO_RFNET_INDICES]

    @torch.no_grad()
    def predict(
        self,
        input_data: Union[np.ndarray, torch.Tensor, Dict[str, Any]],
        mask: Optional[Union[str, Sequence[bool], np.ndarray, torch.Tensor]] = None,
        return_logits: bool = False,
        roi_size: Optional[Tuple[int, int, int]] = None,
        overlap: float = 0.5,
        blend_mode: str = "gaussian",
        postprocess_et: bool = False,
        et_threshold: int = 500,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Runs 3D volumetric segmentation inference using RFNet.

        Args:
            input_data: (4, H, W, D) or (B, 4, H, W, D) volume or sample dict.
            mask: Optional missing-modality indicator ('S1'-'S4', boolean list, or tensor).
            return_logits: If True, returns (pred_mask, logits).
            roi_size: Sliding window patch size. Defaults to self.patch_size (80, 80, 80).
            overlap: Sliding window patch overlap ratio (0.0 - 1.0).
            blend_mode: Sliding window blending mode ('gaussian' or 'constant').
            postprocess_et: If True, applies official paper post-processing (zeros out ET if < et_threshold).
            et_threshold: Voxel threshold for ET post-processing (default: 500).

        Returns:
            pred_labels: uint8 ndarray of shape (H, W, D) or (B, H, W, D) with standard BraTS labels (0, 1, 2, 4).
            logits (optional): float32 ndarray of raw class logits of shape (4, H, W, D) or (B, 4, H, W, D).
        """
        x_bench, was_4d, dict_mask = self._prepare_input_tensor(input_data, mask=mask)
        active_mask = mask if mask is not None else dict_mask

        # Resolve boolean presence mask aligned with RFNet order [FLAIR, T1ce, T1, T2]
        mask_rf = self._resolve_mask(x_bench, active_mask)

        # Permute input channels: benchmark [T1, T1ce, T2, FLAIR] -> RFNet [FLAIR, T1ce, T1, T2]
        x_rf = x_bench[:, self.BENCHMARK_TO_RFNET_INDICES, :, :, :]

        roi = roi_size or self.patch_size
        spatial_shape = x_rf.shape[2:]
        needs_sliding_window = any(s > r for s, r in zip(spatial_shape, roi))

        # Ensure network is in evaluation mode
        if hasattr(self.network, "is_training"):
            self.network.is_training = False
        self.network.eval()

        # Predictor closure for sliding-window inference
        def predictor_fn(patch: torch.Tensor) -> torch.Tensor:
            b_sw = patch.shape[0]
            curr_mask = mask_rf if mask_rf.shape[0] == b_sw else mask_rf[:1].repeat(b_sw, 1)
            out = self.network(patch, curr_mask)
            return out[0] if isinstance(out, (list, tuple)) else out

        if needs_sliding_window:
            logits = sliding_window_inference(
                inputs=x_rf,
                roi_size=roi,
                sw_batch_size=1,
                predictor=predictor_fn,
                overlap=overlap,
                mode=blend_mode,
            )
        else:
            out = self.network(x_rf, mask_rf)
            logits = out[0] if isinstance(out, (list, tuple)) else out

        # Convert logits to standard BraTS categorical labels (0, 1, 2, 4)
        pred_class = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
        pred_labels = self.CLASS_TO_BRATS_LABEL[pred_class]
        logits_np = logits.cpu().numpy()

        # Paper ET post-processing: If ET voxel count < threshold (default 500), suppress ET
        if postprocess_et:
            if pred_labels.ndim == 4:
                for b in range(pred_labels.shape[0]):
                    if np.sum(pred_labels[b] == 4) < et_threshold:
                        pred_labels[b][pred_labels[b] == 4] = 0
            else:
                if np.sum(pred_labels == 4) < et_threshold:
                    pred_labels[pred_labels == 4] = 0

        if was_4d:
            pred_labels = pred_labels[0]
            logits_np = logits_np[0]

        if return_logits:
            return pred_labels, logits_np
        return pred_labels

    def evaluate_sample(
        self,
        input_data: Union[np.ndarray, torch.Tensor, Dict[str, Any]],
        target_mask: Union[np.ndarray, torch.Tensor],
        mask: Optional[Union[str, Sequence[bool], np.ndarray, torch.Tensor]] = None,
        voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        roi_size: Optional[Tuple[int, int, int]] = None,
        postprocess_et: bool = False,
        et_threshold: int = 500,
    ) -> Dict[str, float]:
        """
        Predicts segmentation and computes standardized BraTS subregion metrics.

        Returns:
            Dict containing Dice_WT, Dice_TC, Dice_ET, Dice_Mean,
                           HD95_WT, HD95_TC, HD95_ET, HD95_Mean.
        """
        pred_mask = self.predict(
            input_data,
            mask=mask,
            return_logits=False,
            roi_size=roi_size,
            postprocess_et=postprocess_et,
            et_threshold=et_threshold,
        )
        return compute_segmentation_metrics(
            target_mask=target_mask,
            pred_mask=pred_mask,
            voxel_spacing=voxel_spacing,
        )

    def evaluate_batch(
        self,
        batch: Dict[str, Any],
        scenario: Optional[str] = None,
        voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        return_logits: bool = False,
        postprocess_et: bool = False,
        et_threshold: int = 500,
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], np.ndarray]]:
        """
        Evaluates a batch of samples from DataLoader and returns per-patient metrics.
        If return_logits is True, also returns full predicted logits array.
        """
        inputs = batch.get("modalities", batch.get("inputs"))
        targets = batch.get("mask")
        patient_ids = batch.get("patient_id", [f"patient_{i}" for i in range(len(inputs))])
        batch_mask = scenario if scenario is not None else batch.get("missing_mask", batch.get("scenario", None))

        if isinstance(targets, torch.Tensor) and targets.dim() == 5 and targets.shape[1] == 1:
            targets = targets.squeeze(1)

        results = []
        all_logits = []
        for i in range(len(inputs)):
            sample_in = inputs[i]
            sample_target = targets[i]
            pid = patient_ids[i] if isinstance(patient_ids, (list, tuple)) else str(patient_ids)

            sample_mask = None
            if batch_mask is not None:
                sample_mask = (
                    batch_mask[i]
                    if isinstance(batch_mask, (list, tuple, torch.Tensor, np.ndarray))
                    else batch_mask
                )

            if return_logits:
                pred_mask, logits_np = self.predict(
                    sample_in,
                    mask=sample_mask,
                    return_logits=True,
                    postprocess_et=postprocess_et,
                    et_threshold=et_threshold,
                )
                all_logits.append(logits_np)
            else:
                pred_mask = self.predict(
                    sample_in,
                    mask=sample_mask,
                    return_logits=False,
                    postprocess_et=postprocess_et,
                    et_threshold=et_threshold,
                )

            metrics = compute_segmentation_metrics(
                target_mask=sample_target,
                pred_mask=pred_mask,
                voxel_spacing=voxel_spacing,
            )
            record = {"patient_id": pid, **metrics}
            results.append(record)

        if return_logits:
            stacked_logits = (
                np.stack(all_logits, axis=0) if len(all_logits) > 1 else np.expand_dims(all_logits[0], axis=0)
            )
            return results, stacked_logits
        return results
