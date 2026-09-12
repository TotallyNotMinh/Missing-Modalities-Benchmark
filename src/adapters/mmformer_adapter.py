import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference

from src.metrics.segmentation import compute_segmentation_metrics
from src.utils.pipeline_utils import Config, load_config

# Ensure the external mmFormer repository is importable
MMFORMER_ROOT = Path(__file__).resolve().parent.parent.parent / "externals" / "mmformer" / "mmformer"
if MMFORMER_ROOT.exists() and str(MMFORMER_ROOT) not in sys.path:
    sys.path.insert(0, str(MMFORMER_ROOT))


class MMFormerAdapter:
    """
    Adapter for mmFormer downstream missing-modality segmentation evaluator.

    Handles:
      1. Canonical 4-channel BraTS input ordering: (T1, T1ce, T2, FLAIR).
      2. Automated permutation to mmFormer's internal ordering: (FLAIR, T1ce, T1, T2).
      3. Missing-modality mask construction: supports Scenario IDs ('S1'-'S4'),
         explicit boolean masks, or automatic non-zero channel detection.
      4. Sliding-window 3D volumetric inference via MONAI.
      5. Label re-mapping to standard BraTS convention:
           0: Background
           1: Necrotic / Non-enhancing tumor (NCR/NET)
           2: Peritumoral edema (ED)
           4: Enhancing tumor (ET)\
      6. Subregion metric computation (WT, TC, ET Dice and HD95).
    """

    # Mapping from mmFormer class index (0, 1, 2, 3) to standard BraTS label (0, 1, 2, 4)
    # 0: BG, 1: NCR/NET, 2: ED, 3: ET -> mapped to BraTS 0, 1, 2, 4
    CLASS_TO_BRATS_LABEL = np.array([0, 1, 2, 4], dtype=np.uint8)

    # Permutation from benchmark ordering [T1 (0), T1ce (1), T2 (2), FLAIR (3)]
    # to mmFormer ordering [FLAIR (3), T1ce (1), T1 (0), T2 (2)]
    BENCHMARK_TO_MMFORMER_INDICES = [3, 1, 0, 2]

    # Scenario missing modality definitions in benchmark order [T1, T1ce, T2, FLAIR]
    # False indicates the missing sequence
    SCENARIO_MASKS = {
        "S1": [True, True, True, False],   # Missing FLAIR
        "S2": [True, False, True, True],   # Missing T1ce
        "S3": [False, True, True, True],   # Missing T1
        "S4": [True, True, False, True],   # Missing T2
        "FULL": [True, True, True, True],  # All 4 modalities
    }

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        num_classes: int = 4,
        network: Optional[nn.Module] = None,
    ):
        """
        Args:
            weights_path: Path to checkpoint (.pth/.pt) trained weights.
            device: Device to run inference on ('cuda', 'cpu', or torch.device).
            patch_size: 3D patch ROI size for sliding-window evaluation (default: (128, 128, 128)).
            num_classes: Number of output classes (default: 4 for BG, NCR, ED, ET).
            network: Optional pre-instantiated PyTorch nn.Module. If None, builds mmFormer Model.
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
                print(f"[MMFormerAdapter] Loaded weights into supplied network from '{self.weights_path}' on {self.device}.")
            else:
                print(f"[MMFormerAdapter] Using supplied custom PyTorch network on {self.device}.")
            self.network.eval()
        else:
            self.network = self._build_model()
            if self.weights_path is not None and self.weights_path.is_file():
                self._load_state_dict(self.weights_path)
                print(f"[MMFormerAdapter] Loaded weights from '{self.weights_path}' on {self.device}.")
            elif self.weights_path is not None and not self.weights_path.exists():
                print(f"[MMFormerAdapter] Initialized mmFormer Model on {self.device} (weights path '{self.weights_path}' does not exist yet).")
            else:
                print(f"[MMFormerAdapter] Initialized mmFormer Model on {self.device} with fresh weights.")
            self.network.to(self.device)
            self.network.eval()

    def _build_model(self) -> nn.Module:
        """Dynamically imports and constructs the mmFormer Model architecture."""
        try:
            import mmformer
            model = mmformer.Model(num_cls=self.num_classes)
            return model
        except Exception as e:
            raise ImportError(
                f"[MMFormerAdapter] Could not import or build mmformer.Model from {MMFORMER_ROOT}: {e}"
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
        self.network.load_state_dict(clean_state_dict, strict=False)

    @classmethod
    def from_config(
        cls,
        cfg: Optional[Config] = None,
        weights_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ) -> "MMFormerAdapter":
        """Factory constructor instantiating adapter from project Config."""
        if cfg is None:
            cfg = load_config()

        target_weights = (
            weights_path
            or cfg.paths.get("mmformer_weights", None)
            or cfg.paths.get("segmentation_weights", None)
        )
        target_device = device or cfg.get("device", "cuda")
        target_patch = tuple(cfg.patch.get("size", (128, 128, 128)))
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

        if tensor.shape[1] != self.num_input_channels:
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
        aligned with mmFormer's internal modality order: [FLAIR, T1ce, T1, T2].

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
            # Map benchmark order [T1, T1ce, T2, FLAIR] -> mmFormer [FLAIR, T1ce, T1, T2]
            mm_bools = [bench_bools[i] for i in self.BENCHMARK_TO_MMFORMER_INDICES]
            return torch.tensor([mm_bools] * B, dtype=torch.bool, device=self.device)

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

            # Reorder from benchmark order [T1, T1ce, T2, FLAIR] to mmFormer order [FLAIR, T1ce, T1, T2]
            return mask_t[:, self.BENCHMARK_TO_MMFORMER_INDICES]

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

            # Permute to mmFormer order
            return bench_mask[:, self.BENCHMARK_TO_MMFORMER_INDICES]

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
        Runs 3D volumetric segmentation inference using mmFormer.

        Args:
            input_data: (4, H, W, D) or (B, 4, H, W, D) volume or sample dict.
            mask: Optional missing-modality indicator ('S1'-'S4', boolean list, or tensor).
            return_logits: If True, returns (pred_mask, logits).
            roi_size: Sliding window patch size. Defaults to self.patch_size (128, 128, 128).
            overlap: Sliding window patch overlap ratio (0.0 - 1.0).
            blend_mode: Sliding window blending mode ('gaussian' or 'constant').
            postprocess_et: If True, applies official paper post-processing (zeros out ET if < et_threshold).
            et_threshold: Voxel threshold for ET post-processing (default: 500, matching paper).

        Returns:
            pred_labels: uint8 ndarray of shape (H, W, D) or (B, H, W, D) with standard BraTS labels (0, 1, 2, 4).
            logits (optional): float32 ndarray of raw class logits of shape (4, H, W, D) or (B, 4, H, W, D).
        """
        x_bench, was_4d, dict_mask = self._prepare_input_tensor(input_data)
        active_mask = mask if mask is not None else dict_mask

        # Resolve boolean presence mask aligned with mmFormer order [FLAIR, T1ce, T1, T2]
        mask_mm = self._resolve_mask(x_bench, active_mask)

        # Permute input channels: benchmark [T1, T1ce, T2, FLAIR] -> mmFormer [FLAIR, T1ce, T1, T2]
        x_mm = x_bench[:, self.BENCHMARK_TO_MMFORMER_INDICES, :, :, :]

        roi = roi_size or self.patch_size
        spatial_shape = x_mm.shape[2:]
        needs_sliding_window = any(s > r for s, r in zip(spatial_shape, roi))

        # Ensure network is in evaluation mode
        if hasattr(self.network, "is_training"):
            self.network.is_training = False
        self.network.eval()

        # Predictor closure for sliding-window inference
        def predictor_fn(patch: torch.Tensor) -> torch.Tensor:
            b_sw = patch.shape[0]
            curr_mask = mask_mm if mask_mm.shape[0] == b_sw else mask_mm[:1].repeat(b_sw, 1)
            out = self.network(patch, curr_mask)
            return out[0] if isinstance(out, (list, tuple)) else out

        if needs_sliding_window:
            logits = sliding_window_inference(
                inputs=x_mm,
                roi_size=roi,
                sw_batch_size=1,
                predictor=predictor_fn,
                overlap=overlap,
                mode=blend_mode,
            )
        else:
            out = self.network(x_mm, mask_mm)
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
        batch_mask = batch.get("missing_mask", batch.get("scenario", None))

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
