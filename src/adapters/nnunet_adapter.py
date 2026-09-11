import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference

from dynamic_network_architectures.architectures.unet import PlainConvUNet

from src.metrics.segmentation import compute_segmentation_metrics
from src.utils.pipeline_utils import Config, load_config


class nnUNetAdapter:
    """
    Adapter for nnU-Net v2 downstream segmentation evaluator and oracle.

    Handles:
      1. Canonical 4-channel BraTS input: (T1, T1ce, T2, FLAIR).
      2. Inference with official nnU-Net weights or native PlainConvUNet fallback.
      3. Sliding-window 3D volumetric inference (via MONAI or nnUNetPredictor).
      4. Label re-mapping to standard BraTS convention:
           0: Background
           1: Necrotic / Non-enhancing tumor (NCR/NET)
           2: Peritumoral edema (ED)
           4: Enhancing tumor (ET)
      5. Subregion metric computation (WT, TC, ET Dice and HD95).
    """

    # Mapping from model class index (0, 1, 2, 3) to standard BraTS label (0, 1, 2, 4)
    CLASS_TO_BRATS_LABEL = np.array([0, 1, 2, 4], dtype=np.uint8)

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        num_input_channels: int = 4,
        num_classes: int = 3,
        deep_supervision: bool = True,
        network: Optional[nn.Module] = None,
    ):
        """
        Args:
            weights_path: Path to checkpoint (.pth/.pt) or nnU-Net trained model directory.
            device: Device to run inference on ('cuda', 'cpu', or torch.device).
            patch_size: Default 3D patch ROI size for sliding-window evaluation.
            num_input_channels: Number of input MRI modalities (default 4: T1, T1ce, T2, FLAIR).
            num_classes: Number of output classes (default 3: WT, TC, ET for region-based; or 4 for categorical).
            deep_supervision: Whether multi-scale deep supervision is enabled in the network.
            network: Optional custom PyTorch nn.Module. If None, builds PlainConvUNet.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        else:
            self.device = device

        self.patch_size = tuple(patch_size)
        self.num_input_channels = num_input_channels
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.weights_path = Path(weights_path) if weights_path is not None else None

        self.is_official_predictor = False
        self.predictor = None

        if network is not None:
            self.network = network.to(self.device)
            self.network.eval()
            print(f"[nnUNetAdapter] Using supplied custom PyTorch network on {self.device}.")
        elif self.weights_path is not None and self._is_nnunet_folder(self.weights_path):
            self._init_official_predictor(self.weights_path)
        else:
            self.network = self._build_default_network()
            if self.weights_path is not None and self.weights_path.is_file():
                self._load_state_dict(self.weights_path)
                print(f"[nnUNetAdapter] Initialized 6-stage PlainConvUNet with weights loaded from '{self.weights_path}' on {self.device}.")
            elif self.weights_path is not None and not self.weights_path.exists():
                print(f"[nnUNetAdapter] Initialized primary 6-stage PlainConvUNet (31.2M params) for training on {self.device} (weights path '{self.weights_path}' does not exist yet).")
            else:
                print(f"[nnUNetAdapter] Initialized primary 6-stage PlainConvUNet (31.2M params) on {self.device}.")
            self.network.to(self.device)
            self.network.eval()

    def _is_nnunet_folder(self, path: Path) -> bool:
        """Checks if directory contains standard nnUNet v2 model folder artifacts."""
        return path.is_dir() and (path / "plans.json").exists() and (path / "dataset.json").exists()

    def _init_official_predictor(self, model_folder: Path) -> None:
        """Initializes the official nnUNetPredictor from a trained model folder."""
        try:
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
            self.predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=True,
                perform_everything_on_device=(self.device.type == "cuda"),
                device=self.device,
                verbose=False,
                allow_tqdm=False,
            )
            self.predictor.initialize_from_trained_model_folder(
                model_training_output_dir=str(model_folder),
                use_folds=None,
                checkpoint_name="checkpoint_final.pth",
            )
            self.network = self.predictor.network
            self.network.to(self.device)
            self.network.eval()
            self.is_official_predictor = True
            print(f"[nnUNetAdapter] Initialized official nnUNetPredictor from model folder '{model_folder}' on {self.device}.")
        except Exception as e:
            print(f"[nnUNetAdapter] [FALLBACK ACTIVATED] Could not initialize official nnUNetPredictor from '{model_folder}' ({e}). "
                  f"Falling back to native 6-stage PlainConvUNet architecture on {self.device}.")
            self.network = self._build_default_network()
            self.network.to(self.device)
            self.network.eval()
            self.is_official_predictor = False

    def _build_default_network(self) -> nn.Module:
        """
        Builds a 3D PlainConvUNet matching the official nnU-Net v2 architecture.
        For standard BraTS patch sizes (e.g. 128x128x128), this builds the full
        6-stage backbone with feature channels (32, 64, 128, 256, 320, 320)
        comprising ~31.2M parameters.
        For smaller test patch sizes, stages are dynamically adapted so that
        spatial downsampling remains valid.
        """
        min_dim = min(self.patch_size) if hasattr(self, "patch_size") and self.patch_size else 128
        stages = 1
        curr = min_dim
        while stages < 6 and curr >= 4:
            curr //= 2
            stages += 1
        stages = max(3, stages)

        BASE_FEATURES = (32, 64, 128, 256, 320, 320)
        features = BASE_FEATURES[:stages]
        strides = ((1, 1, 1),) + tuple((2, 2, 2) for _ in range(stages - 1))
        kernel_sizes = tuple((3, 3, 3) for _ in range(stages))

        net = PlainConvUNet(
            input_channels=self.num_input_channels,
            n_stages=stages,
            features_per_stage=features,
            conv_op=nn.Conv3d,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=tuple(2 for _ in range(stages)),
            num_classes=self.num_classes,
            n_conv_per_stage_decoder=tuple(2 for _ in range(stages - 1)),
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=self.deep_supervision,
        )
        return net

    def _load_state_dict(self, checkpoint_path: Path) -> None:
        """Loads model weights from a .pth or .pt checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and "network_weights" in ckpt:
            state_dict = ckpt["network_weights"]
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif isinstance(ckpt, dict):
            state_dict = ckpt
        else:
            raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}")

        # Clean DDP / module prefixes if present
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
    ) -> "nnUNetAdapter":
        """Factory constructor instantiating adapter from project Config."""
        if cfg is None:
            cfg = load_config()

        target_weights = weights_path or cfg.paths.get("oracle_weights", None)
        target_device = device or cfg.get("device", "cuda")
        target_patch = tuple(cfg.patch.get("size", (128, 128, 128)))
        model_cfg = cfg.get("model", {})
        target_classes = model_cfg.get("num_classes", 3)
        target_ds = model_cfg.get("deep_supervision", True)

        return cls(
            weights_path=target_weights,
            device=target_device,
            patch_size=target_patch,
            num_input_channels=len(cfg.modalities.get("order", ["t1", "t1ce", "t2", "flair"])),
            num_classes=target_classes,
            deep_supervision=target_ds,
        )

    def _prepare_input_tensor(
        self,
        input_data: Union[np.ndarray, torch.Tensor, Dict[str, Any]],
    ) -> Tuple[torch.Tensor, bool]:
        """
        Normalizes input data into a 5D PyTorch Tensor (B, C, H, W, D).

        Returns:
            tensor: (B, C, H, W, D) on self.device
            was_4d: Boolean indicating if input was originally 4D (single sample)
        """
        if isinstance(input_data, dict):
            if "modalities" in input_data:
                tensor = input_data["modalities"]
            elif "inputs" in input_data:
                tensor = input_data["inputs"]
            else:
                raise KeyError(f"Expected 'modalities' or 'inputs' in dictionary, got {list(input_data.keys())}")
        elif isinstance(input_data, np.ndarray):
            tensor = torch.from_numpy(input_data)
        elif isinstance(input_data, torch.Tensor):
            tensor = input_data
        else:
            raise TypeError(f"Unsupported input type: {type(input_data)}")

        tensor = tensor.float()

        if tensor.dim() == 4:
            # (C, H, W, D) -> (1, C, H, W, D)
            was_4d = True
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() == 5:
            was_4d = False
        else:
            raise ValueError(f"Expected 4D (C, H, W, D) or 5D (B, C, H, W, D) tensor, got {tensor.shape}")

        if tensor.shape[1] != self.num_input_channels:
            raise ValueError(
                f"Expected {self.num_input_channels} input channels, but got {tensor.shape[1]}"
            )

        return tensor.to(self.device), was_4d

    @torch.no_grad()
    def predict(
        self,
        input_data: Union[np.ndarray, torch.Tensor, Dict[str, Any]],
        return_logits: bool = False,
        roi_size: Optional[Tuple[int, int, int]] = None,
        overlap: float = 0.5,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Runs 3D segmentation inference.

        Args:
            input_data: (4, H, W, D) or (B, 4, H, W, D) volume or sample dict.
            return_logits: If True, returns (pred_mask, logits).
            roi_size: Sliding window patch size. Defaults to self.patch_size.
            overlap: Sliding window patch overlap ratio (0.0 - 1.0).

        Returns:
            pred_labels: uint8 ndarray of shape (H, W, D) or (B, H, W, D) with BraTS labels (0, 1, 2, 4).
            logits (optional): float32 ndarray of raw class logits.
        """
        x, was_4d = self._prepare_input_tensor(input_data)
        roi = roi_size or self.patch_size

        spatial_shape = x.shape[2:]
        needs_sliding_window = any(s > r for s, r in zip(spatial_shape, roi))

        # Handle deep supervision output: extract full-resolution head (index 0)
        has_ds = hasattr(self.network, "decoder") and getattr(self.network.decoder, "deep_supervision", False)
        predictor_fn = (lambda inp: self.network(inp)[0]) if has_ds else self.network

        if needs_sliding_window:
            logits = sliding_window_inference(
                inputs=x,
                roi_size=roi,
                sw_batch_size=1,
                predictor=predictor_fn,
                overlap=overlap,
                mode="gaussian",
            )
        else:
            out = self.network(x)
            logits = out[0] if isinstance(out, (list, tuple)) else out

        if logits.shape[1] == 3:
            # Region-based prediction (official nnU-Net BraTS standard)
            # Channel 0: Whole Tumor (WT), 1: Tumor Core (TC), 2: Enhancing Tumor (ET)
            probs = torch.sigmoid(logits)
            wt = (probs[:, 0] > 0.5).cpu().numpy()
            tc = (probs[:, 1] > 0.5).cpu().numpy()
            et = (probs[:, 2] > 0.5).cpu().numpy()

            pred_labels = np.zeros(wt.shape, dtype=np.uint8)
            pred_labels[wt] = 2  # Edema
            pred_labels[tc] = 1  # Necrosis / Non-enhancing
            pred_labels[et] = 4  # Enhancing tumor
        else:
            # Categorical softmax fallback
            pred_class = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
            pred_labels = self.CLASS_TO_BRATS_LABEL[pred_class]

        logits_np = logits.cpu().numpy()

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
        voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        roi_size: Optional[Tuple[int, int, int]] = None,
    ) -> Dict[str, float]:
        """
        Predicts segmentation and computes standardized BraTS subregion metrics.

        Returns:
            Dict containing Dice_WT, Dice_TC, Dice_ET, Dice_Mean,
                           HD95_WT, HD95_TC, HD95_ET, HD95_Mean.
        """
        pred_mask = self.predict(input_data, return_logits=False, roi_size=roi_size)
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
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], np.ndarray]]:
        """
        Evaluates a batch of samples from DataLoader and returns per-patient metrics.
        If return_logits is True, also returns full predicted logits array of shape (B, num_classes, H, W, D).
        """
        inputs = batch.get("modalities", batch.get("inputs"))
        targets = batch.get("mask")
        patient_ids = batch.get("patient_id", [f"patient_{i}" for i in range(len(inputs))])

        if isinstance(targets, torch.Tensor) and targets.dim() == 5 and targets.shape[1] == 1:
            targets = targets.squeeze(1)

        results = []
        all_logits = []
        for i in range(len(inputs)):
            sample_in = inputs[i]
            sample_target = targets[i]
            pid = patient_ids[i] if isinstance(patient_ids, (list, tuple)) else str(patient_ids)

            if return_logits:
                pred_mask, logits_np = self.predict(sample_in, return_logits=True)
                all_logits.append(logits_np)
            else:
                pred_mask = self.predict(sample_in, return_logits=False)

            metrics = compute_segmentation_metrics(
                target_mask=sample_target,
                pred_mask=pred_mask,
                voxel_spacing=voxel_spacing,
            )
            record = {"patient_id": pid, **metrics}
            results.append(record)

        if return_logits:
            stacked_logits = np.stack(all_logits, axis=0) if len(all_logits) > 1 else np.expand_dims(all_logits[0], axis=0)
            return results, stacked_logits
        return results
