from dataclasses import dataclass
from typing import Dict, List, Tuple, Union
import numpy as np
import torch

# Canonical channel ordering for BraTS 2020 (index -> modality)
MODALITY_NAMES = ["T1", "T1ce", "T2", "FLAIR"]
MODALITY_SUFFIXES = ["t1", "t1ce", "t2", "flair"]  # file suffix order


@dataclass(frozen=True)
class Scenario:
    """
    Defines a single missing-modality scenario.

    Attributes:
        name: Scenario identifier (e.g., 'S1').
        input_indices: Channel indices of available modalities (length=3).
        target_index: Channel index of the missing (target) modality.
        clinical_motivation: Human-readable description.
    """
    name: str
    input_indices: Tuple[int, ...]
    target_index: int
    clinical_motivation: str

    @property
    def input_names(self) -> List[str]:
        return [MODALITY_NAMES[i] for i in self.input_indices]

    @property
    def target_name(self) -> str:
        return MODALITY_NAMES[self.target_index]

    def __repr__(self):
        return (
            f"Scenario({self.name}: inputs={self.input_names}, "
            f"target={self.target_name})"
        )


# Frozen scenario registry — the single source of truth for all 4 scenarios
SCENARIOS: Dict[str, Scenario] = {
    "S1": Scenario(
        name="S1",
        input_indices=(0, 1, 2),   # T1, T1ce, T2
        target_index=3,            # FLAIR
        clinical_motivation="FLAIR absent — most commonly missing in retrospective data.",
    ),
    "S2": Scenario(
        name="S2",
        input_indices=(0, 2, 3),   # T1, T2, FLAIR
        target_index=1,            # T1ce
        clinical_motivation="T1ce absent — contrast skipped (allergy or cost).",
    ),
    "S3": Scenario(
        name="S3",
        input_indices=(1, 2, 3),   # T1ce, T2, FLAIR
        target_index=0,            # T1
        clinical_motivation="T1 absent — pre-contrast occasionally omitted.",
    ),
    "S4": Scenario(
        name="S4",
        input_indices=(0, 1, 3),   # T1, T1ce, FLAIR
        target_index=2,            # T2
        clinical_motivation="T2 absent — emergency scanning protocol.",
    ),
}


def validate_scenarios_against_config(cfg: dict) -> None:
    """Validates that the hardcoded SCENARIOS registry is consistent with config.yaml.
    Call this once at startup to catch drift between code and config."""
    config_scenarios = {s["name"]: s for s in cfg["missing_modality"]["scenarios"]}
    modality_order = [m.lower() for m in cfg.get("modalities", {}).get("order", ["t1", "t1ce", "t2", "flair"])]
    
    for name, scenario in SCENARIOS.items():
        if name not in config_scenarios:
            raise ValueError(
                f"Scenario '{name}' exists in code but not in config.yaml. "
                f"Config scenarios: {list(config_scenarios.keys())}"
            )
        cfg_drop = [d.lower() for d in config_scenarios[name]["drop"]]
        code_target = MODALITY_SUFFIXES[scenario.target_index]
        if code_target not in cfg_drop:
            raise ValueError(
                f"Scenario '{name}' mismatch: code drops '{code_target}' "
                f"but config drops {cfg_drop}"
            )


class ScenarioBuilder:
    """
    Applies a missing-modality scenario to a full 4-channel volume.

    Given a full 4-channel tensor (4, H, W, D) and a scenario ID, produces:
        - input_channels (3, H, W, D): The 3 available modality channels.
        - target_channel (1, H, W, D): The single missing modality (for generators).
        - missing_flag (int): Index 0-3 of the missing modality (for segmenters).
    """

    def __init__(self, scenario_id: str):
        if scenario_id not in SCENARIOS:
            raise ValueError(
                f"Unknown scenario '{scenario_id}'. Choose from {list(SCENARIOS.keys())}."
            )
        self.scenario = SCENARIOS[scenario_id]

    def apply(self, volume: Union[torch.Tensor, np.ndarray]) -> Dict[str, Union[torch.Tensor, np.ndarray, int, str]]:
        """
        Applies the scenario to a full 4-channel volume.

        Args:
            volume: Full 4-channel MRI tensor of shape (4, H, W, D) or (B, 4, H, W, D).

        Returns:
            Dict with:
                'inputs'       : available modalities.
                'target'       : missing modality.
                'missing_flag' : int index of missing modality.
                'scenario'     : scenario name string.
        """
        is_numpy = isinstance(volume, np.ndarray)
        if is_numpy:
            volume = torch.from_numpy(volume)

        if volume.dim() == 4:
            if volume.shape[0] != 4:
                raise ValueError(f"Expected 4 channels in volume (4, H, W, D), got {tuple(volume.shape)}")
            inputs = volume[list(self.scenario.input_indices)]
            target = volume[[self.scenario.target_index]]
        elif volume.dim() == 5:
            if volume.shape[1] != 4:
                raise ValueError(f"Expected 4 channels in volume (B, 4, H, W, D), got {tuple(volume.shape)}")
            inputs = volume[:, list(self.scenario.input_indices)]
            target = volume[:, [self.scenario.target_index]]
        else:
            raise ValueError(f"Expected volume of dim 4 or 5, got {volume.dim()} with shape {tuple(volume.shape)}")

        if is_numpy:
            inputs = inputs.numpy()
            target = target.numpy()

        return {
            "inputs": inputs,
            "target": target,
            "missing_flag": self.scenario.target_index,
            "scenario": self.scenario.name,
        }

    def reconstruct_full(
        self,
        inputs: Union[torch.Tensor, np.ndarray],
        synthetic: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Reconstructs a full 4-channel volume from available + synthetic modality.
        Used when feeding nnU-Net / SwinUNETR in Synthetic mode.

        Args:
            inputs: (3, H, W, D) or (B, 3, H, W, D) real available channels.
            synthetic: (1, H, W, D), (H, W, D), (B, 1, H, W, D), or (B, H, W, D) synthesized channel.

        Returns:
            (4, H, W, D) or (B, 4, H, W, D) full volume in canonical T1/T1ce/T2/FLAIR order.
        """
        is_numpy = isinstance(inputs, np.ndarray)
        if is_numpy:
            inputs = torch.from_numpy(inputs)
        if isinstance(synthetic, np.ndarray):
            synthetic = torch.from_numpy(synthetic)

        synthetic = synthetic.to(dtype=inputs.dtype, device=inputs.device)

        is_batched = inputs.dim() == 5
        if is_batched:
            B, C_in, H, W, D = inputs.shape
            if C_in != 3:
                raise ValueError(f"Expected 3 input channels for batched inputs, got {C_in}")
            # Ensure synthetic is shape (B, 1, H, W, D)
            if synthetic.dim() == 4:
                synthetic = synthetic.unsqueeze(1)
            elif synthetic.dim() == 3:
                synthetic = synthetic.unsqueeze(0).unsqueeze(0)

            full = torch.zeros(B, 4, H, W, D, dtype=inputs.dtype, device=inputs.device)
            for out_idx, in_idx in enumerate(self.scenario.input_indices):
                full[:, in_idx] = inputs[:, out_idx]
            full[:, self.scenario.target_index] = synthetic[:, 0]
        else:
            C_in = inputs.shape[0]
            if C_in != 3:
                raise ValueError(f"Expected 3 input channels for unbatched inputs, got {C_in}")
            # Ensure synthetic is shape (1, H, W, D) or (H, W, D)
            if synthetic.dim() == 4 and synthetic.shape[0] == 1:
                synthetic_sq = synthetic[0]
            elif synthetic.dim() == 3:
                synthetic_sq = synthetic
            else:
                synthetic_sq = synthetic.squeeze()

            full = torch.zeros(4, *inputs.shape[1:], dtype=inputs.dtype, device=inputs.device)
            for out_idx, in_idx in enumerate(self.scenario.input_indices):
                full[in_idx] = inputs[out_idx]
            full[self.scenario.target_index] = synthetic_sq

        return full.numpy() if is_numpy else full

    def reconstruct_native(
        self,
        inputs: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Reconstructs a 4-channel volume by zero-padding the missing modality.
        Used to feed a 4-channel model (like nnU-Net) in 'native_missing' mode.

        Args:
            inputs: (3, H, W, D) or (B, 3, H, W, D) real available channels.

        Returns:
            (4, H, W, D) or (B, 4, H, W, D) full volume with zeroed target channel.
        """
        is_numpy = isinstance(inputs, np.ndarray)
        if is_numpy:
            inputs = torch.from_numpy(inputs)

        is_batched = inputs.dim() == 5
        if is_batched:
            B, C_in, H, W, D = inputs.shape
            full = torch.zeros(B, 4, H, W, D, dtype=inputs.dtype, device=inputs.device)
            for out_idx, in_idx in enumerate(self.scenario.input_indices):
                full[:, in_idx] = inputs[:, out_idx]
        else:
            full = torch.zeros(4, *inputs.shape[1:], dtype=inputs.dtype, device=inputs.device)
            for out_idx, in_idx in enumerate(self.scenario.input_indices):
                full[in_idx] = inputs[out_idx]

        return full.numpy() if is_numpy else full
