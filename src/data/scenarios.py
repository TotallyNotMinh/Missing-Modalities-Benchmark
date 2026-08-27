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
        input_indices: Channel indices of available modalities.
        target_indices: Channel indices of the missing (target) modalities.
        clinical_motivation: Human-readable description.
    """
    name: str
    input_indices: Tuple[int, ...]
    target_indices: Tuple[int, ...]
    clinical_motivation: str

    @property
    def input_names(self) -> List[str]:
        return [MODALITY_NAMES[i] for i in self.input_indices]

    @property
    def target_names(self) -> List[str]:
        return [MODALITY_NAMES[i] for i in self.target_indices]

    def __repr__(self):
        return (
            f"Scenario({self.name}: inputs={self.input_names}, "
            f"targets={self.target_names})"
        )


# Frozen scenario registry — the single source of truth for all scenarios
SCENARIOS: Dict[str, Scenario] = {
    "S1": Scenario(
        name="S1",
        input_indices=(0, 1, 2),   # T1, T1ce, T2
        target_indices=(3,),       # FLAIR
        clinical_motivation="FLAIR absent — most commonly missing in retrospective data.",
    ),
    "S2": Scenario(
        name="S2",
        input_indices=(0, 2, 3),   # T1, T2, FLAIR
        target_indices=(1,),       # T1ce
        clinical_motivation="T1ce absent — contrast skipped (allergy or cost).",
    ),
    "S3": Scenario(
        name="S3",
        input_indices=(1, 2, 3),   # T1ce, T2, FLAIR
        target_indices=(0,),       # T1
        clinical_motivation="T1 absent — pre-contrast occasionally omitted.",
    ),
    "S4": Scenario(
        name="S4",
        input_indices=(0, 1, 3),   # T1, T1ce, FLAIR
        target_indices=(2,),       # T2
        clinical_motivation="T2 absent — emergency scanning protocol.",
    ),
    "two_missing": Scenario(
        name="two_missing",
        input_indices=(0, 2),      # T1, T2
        target_indices=(1, 3),     # T1ce, FLAIR
        clinical_motivation="Accelerated or abbreviated protocol.",
    ),
    "three_missing": Scenario(
        name="three_missing",
        input_indices=(0,),        # T1
        target_indices=(1, 2, 3),  # T1ce, T2, FLAIR
        clinical_motivation="Extreme emergency / triage.",
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
        cfg_drop = set(d.lower() for d in config_scenarios[name]["drop"])
        code_targets = set(MODALITY_SUFFIXES[i] for i in scenario.target_indices)
        if code_targets != cfg_drop:
            raise ValueError(
                f"Scenario '{name}' mismatch: code drops {code_targets} "
                f"but config drops {cfg_drop}"
            )


class ScenarioBuilder:
    """
    Applies a missing-modality scenario to a full 4-channel volume.

    Given a full 4-channel tensor (4, H, W, D) and a scenario ID, produces:
        - input_channels (K, H, W, D): The available modality channels.
        - target_channels (M, H, W, D): The missing modality channels.
        - missing_flag (Tuple[int, ...]): Indices 0-3 of the missing modalities.
    """

    def __init__(self, scenario_id: str):
        if scenario_id not in SCENARIOS:
            raise ValueError(
                f"Unknown scenario '{scenario_id}'. Choose from {list(SCENARIOS.keys())}."
            )
        self.scenario = SCENARIOS[scenario_id]

    def apply(self, volume: Union[torch.Tensor, np.ndarray]) -> Dict[str, Union[torch.Tensor, np.ndarray, Tuple[int, ...], str]]:
        """
        Applies the scenario to a full 4-channel volume.

        Args:
            volume: Full 4-channel MRI tensor of shape (4, H, W, D) or (B, 4, H, W, D).

        Returns:
            Dict with:
                'inputs'       : available modalities.
                'target'       : missing modality.
                'missing_flag' : tuple of ints index of missing modalities.
                'scenario'     : scenario name string.
        """
        is_numpy = isinstance(volume, np.ndarray)
        if is_numpy:
            volume = torch.from_numpy(volume)

        if volume.dim() == 4:
            if volume.shape[0] != 4:
                raise ValueError(f"Expected 4 channels in volume (4, H, W, D), got {tuple(volume.shape)}")
            inputs = volume[list(self.scenario.input_indices)]
            target = volume[list(self.scenario.target_indices)]
        elif volume.dim() == 5:
            if volume.shape[1] != 4:
                raise ValueError(f"Expected 4 channels in volume (B, 4, H, W, D), got {tuple(volume.shape)}")
            inputs = volume[:, list(self.scenario.input_indices)]
            target = volume[:, list(self.scenario.target_indices)]
        else:
            raise ValueError(f"Expected volume of dim 4 or 5, got {volume.dim()} with shape {tuple(volume.shape)}")

        if is_numpy:
            inputs = inputs.numpy()
            target = target.numpy()

        return {
            "inputs": inputs,
            "target": target,
            "missing_flag": self.scenario.target_indices,
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
            inputs: (K, H, W, D) or (B, K, H, W, D) real available channels.
            synthetic: (M, H, W, D) or (B, M, H, W, D) synthesized channel(s).

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
        
        # Determine number of expected inputs and targets based on scenario
        expected_inputs = len(self.scenario.input_indices)
        expected_targets = len(self.scenario.target_indices)

        if is_batched:
            B, C_in, H, W, D = inputs.shape
            if C_in != expected_inputs:
                raise ValueError(f"Expected {expected_inputs} input channels for batched inputs, got {C_in}")
            
            # Ensure synthetic has correct shape (B, M, H, W, D)
            if synthetic.dim() == 4 and expected_targets == 1:
                synthetic = synthetic.unsqueeze(1)
            elif synthetic.dim() == 3 and expected_targets == 1:
                synthetic = synthetic.unsqueeze(0).unsqueeze(0)
            
            if synthetic.shape[1] != expected_targets:
                raise ValueError(f"Expected {expected_targets} synthetic channels, got {synthetic.shape[1]}")

            full = torch.zeros(B, 4, H, W, D, dtype=inputs.dtype, device=inputs.device)
            for out_idx, in_idx in enumerate(self.scenario.input_indices):
                full[:, in_idx] = inputs[:, out_idx]
            for out_idx, target_idx in enumerate(self.scenario.target_indices):
                full[:, target_idx] = synthetic[:, out_idx]
        else:
            C_in = inputs.shape[0]
            if C_in != expected_inputs:
                raise ValueError(f"Expected {expected_inputs} input channels for unbatched inputs, got {C_in}")
            
            # Ensure synthetic has correct shape (M, H, W, D)
            if synthetic.dim() == 4 and synthetic.shape[0] == 1 and expected_targets == 1:
                synthetic_sq = synthetic[0].unsqueeze(0)
            elif synthetic.dim() == 3 and expected_targets == 1:
                synthetic_sq = synthetic.unsqueeze(0)
            else:
                synthetic_sq = synthetic

            if synthetic_sq.shape[0] != expected_targets:
                raise ValueError(f"Expected {expected_targets} synthetic channels, got {synthetic_sq.shape[0]}")

            full = torch.zeros(4, *inputs.shape[1:], dtype=inputs.dtype, device=inputs.device)
            for out_idx, in_idx in enumerate(self.scenario.input_indices):
                full[in_idx] = inputs[out_idx]
            for out_idx, target_idx in enumerate(self.scenario.target_indices):
                full[target_idx] = synthetic_sq[out_idx]

        return full.numpy() if is_numpy else full

    def reconstruct_native(
        self,
        inputs: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Reconstructs a 4-channel volume by zero-padding the missing modality.
        Used to feed a 4-channel model (like nnU-Net) in 'native_missing' mode.

        Args:
            inputs: (K, H, W, D) or (B, K, H, W, D) real available channels.

        Returns:
            (4, H, W, D) or (B, 4, H, W, D) full volume with zeroed target channel(s).
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
