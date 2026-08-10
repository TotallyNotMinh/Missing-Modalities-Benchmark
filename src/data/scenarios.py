from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch

# Canonical channel ordering for BraTS 2020 (index → modality)
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

    def apply(self, volume: torch.Tensor) -> Dict[str, torch.Tensor | int]:
        """
        Applies the scenario to a full 4-channel volume.

        Args:
            volume: Full 4-channel MRI tensor of shape (4, H, W, D).

        Returns:
            Dict with:
                'inputs'       : (3, H, W, D) available modalities.
                'target'       : (1, H, W, D) missing modality.
                'missing_flag' : int index of missing modality.
                'scenario'     : scenario name string.
        """
        if volume.dim() != 4 or volume.shape[0] != 4:
            raise ValueError(
                f"Expected volume of shape (4, H, W, D), got {tuple(volume.shape)}"
            )

        inputs = volume[list(self.scenario.input_indices)]      # (3, H, W, D)
        target = volume[[self.scenario.target_index]]           # (1, H, W, D)

        return {
            "inputs": inputs,
            "target": target,
            "missing_flag": self.scenario.target_index,
            "scenario": self.scenario.name,
        }

    def reconstruct_full(self, inputs: torch.Tensor, synthetic: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs a full 4-channel volume from available + synthetic modality.
        Used when feeding nnU-Net / SwinUNETR in Synthetic mode.

        Args:
            inputs: (3, H, W, D) or (B, 3, H, W, D) real available channels.
            synthetic: (1, H, W, D) or (B, 1, H, W, D) synthesized missing channel.

        Returns:
            (4, H, W, D) or (B, 4, H, W, D) full volume in canonical T1/T1ce/T2/FLAIR order.
        """
        is_batched = inputs.dim() == 5
        if is_batched:
            B, _, H, W, D = inputs.shape
            full = torch.zeros(B, 4, H, W, D, dtype=inputs.dtype, device=inputs.device)
            for out_idx, in_idx in enumerate(self.scenario.input_indices):
                full[:, in_idx] = inputs[:, out_idx]
            full[:, self.scenario.target_index] = synthetic[:, 0]
        else:
            full = torch.zeros(4, *inputs.shape[1:], dtype=inputs.dtype, device=inputs.device)
            for out_idx, in_idx in enumerate(self.scenario.input_indices):
                full[in_idx] = inputs[out_idx]
            full[self.scenario.target_index] = synthetic[0]

        return full

    def reconstruct_native(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs a 4-channel volume by zero-padding the missing modality.
        Used to feed a 4-channel model (like nnU-Net) in 'native_missing' mode.

        Args:
            inputs: (3, H, W, D) or (B, 3, H, W, D) real available channels.

        Returns:
            (4, H, W, D) or (B, 4, H, W, D) full volume with zeroed target channel.
        """
        is_batched = inputs.dim() == 5
        if is_batched:
            B, _, H, W, D = inputs.shape
            full = torch.zeros(B, 4, H, W, D, dtype=inputs.dtype, device=inputs.device)
            for out_idx, in_idx in enumerate(self.scenario.input_indices):
                full[:, in_idx] = inputs[:, out_idx]
        else:
            full = torch.zeros(4, *inputs.shape[1:], dtype=inputs.dtype, device=inputs.device)
            for out_idx, in_idx in enumerate(self.scenario.input_indices):
                full[in_idx] = inputs[out_idx]

        return full
