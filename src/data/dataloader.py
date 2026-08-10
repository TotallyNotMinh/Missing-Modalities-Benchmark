import random
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .brats_dataset import BraTSDataset
from .splits import SplitManager
from .scenarios import ScenarioBuilder, SCENARIOS
from .augmentation import get_train_transforms, get_val_transforms


class ScenarioDataset(BraTSDataset):
    """
    Extends BraTSDataset to apply a missing-modality scenario
    to each sample before returning it.

    Returned sample dict adds:
        'inputs'       : (3, H, W, D) available modality channels
        'target'       : (1, H, W, D) missing modality channel
        'missing_flag' : int index of missing modality
        'scenario'     : scenario name string
    """

    def __init__(self, scenario_id: str, **kwargs):
        super().__init__(**kwargs)
        self.builder = ScenarioBuilder(scenario_id)

    def __getitem__(self, idx: int) -> dict:
        sample = super().__getitem__(idx)
        scenario_data = self.builder.apply(sample["modalities"])
        sample.update(scenario_data)
        return sample


def get_dataloaders(
    scenario: str,
    processed_dir: str = "data/processed",
    splits_file: str = "data/splits/splits.json",
    patch_size: list = [128, 128, 128],
    batch_size: int = 2,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Factory function that returns train, val, and test DataLoaders for a given
    missing-modality scenario.

    Each yielded batch dict contains:
        patient_id  : List[str]
        modalities  : (B, 4, H, W, D) — full 4-channel volume
        mask        : (B, 1, H, W, D) — segmentation label map
        inputs      : (B, 3, H, W, D) — available modalities (scenario)
        target      : (B, 1, H, W, D) — missing modality (scenario)
        missing_flag: (B,) int — missing channel index
        scenario    : List[str]

    Args:
        scenario: Missing modality scenario ('S1', 'S2', 'S3', 'S4').
        processed_dir: Root of preprocessed patient volumes.
        splits_file: Path to the frozen splits JSON.
        patch_size: 3D patch size for cropping.
        batch_size: Training batch size.
        num_workers: DataLoader worker processes.
        pin_memory: Pin memory for faster GPU transfer.
        seed: Random seed (for reproducibility of worker init).

    Returns:
        (train_loader, val_loader, test_loader)
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario}'. Choose from {list(SCENARIOS.keys())}.")

    split_mgr = SplitManager(processed_dir=processed_dir, splits_file=splits_file)

    train_ids = split_mgr.get_split("train")
    val_ids = split_mgr.get_split("val")
    test_ids = split_mgr.get_split("test")

    train_ds = ScenarioDataset(
        scenario_id=scenario,
        data_dir=processed_dir,
        patient_ids=train_ids,
        transform=get_train_transforms(patch_size=patch_size),
    )
    val_ds = ScenarioDataset(
        scenario_id=scenario,
        data_dir=processed_dir,
        patient_ids=val_ids,
        transform=get_val_transforms(patch_size=patch_size),
    )
    test_ds = ScenarioDataset(
        scenario_id=scenario,
        data_dir=processed_dir,
        patient_ids=test_ids,
        transform=get_val_transforms(patch_size=patch_size),
    )

    def make_loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            worker_init_fn=_seed_worker,
            generator=torch.Generator().manual_seed(seed),
        )

    return (
        make_loader(train_ds, shuffle=True),
        make_loader(val_ds, shuffle=False),
        make_loader(test_ds, shuffle=False),
    )


def _seed_worker(worker_id: int) -> None:
    """Seeds numpy/random/torch inside each DataLoader worker for reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
