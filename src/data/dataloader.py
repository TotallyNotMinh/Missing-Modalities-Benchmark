import random
from typing import Optional, Tuple, Literal, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from .brats_dataset import BraTSDataset
from .splits import SplitManager
from .scenarios import ScenarioBuilder, SCENARIOS
from .augmentation import (
    get_synthesis_train_transforms,
    get_segmentation_train_transforms,
    get_val_transforms,
)


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
    processed_dir: Optional[str] = None,
    splits_file: Optional[str] = None,
    patch_size: Optional[list] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    seed: Optional[int] = None,
    task: Literal["synthesis", "segmentation"] = "segmentation",
    train_transform: Optional[Callable] = None,
    eval_transform: Optional[Callable] = None,
    cfg: Optional[dict] = None,
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
        splits_file: Path to splits.json.
        patch_size: 3D patch size for spatial cropping.
        batch_size: DataLoader batch size.
        num_workers: Subprocess workers for data loading.
        pin_memory: Pin memory for faster GPU transfer.
        seed: Random seed for loader worker initialization.
        task: 'segmentation' or 'synthesis'.
        train_transform: Custom training transform.
        eval_transform: Custom validation/test transform.
        cfg: Optional dict/Config to read defaults from.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Resolve parameters: explicit argument > cfg > hardcoded default
    if cfg is not None:
        processed_dir = processed_dir or cfg.get("paths", {}).get("processed_dir", "data/processed")
        splits_file = splits_file or cfg.get("paths", {}).get("splits_file", "data/splits/splits.json")
        patch_size = patch_size or cfg.get("patch", {}).get("size", [128, 128, 128])
        batch_size = batch_size if batch_size is not None else cfg.get("training", {}).get("batch_size", 2)
        num_workers = num_workers if num_workers is not None else cfg.get("num_workers", 4)
        pin_memory = pin_memory if pin_memory is not None else cfg.get("pin_memory", True)
        seed = seed if seed is not None else cfg.get("seed", 42)
    else:
        processed_dir = processed_dir or "data/processed"
        splits_file = splits_file or "data/splits/splits.json"
        patch_size = patch_size or [128, 128, 128]
        batch_size = batch_size if batch_size is not None else 2
        num_workers = num_workers if num_workers is not None else 4
        pin_memory = pin_memory if pin_memory is not None else True
        seed = seed if seed is not None else 42

    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario}'. Choose from {list(SCENARIOS.keys())}.")

    split_mgr = SplitManager(processed_dir=processed_dir, splits_file=splits_file)

    train_ids = split_mgr.get_split("train")
    val_ids = split_mgr.get_split("val")
    test_ids = split_mgr.get_split("test")

    # Resolve transforms according to task and protocol
    if train_transform is None:
        if task == "synthesis":
            train_transform = get_synthesis_train_transforms(patch_size=patch_size, cfg=cfg)
        else:
            train_transform = get_segmentation_train_transforms(patch_size=patch_size, cfg=cfg)

    if eval_transform is None:
        eval_transform = get_val_transforms(patch_size=patch_size, cfg=cfg)

    train_ds = ScenarioDataset(
        scenario_id=scenario,
        data_dir=processed_dir,
        patient_ids=train_ids,
        transform=train_transform,
    )
    val_ds = ScenarioDataset(
        scenario_id=scenario,
        data_dir=processed_dir,
        patient_ids=val_ids,
        transform=eval_transform,
    )
    test_ds = ScenarioDataset(
        scenario_id=scenario,
        data_dir=processed_dir,
        patient_ids=test_ids,
        transform=eval_transform,
    )

    eval_batch_size = 1 if task == "segmentation" else batch_size

    def make_loader(ds, shuffle, bsz=batch_size):
        return DataLoader(
            ds,
            batch_size=bsz,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            worker_init_fn=_seed_worker,
            generator=torch.Generator().manual_seed(seed),
        )

    return (
        make_loader(train_ds, shuffle=True, bsz=batch_size),
        make_loader(val_ds, shuffle=False, bsz=eval_batch_size),
        make_loader(test_ds, shuffle=False, bsz=eval_batch_size),
    )


def _seed_worker(worker_id: int) -> None:
    """Seeds numpy/random inside each DataLoader worker for reproducibility.
    
    Note: Uses torch.initial_seed() (derived from the DataLoader's Generator)
    rather than pipeline_utils.worker_init_fn which takes an explicit base_seed.
    Both approaches are valid; this one integrates with PyTorch's Generator seeding.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
