import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure external mmFormer is importable
MMFORMER_DIR = PROJECT_ROOT / "externals" / "mmformer" / "mmformer"
if MMFORMER_DIR.exists() and str(MMFORMER_DIR) not in sys.path:
    sys.path.insert(0, str(MMFORMER_DIR))

from scripts.train_mmformer import (
    COMBINATORIAL_MASKS,
    BENCHMARK_TO_MMFORMER,
    train_one_epoch,
)


class MockBraTSBatch:
    """Provides a synthetic batch matching the format of BraTSDataset."""
    @staticmethod
    def create(b_size=1, patch_size=(32, 32, 32)):
        modalities = torch.randn(b_size, 4, *patch_size)
        # Masks with labels in {0, 1, 2, 4}
        mask = torch.randint(0, 3, (b_size, 1, *patch_size), dtype=torch.long)
        mask[mask == 3] = 4
        return {"modalities": modalities, "mask": mask}


def test_combinatorial_masks_shape():
    """Verify that combinatorial masks has exactly 15 valid 4-modality combinations."""
    assert COMBINATORIAL_MASKS.shape == (15, 4)
    # Check that each has at least 1 modality present
    assert (COMBINATORIAL_MASKS.sum(axis=1) >= 1).all()
    # Check full modality mask exists
    assert COMBINATORIAL_MASKS[-1].all()


def test_benchmark_to_mmformer_permutation():
    """
    Verify permutation maps:
    Index 0 (T1)    -> Index 2 in mmFormer
    Index 1 (T1ce)  -> Index 1 in mmFormer
    Index 2 (T2)    -> Index 3 in mmFormer
    Index 3 (FLAIR) -> Index 0 in mmFormer
    """
    # Benchmark indices: 0: T1, 1: T1ce, 2: T2, 3: FLAIR
    # BENCHMARK_TO_MMFORMER = [3, 1, 0, 2]
    assert BENCHMARK_TO_MMFORMER == [3, 1, 0, 2]
    # In mmFormer:
    # 0 should be FLAIR (benchmark index 3)
    assert BENCHMARK_TO_MMFORMER[0] == 3
    # 1 should be T1ce (benchmark index 1)
    assert BENCHMARK_TO_MMFORMER[1] == 1
    # 2 should be T1 (benchmark index 0)
    assert BENCHMARK_TO_MMFORMER[2] == 0
    # 3 should be T2 (benchmark index 2)
    assert BENCHMARK_TO_MMFORMER[3] == 2


def test_one_hot_target_encoding():
    """Verify ground truth remapping from 4 -> 3 and one-hot encoding."""
    targets = torch.tensor([[[[0, 1], [2, 4]]]], dtype=torch.long)  # (1, 1, 2, 2)
    target_mapped = targets.clone()
    target_mapped[target_mapped == 4] = 3

    assert target_mapped.max().item() == 3
    assert set(target_mapped.flatten().tolist()) == {0, 1, 2, 3}

    one_hot = torch.zeros((1, 4, 2, 2), dtype=torch.float32)
    one_hot.scatter_(1, target_mapped, 1.0)

    assert one_hot.shape == (1, 4, 2, 2)
    assert (one_hot.sum(dim=1) == 1.0).all()
