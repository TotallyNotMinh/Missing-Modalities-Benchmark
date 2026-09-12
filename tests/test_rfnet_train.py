import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure external RFNet is importable
RFNET_DIR = PROJECT_ROOT / "externals" / "rfnet"
if RFNET_DIR.exists() and str(RFNET_DIR) not in sys.path:
    sys.path.insert(0, str(RFNET_DIR))

from scripts.train_rfnet import (
    COMBINATORIAL_MASKS,
    BENCHMARK_TO_RFNET,
    train_one_epoch,
)


class MockBraTSBatch:
    """Provides a synthetic batch matching the format of BraTSDataset."""
    @staticmethod
    def create(b_size=1, patch_size=(16, 16, 16)):
        modalities = torch.randn(b_size, 4, *patch_size)
        # Masks with labels in {0, 1, 2, 4}
        mask = torch.randint(0, 3, (b_size, 1, *patch_size), dtype=torch.long)
        mask[mask == 3] = 4
        return {"modalities": modalities, "mask": mask}


def test_rfnet_combinatorial_masks_shape():
    """Verify that combinatorial masks has exactly 15 valid 4-modality combinations."""
    assert COMBINATORIAL_MASKS.shape == (15, 4)
    # Check that each has at least 1 modality present
    assert (COMBINATORIAL_MASKS.sum(axis=1) >= 1).all()
    # Check full modality mask exists
    assert COMBINATORIAL_MASKS[-1].all()


def test_benchmark_to_rfnet_permutation():
    """
    Verify permutation maps:
    Index 0 (T1)    -> Index 2 in RFNet
    Index 1 (T1ce)  -> Index 1 in RFNet
    Index 2 (T2)    -> Index 3 in RFNet
    Index 3 (FLAIR) -> Index 0 in RFNet
    """
    assert BENCHMARK_TO_RFNET == [3, 1, 0, 2]
    # 0 in RFNet should be FLAIR (benchmark index 3)
    assert BENCHMARK_TO_RFNET[0] == 3
    # 1 in RFNet should be T1ce (benchmark index 1)
    assert BENCHMARK_TO_RFNET[1] == 1
    # 2 in RFNet should be T1 (benchmark index 0)
    assert BENCHMARK_TO_RFNET[2] == 0
    # 3 in RFNet should be T2 (benchmark index 2)
    assert BENCHMARK_TO_RFNET[3] == 2


class MockTrainRFNet(nn.Module):
    def __init__(self, num_cls=4):
        super().__init__()
        self.conv = nn.Conv3d(4, num_cls, kernel_size=1)
        self.is_training = True

    def forward(self, x, mask):
        B = x.shape[0]
        mask_exp = mask.view(B, 4, 1, 1, 1).float()
        x_m = x * mask_exp
        logits = self.conv(x_m)
        pred = torch.softmax(logits, dim=1)
        sep_preds = [pred, pred, pred, pred]
        prm_preds = [pred, pred, pred, pred]
        return pred, sep_preds, prm_preds


def test_rfnet_train_one_epoch_execution():
    """Simulates 1 epoch of RFNet training using mock batch and mock model."""
    mock_model = MockTrainRFNet(num_cls=4)
    optimizer = torch.optim.Adam(mock_model.parameters(), lr=1e-4)
    mock_batch = MockBraTSBatch.create(b_size=1, patch_size=(16, 16, 16))
    loader = [mock_batch, mock_batch]

    metrics = train_one_epoch(
        model=mock_model,
        loader=loader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        epoch=0,
        region_fusion_start_epoch=0,
    )

    assert "loss" in metrics
    assert "fuse_loss" in metrics
    assert "sep_loss" in metrics
    assert "prm_loss" in metrics
    assert metrics["loss"] > 0.0
