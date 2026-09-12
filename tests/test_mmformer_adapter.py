import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.adapters.mmformer_adapter import MMFormerAdapter
from src.utils.pipeline_utils import Config


class DummyMMFormer(nn.Module):
    """Lightweight mock network imitating mmformer.Model signature for fast unit tests."""

    def __init__(self, num_cls=4):
        super().__init__()
        self.conv = nn.Conv3d(4, num_cls, kernel_size=1)
        self.is_training = False

    def forward(self, x, mask):
        # x is in mmFormer order [FLAIR, T1ce, T1, T2]
        # mask is (B, 4) boolean
        # Apply mask by zeroing inactive channels
        B = x.shape[0]
        mask_expanded = mask.view(B, 4, 1, 1, 1).float()
        x_masked = x * mask_expanded
        return self.conv(x_masked)


def test_mmformer_adapter_init():
    dummy = DummyMMFormer(num_cls=4)
    adapter = MMFormerAdapter(device="cpu", patch_size=(32, 32, 32), network=dummy)
    assert adapter is not None
    assert adapter.num_input_channels == 4
    assert adapter.num_classes == 4
    assert adapter.device.type == "cpu"


def test_mmformer_adapter_from_config():
    cfg = Config({
        "paths": {"mmformer_weights": "checkpoints/mmformer_best.pth"},
        "device": "cpu",
        "patch": {"size": [32, 32, 32]},
        "model": {"num_classes": 4},
    })
    dummy = DummyMMFormer(num_cls=4)
    adapter = MMFormerAdapter.from_config(cfg)
    assert adapter.patch_size == (32, 32, 32)
    assert adapter.num_classes == 4


def test_mmformer_adapter_predict_4d_and_5d():
    dummy = DummyMMFormer(num_cls=4)
    adapter = MMFormerAdapter(device="cpu", patch_size=(32, 32, 32), network=dummy)

    # 4D input in benchmark order: (4, 32, 32, 32)
    x_4d = np.random.randn(4, 32, 32, 32).astype(np.float32)
    pred_4d = adapter.predict(x_4d)
    assert pred_4d.shape == (32, 32, 32)
    assert pred_4d.dtype == np.uint8
    # Assert predicted labels are subset of standard BraTS labels (0, 1, 2, 4)
    unique_labels = set(np.unique(pred_4d))
    assert unique_labels.issubset({0, 1, 2, 4})

    # 5D input: (1, 4, 32, 32, 32)
    x_5d = torch.randn(1, 4, 32, 32, 32)
    pred_5d = adapter.predict(x_5d)
    assert pred_5d.shape == (1, 32, 32, 32)
    assert pred_5d.dtype == np.uint8


def test_mmformer_adapter_predict_logits():
    dummy = DummyMMFormer(num_cls=4)
    adapter = MMFormerAdapter(device="cpu", patch_size=(32, 32, 32), network=dummy)
    x = torch.randn(4, 32, 32, 32)
    pred, logits = adapter.predict(x, return_logits=True)
    assert pred.shape == (32, 32, 32)
    assert logits.shape == (4, 32, 32, 32)


def test_mmformer_adapter_scenarios_mask():
    dummy = DummyMMFormer(num_cls=4)
    adapter = MMFormerAdapter(device="cpu", patch_size=(32, 32, 32), network=dummy)
    x = torch.randn(1, 4, 32, 32, 32)

    # Test Scenario strings S1 to S4
    for sc in ["S1", "S2", "S3", "S4", "FULL"]:
        pred = adapter.predict(x, mask=sc)
        assert pred.shape == (1, 32, 32, 32)

    # Test boolean list mask in benchmark order [T1, T1ce, T2, FLAIR]
    # S1: FLAIR missing -> [True, True, True, False]
    mask_list = [True, True, True, False]
    resolved_mask = adapter._resolve_mask(x, mask_list)
    # in mmFormer order [FLAIR, T1ce, T1, T2], FLAIR is index 0 -> False
    assert resolved_mask[0, 0] == False
    assert resolved_mask[0, 1] == True  # T1ce
    assert resolved_mask[0, 2] == True  # T1
    assert resolved_mask[0, 3] == True  # T2


def test_mmformer_adapter_auto_missing_mask():
    dummy = DummyMMFormer(num_cls=4)
    adapter = MMFormerAdapter(device="cpu", patch_size=(32, 32, 32), network=dummy)

    # Simulate missing T1ce (channel 1 in benchmark order)
    x = torch.randn(1, 4, 16, 16, 16)
    x[:, 1] = 0.0  # Zero out T1ce

    resolved = adapter._resolve_mask(x, mask=None)
    # In mmFormer order [FLAIR, T1ce, T1, T2], T1ce is index 1
    assert resolved[0, 1] == False
    assert resolved[0, 0] == True  # FLAIR
    assert resolved[0, 2] == True  # T1
    assert resolved[0, 3] == True  # T2


def test_mmformer_adapter_dict_input():
    dummy = DummyMMFormer(num_cls=4)
    adapter = MMFormerAdapter(device="cpu", patch_size=(32, 32, 32), network=dummy)
    sample_dict = {
        "modalities": torch.randn(4, 32, 32, 32),
        "scenario": "S1"
    }
    pred = adapter.predict(sample_dict)
    assert pred.shape == (32, 32, 32)


def test_mmformer_adapter_evaluate_sample():
    dummy = DummyMMFormer(num_cls=4)
    adapter = MMFormerAdapter(device="cpu", patch_size=(32, 32, 32), network=dummy)
    x = torch.randn(4, 32, 32, 32)
    target = np.zeros((32, 32, 32), dtype=np.uint8)
    target[10:20, 10:20, 10:20] = 1  # NCR
    target[12:18, 12:18, 12:18] = 4  # ET

    metrics = adapter.evaluate_sample(x, target, mask="S2")
    expected_keys = [
        "Dice_WT", "Dice_TC", "Dice_ET", "Dice_Mean",
        "HD95_WT", "HD95_TC", "HD95_ET", "HD95_Mean"
    ]
    for key in expected_keys:
        assert key in metrics
        assert isinstance(metrics[key], float)


def test_mmformer_adapter_evaluate_batch():
    dummy = DummyMMFormer(num_cls=4)
    adapter = MMFormerAdapter(device="cpu", patch_size=(32, 32, 32), network=dummy)
    batch = {
        "modalities": torch.randn(2, 4, 32, 32, 32),
        "mask": torch.zeros(2, 1, 32, 32, 32, dtype=torch.long),
        "patient_id": ["BraTS20_Training_001", "BraTS20_Training_002"],
        "scenario": "S1"
    }
    results = adapter.evaluate_batch(batch)
    assert len(results) == 2
    assert results[0]["patient_id"] == "BraTS20_Training_001"
    assert results[1]["patient_id"] == "BraTS20_Training_002"
    assert "Dice_Mean" in results[0]


def test_mmformer_adapter_checkpoint_loading():
    dummy1 = DummyMMFormer(num_cls=4)
    adapter1 = MMFormerAdapter(device="cpu", patch_size=(16, 16, 16), network=dummy1)

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        ckpt_path = Path(f.name)

    try:
        torch.save({"state_dict": dummy1.state_dict()}, ckpt_path)

        dummy2 = DummyMMFormer(num_cls=4)
        adapter2 = MMFormerAdapter(weights_path=ckpt_path, device="cpu", patch_size=(16, 16, 16), network=dummy2)
        x = torch.randn(4, 16, 16, 16)
        pred1 = adapter1.predict(x)
        pred2 = adapter2.predict(x)
        assert np.array_equal(pred1, pred2)
    finally:
        if ckpt_path.exists():
            ckpt_path.unlink()


def test_mmformer_adapter_postprocess_et():
    dummy = DummyMMFormer(num_cls=4)
    adapter = MMFormerAdapter(device="cpu", patch_size=(16, 16, 16), network=dummy)

    # Mock output where ET (label 4) has only 10 voxels
    x = torch.randn(4, 16, 16, 16)
    pred_raw = adapter.predict(x, postprocess_et=False)

    # Force a small ET cluster (< 500 voxels)
    pred_post = adapter.predict(x, postprocess_et=True, et_threshold=50000)
    # When threshold is higher than total volume (16^3 = 4096), ET should be suppressed to 0
    assert np.sum(pred_post == 4) == 0
