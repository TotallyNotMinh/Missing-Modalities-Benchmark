import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
import torch

from src.adapters.nnunet_adapter import nnUNetAdapter
from src.utils.pipeline_utils import Config


def test_nnunet_adapter_init():
    adapter = nnUNetAdapter(device="cpu", patch_size=(32, 32, 32))
    assert adapter is not None
    assert adapter.num_input_channels == 4
    assert adapter.num_classes == 3
    assert adapter.device.type == "cpu"


def test_nnunet_adapter_from_config():
    cfg = Config({
        "paths": {"oracle_weights": "checkpoints/oracle_nnunet"},
        "device": "cpu",
        "patch": {"size": [32, 32, 32]},
        "modalities": {"order": ["t1", "t1ce", "t2", "flair"]},
    })
    adapter = nnUNetAdapter.from_config(cfg)
    assert adapter.patch_size == (32, 32, 32)
    assert adapter.num_input_channels == 4


def test_nnunet_adapter_predict_4d_and_5d():
    adapter = nnUNetAdapter(device="cpu", patch_size=(32, 32, 32))

    # 4D input: (4, 32, 32, 32)
    x_4d = np.random.randn(4, 32, 32, 32).astype(np.float32)
    pred_4d = adapter.predict(x_4d)
    assert pred_4d.shape == (32, 32, 32)
    assert pred_4d.dtype == np.uint8
    # Assert predicted labels are subset of BraTS labels (0, 1, 2, 4)
    unique_labels = set(np.unique(pred_4d))
    assert unique_labels.issubset({0, 1, 2, 4})

    # 5D input: (1, 4, 32, 32, 32)
    x_5d = torch.randn(1, 4, 32, 32, 32)
    pred_5d = adapter.predict(x_5d)
    assert pred_5d.shape == (1, 32, 32, 32)
    assert pred_5d.dtype == np.uint8


def test_nnunet_adapter_predict_logits():
    adapter = nnUNetAdapter(device="cpu", patch_size=(32, 32, 32))
    x = torch.randn(4, 32, 32, 32)
    pred, logits = adapter.predict(x, return_logits=True)
    assert pred.shape == (32, 32, 32)
    assert logits.shape == (3, 32, 32, 32)


def test_nnunet_adapter_dict_input():
    adapter = nnUNetAdapter(device="cpu", patch_size=(32, 32, 32))
    sample_dict = {"modalities": torch.randn(4, 32, 32, 32)}
    pred = adapter.predict(sample_dict)
    assert pred.shape == (32, 32, 32)


def test_nnunet_adapter_sliding_window():
    adapter = nnUNetAdapter(device="cpu", patch_size=(16, 16, 16))
    # Input spatial dimensions larger than patch size
    x_large = torch.randn(4, 32, 32, 32)
    pred = adapter.predict(x_large, roi_size=(16, 16, 16), overlap=0.5)
    assert pred.shape == (32, 32, 32)


def test_nnunet_adapter_evaluate_sample():
    adapter = nnUNetAdapter(device="cpu", patch_size=(32, 32, 32))
    x = torch.randn(4, 32, 32, 32)
    target = np.zeros((32, 32, 32), dtype=np.uint8)
    target[10:20, 10:20, 10:20] = 1  # NCR
    target[12:18, 12:18, 12:18] = 4  # ET

    metrics = adapter.evaluate_sample(x, target)
    expected_keys = [
        "Dice_WT", "Dice_TC", "Dice_ET", "Dice_Mean",
        "HD95_WT", "HD95_TC", "HD95_ET", "HD95_Mean"
    ]
    for key in expected_keys:
        assert key in metrics
        assert isinstance(metrics[key], float)


def test_nnunet_adapter_evaluate_batch():
    adapter = nnUNetAdapter(device="cpu", patch_size=(32, 32, 32))
    batch = {
        "modalities": torch.randn(2, 4, 32, 32, 32),
        "mask": torch.zeros(2, 1, 32, 32, 32, dtype=torch.long),
        "patient_id": ["BraTS20_Training_001", "BraTS20_Training_002"]
    }
    results = adapter.evaluate_batch(batch)
    assert len(results) == 2
    assert results[0]["patient_id"] == "BraTS20_Training_001"
    assert results[1]["patient_id"] == "BraTS20_Training_002"
    assert "Dice_Mean" in results[0]


def test_nnunet_adapter_checkpoint_loading():
    adapter1 = nnUNetAdapter(device="cpu", patch_size=(16, 16, 16))

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        ckpt_path = Path(f.name)

    try:
        torch.save({"network_weights": adapter1.network.state_dict()}, ckpt_path)

        adapter2 = nnUNetAdapter(weights_path=ckpt_path, device="cpu", patch_size=(16, 16, 16))
        x = torch.randn(4, 16, 16, 16)
        pred1 = adapter1.predict(x)
        pred2 = adapter2.predict(x)
        assert np.array_equal(pred1, pred2)
    finally:
        if ckpt_path.exists():
            ckpt_path.unlink()
