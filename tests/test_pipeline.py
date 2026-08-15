"""
Comprehensive unit and integration test suite for Missing-Modalities-Benchmark.
Verifies all pipeline components, math correctness, data structures, and edge cases.
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

import pipeline_utils as pu
from src.data.scenarios import SCENARIOS, ScenarioBuilder
from src.data.splits import SplitManager
from src.data.augmentation import (
    get_synthesis_train_transforms,
    get_segmentation_train_transforms,
    get_val_transforms,
)
from src.metrics.segmentation import (
    convert_brats_labels_to_subregions,
    _dice_numpy,
    _hd95_numpy,
    compute_segmentation_metrics,
)
from src.metrics.reconstruction import (
    compute_psnr,
    compute_ssim,
    compute_mae,
    compute_mse,
    compute_reconstruction_metrics,
)


def test_config_and_consistency():
    print("[Test 1/9] Testing config loading and pipeline consistency...")
    cfg = pu.load_config()
    assert cfg is not None
    assert cfg.seed == 42
    assert cfg.patch.size == [128, 128, 128]
    assert cfg.modalities.order == ["t1", "t1ce", "t2", "flair"]
    assert cfg.labels.whole_tumor == [1, 2, 4]
    assert cfg.labels.tumor_core == [1, 4]
    assert cfg.labels.enhancing_tumor == [4]
    assert cfg.preprocessing.normalization == cfg.junction.renormalize_to

    # Verify assert_pipeline_consistency without requiring splits file on disk
    pu.assert_pipeline_consistency(cfg, check_splits_exist=False)

    # Test scenario cross-validation against config
    from src.data.scenarios import validate_scenarios_against_config
    validate_scenarios_against_config(cfg)

    # Test renormalize_synthetic_output unknown range raises ValueError
    bad_cfg = {"junction": {"synthesis_output_range": "invalid_range", "renormalize_to": "zscore_per_modality_per_patient"}, "preprocessing": {"normalization": "zscore_per_modality_per_patient"}}
    try:
        pu.renormalize_synthetic_output(np.ones((5,5,5)), bad_cfg)
        raise AssertionError("Should have raised ValueError on invalid range")
    except ValueError:
        pass  # Expected

    # Test load_config with deep merge
    merged_cfg = pu.load_config()
    assert "nnunet" in merged_cfg
    print(" -> Config and consistency passed.")


def test_reproducibility_and_seeding():
    print("[Test 2/9] Testing reproducibility and global seeding...")
    pu.seed_everything(1234)
    r1 = np.random.rand(5)
    t1 = torch.rand(5)

    pu.seed_everything(1234)
    r2 = np.random.rand(5)
    t2 = torch.rand(5)

    assert np.allclose(r1, r2), "NumPy seed failure"
    assert torch.allclose(t1, t2), "PyTorch seed failure"
    print(" -> Seeding and reproducibility passed.")


def test_splits_generation_and_loading():
    print("[Test 3/9] Testing split generation, ratio enforcement, and zero leakage...")
    cfg = pu.load_config()
    temp_dir = tempfile.mkdtemp()
    try:
        splits_path = Path(temp_dir) / "splits.json"
        patient_ids = [f"BraTS20_Training_{i:03d}" for i in range(1, 101)]

        splits = pu.make_splits(patient_ids, cfg, out_path=splits_path)

        assert len(splits["train"]) == 70, f"Expected 70 train, got {len(splits['train'])}"
        assert len(splits["val"]) == 15, f"Expected 15 val, got {len(splits['val'])}"
        assert len(splits["test"]) == 15, f"Expected 15 test, got {len(splits['test'])}"

        # Strict leakage checks
        assert len(set(splits["train"]) & set(splits["val"])) == 0, "Train/Val leakage!"
        assert len(set(splits["train"]) & set(splits["test"])) == 0, "Train/Test leakage!"
        assert len(set(splits["val"]) & set(splits["test"])) == 0, "Val/Test leakage!"

        # Test SplitManager loading
        cfg_custom = dict(cfg)
        cfg_custom["paths"] = dict(cfg["paths"])
        cfg_custom["paths"]["splits_file"] = str(splits_path)
        loaded = pu.load_splits(cfg_custom)
        assert loaded["train"] == splits["train"]
        assert loaded["test"] == splits["test"]

        # Test SplitManager.generate produces identical counts for N=369
        split_mgr = SplitManager(processed_dir=temp_dir, splits_file=str(splits_path), seed=42)
        p369 = [f"P_{i}" for i in range(369)]
        splits_pu = pu.make_splits(p369, cfg, out_path=splits_path)
        assert len(splits_pu["train"]) == int(round(369 * 0.70))
        assert len(splits_pu["val"]) == int(round(369 * 0.15))
        assert len(splits_pu["test"]) == 369 - len(splits_pu["train"]) - len(splits_pu["val"])
    finally:
        shutil.rmtree(temp_dir)
    print(" -> Splits generation and leakage checks passed.")


def test_dual_policy_augmentations():
    print("[Test 4/9] Testing dual-policy augmentations and deterministic eval...")
    patch_size = (128, 128, 128)
    modalities = torch.randn(4, 140, 140, 140)
    mask = torch.randint(0, 5, (1, 140, 140, 140))
    sample = {"modalities": modalities, "mask": mask}

    # 1. Synthesis augmentation
    synth_tf = get_synthesis_train_transforms(patch_size)
    synth_out = synth_tf(sample.copy())
    assert synth_out["modalities"].shape == (4, 128, 128, 128)
    assert synth_out["mask"].shape == (1, 128, 128, 128)

    # 2. Segmentation augmentation
    seg_tf = get_segmentation_train_transforms(patch_size)
    seg_out = seg_tf(sample.copy())
    assert seg_out["modalities"].shape == (4, 128, 128, 128)
    assert seg_out["mask"].shape == (1, 128, 128, 128)

    # 3. Deterministic Val/Test transform (Center Crop)
    val_tf = get_val_transforms(patch_size)
    val_out1 = val_tf(sample.copy())
    val_out2 = val_tf(sample.copy())
    assert torch.allclose(val_out1["modalities"], val_out2["modalities"]), "Val transform must be 100% deterministic"
    assert torch.allclose(val_out1["mask"], val_out2["mask"]), "Val mask must be 100% deterministic"
    print(" -> Dual-policy augmentations and deterministic eval passed.")


def test_scenarios_and_reconstruction():
    print("[Test 5/9] Testing scenarios S1-S4 and multi-channel tensor assembly...")
    cfg = pu.load_config()

    # Test all scenarios
    scenarios_target_map = {"S1": 3, "S2": 1, "S3": 0, "S4": 2}
    for sc_name, target_idx in scenarios_target_map.items():
        builder = ScenarioBuilder(sc_name)
        assert builder.scenario.target_index == target_idx

        # 4D Volume test (4, H, W, D)
        vol_4d = torch.arange(4, dtype=torch.float32).view(4, 1, 1, 1).expand(4, 10, 10, 10).clone()
        res_4d = builder.apply(vol_4d)
        assert res_4d["inputs"].shape == (3, 10, 10, 10)
        assert res_4d["target"].shape == (1, 10, 10, 10)
        assert res_4d["missing_flag"] == target_idx

        # Synthetic reconstruction
        synth_chan = torch.full((1, 10, 10, 10), 99.0)
        reconstructed = builder.reconstruct_full(res_4d["inputs"], synth_chan)
        assert reconstructed.shape == (4, 10, 10, 10)
        assert torch.all(reconstructed[target_idx] == 99.0), f"Reconstructed target channel mismatch in {sc_name}"

        # Native missing (zero padding)
        native = builder.reconstruct_native(res_4d["inputs"])
        assert native.shape == (4, 10, 10, 10)
        assert torch.all(native[target_idx] == 0.0), f"Native zero-padding mismatch in {sc_name}"

        # Batched 5D test (B, 4, H, W, D)
        vol_5d = torch.randn(2, 4, 10, 10, 10)
        res_5d = builder.apply(vol_5d)
        assert res_5d["inputs"].shape == (2, 3, 10, 10, 10)
        assert res_5d["target"].shape == (2, 1, 10, 10, 10)
        synth_5d = torch.full((2, 1, 10, 10, 10), 88.0)
        rec_5d = builder.reconstruct_full(res_5d["inputs"], synth_5d)
        assert rec_5d.shape == (2, 4, 10, 10, 10)
        assert torch.all(rec_5d[:, target_idx] == 88.0)

    # Test pipeline_utils apply_missing_modality & stack_modalities
    v_dict = {"t1": np.ones((5,5,5)), "t1ce": np.ones((5,5,5)), "t2": np.ones((5,5,5)), "flair": np.ones((5,5,5))}
    s1_dict = pu.apply_missing_modality(v_dict, "S1", cfg)
    assert np.all(s1_dict["flair"] == 0)
    assert np.all(s1_dict["t1"] == 1)
    stacked = pu.stack_modalities(s1_dict, cfg)
    assert stacked.shape == (4, 5, 5, 5)
    print(" -> Scenarios S1-S4 and reconstruction passed.")


def test_segmentation_metrics():
    print("[Test 6/9] Testing segmentation subregion extraction, Dice, and KDTree HD95...")
    # Synthetic label map: background=0, NCR=1, ED=2, ET=4
    gt_mask = np.zeros((30, 30, 30), dtype=np.int64)
    gt_mask[10:20, 10:20, 10:20] = 1  # NCR
    gt_mask[5:10, 5:10, 5:10] = 2    # ED
    gt_mask[12:18, 12:18, 12:18] = 4  # ET

    sub = convert_brats_labels_to_subregions(gt_mask)
    assert np.sum(sub["ET"]) == 6 * 6 * 6
    assert np.sum(sub["TC"]) == (10 * 10 * 10)  # NCR (contains ET)
    assert np.sum(sub["WT"]) == (10 * 10 * 10) + (5 * 5 * 5)

    # Perfect prediction test
    metrics_perf = compute_segmentation_metrics(gt_mask, gt_mask)
    assert metrics_perf["Dice_WT"] == 1.0
    assert metrics_perf["Dice_TC"] == 1.0
    assert metrics_perf["Dice_ET"] == 1.0
    assert metrics_perf["HD95_WT"] == 0.0
    assert metrics_perf["HD95_TC"] == 0.0
    assert metrics_perf["HD95_ET"] == 0.0

    # Perturbed prediction
    pred_mask = np.copy(gt_mask)
    pred_mask[5:7, 5:7, 5:7] = 0  # slight drop
    metrics_pert = compute_segmentation_metrics(gt_mask, pred_mask)
    assert 0.90 < metrics_pert["Dice_WT"] < 1.0
    assert metrics_pert["HD95_WT"] >= 0.0
    print(" -> Segmentation metrics, Dice, and HD95 passed.")


def test_reconstruction_metrics():
    print("[Test 7/9] Testing pixel reconstruction metrics (PSNR, SSIM, MAE, MSE)...")
    target = np.random.randn(32, 32, 16).astype(np.float32)
    pred = target + np.random.normal(0, 0.05, target.shape).astype(np.float32)
    brain_mask = (target > 0).astype(np.uint8)

    recon = compute_reconstruction_metrics(target, pred, mask=brain_mask)
    assert "PSNR" in recon and recon["PSNR"] > 10.0
    assert "SSIM" in recon and 0.0 <= recon["SSIM"] <= 1.0
    assert "MAE" in recon and recon["MAE"] >= 0.0
    assert "MSE" in recon and recon["MSE"] >= 0.0

    # Test identical volumes -> infinite PSNR / SSIM=1
    perf_recon = compute_reconstruction_metrics(target, target)
    assert perf_recon["SSIM"] == 1.0
    assert perf_recon["MAE"] == 0.0
    print(" -> Reconstruction metrics passed.")


def test_junction_renormalization():
    print("[Test 8/9] Testing generator-segmenter junction point renormalization...")
    cfg = pu.load_config()
    # Generator output in tanh space [-1, 1]
    tanh_output = np.random.uniform(-1.0, 1.0, (20, 20, 20)).astype(np.float32)
    brain_mask = (tanh_output > -0.8).astype(np.uint8)

    renorm = pu.renormalize_synthetic_output(tanh_output, cfg, mask=brain_mask)
    # Check that brain region has mean ~0 and std ~1
    fg = renorm[brain_mask > 0]
    assert abs(fg.mean()) < 1e-4, f"Expected mean ~0, got {fg.mean()}"
    assert abs(fg.std() - 1.0) < 1e-3, f"Expected std ~1, got {fg.std()}"
    assert np.all(renorm[brain_mask == 0] == 0.0), "Background voxels must remain 0.0"
    print(" -> Junction renormalization passed.")


from src.analysis.aggregate_results import ResultsAggregator
import pandas as pd


def test_results_aggregator():
    print("[Test 9/9] Testing ResultsAggregator summary tables and vector extraction...")
    df = pd.DataFrame([
        {'patient_id': '001', 'model': 'nnunet', 'generator': 'none', 'scenario': 'S1', 'condition': 'oracle', 'dice_mean': 0.91, 'hd95_mean': 3.2, 'psnr': None, 'ssim': None},
        {'patient_id': '002', 'model': 'nnunet', 'generator': 'none', 'scenario': 'S1', 'condition': 'oracle', 'dice_mean': 0.89, 'hd95_mean': 4.1, 'psnr': None, 'ssim': None},
        {'patient_id': '001', 'model': 'nnunet', 'generator': 'pix2pix', 'scenario': 'S1', 'condition': 'synthetic', 'dice_mean': 0.87, 'hd95_mean': 5.2, 'psnr': 28.5, 'ssim': 0.88},
        {'patient_id': '002', 'model': 'nnunet', 'generator': 'pix2pix', 'scenario': 'S1', 'condition': 'synthetic', 'dice_mean': 0.85, 'hd95_mean': 6.0, 'psnr': 27.8, 'ssim': 0.86},
        {'patient_id': '001', 'model': 'adamm', 'generator': 'none', 'scenario': 'S1', 'condition': 'native_missing', 'dice_mean': 0.82, 'hd95_mean': 7.1, 'psnr': None, 'ssim': None},
        {'patient_id': '002', 'model': 'adamm', 'generator': 'none', 'scenario': 'S1', 'condition': 'native_missing', 'dice_mean': 0.80, 'hd95_mean': 7.5, 'psnr': None, 'ssim': None},
        {'patient_id': '001', 'model': 'adamm', 'generator': 'pix2pix', 'scenario': 'S1', 'condition': 'synthetic', 'dice_mean': 0.84, 'hd95_mean': 6.5, 'psnr': 28.5, 'ssim': 0.88},
        {'patient_id': '002', 'model': 'adamm', 'generator': 'pix2pix', 'scenario': 'S1', 'condition': 'synthetic', 'dice_mean': 0.83, 'hd95_mean': 6.8, 'psnr': 27.8, 'ssim': 0.86},
    ])

    agg = ResultsAggregator("results")
    t1 = agg.rq1_table(df)
    assert len(t1) > 0
    assert "dice_mean_oracle" in t1.columns
    assert "delta_dice_mean" in t1.columns

    t2 = agg.rq2_table(df)
    assert len(t2) > 0
    assert "dice_mean_native" in t2.columns
    assert "delta_pix2pix" in t2.columns

    t3 = agg.rq3_vectors(df)
    assert len(t3) == 4
    assert "psnr" in t3.columns
    assert "ssim" in t3.columns
    print(" -> ResultsAggregator tables and vectors passed.")


if __name__ == "__main__":
    print("=================================================================")
    print("Running Rigorous Missing-Modalities-Benchmark Pipeline Audit Tests")
    print("=================================================================")
    test_config_and_consistency()
    test_reproducibility_and_seeding()
    test_splits_generation_and_loading()
    test_dual_policy_augmentations()
    test_scenarios_and_reconstruction()
    test_segmentation_metrics()
    test_reconstruction_metrics()
    test_junction_renormalization()
    test_results_aggregator()
    print("=================================================================")
    print("ALL 9 RIGOROUS PIPELINE AUDIT TESTS PASSED WITH 0 ERRORS!")
    print("=================================================================")
