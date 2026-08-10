import numpy as np
import torch
from src.metrics import (
    compute_reconstruction_metrics,
    compute_segmentation_metrics,
    compute_spearman_correlation,
    compute_pearson_correlation,
    paired_statistical_test,
    tost_equivalence_test
)

def test_metrics_pipeline():
    print("=== Testing Validation Metrics Pipeline ===")
    
    # 1. Test Reconstruction Metrics
    real_volume = np.random.rand(128, 128, 64).astype(np.float32)
    synth_volume = real_volume + np.random.normal(0, 0.05, size=real_volume.shape).astype(np.float32)
    
    recon_results = compute_reconstruction_metrics(real_volume, synth_volume)
    print("\n1. Reconstruction Metrics Output:")
    for k, v in recon_results.items():
        print(f"   - {k}: {v:.4f}")
        
    # 2. Test Segmentation Metrics
    # Synthetic BraTS label map (0: BG, 1: NCR, 2: ED, 4: ET)
    gt_mask = np.zeros((128, 128, 64), dtype=np.int64)
    gt_mask[40:80, 40:80, 20:40] = 2 # Edema
    gt_mask[50:70, 50:70, 25:35] = 1 # Core
    gt_mask[55:65, 55:65, 28:32] = 4 # Enhancing
    
    # Slightly perturbed prediction mask
    pred_mask = gt_mask.copy()
    pred_mask[39:41, 39:41, 19:21] = 0
    
    seg_results = compute_segmentation_metrics(gt_mask, pred_mask)
    print("\n2. Segmentation Metrics Output:")
    for k, v in seg_results.items():
        print(f"   - {k}: {v:.4f}")
        
    # 3. Test Statistical & Correlation Metrics
    psnr_scores = np.random.uniform(20, 35, 50)
    dice_scores = psnr_scores * 0.02 + np.random.normal(0, 0.05, 50)
    
    rho, p_rho = compute_spearman_correlation(psnr_scores, dice_scores)
    r, p_r = compute_pearson_correlation(psnr_scores, dice_scores)
    print("\n3. Correlation Analysis (RQ3):")
    print(f"   - Spearman rho: {rho:.4f} (p={p_rho:.4e})")
    print(f"   - Pearson r:    {r:.4f} (p={p_r:.4e})")
    
    # 4. Test Paired Stats & TOST Equivalence (RQ1 & RQ2)
    oracle_dices = np.random.uniform(0.85, 0.95, 50)
    synth_dices = oracle_dices - np.random.uniform(0.001, 0.015, 50)
    
    paired_res = paired_statistical_test(oracle_dices, synth_dices)
    tost_res = tost_equivalence_test(oracle_dices, synth_dices, margin=0.01)
    
    print("\n4. Paired Statistical Testing & Equivalence:")
    print(f"   - Test Type: {paired_res['test_type']}")
    print(f"   - p-value:   {paired_res['p_value']:.4e}")
    print(f"   - Cohen's d: {paired_res['cohens_d']:.4f}")
    print(f"   - TOST Equivalence (margin=1%): {tost_res['is_equivalent']} (p={tost_res['p_tost']:.4e})")
    
    print("\n=== Validation Metrics Test Completed Successfully! ===")

if __name__ == "__main__":
    test_metrics_pipeline()
