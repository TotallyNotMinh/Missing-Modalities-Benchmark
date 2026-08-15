# Missing-Modalities-Benchmark: Complete Codebase & Architectural Guide

> **Repository Purpose:** A unified, reproducible medical imaging benchmark investigating whether existing generative MRI modality synthesis models (GANs, native 3D diffusion, 3D latent diffusion) produce viable substitutes for real sequences in downstream brain tumor segmentation (**RQ1**), whether they benefit dedicated missing-modality networks (**RQ2**), and whether pixel-level reconstruction metrics predict downstream segmentation performance (**RQ3**).

---

## 1. High-Level Architecture & Data Flow

```
                                  [ BraTS 2020 Raw NIfTI ]
                                             │
                                             ▼
                               [ src/data/preprocess.py ]
                         (N4 Correction -> 1mm³ Isotropic Resample
                          -> Z-Score Normalization -> Skull-Stripped)
                                             │
                                             ▼
                                    [ data/processed/ ]
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
             [ make_splits() / splits.py ]               [ scenarios.py ]
          (Frozen 70/15/15 Patient Split)             (Canonical S1-S4 Scenarios)
                       │                                           │
                       └─────────────────────┬─────────────────────┘
                                             │
                                             ▼
                                  [ brats_dataset.py ]
                               (Loads 4 channels: (4, H, W, D))
                                             │
                                             ▼
                                  [ augmentation.py ]
                         ┌───────────────────┴───────────────────┐
                         ▼                                       ▼
             Synthesis Augmentation                  Segmentation Augmentation
           (Conservative, Pairing-Safe)               (nnU-Net Standard 3D Policy)
                         │                                       │
                         └───────────────────┬───────────────────┘
                                             │
                                             ▼
                                   [ dataloader.py ]
                       (Yields batches with inputs, target, mask)
                                             │
        ┌────────────────────────────────────┼────────────────────────────────────┐
        ▼                                    ▼                                    ▼
[ Synthesis Training ]             [ RQ1 Evaluation ]                   [ RQ2 Evaluation ]
(Pix2Pix / Med-DDPM /              (Frozen 4-Ch Segmenters:             (Missing-Modality Models:
 3D-MedDiffusion)                   nnU-Net v2, SwinUNETR)               AdaMM, mmFormer, RFNet)
        │                                    │                                    │
        ▼                                    │                                    │
[ Synthetic Modality Volume ]                │                                    │
        │                                    │                                    │
        ▼                                    │                                    │
[ renormalize_synthetic_output() ]           │                                    │
(tanh/sigmoid -> z-score junction)           │                                    │
        │                                    │                                    │
        └────────────────────────────────────┴────────────────────────────────────┘
                                             │
                                             ▼
                                 [ src/metrics/ ]
                     ┌───────────────────────┴───────────────────────┐
                     ▼                                               ▼
          [ segmentation.py ]                              [ reconstruction.py ]
    (Subregions: WT, TC, ET;                         (PSNR, SSIM, MAE, MSE
     Dice & Surface-Voxel KDTree HD95)                over Foreground Brain)
                                             │
                                             ▼
                                [ aggregate_results.py ]
                     (Generates Tables 1-3 & RQ3 Correlation Vectors)
```

---

## 2. Configuration & Core Utilities

### `config.yaml`
* **Path:** `config.yaml`
* **Role:** **Single Source of Truth** for the entire repository.
* **Key Contents:**
  - `seed: 42`: Global deterministic RNG seed.
  - `paths`: Workspace-relative paths for `raw_data_root`, `splits_file`, `preprocessed_cache`, and `oracle_weights`.
  - `modalities.order`: Strict canonical 4-channel ordering `["t1", "t1ce", "t2", "flair"]`.
  - `labels`: BraTS label sets for Whole Tumor (`[1, 2, 4]`), Tumor Core (`[1, 4]`), and Enhancing Tumor (`[4]`).
  - `split.ratios`: `{train: 0.70, val: 0.15, test: 0.15}`.
  - `patch.size`: `[128, 128, 128]` 3D patch dimensions.
  - `augmentation_synthesis`: Conservative augmentation parameters (sagittal flip only, subtle $\pm 10^\circ$ rotation, noise $\sigma \in [0.01, 0.03]$, gamma jitter).
  - `augmentation_segmentation`: Aggressive nnU-Net standard augmentation parameters (3-axis flips, $\pm 30^\circ$ 3D rotation, scaling $[0.7, 1.4]$, 3D elastic deformation, blur, low-res simulation).
  - `missing_modality.scenarios`: Formal declarations of missing-modality scenarios S1–S4.
  - `junction`: Normalization target (`zscore_per_modality_per_patient`) and synthesis output format (`tanh_[-1,1]`).

---

### `pipeline_utils.py` (Root) & `src/utils/pipeline_utils.py`
* **Paths:** `pipeline_utils.py` and `src/utils/pipeline_utils.py`
* **Role:** Single import interface providing core benchmark guarantees. The root file is a thin re-export of `src/utils/pipeline_utils.py`.
* **Key Functions & Classes:**
  - `Config(dict)`: Subclassed dictionary enabling dot-notation attribute access (e.g., `cfg.training.batch_size`).
  - `load_config(path, model_name=None)`: Loads `config.yaml`, recursively deep-merges optional model overrides (`configs/models/<model_name>.yaml`), and dynamically injects `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results` environment variables.
  - `seed_everything(seed)`: Sets Python `random`, `numpy`, and PyTorch CPU/CUDA RNG seeds.
  - `worker_init_fn(worker_id, base_seed)`: DataLoader worker initializer ensuring multi-worker processes don't duplicate stochastic streams.
  - `make_splits(patient_ids, cfg, out_path)`: Creates and saves frozen 70/15/15 splits using deterministic sorting + shuffling.
  - `load_splits(cfg)`: Loads the frozen `splits.json` file across any training or evaluation script.
  - `zscore_normalize(volume, mask)`: Computes foreground-restricted z-score normalization ($(\text{vol} - \mu) / \sigma$).
  - `renormalize_synthetic_output(volume, cfg, mask)`: **Junction-point mapper** that safely converts generator outputs (e.g., $\tanh \in [-1, 1]$ or sigmoid $\in [0, 1]$) into the exact z-score space expected by downstream segmentation models.
  - `sample_patch_coords(volume_shape, patch_size, mode, rng)`: Extracts 3D coordinates — random crop for training, deterministic center crop for val/test.
  - `apply_missing_modality(volume_dict, scenario_name, cfg)`: Zeros out the designated missing modality channel per scenario.
  - `get_modality_order(cfg)` & `stack_modalities(volume_dict, cfg)`: Assembles individual channel volumes into canonical $(4, H, W, D)$ arrays.
  - `compute_dice(pred_mask, gt_mask, label_ids)`: Standalone NumPy implementation of binary/multi-label Dice similarity.
  - `assert_pipeline_consistency(cfg)`: Startup guard-rail verifying patch size integrity, normalization compatibility, and split existence.

---

## 3. Data Processing & Pipeline (`src/data/`)

### `src/data/__init__.py`
* **Path:** `src/data/__init__.py`
* **Role:** Exposes data module classes and factories: `SplitManager`, `ScenarioBuilder`, `SCENARIOS`, `BraTSDataset`, transform factories, and `get_dataloaders`.

---

### `src/data/preprocess.py`
* **Path:** `src/data/preprocess.py`
* **Role:** Offline, one-time preprocessing script converting raw BraTS cases into standardized `.nii.gz` files.
* **Pipeline Steps:**
  1. `n4_bias_correction`: Corrects magnetic field radiofrequency inhomogeneity via SimpleITK.
  2. `resample_to_spacing`: Resamples images via BSpline and masks via Nearest Neighbor to isotropic $1.0\text{ mm}^3$.
  3. `zscore_normalize`: Standardizes brain voxel intensities to zero mean and unit variance.
  4. Saves preprocessed volumes under `data/processed/<patient_id>/<patient_id>_<modality>.nii.gz`.

---

### `src/data/scenarios.py`
* **Path:** `src/data/scenarios.py`
* **Role:** Missing-modality simulation registry and tensor assembler.
* **Key Components:**
  - `MODALITY_NAMES` (`["T1", "T1ce", "T2", "FLAIR"]`) & `MODALITY_SUFFIXES` (`["t1", "t1ce", "t2", "flair"]`).
  - `SCENARIOS`: Frozen dataclass dictionary defining S1 (drop FLAIR), S2 (drop T1ce), S3 (drop T1), and S4 (drop T2).
  - `validate_scenarios_against_config(cfg)`: Cross-validates code definitions against `config.yaml` at startup.
  - `ScenarioBuilder`:
    - `apply(volume)`: Splits full $(4, H, W, D)$ volume into 3 available input channels, 1 target channel, and an integer missing flag.
    - `reconstruct_full(inputs, synthetic)`: Inserts synthesized modality into the missing slot, rebuilding $(4, H, W, D)$ for full-modality segmenters.
    - `reconstruct_native(inputs)`: Inserts zeros at the missing channel position for native missing-modality evaluation.

---

### `src/data/brats_dataset.py`
* **Path:** `src/data/brats_dataset.py`
* **Role:** Base PyTorch `Dataset` loading 4 preprocessed MRI volumes and the segmentation mask.
* **Returns per sample:**
  - `"patient_id"`: String ID.
  - `"modalities"`: `torch.FloatTensor` of shape $(4, H, W, D)$.
  - `"mask"`: `torch.LongTensor` of shape $(1, H, W, D)$ containing raw BraTS labels $(0, 1, 2, 4)$.
  - `"spacing"`: Tuple $(s_x, s_y, s_z)$ from the NIfTI header.

---

### `src/data/augmentation.py`
* **Path:** `src/data/augmentation.py`
* **Role:** Controlled augmentation pipelines with automatic MONAI transforms and pure PyTorch fallbacks.
* **Key Functions:**
  - `get_synthesis_train_transforms(patch_size, cfg)`: **Conservative pairing-safe policy**. Applies joint spatial transforms across all sequences (preserving anatomical geometry) and subtle intensity noise.
  - `get_segmentation_train_transforms(patch_size, cfg)`: **nnU-Net-derived policy**. Heavy 3D rotations, scaling, elastic deformation, low-res simulation, and gamma inversion.
  - `get_val_transforms(patch_size, cfg)`: **Deterministic evaluation policy**. Center spatial crop with zero random augmentations for 100% reproducible validation and testing.

---

### `src/data/splits.py`
* **Path:** `src/data/splits.py`
* **Role:** Patient-level split manager guaranteeing zero patient leakage across train, val, and test splits.
* **Methods:**
  - `generate(overwrite)`: Computes patient split with `int(round(n * ratio))` and persists to `splits.json`.
  - `load()` & `get_split(split)`: Reads precomputed split IDs.

---

### `src/data/dataloader.py`
* **Path:** `src/data/dataloader.py`
* **Role:** DataLoader factory and dataset wrapper.
* **Key Components:**
  - `ScenarioDataset`: Wraps `BraTSDataset` and dynamically calls `ScenarioBuilder.apply()` per sample.
  - `get_dataloaders(scenario, ..., cfg)`: Returns `(train_loader, val_loader, test_loader)` preconfigured with reproducible worker seeders and task-specific augmentations.

---

## 4. Evaluation Metrics (`src/metrics/`)

### `src/metrics/__init__.py`
* **Path:** `src/metrics/__init__.py`
* **Role:** Clean re-export of all pixel reconstruction and segmentation metrics.

---

### `src/metrics/segmentation.py`
* **Path:** `src/metrics/segmentation.py`
* **Role:** Downstream clinical evaluation metrics for BraTS tumor subregions.
* **Key Functions:**
  - `convert_brats_labels_to_subregions(mask)`: Converts raw BraTS label integers $(0, 1, 2, 4)$ into clinical overlapping subregions:
    - **Whole Tumor (WT):** Labels 1 (NCR) + 2 (ED) + 4 (ET).
    - **Tumor Core (TC):** Labels 1 (NCR) + 4 (ET).
    - **Enhancing Tumor (ET):** Label 4 (ET).
  - `_dice_numpy(pred, target)`: Computes binary Dice overlap.
  - `_hd95_numpy(pred, target, voxel_spacing)`: **Surface-Voxel 95th Percentile Hausdorff Distance**. Uses `scipy.ndimage.binary_erosion` to extract boundary surfaces, converts points to millimeter physical coordinates, and computes nearest-neighbor distances via `scipy.spatial.cKDTree`.
  - `compute_segmentation_metrics(gt_mask, pred_mask, voxel_spacing)`: Computes Dice and HD95 across WT, TC, and ET, plus macro means.

---

### `src/metrics/reconstruction.py`
* **Path:** `src/metrics/reconstruction.py`
* **Role:** Pixel/voxel-level reconstruction fidelity metrics between real ground-truth modalities and synthetic outputs.
* **Key Functions:**
  - `compute_psnr(target, pred, data_range, mask)`: Peak Signal-to-Noise Ratio over foreground brain tissue.
  - `compute_ssim(target, pred, data_range, slice_wise, mask)`: Structural Similarity Index averaged over non-empty 2D axial slices.
  - `compute_mae(target, pred, mask)`: Mean Absolute Error over foreground voxels.
  - `compute_mse(target, pred, mask)`: Mean Squared Error over foreground voxels.
  - `compute_reconstruction_metrics(target, pred, mask)`: Convenience wrapper returning all 4 metrics in a dictionary.

---

## 5. Experiment Utilities (`src/utils/`)

### `src/utils/__init__.py`
* **Path:** `src/utils/__init__.py`
* **Role:** Exports `Config`, `load_config`, `CheckpointManager`, `EarlyStopping`, `ExperimentLogger`, and pipeline utilities.

---

### `src/utils/checkpoint.py`
* **Path:** `src/utils/checkpoint.py`
* **Role:** Model state persistence and early stopping.
* **Key Classes:**
  - `CheckpointManager`: Saves/loads model state, optimizer state, epoch, and metric history (`best.pth` and `last.pth`). Uses `weights_only=False` with explicit documentation for training state dicts.
  - `EarlyStopping`: Monitors validation metric improvements (e.g., minimizing validation loss or maximizing validation PSNR/Dice) with configurable patience.

---

### `src/utils/logger.py`
* **Path:** `src/utils/logger.py`
* **Role:** Standardized experiment tracking. Writes structured metric logs to JSONL files and formatted text logs to disk with UTC timestamps.

---

## 6. Analysis & Table Aggregation (`src/analysis/`)

### `src/analysis/aggregate_results.py`
* **Path:** `src/analysis/aggregate_results.py`
* **Role:** Post-hoc results processor that loads per-patient evaluation CSVs and builds summary tables:
  - `rq1_table(df, metric)`: Table 1 & 2 — Evaluator $\times$ Generator $\times$ Scenario ($\text{Oracle}, \text{Synthetic}, \Delta\text{Metric}$).
  - `rq2_table(df, metric)`: Table 3 — Missing-Modality Model $\times$ Condition $\times$ Scenario ($\text{Native}, \text{Synthetic}, \Delta\text{Gain}$).
  - `rq3_vectors(df)`: Table 4 — Per-patient $(PSNR, SSIM)$ vs $(Dice, HD95)$ paired vectors for correlation analysis.
  - `save_tables(df, out_dir)`: Exports formatted summary CSVs to `results/tables/`.

---

## 7. Verification & Test Suite (`tests/`)

### `tests/test_pipeline.py`
* **Path:** `tests/test_pipeline.py`
* **Role:** 9-point unit and integration test suite verifying every component without external data dependencies:
  1. Config loading, model merging, and consistency assertions.
  2. Deterministic seeding and reproducibility.
  3. Patient split generation, ratio integrity, and zero leakage.
  4. Dual-policy augmentations and deterministic validation cropping.
  5. Scenarios S1–S4 channel separation, native padding, and synthetic reconstruction.
  6. BraTS subregion extraction, Dice overlap, and KDTree surface HD95.
  7. Pixel reconstruction metrics (PSNR, SSIM, MAE, MSE).
  8. Junction-point $\tanh \to \text{z-score}$ re-normalization.
  9. ResultsAggregator table generation and vector formatting.

---

## 8. Summary of File Relationships

| File | Depends On | Used By |
|---|---|---|
| `config.yaml` | None (Declaration) | `pipeline_utils.py`, `dataloader.py`, `scenarios.py` |
| `src/utils/pipeline_utils.py` | `config.yaml` | Entire repository |
| `pipeline_utils.py` (root) | `src/utils/pipeline_utils.py` | Root training / evaluation scripts |
| `src/data/scenarios.py` | `pipeline_utils.py` | `brats_dataset.py`, `dataloader.py`, model runners |
| `src/data/splits.py` | `pipeline_utils.py` | `dataloader.py`, `make_splits.py` |
| `src/data/augmentation.py` | `config.yaml` | `dataloader.py` |
| `src/data/brats_dataset.py` | `src/data/scenarios.py` | `dataloader.py` |
| `src/data/dataloader.py` | `brats_dataset.py`, `augmentation.py`, `scenarios.py`, `splits.py` | Training & evaluation scripts |
| `src/metrics/segmentation.py` | NumPy, SciPy (KDTree, binary erosion) | Evaluation scripts, `test_pipeline.py` |
| `src/metrics/reconstruction.py` | NumPy, scikit-image | Evaluation scripts, `test_pipeline.py` |
| `src/analysis/aggregate_results.py` | Pandas, NumPy | Post-hoc analysis & reporting |
