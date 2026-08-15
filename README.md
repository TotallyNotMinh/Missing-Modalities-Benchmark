# Missing Modalities Benchmark: Synthesis vs. Native Handling in Brain MRI Segmentation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-1.3+-blueviolet.svg)](https://monai.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An empirical benchmark and methodology framework evaluating whether modern **Generative Modality Synthesis** models (Conditional GANs, 3D DDPMs, 3D Latent Diffusion) produce viable substitutes for real MRI sequences in downstream clinical segmentation, compared against **Purpose-Built Missing-Modality Architectures** (AdaMM, mmFormer, RFNet).

For the full theoretical study design and experimental hypothesis matrix, see [`PLAN.md`](PLAN.md).

---

## Table of Contents
- [1. Benchmark Overview](#1-benchmark-overview)
- [2. Repository Structure](#2-repository-structure)
- [3. Missing-Modality Scenarios](#3-missing-modality-scenarios)
- [4. Data Preprocessing Pipeline](#4-data-preprocessing-pipeline)
- [5. Controlled Augmentation Protocols](#5-controlled-augmentation-protocols)
  - [5.1 Synthesis Models Augmentation Protocol](#51-synthesis-models-augmentation-protocol-controlled-variable)
  - [5.2 Downstream Segmentation Augmentation Protocol](#52-downstream-segmentation-augmentation-protocol-nnu-net-standard)
  - [5.3 Deterministic Validation & Test Policy](#53-deterministic-validation--test-policy)
- [6. Universal 8-Channel Modality Synthesis Formulation](#6-universal-8-channel-modality-synthesis-formulation)
- [7. Getting Started & Reproducibility](#7-getting-started--reproducibility)
  - [7.1 Environment Setup](#71-environment-setup)
  - [7.2 Deterministic 70/15/15 Data Split](#72-deterministic-701515-data-split)
  - [7.3 Training Synthesis Models](#73-training-synthesis-models)
  - [7.4 Downstream Segmentation Evaluation](#74-downstream-segmentation-evaluation)
- [8. Evaluation Metrics](#8-evaluation-metrics)

---

## 1. Benchmark Overview

Clinical MRI protocols frequently suffer from missing sequences due to patient motion, scan-time constraints, allergic reactions to gadolinium contrast, or emergency triage. 

This benchmark evaluates two competing paradigms across 4 standard single-modality-missing scenarios on **BraTS 2020**:
1. **"Synthesise-then-Segment"**: Use a generator (e.g., 3D Pix2Pix, Med-DDPM, 3D-MedDiffusion) to reconstruct the missing volume, then feed the full 4-channel stack into a frozen oracle segmenter (`nnU-Net v2`, `SwinUNETR`).
2. **"Native Missing-Modality Handling"**: Feed the available 3 sequences directly into architectures with built-in missingness compensation (`AdaMM`, `mmFormer`, `RFNet`).
3. **Metric Decoupling Analysis**: Investigate whether pixel-level fidelity metrics (PSNR, SSIM) genuinely correlate with or decouple from downstream clinical task utility (Dice, HD95).

```
                 ┌─────────────────────────────────────────────────────────┐
                 │       Incomplete 3-Modality Patient MRI Volume          │
                 │         e.g., S1: [T1, T1ce, T2] (Missing FLAIR)         │
                 └────────────────────────────┬────────────────────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
     ┌───────────────────────────────┐                 ┌───────────────────────────────┐
     │  Paradigms 1: Modality Synth  │                 │  Paradigm 2: Native Handling  │
     │  (Pix2Pix, Med-DDPM, LDM)     │                 │  (AdaMM, mmFormer, RFNet)     │
     └───────────────┬───────────────┘                 └───────────────┬───────────────┘
                     ▼                                                 │
     ┌───────────────────────────────┐                                 │
     │ Synthesized 4-Channel Stack   │                                 │
     │ [T1, T1ce, T2, FLAIR_syn]     │                                 │
     └───────────────┬───────────────┘                                 │
                     ▼                                                 │
     ┌───────────────────────────────┐                                 │
     │ Frozen Oracle Segmenter       │                                 │
     │ (nnU-Net v2 / SwinUNETR)      │                                 │
     └───────────────┬───────────────┘                                 │
                     ▼                                                 ▼
     ┌─────────────────────────────────────────────────────────────────────────────────┐
     │                     Downstream Tumor Segmentation (WT, TC, ET)                  │
     │                   Evaluation: Dice, HD95, Volumetric Error, TOST                │
     └─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Repository Structure

```
Missing-Modalities-Benchmark/
├── configs/                     # Centralized YAML configuration files
│   └── default.yaml             # Global paths, patch sizes, seeds, hyperparameters
├── data/
│   ├── raw/                     # Raw BraTS 2020 data (MICCAI_BraTS2020_TrainingData)
│   ├── processed/               # Standardized preprocessed volumes (train/val/test)
│   └── splits/
│       └── splits.json          # Frozen deterministic 70/15/15 patient split
├── figures/                     # Methodological diagrams and benchmark charts
├── plan.md                      # Complete study plan, research questions & hypotheses
├── src/
│   ├── data/
│   │   ├── __init__.py          # Data module exports
│   │   ├── brats_dataset.py     # Volumetric BraTS NIfTI dataset loader
│   │   ├── augmentation.py      # Dual-policy augmentation protocols (Synthesis & Segmentation)
│   │   ├── dataloader.py        # PyTorch DataLoaders with scenario builders
│   │   ├── scenarios.py         # Missing-modality scenario masks (S1, S2, S3, S4)
│   │   └── splits.py            # Deterministic split manager and symlinker
│   ├── metrics/
│   │   ├── __init__.py          # Metrics exports
│   │   ├── segmentation.py      # Multi-class Dice and Hausdorff Distance 95 (HD95)
│   │   └── synthesis.py         # 3D PSNR and 3D Structural Similarity Index (SSIM)
│   ├── models/                  # Evaluator and missing-modality wrappers
│   ├── utils/                   # General utility functions and logging
│   └── visualization/           # Slice plotting and metric correlation visualizers
├── requirements.txt             # Python dependency requirements
└── README.md                    # Project documentation
```

---

## 3. Missing-Modality Scenarios

All experiments are evaluated across the 4 canonical single-missing-modality scenarios:

| Scenario | Input Available Modalities | Target Missing Modality | Clinical / Acquisition Context |
| :--- | :--- | :--- | :--- |
| **$S_1$** | `[T1, T1ce, T2]` | **`FLAIR`** | Most frequently missing in retrospective archives. |
| **$S_2$** | `[T1, T2, FLAIR]` | **`T1ce`** | Contrast omitted due to renal impairment, pregnancy, or cost. |
| **$S_3$** | `[T1ce, T2, FLAIR]` | **`T1`** | Pre-contrast T1 occasionally skipped during rapid protocols. |
| **$S_4$** | `[T1, T1ce, FLAIR]` | **`T2`** | Emergency/stroke triage protocol variations. |

---

## 4. Data Preprocessing Pipeline

To eliminate data processing discrepancies as a confound, all MRI volumes pass through a standardized pipeline:

1. **Anatomical Reorientation (`IPL`)**:
   * Reorients all incoming NIfTI scans to standard **Inferior-Posterior-Left (IPL)** space using `nibabel.orientations.ornt_transform`.
2. **Whole-Brain Bounding Box Extraction**:
   * Raw BraTS scans ($240 \times 240 \times 155 = 8.93\text{M voxels}$) contain $>60\%$ empty black background air outside the skull.
   * Volumes are tightly cropped to the anatomical brain bounding box ($144 \times 192 \times 192 = 5.31\text{M voxels}$), yielding a **$40.5\%$ reduction in voxel volume** and proportionally reducing GPU memory overhead per batch without discarding any brain parenchyma.
3. **Voxel Intensity Normalization**:
   * Scaled per-modality to $[0.0, 1.0]$ via min-max normalization:
     $$X_{\text{norm}} = \frac{X - X_{\min}}{X_{\max} - X_{\min}}$$
4. **Power-of-2 Symmetric Padding (`PadIfNecessary(5)`)**:
   * Padded to $160 \times 192 \times 192$ (multiples of $2^5 = 32$) to ensure clean recursive downsamplings without skip-connection shape mismatches.
5. **Zero-Distortion Test Reconstruction**:
   * When saving synthesized modalities, volumes are un-padded back to native **$240 \times 240 \times 155$** matrices. Downstream segmentation models load the real and synthesized scans in identical native spatial coordinates.

> [!IMPORTANT]
> **Standardized Segmentation Input ($128^3$)**:
> All downstream segmentation models (`nnU-Net v2`, `SwinUNETR`, `AdaMM`, `mmFormer`, `RFNet`) strictly operate on **$128 \times 128 \times 128$** volumetric inputs.
> * **Training**: The dataloader extracts random $128^3$ spatial patches with joint spatial augmentations.
> * **Validation & Testing**: Evaluated on deterministic $128^3$ center crops (or $128^3$ sliding-window inference), ensuring exact spatial alignment across real and synthesized modalities against ground truth segmentation masks.

---

## 5. Controlled Augmentation Protocols

We implement a **dual-policy augmentation framework** in [`src/data/augmentation.py`](src/data/augmentation.py) to preserve scientific fairness:

### 5.1 Synthesis Models Augmentation Protocol (Controlled Variable)
Applied identically across all generative synthesis models (`Pix2Pix`, `Med-DDPM`, `3D-MedDiffusion`). Spatial transforms are sampled once per volume and applied jointly across all 4 sequences to preserve physical tissue alignment.

| Transform | Parameters | Probability | Rationale / Exclusions |
| :--- | :--- | :--- | :--- |
| **Sagittal Flip** | Left-Right axis ($x$) | $p = 0.50$ | Preserves physiological brain symmetry. |
| **In-Plane Axial Rotation** | $\pm 10^\circ$ on axial plane ($z$) | $p = 0.50$ | Conservative rotation without non-axial distortion. |
| **Contrast / Brightness Jitter** | Multiplier scale $\in [0.9, 1.1]$ | $p = 0.30$ | Subtle per-modality variation; prevents artificial drift. |
| **Gaussian Noise** | Additive $\sigma = 0.02$ on $[0, 1]$ | $p = 0.20$ | Simulates minor RF-coil noise. |
| **Train Patch Sampling** | $128 \times 128 \times 128$ | $1.00$ | Fixed spatial sub-volume crop during training. |
| **Elastic Deformation** | *Explicitly Excluded* | $p = 0.00$ | Avoids non-anatomical cross-modal warping. |
| **Multi-Axis Flipping** | *Explicitly Excluded* | $p = 0.00$ | Excludes non-sagittal A-P and S-I flips. |

### 5.2 Downstream Segmentation Augmentation Protocol (nnU-Net Standard)
Applied to downstream segmentation networks (`nnUNet`, `SwinUNETR`, `MMFormer`, `RFNet`) to maintain **Oracle consistency** with established pretraining regimes.

| Transform | Parameters | Probability |
| :--- | :--- | :--- |
| **3D Random Rotation** | $\pm 30^\circ$ independently per axis ($x, y, z$) | $p = 0.20$ |
| **Random 3D Scaling** | Zoom factor $\in [0.7, 1.4]$ | $p = 0.20$ |
| **3D Elastic Deformation** | nnU-Net-derived displacement ($\sigma \in [5, 8]$, mag $\in [50, 150]$) | $p = 0.15$ |
| **3-Axis Spatial Mirroring** | Random flip along $x, y, z$ axes | $p = 0.50$ per axis |
| **Additive Gaussian Noise** | Zero-mean Gaussian with $\sigma = 0.10$ | $p = 0.20$ |
| **3D Gaussian Blur** | Kernel smoothing with $\sigma \in [0.5, 1.0]$ | $p = 0.20$ |
| **Brightness Scaling** | Multiplier $\in [0.75, 1.25]$ | $p = 0.15$ |
| **Contrast Jitter** | Exponential scaling $\gamma \in [0.75, 1.25]$ | $p = 0.15$ |
| **Gamma Inversion** | $\gamma \in [0.7, 1.5]$ with intensity inversion | $p = 0.10$ |
| **Gamma (Standard)** | $\gamma \in [0.7, 1.5]$ without inversion | $p = 0.30$ |
| **Simulated Low Resolution** | Nearest-neighbor downsampling scale $\in [0.5, 1.0]$ | $p = 0.20$ |
| **Train Patch Sampling** | $128 \times 128 \times 128$ | $1.00$ |

### 5.3 Deterministic Validation & Test Policy
* **Zero stochastic augmentations** applied during validation or testing.
* Evaluates strictly on deterministic **$128 \times 128 \times 128$ center crops** (or full-volume sliding-window inference), ensuring exact metric reproducibility across all runs.

---

## 6. Universal 8-Channel Modality Synthesis Formulation

Rather than training 4 separate generator models for each missing modality scenario, we implement the **Universal 8-Channel Conditional Formulation**:

$$\mathbf{A} = \big[\, \mathbf{X}_{\text{img}} \;\Vert\; \mathbf{M}_{\text{ind}} \,\big] \in \mathbb{R}^{B \times 8 \times D \times H \times W}$$

* **Channels $0\text{--}3$ ($\mathbf{X}_{\text{img}}$)**: The 4 canonical image slots `[T1, T1ce, T2, FLAIR]`. The missing target modality channel $k$ is masked to **zeros**.
* **Channels $4\text{--}7$ ($\mathbf{M}_{\text{ind}}$)**: 4 binary indicator channels ($1.0 = \text{present}, 0.0 = \text{missing}$).
* **Target $\mathbf{B}$**: Single-channel tensor $\mathbb{R}^{B \times 1 \times D \times H \times W}$ containing the genuine ground-truth missing sequence.

```
Input Tensor A: [ T1, T1ce, T2, FLAIR | M_T1, M_T1ce, M_T2, M_FLAIR ]
For Scenario S1 (FLAIR Missing):
  Image Channels     : [ T1_vol, T1ce_vol, T2_vol,    0.0    ]
  Indicator Channels : [   1.0 ,    1.0  ,   1.0 ,    0.0    ]
  Target Output B    : [            FLAIR_vol                ]
```

---

## 7. Getting Started & Reproducibility

### 7.1 Environment Setup
```bash
git clone https://github.com/TotallyNotMinh/Missing-Modalities-Benchmark.git
cd Missing-Modalities-Benchmark

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 7.2 Deterministic 70/15/15 Data Split
Generate the frozen, reproducible patient split (Seed 42):
```python
from src.data.splits import SplitManager

manager = SplitManager(
    raw_dir="data/raw/MICCAI_BraTS2020_TrainingData",
    splits_file="data/splits/splits.json",
    seed=42
)
manager.create_splits(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
manager.setup_split_directories(processed_dir="data/processed")
```

---

## 8. Evaluation Metrics

### 8.1 Voxel-Level Synthesis Fidelity
* **Peak Signal-to-Noise Ratio (PSNR)**: Measures voxel-level mean squared error across the brain mask.
* **3D Structural Similarity Index (SSIM)**: Evaluates structural, luminance, and contrast degradation.

### 8.2 Task-Level Clinical Utility (Primary)
* **Dice Similarity Coefficient (DSC)**: Evaluated across standard BraTS subregions:
  * **Whole Tumor (WT)**: Edema + Enhancing + Necrotic Core
  * **Tumor Core (TC)**: Enhancing + Necrotic Core
  * **Enhancing Tumor (ET)**: Active Gadolinium-enhancing rim
* **95th Percentile Hausdorff Distance (HD95)**: Quantifies boundary surface distance in millimeters.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
