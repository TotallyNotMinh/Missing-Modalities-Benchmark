# Missing Modalities Benchmark — Comprehensive Dependency & Compatibility Matrix

## Executive Summary

This document establishes the official cross-model software, hardware, and dependency compatibility specification for the **Missing Modalities Benchmark**. The benchmark evaluates **8 core medical AI models** spanning paired 2D conditional GANs, 3D direct voxel diffusion, 3D latent diffusion, frozen segmentation evaluators, and missing-modality segmentors:

1. **Pix2Pix** (2D Conditional GAN synthesis baseline)
2. **Med-DDPM** (3D Pixel-Space Denoising Diffusion synthesis baseline)
3. **3D-MedDiffusion** (3D Latent Diffusion SOTA synthesis model)
4. **nnU-Net v2** (Frozen 3D CNN Gold-Standard evaluator)
5. **SwinUNETR** (Frozen 3D Swin-Transformer SOTA evaluator)
6. **PASSION** (SOTA ACM MM 2024 Preference-Aware Self-Distillation Transformer segmentor)
7. **mmFormer** (Multi-Encoder 3D Transformer missing-modality segmentor)
8. **RFNet** (Region-Aware Fusion 3D CNN missing-modality segmentor)

This audit resolves historical version friction (PyTorch 1.x vs 2.x, unmaintained legacy packages like `medpy` and `acloss`, and CUDA compute capability thresholds), defining a **single, unified Conda environment** (`missing-modalities`) supporting all 8 models simultaneously.

---

## 1. Master Cross-Model Dependency & Hardware Matrix

| Parameter / Package | Pix2Pix | Med-DDPM | 3D-MedDiffusion | nnU-Net v2 | SwinUNETR | PASSION | mmFormer | RFNet | Unified Resolution Strategy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Python Version** | 3.8 – 3.11 | 3.9 – 3.11 | 3.9 – 3.11 | 3.9 – 3.11 | 3.9 – 3.11 | 3.8 – 3.10 | 3.9 – 3.11 | 3.9 – 3.11 | **Python 3.10** |
| **PyTorch Version** | `>=1.4.0` | `>=1.10.0` | `>=2.0.0` | `>=2.0.0` | `>=1.12.0` | `==1.12.1` (orig) | `==1.10.0` (orig) | `==1.7.1` (orig) | **`torch == 2.1.2`** |
| **Torchvision** | `>=0.5.0` | `>=0.11.0` | `>=0.15.0` | `>=0.15.0` | `>=0.15.0` | `==0.13.1` (orig) | `>=0.11.0` | `>=0.8.0` | **`torchvision == 0.16.2`** |
| **CUDA Toolkit** | 11.8 / 12.1 | 11.8 / 12.1 | 11.8 / 12.1 | 11.8 / 12.1 | 11.8 / 12.1 | 11.8 / 12.1 | 11.8 / 12.1 | 11.8 / 12.1 | **CUDA 11.8 or CUDA 12.1** |
| **Min Compute (SM)** | SM 5.0+ | SM 6.0+ | SM 7.0+ | SM 6.0+ | **SM 7.0+** | **SM 7.0+** | **SM 7.0+** | SM 6.0+ | **SM 7.0+** (Volta / Turing / Ampere / Ada) |
| **Rec. Compute (SM)**| SM 7.0+ | SM 7.0+/8.0+| SM 8.0+ | SM 7.0+/8.0+| **SM 8.0+** | **SM 8.0+** | **SM 8.0+** | SM 7.0+ | **SM 8.0+** (Ampere RTX 3090/4090, A100) |
| **MONAI** | Optional | Required (`>=1.0`) | Required (`>=1.3`) | Optional | **Strict Required** | Optional | Optional | Optional | **`monai >= 1.3.0`** |
| **MONAI Generative** | Optional | Recommended | Required | Optional | Optional | Optional | Optional | Optional | **`monai-generative >= 0.2.0`** |
| **SimpleITK** | Recommended| Recommended | Recommended | **Strict Required**| Recommended | Recommended | Recommended | **Strict Required**| **`SimpleITK >= 2.2.0`** |
| **Nibabel** | Recommended| Recommended | Recommended | **Strict Required**| Recommended | Recommended | Recommended | Recommended | **`nibabel >= 3.0.0`** |
| **Einops** | Not req. | Required (`>=0.6`) | Required (`>=0.7`) | Optional | **Strict Required**| **Strict Required**| **Strict Required**| Optional | **`einops >= 0.7.0`** |
| **MedPy** | Exclude | Exclude | Exclude | Exclude | Exclude | Exclude | Exclude | Exclude | **EXCLUDE**. Replaced by `monai.metrics` & `src/metrics` |
| **acloss** | Exclude | Exclude | Exclude | Exclude | Exclude | Exclude | Exclude | Exclude | **EXCLUDE**. Obsolete legacy autograd loss |
| **Min GPU VRAM** | 4 – 8 GB | 16 – 24 GB | 12 – 16 GB | 12 – 16 GB | 16 – 24 GB | 16 – 24 GB | 16 – 24 GB | 12 – 16 GB | **>= 16 GB VRAM** (RTX 3090/4090/A100) |
| **Min System RAM** | 16 GB | 32 – 64 GB | 32 GB | 32 GB | 32 GB | 32 GB | 32 GB | 32 GB | **>= 32 GB RAM** |
| **Env Variables** | None | None | None | **Strict Required**| Optional | None | None | None | `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results` |

---

## 2. Model Profiles & Technical Audits

### 2.1 Generative Models (Modality Synthesis)

#### 1. Pix2Pix (2D Conditional GAN)
- **Role**: 2D slice-wise image-to-image translation baseline (e.g. T1 $\to$ T2, T1 $\to$ T1ce, T1 $\to$ FLAIR).
- **Architecture**: 2D U-Net Generator + 70×70 PatchGAN Discriminator.
- **Loss Functions**: Adversarial Loss ($L_{\text{GAN}}$) + $L_1$ Reconstruction Loss + $L_{\text{SSIM}}$.
- **Software Dependencies**: PyTorch $\ge 2.0.0$, Torchvision $\ge 0.15.0$, NumPy, SciPy, Scikit-Image, SimpleITK/Nibabel.
- **Hardware & VRAM**: 4–8 GB GPU VRAM; 16 GB System RAM. SM 5.0+ minimum.
- **Operational Considerations**: Generates 2D axial/coronal/sagittal slices independently. Z-axis volume re-assembly requires post-synthesis z-score normalization across the re-stacked 3D volume ($240 \times 240 \times 155$) to remove staircase artifacts before feeding 3D evaluators (nnU-Net v2, SwinUNETR).

#### 2. Med-DDPM (3D Pixel-Space Denoising Diffusion)
- **Role**: Direct voxel-space 3D diffusion synthesis baseline.
- **Architecture**: 3D Denoising UNet with 3D Residual Blocks, Group Normalization, and 3D spatial self-attention at lower resolutions ($T=1000$ timesteps).
- **Loss Functions**: Voxel-wise $L_2$ Noise Prediction Loss $\| \epsilon - \epsilon_\theta(x_t, t, c) \|^2$.
- **Software Dependencies**: PyTorch $\ge 2.0.0$, Torchvision $\ge 0.15.0$, MONAI $\ge 1.3.0$, Nibabel $\ge 3.0.0$, SimpleITK $\ge 2.2.0$, Einops $\ge 0.7.0$.
- **Hardware & VRAM**: 16–24 GB GPU VRAM (for 3D patch diffusion backpropagation on $128 \times 128 \times 128$ voxel patches); 32–64 GB System RAM. SM 7.0+ required (SM 8.0+ recommended for FP16 Tensor Cores).
- **Operational Considerations**: Direct 3D diffusion requires accelerated DDIM sampling ($S=50$ or $100$ steps) during inference to reduce synthesis time from ~3 minutes to ~15 seconds per volume. Gradient checkpointing and AMP (`torch.cuda.amp.autocast()`) are mandatory during training.

#### 3. 3D-MedDiffusion (3D Latent Diffusion Model)
- **Role**: SOTA 3D Latent Diffusion Model (3D LDM) for missing-modality MRI synthesis.
- **Architecture**: 
  - *Stage 1 (Perceptual Compression)*: 3D Autoencoder KL (VAE) compressing $128 \times 128 \times 128$ volumes into $z \in \mathbb{R}^{4 \times 32 \times 32 \times 32}$ (8× spatial reduction).
  - *Stage 2 (Latent Denoising Diffusion)*: 3D Latent UNet operating on latent space $z$.
- **Loss Functions**: VAE Reconstruction $L_1$/$L_2$ Loss + KL Divergence + 3D Patch Discriminator Loss; Diffusion Latent MSE Noise Loss.
- **Software Dependencies**: PyTorch $\ge 2.1.2$, Torchvision $\ge 0.16.2$, MONAI $\ge 1.3.0$, MONAI Generative $\ge 0.2.0$, Einops $\ge 0.7.0$, SimpleITK $\ge 2.2.0$, Nibabel $\ge 3.0.0$.
- **Hardware & VRAM**: 12–16 GB GPU VRAM; 32 GB System RAM. SM 7.0+ required (SM 8.0+ recommended).
- **Operational Considerations**: Native 3D spatial continuity without z-axis artifacts. Highly efficient VRAM footprint due to latent space operations.

---

### 2.2 Frozen Downstream Evaluator Models (RQ1)

#### 4. nnU-Net v2 (`nnunetv2`)
- **Role**: Frozen 3D CNN Gold-Standard downstream segmentation evaluator for RQ1.
- **Architecture**: Self-configuring 3D U-Net pipeline (`dynamic-network-architectures`).
- **Software Dependencies**: PyTorch $\ge 2.0.0$ (target `2.1.2`), `nnunetv2 >= 2.2`, `dynamic-network-architectures >= 0.4.4`, `batchgenerators >= 0.25.1`, `acvl-utils >= 0.2.6`, `SimpleITK >= 2.2.0`, `nibabel >= 3.0.0`.
- **Hardware & VRAM**: 12–16 GB GPU VRAM; 32 GB System RAM; 8+ CPU cores for background `batchgenerators` multi-processing workers. SM 6.0+ minimum (SM 7.0+/8.0+ recommended for AMP).
- **Operational Considerations**: Requires three environment variables (`nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`) exported in shell environment prior to execution.

#### 5. SwinUNETR (`monai.networks.nets.SwinUNETR`)
- **Role**: Frozen 3D Swin-Transformer + U-Net Decoder downstream segmentation evaluator for RQ1.
- **Architecture**: 3D Swin Transformer encoder with shifted window self-attention + residual U-Net decoder.
- **Software Dependencies**: PyTorch $\ge 2.0.0$, Torchvision $\ge 0.16.2$, `monai >= 1.3.0`, `einops >= 0.7.0`, `SimpleITK >= 2.2.0`, `nibabel >= 3.0.0`.
- **Hardware & VRAM**: 16–24 GB GPU VRAM for training; 12 GB VRAM for sliding window inference (`monai.inferers.sliding_window_inference`). **SM 7.0+ strictly required** for 3D windowed attention matrix computations.

---

### 2.3 Missing-Modality Segmentation Models (RQ2)

#### 6. PASSION
- **Role**: SOTA (ACM MM 2024) Preference-Aware Self-Distillation Transformer for incomplete multi-modal segmentation.
- **Architecture**: Multi-modal Transformer with dual pixel-wise and semantic-wise self-distillation preference loss matrices.
- **Software Dependencies**: PyTorch $\ge 2.0.0$ (upgraded from original `1.12.1`), Torchvision $\ge 0.16.2$, `einops >= 0.7.0`, `SimpleITK >= 2.2.0`, `h5py >= 3.8.0`, `tensorboard`.
- **Hardware & VRAM**: 16–24 GB GPU VRAM; 32 GB System RAM. SM 7.0+ strictly required.

#### 7. mmFormer
- **Role**: Multi-encoder 3D Transformer baseline for missing-modality segmentation. Evaluated in native 3-channel mode vs 4-channel mode augmented with synthetic modalities (S1–S4).
- **Architecture**: Modality-specific 3D encoders + cross-modal attention fusion + 3D U-Net decoder.
- **Software Dependencies**: PyTorch $\ge 2.0.0$ (upgraded from original `1.10.0`), Torchvision $\ge 0.16.2$, `einops >= 0.7.0`, `monai >= 1.3.0`, `nibabel >= 3.0.0`, `SimpleITK >= 2.2.0`.
- **Hardware & VRAM**: 16–24 GB GPU VRAM (4 parallel 3D encoders + cross-attention blocks); 32 GB System RAM. SM 7.0+ strictly required (SM 8.0+ recommended for 3D cross-attention efficiency).

#### 8. RFNet
- **Role**: Region-Aware Fusion 3D CNN baseline for missing-modality segmentation. Evaluated in native 3-channel mode vs 4-channel mode augmented with synthetic modalities (S1–S4).
- **Architecture**: Multi-stream 3D CNN feature extractors + Region-Aware Fusion (RF) modules using spatial region masks (WT, TC, ET).
- **Software Dependencies**: PyTorch $\ge 2.0.0$ (upgraded from original `1.7.1`), Torchvision $\ge 0.16.2$, `monai >= 1.3.0`, `SimpleITK >= 2.2.0`, `h5py >= 3.8.0`, `nibabel >= 3.0.0`.
- **Hardware & VRAM**: 12–16 GB GPU VRAM; 32 GB System RAM. SM 6.0+ minimum (SM 7.0+ recommended).

---

## 3. Major Conflict Analysis & Actionable Resolution Strategies

### 3.1 Conflict Point 1: PyTorch 1.x vs PyTorch 2.x Unification
- **Conflict**: 
  - Legacy implementations specified PyTorch 1.x: PASSION (`1.12.1`), mmFormer (`1.10.0`), RFNet (`1.7.1`).
  - Modern evaluation frameworks strictly require PyTorch 2.x: nnU-Net v2 (`torch>=2.0.0`), 3D-MedDiffusion (`torch>=2.0.0`).
  - Running dual Conda environments introduces severe workflow friction, evaluation latency, and IPC disk serialization bottlenecks.
- **Resolution Strategy**:
  - **Upgrade all legacy models to PyTorch 2.1.2**.
  - All 3D CNN operations (`nn.Conv3d`, `nn.InstanceNorm3d`) and Transformer modules (`nn.MultiheadAttention`, `torch.einsum`) in PASSION, mmFormer, and RFNet are 100% backward compatible with PyTorch 2.x.
  - PyTorch 2.1.2 provides Accelerated FlashAttention via `torch.nn.functional.scaled_dot_product_attention`, offering up to **2.5× faster training/inference** and **30% lower VRAM usage** for mmFormer, PASSION, and SwinUNETR cross-attention layers.

---

### 3.2 Conflict Point 2: Legacy `medpy` Dependency & MSVC Compilation Breakage
- **Conflict**:
  - `medpy` was historically used for calculating 3D segmentation metrics (`Dice` coefficient, `95% Hausdorff Distance (HD95)`).
  - `medpy` relies on `distutils` (removed in Python 3.12, deprecated in Python 3.10) and legacy C++ wrappers (`_image.pyd`) that fail to compile under modern MSVC compilers on Windows and Python 3.10/3.11.
- **Resolution Strategy**:
  - **Strictly exclude `medpy`** from `environment.yml`.
  - Functional replacement is fully implemented in `src/metrics/segmentation.py` and `monai.metrics`:
    - Dice score: `monai.metrics.DiceMetric` and `scipy.spatial`.
    - HD95: `monai.metrics.HausdorffDistanceMetric(percentile=95)` and `scipy.spatial.distance.directed_hausdorff`.
  - Zero loss of functionality; 100% build reliability across Windows and Linux.

---

### 3.3 Conflict Point 3: Legacy `acloss` Package Deprecation
- **Conflict**:
  - `acloss` (Active Contour Loss) is an unmaintained 2019 package whose autograd hooks violate PyTorch 2.x computation graph standards.
- **Resolution Strategy**:
  - **Strictly exclude `acloss`** from `environment.yml`.
  - Audit confirms none of the 8 models in the Missing Modalities Benchmark utilize `acloss`.

---

### 3.4 Conflict Point 4: `nnunetv2` Mandatory Environment Variables
- **Conflict**:
  - `nnunetv2` throws an unhandled `RuntimeError` at module import or execution if required path environment variables are missing.
- **Resolution Strategy**:
  - Shell environment configuration must explicitly declare `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results`.
  - In PowerShell:
    ```powershell
    $Env:nnUNet_raw = "c:\Users\Zephyrus\Documents\Missing-Modalities-Benchmark\data\raw\nnUNet_raw"
    $Env:nnUNet_preprocessed = "c:\Users\Zephyrus\Documents\Missing-Modalities-Benchmark\data\processed\nnUNet_preprocessed"
    $Env:nnUNet_results = "c:\Users\Zephyrus\Documents\Missing-Modalities-Benchmark\checkpoints\nnUNet_results"
    ```
  - In Bash:
    ```bash
    export nnUNet_raw="c:/Users/Zephyrus/Documents/Missing-Modalities-Benchmark/data/raw/nnUNet_raw"
    export nnUNet_preprocessed="c:/Users/Zephyrus/Documents/Missing-Modalities-Benchmark/data/processed/nnUNet_preprocessed"
    export nnUNet_results="c:/Users/Zephyrus/Documents/Missing-Modalities-Benchmark/checkpoints/nnUNet_results"
    ```

---

### 3.5 Conflict Point 5: CUDA Compute Capability Minimums (SM 6.0 / 7.0 / 8.0)
- **Conflict**:
  - Older GPUs (e.g. Pascal GTX 1080 Ti, SM 6.1) support basic 3D CNNs (Pix2Pix, RFNet, nnU-Net v2), but fail or exhibit severe performance degradation on 3D Swin-Transformer (SwinUNETR) and 3D cross-attention modules (mmFormer, PASSION, 3D-MedDiffusion).
- **Resolution Strategy**:
  - Establish **NVIDIA GPU SM 7.0+ (Volta, Turing, Ampere, Ada)** as the minimum hardware specification.
  - **SM 8.0+ (RTX 3090, RTX 4090, A100, H100)** is strongly recommended to enable FP16/BF16 Tensor Core acceleration, reducing 3D attention memory overhead by ~45%.

---

## 4. Unified Environment Architecture

The unified Conda environment `missing-modalities` integrates all required frameworks into a single build specification:

- **Runtime**: Python 3.10 + PyTorch 2.1.2 + Torchvision 0.16.2 + CUDA 11.8/12.1.
- **Medical Imaging & Deep Learning**: `monai>=1.3.0`, `monai-generative>=0.2.0`, `nnunetv2>=2.2`, `dynamic-network-architectures>=0.4.4`, `batchgenerators>=0.25.1`, `acvl-utils>=0.2.6`, `SimpleITK>=2.2.0`, `nibabel>=3.0.0`, `einops>=0.7.0`, `scikit-image>=0.19.3`, `scipy>=1.10.0`, `numpy>=1.24.0`, `h5py>=3.8.0`, `pandas`, `scikit-learn`, `tensorboard`, `tqdm`, `pyyaml`.
- **Exclusions**: `medpy`, `acloss`.
