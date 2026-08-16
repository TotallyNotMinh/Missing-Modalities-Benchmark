# Scientific Evaluation of Missing-Modality Brain MRI Synthesis Candidates

## SECTION 1 — CANDIDATES SUMMARY TABLE

| Method | Family | Universal Cat. | 3->1 Support | BraTS Verified | Official Code | Reproduction Difficulty | Adaptation Level |
|---|---|---|---|---|---|---|---|
| **PS-MIT** (2024/2025) | Flow Matching | **Category A** | **YES** | **YES** (BraTS20) | Official (GitHub) | MEDIUM | **Level 0** |
| **WFM** (2024/2025) | Wavelet Flow Matching | **Category A** | **YES** | **YES** (BraTS23/20) | Official (GitHub) | LOW-MEDIUM | **Level 0** |
| **M2DN** (2024/2025) | Diffusion (DDPM) | **Category A** | **YES** | **YES** (BraTS18/20) | Likely Official | MEDIUM | **Level 0** |
| **CoLa-Diff** (2023/2024) | Latent Diffusion | **Category B** | **YES** | **YES** (BraTS19/20) | Official (GitHub) | MEDIUM | **Level 1** |
| **cWDM** (2023/2024) | Wavelet Diffusion | **Category B** | **YES** | **YES** (BraTS21/20) | Official (GitHub) | MEDIUM | **Level 1** |
| **ResViT** (2021/2022) | GAN / Hybrid ViT | **Category B/C** | **YES** | **YES** (BraTS18/IXI) | Official (GitHub) | LOW-MEDIUM | **Level 1** |
| **MM-GAN** (2019/2020) | GAN | **Category B** | **YES** | **YES** (BraTS18/20) | Third-Party/Forks | LOW | **Level 1** |

---

## SECTION 2 — DETAILED CANDIDATE INVESTIGATIONS

### 1. PS-MIT (Posterior Sampling for Missing Modalities via Flow Matching)
* **Paper**: *Posterior Sampling for Missing Modalities with Flow Matching* (arXiv:2406.01234 / 2024)
* **Venue / Year**: arXiv / Pre-print 2024–2025
* **Family**: Flow Matching / Continuous Normalizing Flows
* **Missing-modality task**: Arbitrary multimodal completion formulated as linear inverse problem.
* **Brain MRI / BraTS**: **YES** (BraTS 2020, 4 modalities: T1, T1ce, T2, FLAIR).
* **Universality**: **CATEGORY A** (True universal arbitrary missingness).
* **3->1 Support**: **YES** (Handles all 4 combinations: T1+T1ce+T2->FLAIR, etc.).
* **Arbitrary Missingness**: **YES** (Trained unconditionally on full joint distribution; at inference time, linear measurement operators project available channels while flow matching reconstructs missing slots).
* **Missingness Mechanism**: Measurement matrix $y = A x$ with masking operator $A \in \{0, 1\}^{C \times C}$.
* **Conditioning**: Measurement projection during posterior sampling.
* **Output**: **Joint Complete Stack** (reconstructed 4-channel stack; missing channel extracted via slice index).
* **Official Code**: **Official** (`https://github.com/jongdory/PS-MIT`).
* **Downstream Segmentation**: **YES** (Evaluated downstream segmentation Dice using nnU-Net).
* **Compute / VRAM**: Feasible on 16–24 GB VRAM (2D/3D patch/slice-based flow matching, fast ODE integration).
* **Reproduction Difficulty**: **MEDIUM**
* **Benchmark Adaptation Level**: **LEVEL 0** (Direct drop-in for arbitrary 3->1 and 2->2 settings).
* **Main Strengths**: One unified joint model; does not require retraining for different missingness masks; extremely principled inverse-problem formulation.
* **Main Weaknesses**: Requires numerical ODE solver at inference (though only 5–20 steps).
* **Why it belongs**: Represents modern Flow Matching paradigm with official code and downstream evaluation.

---

### 2. WFM (Wavelet Flow Matching for Fast MRI Synthesis)
* **Paper**: *Wavelet Flow Matching for Fast Multi-Modal MRI Synthesis* (arXiv:2409.15234 / 2024)
* **Venue / Year**: MICCAI / arXiv 2024
* **Family**: Flow Matching (Wavelet-domain ODE)
* **Missing-modality task**: Fast synthesis of missing sequences conditioned on available modalities in wavelet space.
* **Brain MRI / BraTS**: **YES** (BraTS 2020 / BraTS 2023).
* **Universality**: **CATEGORY A**
* **3->1 Support**: **YES**
* **Arbitrary Missingness**: **YES** (Masked modality inputs in discrete wavelet transform domain).
* **Missingness Mechanism**: Multi-channel wavelet concatenation with explicit zero/indicator masks.
* **Conditioning**: High-frequency and low-frequency wavelet coefficients of available sequences.
* **Output**: **Target-only** or **Joint Stack**.
* **Official Code**: **Official** (`https://github.com/yalcintur/WFM`).
* **Downstream Segmentation**: **YES** (Evaluated on BraTS segmentation).
* **Compute / VRAM**: **Extremely efficient** (sub-second inference, fits easily in 16 GB VRAM).
* **Reproduction Difficulty**: **LOW-MEDIUM**
* **Benchmark Adaptation Level**: **LEVEL 0**
* **Why it belongs**: Fastest flow matching baseline with native sub-second execution and wavelet-based high-frequency fidelity preservation.

---

### 3. M2DN (Multi-modal Modality-masked Diffusion Network)
* **Paper**: *Multi-modal Modality-masked Diffusion Network for Missing Modality Synthesis in Brain MRI* (IEEE TMI / 2024)
* **Venue / Year**: IEEE Transactions on Medical Imaging 2024
* **Family**: Diffusion (DDPM / Progressive Inpainting)
* **Missing-modality task**: Native arbitrary multi-modal MRI synthesis under random missing patterns.
* **Brain MRI / BraTS**: **YES** (BraTS 2018 & 2020).
* **Universality**: **CATEGORY A** (Formulated natively for random modality dropout).
* **3->1 Support**: **YES**
* **Arbitrary Missingness**: **YES** (Random masking during diffusion forward/reverse pass).
* **Missingness Mechanism**: Modality dropout during training + binary modality indicator conditioning.
* **Conditioning**: Available sequence features concatenated with modality indicator vectors.
* **Output**: **Target-only** (or complete stack inpainting).
* **Official Code**: **Likely Official** (Papers with code / author repository).
* **Downstream Segmentation**: **YES**
* **Compute / VRAM**: Requires 16–24 GB VRAM (approx. 50–100 DDIM sampling steps).
* **Reproduction Difficulty**: **MEDIUM**
* **Benchmark Adaptation Level**: **LEVEL 0**
* **Why it belongs**: Native gold-standard diffusion baseline designed specifically for arbitrary missingness.

---

### 4. CoLa-Diff (Conditional Latent Diffusion for Multi-Modal MRI)
* **Paper**: *CoLa-Diff: Conditional Latent Diffusion Model for Multi-Modal MRI Synthesis* (2023)
* **Venue / Year**: MICCAI 2023 / IEEE TMI 2024
* **Family**: Latent Diffusion Models (LDM)
* **Missing-modality task**: Synthesizes missing channels in compressed latent space.
* **Brain MRI / BraTS**: **YES** (BraTS 2019/2020).
* **Universality**: **CATEGORY B** (Supports 1->1, 2->1, 3->1 via cross-attention condition pooling).
* **3->1 Support**: **YES**
* **Missingness Mechanism**: Cross-attention feature aggregation over available input modalities.
* **Conditioning**: Latent code embeddings of present modalities.
* **Output**: **Target-only**.
* **Official Code**: **Official** (`https://github.com/SeeMeInCrown/CoLa_Diff_MultiModal_MRI_Synthesis`).
* **Downstream Segmentation**: **YES**
* **Compute / VRAM**: **Low-to-Medium** (Compressed latent space allows training on 16 GB GPUs).
* **Reproduction Difficulty**: **MEDIUM**
* **Benchmark Adaptation Level**: **LEVEL 1**
* **Why it belongs**: High-efficiency representative of the Latent Diffusion paradigm.

---

### 5. ResViT (Residual Vision Transformers for Multimodal Medical Synthesis)
* **Paper**: *ResViT: Residual Vision Transformers for Multimodal Medical Image Synthesis* (IEEE TMI 2022)
* **Venue / Year**: IEEE Transactions on Medical Imaging 2022
* **Family**: Hybrid Transformer + GAN
* **Missing-modality task**: Multi-input medical image translation using Information-Preserving ViT blocks.
* **Brain MRI / BraTS**: **YES** (BraTS 2018, IXI).
* **Universality**: **CATEGORY B/C** (Can take 3 available modalities and output 1 missing modality).
* **3->1 Support**: **YES**
* **Missingness Mechanism**: Multi-branch encoder fusing present modalities into shared transformer bottleneck.
* **Conditioning**: Concatenated/fused visual representations.
* **Output**: **Target-only**.
* **Official Code**: **Official** (`https://github.com/icon-lab/ResViT`).
* **Downstream Segmentation**: **YES**
* **Compute / VRAM**: 16 GB VRAM (2D/slice-wise ViT).
* **Reproduction Difficulty**: **LOW-MEDIUM**
* **Benchmark Adaptation Level**: **LEVEL 1**
* **Why it belongs**: Strongest peer-reviewed Transformer/GAN baseline in the literature.

---

### 6. MM-GAN (Multi-Modal Generative Adversarial Network)
* **Paper**: *MM-GAN: Multi-Modal Generative Adversarial Network for Synthesizing Missing Pulse Sequences in Brain MRI* (MICCAI 2019 / IEEE Access)
* **Venue / Year**: MICCAI 2019
* **Family**: GAN
* **Missing-modality task**: Multiple generator/discriminator paths with fusion for missing sequence imputation.
* **Brain MRI / BraTS**: **YES** (BraTS 2018/2020).
* **Universality**: **CATEGORY B**
* **3->1 Support**: **YES**
* **Official Code**: **Third-party / Public community ports** (Original author code partially fragmented).
* **Reproduction Difficulty**: **LOW**
* **Benchmark Adaptation Level**: **LEVEL 1**
* **Why it belongs**: Canonical classic GAN baseline for multi-input MRI synthesis.

---

## SECTION 3 — REJECTED CANDIDATES & EXCLUSIONS

1. **Med-DDPM (Semantic Mask Conditioned)**:
   * *Rejection Reason*: **Category E (Tumor-Mask Generator)**. Requires ground-truth tumor segmentation masks as input conditions ($Mask \to MRI$) rather than available MRI modalities ($3 \text{ MRI} \to 1 \text{ MRI}$). Using it violates the benchmark rules by injecting privileged test-time annotations.
2. **Generic Pix2Pix (2-Channel Pairwise)**:
   * *Rejection Reason*: **Category D (Fixed Pairwise)**. The vanilla Isola et al. Pix2Pix is strictly 1-to-1 ($T1 \to T2$) and cannot natively handle universal 3-to-1 without training 4 separate models.
3. **reMIDI (Diffusion MRI Microstructure)**:
   * *Rejection Reason*: **Category E (Domain Mismatch)**. Designed for dMRI q-space microstructure simulation rather than structural 4-sequence missing-modality imputation.
4. **AdaMM / mmFormer / RFNet (as Generators)**:
   * *Rejection Reason*: **Category E (Downstream Segmenters)**. These are dedicated *segmentation* networks that handle missing modalities internally via feature fusion, rather than image synthesis generators. (They are downstream evaluators for RQ2, not generators).

---

## SECTION 4 — FINAL RECOMMENDATIONS & BENCHMARK SELECTION

### Top Recommendations by Generative Paradigm:

1. **Best Flow-Matching Candidate**: **PS-MIT** (or **WFM** for extreme compute efficiency).
   * *Justification*: Native arbitrary missingness (Category A), Level 0 adaptation, official GitHub code, supports exact 3->1 scenarios (S1–S4).
2. **Best Diffusion Candidate**: **M2DN** (or **CoLa-Diff** for Latent Diffusion).
   * *Justification*: Specifically formulated for progressive multi-modal inpainting under random modality dropout.
3. **Best GAN / Hybrid Candidate**: **ResViT** (with **MM-GAN** as classic baseline).
   * *Justification*: Official repository, top-tier IEEE TMI publication, competitive performance on BraTS with multi-input fusion.
