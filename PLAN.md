# Experiment Plan - Can Existing Generative Models Produce Viable MRI Modality Substitutes for Segmentation?

---

## Background & Motivation

Existing MRI modality synthesis research primarily evaluates image reconstruction quality or introduces new generation methods. There is limited systematic evidence on whether current generators produce synthetic modalities that are suitable substitutes for real MRI sequences in downstream segmentation, particularly across both conventional segmentation models and dedicated missing-modality architectures. Furthermore, the relationship between image fidelity metrics and downstream clinical utility remains poorly understood.

## Study Type

**Empirical comparative and methodological study.** No novel models are proposed. All generators and segmenters are existing published methods. The contributions are:

1. A controlled, within‑model comparative evaluation of the "synthesise then segment" paradigm against purpose‑built missing‑modality architectures.
2. A methodological evaluation of whether traditional pixel-level reconstruction metrics (PSNR, SSIM) are reliable predictors of downstream clinical task performance (segmentation).
3. A systematic failure-mode analysis characterizing the physical/biological reasons behind generative model translation failures.

## Research Questions

| #             | Question                                                                                                                                      | How tested                                                                                                                             |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| **RQ1** | Do existing generative models produce synthetic modalities that are viable substitutes for real ones in downstream segmentation?              | Freeze a full‑modality segmenter. Compare its performance on (4 real) vs (3 real + 1 synthetic). Small Dice drop = viable substitute. |
| **RQ2** | Do models with built‑in missing‑modality handling benefit from receiving a synthesised modality instead of using their native compensation? | Same missing‑modality model, compared against itself: native 3‑channel mode vs 3 real + 1 synthetic as full 4‑channel input.        |
| **RQ3** | Do traditional pixel-level quality metrics (PSNR, SSIM) correlate with and predict downstream segmentation performance (Dice, HD95)?          | Compute Pearson/Spearman correlation coefficients between (PSNR, SSIM) and (Dice, HD95) across all test cases and scenarios.           |

> [!IMPORTANT]
> **The generators are the subject of evaluation, not a contribution.** Segmentation models serve as measuring instruments — downstream, task‑based quality metrics for the synthetic modality. We are not benchmarking segmenters against each other, nor proposing new generative architectures.

![Experimental Pipeline Flowchart](figures/01-flowchart-experimental-pipeline.jpg)

---

## 1 Dataset & Pre‑processing

### 1.1 BraTS 2020

| Property         | Detail                                                                                                              |
| ---------------- | ------------------------------------------------------------------------------------------------------------------- |
| Modalities       | T1, T1ce, T2, FLAIR (all present per patient).                                                                      |
| Annotations      | Expert‑revised masks for WT, TC, ET.                                                                               |
| Size             | 369 training cases.                                                                                                 |
| Why this dataset | Standard benchmark; full‑modality availability enables controlled missingness; official nnU‑Net v2 weights exist. |

### 1.2 Pre‑processing

| Step                      | Justification                                                        |
| ------------------------- | -------------------------------------------------------------------- |
| Skull‑stripping          | Removes non‑brain voxels.                                           |
| N4 bias‑field correction | Corrects RF‑coil intensity inhomogeneity.                           |
| Per‑modality z‑score    | Standardises intensity distributions across patients and modalities. |

### 1.3 Split

- **70 / 15 / 15 %** patient‑wise (≈ 259 / 56 / 54).
- Same split for all experiments.

### 1.4 Missing‑Modality Scenarios

| Scenario | Available       | Synthesised | Clinical motivation                         |
| -------- | --------------- | ----------- | ------------------------------------------- |
| S1       | T1, T1ce, T2    | FLAIR       | Most commonly absent in retrospective data. |
| S2       | T1, T2, FLAIR   | T1ce        | Contrast skipped (allergy, cost).           |
| S3       | T1ce, T2, FLAIR | T1          | Pre‑contrast T1 occasionally omitted.      |
| S4       | T1, T1ce, FLAIR | T2          | Emergency protocol.                         |
| two_missing | T1, T2       | T1ce, FLAIR | Accelerated or abbreviated protocol.        |
| three_missing | T1         | T1ce, T2, FLAIR | Extreme emergency / triage.             |

---

## 2 Generators (Evaluated Models)

| Generator                  | Type                  | Venue / Year          | Universality | Why included                                                                                    |
| -------------------------- | --------------------- | --------------------- | ------------ | ----------------------------------------------------------------------------------------------- |
| **PS-MIT**           | Flow Matching         | arXiv 2024            | Category A   | Native arbitrary missingness via posterior sampling; BraTS 2020 verified; official code ([jongdory/PS-MIT](https://github.com/jongdory/PS-MIT)). |
| **M2DN**             | Diffusion (DDPM)      | IEEE TMI 2024         | Category A   | Modality-masked diffusion designed for random dropout; BraTS 2018/2020 verified; Level 0 adaptation. |
| **ResViT**           | Hybrid Transformer/GAN| IEEE TMI 2022         | Category B/C | Multi-input ViT + GAN fusion; BraTS 2018/IXI verified; official code ([icon-lab/ResViT](https://github.com/icon-lab/ResViT)). Requires Level 1 adaptation (retraining on BraTS 2020). |
| **CoLa-Diff**        | Latent Diffusion (3D) | MICCAI 2023 / TMI 2024| Category B   | Latent-space diffusion with cross-attention conditioning; BraTS 2019/2020 verified; official code. Requires Level 1 adaptation. |

Four generators are evaluated to span distinct modern generative paradigms (Flow Matching, DDPM, Latent Diffusion, and Transformer/GAN). All were selected from the [candidates report](synthesis_benchmark_candidates_report.md) based on official code availability, BraTS compatibility, and 3→1 support. See the candidates report for full evidence.

### Training Protocol

- **Supervision**: paired — (available modalities) → (missing modality).
- **Losses**: adversarial + L1 + SSIM / perceptual.
- **Early stopping**: validation PSNR / SSIM, patience = 20 epochs.
- **Freeze** after training. Generate synthetic modality for all val and test patients under all 6 scenarios (S1–S4, two_missing, three_missing).

---

## 3 Experimental Design

### 3.1 RQ1 — "Can existing generators produce a viable substitute?"

The segmentation model is a **frozen downstream evaluator**. It is not being trained or adapted — it simply processes the input and returns a segmentation. The Dice/HD95 difference between oracle and synthetic input is a **task‑based quality score for the generator**.

#### Evaluators

| Model                 | Architecture                       | Weight Source                              | Why this evaluator                                                                                                                           |
| --------------------- | ---------------------------------- | ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **nnU‑Net v2** | Self‑configuring 3D U‑Net (CNN)  | Official BraTS 2020 challenge weights      | Gold‑standard medical segmentation. Fully reproducible, no retraining.                                                                     |
| **SwinUNETR**   | Swin‑Transformer + U‑Net decoder | MONAI research-contributions (BraTS 2021 fine‑tuned) | Transformer‑based evaluator. Architecture‑agnostic comparison with nnU‑Net.                                                               |

> [!WARNING]
> **SwinUNETR weight provenance.** No official frozen BraTS 2020 SwinUNETR checkpoint exists. The best available option is the [Project-MONAI/research-contributions BraTS21](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/BRATS21) checkpoint. Before using it, we must verify:
> - **Training dataset**: BraTS 2021 (structurally compatible with BraTS 2020, but not identical).
> - **Preprocessing**: MONAI default transforms (RandCropByPosNegLabel, NormalizeIntensity per-channel).
> - **Modality ordering**: T1, T1ce, T2, FLAIR (matches our canonical order).
> - **Input dimensions**: 128×128×128 ROI (matches our patch size).
> - **Normalization**: Per-channel z-score (must confirm alignment with our `zscore_per_modality_per_patient`).
> - **Labels**: WT/TC/ET using BraTS convention {1, 2, 4} (matches our label definitions).
>
> If any of these diverge from our pipeline, we must either (a) retrain SwinUNETR on our BraTS 2020 split, or (b) document the mismatch as a confound.

> [!NOTE]
> **Why two evaluators?** If nnU‑Net shows a small Dice drop but SwinUNETR shows a large one (or vice versa), the quality of the synthetic modality is architecture‑dependent — a finding worth reporting. If both agree, the conclusion is robust.

#### Conditions (per evaluator, per scenario)

| Condition           | Input                | Role                                                                        |
| ------------------- | -------------------- | --------------------------------------------------------------------------- |
| **Oracle**    | 4 real modalities    | Ground truth performance — the standard the generator is measured against. |
| **Synthetic** | 3 real + 1 generated | Generator's output under evaluation.                                        |

#### Primary Readout

$$
\Delta\text{Dice} = \text{Dice}_{\text{oracle}} - \text{Dice}_{\text{synthetic}}
$$

- **ΔDice ≈ 0** → synthetic modality is a good substitute; it preserves segmentation‑relevant information.
- **ΔDice large** → synthetic modality loses critical information; the generator is not sufficient.

Same logic applies to ΔHD95.

---

## 3.2 RQ2 — "Does synthesis improve missing‑modality models?"

Each missing‑modality model is compared **against itself** under two input conditions.

#### Models

| Model              | Architecture                               | Why included                                                                                                |
| ------------------ | ------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| **AdaMM**    | Adaptive multi-modal fusion                | SOTA adaptive feature fusion missing-modality baseline.                                                     |
| **mmFormer** | Multi‑modal transformer, cross‑attention | Transformer missing‑modality baseline.                                                                     |
| **RFNet**    | CNN, region‑aware fusion                  | CNN missing‑modality baseline.                                                                             |
| **UniME**    | Unified Masked Image Modeling ViT          | SOTA incomplete-modality prior via MIM. (Included conditionally based on pre-trained weights).              |

#### Conditions (per model, per scenario)

| Condition                | Input                                                   | Role                                                             |
| ------------------------ | ------------------------------------------------------- | ---------------------------------------------------------------- |
| **Native missing** | Available real channels + missing flag                 | Model's own baseline — its designed behaviour.                  |
| **+ PS-MIT**       | Available + PS-MIT synthetic (full 4-ch mode)          | Does Flow Matching beat native compensation?                     |
| **+ M2DN**         | Available + M2DN synthetic (full 4-ch mode)            | Does DDPM beat native compensation?                              |
| **+ ResViT**       | Available + ResViT synthetic (full 4-ch mode)          | Does ViT/GAN beat native compensation?                           |
| **+ CoLa-Diff**    | Available + CoLa-Diff synthetic (full 4-ch mode)       | Does Latent Diffusion beat native compensation?                  |
| **Oracle**         | 4 real channels                                         | Ceiling — how much room exists above native missing?            |

#### Primary Readout

$$
\Delta\text{Dice} = \text{Dice}_{\text{synthetic}} - \text{Dice}_{\text{native missing}}
$$

- **ΔDice > 0** → synthesis helps; the generated modality carries information the model's internal mechanism cannot recover.
- **ΔDice ≈ 0** → the model's built‑in compensation is already sufficient; synthesis adds nothing.
- **ΔDice < 0** → synthetic artefacts actively interfere with the model's learned representations. Native handling is safer.

---

## 3.3 Full Condition Matrix

#### RQ1 — Substitute quality (evaluator models)

| Evaluator   | Oracle    | + PS-MIT | + M2DN | + ResViT | + CoLa-Diff |
| ----------- | --------- | -------- | ------ | -------- | ----------- |
| nnU‑Net v2 | ✅ All 6 | ✅ All 6 | ✅ All 6 | ✅ All 6 | ✅ All 6   |
| SwinUNETR   | ✅ All 6 | ✅ All 6 | ✅ All 6 | ✅ All 6 | ✅ All 6   |

**2 evaluators × 5 conditions × 6 scenarios = 60 cells**

#### RQ2 — Synthesis vs native handling

| Model    | Native missing | + PS-MIT | + M2DN | + ResViT | + CoLa-Diff | Oracle    |
| -------- | -------------- | -------- | ------ | -------- | ----------- | --------- |
| AdaMM    | ✅ All 6      | ✅ All 6 | ✅ All 6 | ✅ All 6 | ✅ All 6   | ✅ All 6 |
| mmFormer | ✅ All 6      | ✅ All 6 | ✅ All 6 | ✅ All 6 | ✅ All 6   | ✅ All 6 |
| RFNet    | ✅ All 6      | ✅ All 6 | ✅ All 6 | ✅ All 6 | ✅ All 6   | ✅ All 6 |
| UniME    | ✅ All 6      | ✅ All 6 | ✅ All 6 | ✅ All 6 | ✅ All 6   | ✅ All 6 |

**4 models × 6 conditions × 6 scenarios = 144 cells**

**Total: 204 evaluation cells** on ~54 test patients.

![RQ1 vs RQ2 Experimental Setup Comparison](figures/02-comparison-rq1-rq2.jpg)

---

## 4 Metrics

### 4.1 Generator Quality — Pixel‑Level

| Metric         | What it measures                                          | Role                                                                  |
| -------------- | --------------------------------------------------------- | --------------------------------------------------------------------- |
| **PSNR** | Pixel‑wise reconstruction fidelity.                      | Sanity check. High PSNR = intensities are close to real.              |
| **SSIM** | Structural similarity (luminance + contrast + structure). | Perceptual quality. Correlates better with human judgement than PSNR. |

### 4.2 Generator Quality — Task‑Level (Primary)

| Metric                      | What it measures                                                   | Role                                                                                                                          |
| --------------------------- | ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| **Dice (WT, TC, ET)** | Volumetric overlap when downstream segmenter uses synthetic input. | **The metric that matters.** Directly measures whether the synthetic modality preserves the features a segmenter needs. |
| **HD95 (WT, TC, ET)** | Boundary accuracy under synthetic input.                           | Complements Dice — catches boundary degradation that Dice may miss.                                                          |

> [!TIP]
> **The relationship between pixel‑level and task‑level metrics is itself a finding.** If PSNR/SSIM are high but Dice drops significantly, the generator is reconstructing the wrong features — it's pixel‑accurate but not task‑relevant. If PSNR/SSIM are mediocre but Dice holds, the generator preserves the features that matter despite cosmetic imperfections.

### 4.3 Methodological Analysis (RQ3): Metric Decoupling

We analyze whether traditional image-quality metrics (PSNR, SSIM) are reliable predictors of downstream clinical utility.

| Method                                   | Objective                                                                                                                  | Rationale                                                                                                                                                                                                                                                                           |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Joint Metric Correlation**       | Compute Spearman's $\rho$ and Pearson's $r$ between per-patient image metrics (PSNR/SSIM) and task metrics (Dice/HD95). | Tests the hypothesis that structural synthesis fidelity correlates with segmentation task performance.                                                                                                                                                                              |
| **Outlier & Discordance Analysis** | Identify cases where: 1. PSNR/SSIM is high but downstream Dice is low. 2. PSNR/SSIM is low but downstream Dice is high.      | Pinpoints *why* traditional voxel-wise metrics fail. For example, a generator might perfectly reconstruct normal brain tissues (high PSNR) but erase/deform the tumor (low Dice), or it might introduce background noise (low PSNR) while preserving tumor boundaries (high Dice). |

![Metric Decoupling Framework Matrix](figures/03-framework-metric-decoupling.jpg)

### 4.4 Systematic Failure-Mode Analysis

Instead of simply reporting that a generator fails, we categorize *how* and *why* generators fail on specific modalities or patient classes:

| Failure Mode                                     | Definition / Metric                                           | Physical / Biological Cause                                                                                                                                        |
| ------------------------------------------------ | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Lesion Erasure / Hallucination**         | Change in detected tumor volume (prediction vs ground truth). | Generator maps pathology to normal tissue distribution (erasure) or maps normal variance to pathology (hallucination) due to mode collapse or over-regularization. |
| **Boundary Blurring / de-differentiation** | High HD95 despite reasonable Dice.                            | Loss of high-frequency details (common in GANs and 2D methods), leading to poor contrast at tumor boundaries (e.g., T1ce enhancing border).                        |
| **Contrast Inversion / Domain Shift**      | Extreme intensity deviation from real target sequence.        | Inability of 2D slice-wise methods to normalize intensity across the full 3D volume, or failure to capture complex scanner-specific bias fields.                   |
| **Spatial / Structural Warping**           | Distortions in ventricular shape or midline shifts.           | Generator alters anatomy due to weak structural constraints (e.g., excessive deformation in latent space).                                                         |

#### Stratification

We stratify downstream segmentation errors (Dice drop) by:

1. **Tumor Size**: Small (<5 cc) vs Medium (5-50 cc) vs Large (>50 cc). Hypothesized that generators struggle to synthesize small local lesions.
2. **Tumor Composition**: Dominantly necrotic/cystic vs active enhancing vs edematous tumor (WT, TC, ET subregions). This highlights scenario-specific limits (e.g., synthesizing T1ce enhancing core S2 vs FLAIR edema S1).

> [!WARNING]
> If a generator shows high average PSNR/SSIM but fails catastrophically (lesion erasure) on small tumors, it is clinically unsafe. Identifying these failure thresholds is the core methodological contribution of the paper.

---

## 5 Statistical Testing

| Step                        | Method                                                         | Justification                                                                                                                                                                                                                                                                                             |
| --------------------------- | -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Normality check             | Shapiro‑Wilk on per‑patient ΔDice/ΔHD95                    | Medical imaging metrics often violate normality.                                                                                                                                                                                                                                                          |
| Paired test                 | Wilcoxon signed‑rank (non‑normal) or paired t‑test (normal) | Same patients under two conditions for the same model. Maximally controlled pairing.                                                                                                                                                                                                                      |
| Multiple comparisons        | Bonferroni                                                     | Controls error rate across sub‑regions × scenarios × generators.                                                                                                                                                                                                                                       |
| Effect size                 | Cohen's d or rank‑biserial r                                  | Quantifies practical significance. A statistically significant 0.3 % Dice drop is not clinically meaningful; effect size makes this clear.                                                                                                                                                                |
| Equivalence test (optional) | TOST (Two One‑Sided Tests)                                    | For RQ1, the goal is to show the synthetic condition is **not worse** than oracle, not that it's better. A standard test can fail to reject H₀ (no difference) without proving equivalence. TOST directly tests whether ΔDice falls within a pre‑specified equivalence margin (e.g., ±1 % Dice). |

> [!IMPORTANT]
> **Why consider TOST for RQ1.** Standard null‑hypothesis testing asks "is there a difference?" But RQ1 is "is the synthetic modality a viable substitute" — that's an equivalence question. Failing to find a significant difference (p > 0.05) does not prove equivalence; it may just mean insufficient power. TOST with a clinically meaningful margin (e.g., ΔDice < 1 %) is the correct test for this question.

---

## 6 Possible Outcomes & Interpretations

### RQ1 (Substitute Quality)

| Outcome                                                           | Interpretation                                                                                                 |
| ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| ΔDice < 1 % for best generator (e.g., PS-MIT or M2DN), both evaluators | Excellent substitute. Synthesis preserves nearly all segmentation‑relevant information.                       |
| ΔDice < 1 % for diffusion/flow but > 3 % for ResViT             | Generator paradigm is decisive. Diffusion/flow matching outperform hybrid ViT/GAN for synthesis.               |
| ΔDice > 3 % for all four generators                              | Current generators are not good enough. The gap is too large to call synthesis a viable substitute.            |
| ΔDice varies by scenario (e.g., small for FLAIR, large for T1ce) | Some modalities are harder to synthesise than others. Claim holds conditionally.                               |
| nnU‑Net and SwinUNETR show different ΔDice patterns             | Synthetic quality is architecture‑dependent — the claim needs qualification.                                 |
| Multi-missing scenarios show catastrophic ΔDice (> 10 %)        | Synthesis viability degrades sharply under extreme missingness.                                                 |

### RQ2 (Synthesis vs Native Handling)

| Outcome                                          | Interpretation                                                                                                                         |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| Synthesis helps all four models                  | Generated modality carries information that even purpose‑built architectures cannot recover internally. Strong result.                |
| Synthesis helps RFNet / mmFormer but not AdaMM   | AdaMM's adaptive fusion mechanism already recovers what the generator provides. Synthesis substitutes for architectural sophistication. |
| Synthesis hurts all four models                  | Synthetic artefacts interfere with learned missing‑modality representations. Native handling is strictly better.                      |
| Flow/diffusion helps, ViT/GAN hurts              | There is a generator quality threshold below which synthesis is harmful.                                                               |
| UniME (if included) is immune to synthesis       | MIM pre-training already hallucinated sufficient features internally.                                                                  |

---

## 7 Stated Limitations

1. **Four generators only.** Spans Flow Matching, DDPM, Latent Diffusion, and Transformer/GAN, but excludes other paradigms (e.g., score-based models, normalising flows, wavelet diffusion).
2. **ResViT is 2D slice‑wise.** Cannot fully disentangle 2D vs 3D effects from the generative paradigm itself. PS-MIT, M2DN, and CoLa-Diff operate in 3D.
3. **Single dataset (BraTS 2020).** May not generalise to other anatomies, field strengths, or vendor protocols.
4. **Simulated missingness.** All four modalities exist for every patient; missingness is artificially imposed. Real‑world missing data may have different characteristics.
5. **SwinUNETR trained on BraTS 2021.** Preprocessing and label mapping are compatible but not identical to BraTS 2020. Any evaluation discrepancy may partially reflect dataset shift rather than synthetic quality alone.
6. **Generator retraining required for two models.** ResViT and CoLa-Diff (Level 1 adaptation) must be retrained on our BraTS 2020 split. This introduces retraining variance not present in Level 0 models (PS-MIT, M2DN).

---

## 8 Checklist

### Data

- [ ] Freeze train/val/test split.
- [ ] Pre‑process all BraTS 2020 volumes.

### Generators

- [ ] Wrap and validate PS-MIT (S1-S4, Multi-Missing).
- [ ] Wrap and validate M2DN (S1-S4, Multi-Missing).
- [ ] Wrap and validate ResViT (S1-S4, Multi-Missing).
- [ ] Wrap and validate CoLa-Diff (S1-S4, Multi-Missing).
- [ ] Generate synthetic modalities for val & test sets.
- [ ] Compute PSNR / SSIM for all generated volumes.

### RQ1 — Substitute Quality

- [ ] Run nnU‑Net v2 on oracle inputs (All 6 scenarios).
- [ ] Run nnU‑Net v2 on synthetic inputs (4 generators × 6 scenarios).
- [ ] Run SwinUNETR on oracle inputs (All 6 scenarios).
- [ ] Run SwinUNETR on synthetic inputs (4 generators × 6 scenarios).
- [ ] Compute Dice / HD95 for all 60 cells.

### RQ2 — Synthesis vs Native Handling

- [ ] Train AdaMM, mmFormer, RFNet, (and UniME) on incomplete training data.
- [ ] Evaluate each in native missing mode (All 6 scenarios).
- [ ] Evaluate each with PS-MIT synthetic input (All 6 scenarios).
- [ ] Evaluate each with M2DN synthetic input (All 6 scenarios).
- [ ] Evaluate each with ResViT synthetic input (All 6 scenarios).
- [ ] Evaluate each with CoLa-Diff synthetic input (All 6 scenarios).
- [ ] Evaluate each on oracle (All 6 scenarios).
- [ ] Failure gallery (3–5 qualitative cases).
- [ ] Statistical tests (Shapiro‑Wilk → Wilcoxon/t‑test → Bonferroni → effect sizes).
- [ ] TOST equivalence test for Claim 1 (optional but recommended).

## 9 Reporting

### Tables

- **Table 1**: RQ1 results — Dice (mean ± SD) per evaluator × generator × scenario. ΔDice from oracle highlighted.
- **Table 2**: RQ1 results — HD95, same layout.
- **Table 3**: RQ2 results — Dice per missing‑modality model × condition × scenario.
- **Table 4**: Generation quality — PSNR / SSIM per generator × scenario.
- **Table 5**: Statistical summary — p‑values, effect sizes, TOST results for key comparisons.

### Figures

- **Fig 1**: RQ1 results — paired bar chart (oracle vs synthetic) per evaluator.
- **Fig 2**: RQ2 results — grouped bars per missing‑modality model.
- **Fig 3**: RQ3 results — Joint scatter plots of SSIM/PSNR vs. $\Delta$Dice across test cases.
- **Fig 4**: Failure Analysis — Stratified bar charts of Dice drops grouped by tumor size and composition.
- **Fig 5**: Qualitative Failure Casebook — example slices annotated with failure types.

### Conclusion Template

> *"Using [nnU‑Net v2 / SwinUNETR] as a downstream evaluator, [PS-MIT / M2DN / ResViT / CoLa-Diff]‑synthesised [modality] achieved a downstream Dice within [X.X ± Y.Y %] of the real‑modality oracle (p = Z.ZZ, equivalence confirmed/not confirmed within a ±1 % margin). Notably, correlation analysis revealed that voxel-wise reconstruction metrics (PSNR, SSIM) [correlated strongly / decoupled] with downstream performance (Spearman's $\rho$ = W.WW), suggesting that pixel-level fidelity [is / is not] a reliable proxy for clinical task utility. Systematic failure analysis highlighted that translation models primarily failed due to [lesion erasure in small tumors / boundary blurring / contrast domain shifts]."*
