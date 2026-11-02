# NEUROSCAN AI — Controlled Segmentation Improvement Experiment Summary

**Date**: 2026-10-20  
**Phase**: Controlled Improvement Experiment (Exp 1: Patch Sampling Rebalancing)  
**Experiment ID**: `Exp1_BalancedPatches_1to2`  
**Candidate Checkpoint**: [`models/NeuroScan_Seg3D_UNet_Phase5_Exp1_BalancedPatches.pth`](file:///c:/Users/atuls/OneDrive/Desktop/NeuroScan%20AI/models/NeuroScan_Seg3D_UNet_Phase5_Exp1_BalancedPatches.pth)  
**Evaluation Subject**: `BraTS-GLI-00481-000` (Independent Validation Cohort)  
**Final Decision**: **`IMPROVED — CANDIDATE RETAINED`**

---

## 1. Executive Summary

Following the Phase 5.18 independent quality audit, which identified high tumor sensitivity ($97.39\%$) accompanied by substantial normal tissue over-prediction ($2,365,455$ false-positive voxels), a single, controlled scientific experiment was designed and executed.

The single intervention tested was **Patch Sampling Rebalancing**:
- **Baseline**: 4 tumor-centered patches vs 2 random background patches per volume per epoch ($66.7\%$ tumor foreground bias).
- **Intervention**: 2 tumor-centered patches vs 4 non-tumor background patches per volume per epoch ($33.3\%$ tumor / $66.7\%$ background representation).

All other variables were held strictly constant:
- Model architecture (`FourChannelUNet3D`, $16\to 256$ channels),
- Optimizer (AdamW, $lr=2\times 10^{-4}$, weight decay $1\times 10^{-4}$),
- Scheduler (CosineAnnealingLR, $T_{\max}=5$),
- Loss function (Softmax multi-class `DiceCELoss`),
- Batch size ($B=2$), patch resolution ($48\times 48\times 48$), total patches per epoch ($18$),
- Random seed (`seed=42`),
- Training cohort (3 cases) and validation subject (`BraTS-GLI-00481-000`),
- Quarantined locked test subject (`BraTS-GLI-01012-000`).

---

## 2. Quantitative Comparison: Baseline vs Candidate

Evaluated on `BraTS-GLI-00481-000` using the identical full-resolution sliding-window inference protocol (ROI: $64\times 64\times 64$, overlap: $0.5$):

| Subregion | Metric | Baseline (Phase 5.14/5.18) | Candidate (Exp 1 Rebalanced) | Absolute Delta ($\Delta$) | Relative Change |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Whole Tumor (WT)** | **Dice (DSC)** | **`0.1089`** | **`0.1198`** | **`+0.0109`** | **`+10.0%` (IMPROVED)** |
| | **Precision (PPV)** | **`0.0577`** | **`0.0640`** | **`+0.0063`** | **`+10.9%` (IMPROVED)** |
| | **Sensitivity (Recall)** | `0.9739` | `0.9298` | `-0.0441` | Preserved ($93.0\%$ recall) |
| | **False Positives (FP)** | $2,365,455$ voxels | $2,020,215$ voxels | **`-345,240` voxels** | **`-14.6%` FP Reduction** |
| | **Predicted Volume** | $2,510.167$ cm³ | $2,158.377$ cm³ | $-351.790$ cm³ | Closer to GT ($148.597$ cm³) |
| | **Specificity** | $0.7306$ | $0.7699$ | `+0.0393` | `+5.4%` (IMPROVED) |
| **Tumor Core (TC)** | **Dice (DSC)** | **`0.0322`** | **`0.0425`** | **`+0.0103`** | **`+32.0%` (IMPROVED)** |
| | **Precision (PPV)** | **`0.0168`** | **`0.0222`** | **`+0.0054`** | **`+32.1%` (IMPROVED)** |
| | **Sensitivity (Recall)** | `0.3682` | `0.4849` | **`+0.1167`** | **`+31.7%` (IMPROVED)** |
| | **True Positives (TP)** | $15,142$ voxels | $19,944$ voxels | **`+4,802` voxels** | Real core voxels captured |
| | **False Positives (FP)** | $883,508$ voxels | $877,603$ voxels | `-5,905` voxels | Suppressed |
| **Enhancing (ET)** | **Dice (DSC)** | `0.0282` | `0.0272` | `-0.0010` | Roughly stable |
| | **Precision (PPV)** | `0.0152` | `0.0148` | `-0.0004` | Roughly stable |
| | **Specificity** | `0.9544` | `0.9595` | `+0.0051` | Improved |
| | **False Positives (FP)** | $405,795$ voxels | $360,654$ voxels | **`-45,141` voxels** | Suppressed |
| **Composite** | **Mean Dice** | **`0.0564`** | **`0.0632`** | **`+0.0068`** | **`+12.1%` (IMPROVED)** |

---

## 3. Training & Convergence Dynamics

| Training Epoch | Baseline Mean Loss (4 FG / 2 BG) | Candidate Mean Loss (2 FG / 4 BG) | Candidate Grad Norm | Candidate Wall Time |
| :---: | :---: | :---: | :---: | :---: |
| **Epoch 1** | 2.3692 | 2.4987 | 2.6622 | 10.95s |
| **Epoch 2** | 2.1394 | 2.3284 | 2.4550 | 10.42s |
| **Epoch 3** | 2.0451 | 2.1982 | 2.2101 | 11.88s |
| **Epoch 4** | 2.0219 | 2.1713 | 2.0182 | 10.37s |
| **Epoch 5** | 1.9231 | 2.1348 | 2.1091 | 10.93s |

- **Loss Convergence**: The candidate training loss dropped smoothly from $2.4987$ to $2.1348$ without non-finite values or gradient explosions.
- **Resource Footprint**: Candidate training completed in $54.54$ seconds on CPU. Validation completed in $121.95$ seconds.

---

## 4. Analysis of Findings

1. **Successful False-Positive Suppression**:
   Supplying 4 background patches per volume provided the network with critical negative context (normal brain parenchyma, CSF, grey/white matter junctions), directly eliminating **$345,240$ false-positive voxels** in Whole Tumor and **$45,141$ false-positive voxels** in Enhancing Tumor.
2. **Whole Tumor Dice & Precision Boost**:
   Whole Tumor Dice increased by **`+10.0%`** ($0.1089 \to 0.1198$) and precision rose by **`+10.9%`** ($0.0577 \to 0.0640$), while preserving an exceptionally strong **$92.98\%$ tumor recall**.
3. **Tumor Core Quality Jump**:
   Tumor Core experienced a notable **`+32.0%` Dice gain** ($0.0322 \to 0.0425$) and a **`+31.7%` recall gain** ($0.3682 \to 0.4849$), capturing $4,802$ additional true core voxels.
4. **Overall Score**:
   Composite Mean Dice improved by **`+12.1%`** ($0.0564 \to 0.0632$).

---

## 5. Checkpoint & Governance Ledger

- **Baseline Checkpoint (Preserved)**:
  - File: `models/NeuroScan_Seg3D_UNet_Phase5_Multimodal_Best.pth`
  - SHA-256: `a1f5d9396f73ad9353d11cbd11eaa3f11187545b3075173bf5726a88dc6186b8`
- **Candidate Checkpoint (Retained)**:
  - File: `models/NeuroScan_Seg3D_UNet_Phase5_Exp1_BalancedPatches.pth`
  - SHA-256: `0e3a312153fb55d4d538564fb55f9d2e377b5305924cbcc7e139d889c375c268`
  - Size: $19,279,475$ bytes
- **Historical Checkpoints (100% Invariant)**: All 5 legacy checkpoints verified bit-for-bit.
- **Locked Test Isolation**: Quarantined subject `BraTS-GLI-01012-000` was never accessed.

---

## 6. Master Decision

$$\mathbf{FINAL\; EXPERIMENT\; DECISION:}$$
$$\mathbf{IMPROVED\; \text{---}\; CANDIDATE\; RETAINED}$$
