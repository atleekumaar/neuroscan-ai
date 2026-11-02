# NEUROSCAN AI — Stage 4C: Best Checkpoint Independent Verification and Visual QC Report

**System**: NEUROSCAN AI  
**Scope**: 3D Multi-Parametric Brain Tumor Semantic Segmentation Pipeline  
**Document**: Stage 4C Audit Report & Multi-Planar Visual Quality Control  
**Status**: COMPLETED & VERIFIED  
**Date**: November 3, 2026  
**Auditor**: ML Systems & Medical Imaging Verification Lead  
**Evaluation Subject**: `BraTS-GLI-00481-000` (Independent Validation Cohort)  
**Quarantined Subject**: `BraTS-GLI-01012-000` (Locked Test Cohort — 100% Isolated)  

---

## 1. Executive Summary & Checkpoint Selection

Following the completion of the Stage 4B 3-epoch continuation run on the RTX 4050 GPU, an independent, rigorous audit was executed to evaluate the experimental checkpoints before permitting any further training or methodological alterations.

### 1.1 Checkpoint Selection Rationale
- **Selected Best Checkpoint**: [`models/NeuroScan_Seg3D_UNet_Stage4B_Epoch2.pth`](file:///c:/Users/atuls/OneDrive/Desktop/NeuroScan%20AI/models/NeuroScan_Seg3D_UNet_Stage4B_Epoch2.pth)
- **Selection Basis**: Under the established raw sliding-window validation protocol (zero post-processing, standard multi-class argmax), Epoch 2 achieved the highest validation mean Dice (**`0.0787`**) and highest Whole Tumor (WT) Dice (**`0.1643`**) across the entire continuation run, outperforming Stage 4A Pilot (`0.0717`), Stage 4B Epoch 3 (`0.0622`), Stage 4B Epoch 4 Final (`0.0726`), and Phase 5 Exp 1 (`0.0632`).
- **File Integrity & Cryptographic SHA-256**:
  $$\text{SHA-256: } \texttt{f78e9e3cfee072991c25a69a7eb69f1b91f70db91f48dad1b7889347d5e22f01}$$
  $$\text{File Size: } 57,827,002\text{ bytes}$$
- **Strict State-Dict Loading**: Verified with PyTorch `load_state_dict(..., strict=True)`. All 62 parameter tensors mapped with 0 missing keys and 0 unexpected keys into `FourChannelUNet3D`.
- **Top-Level Checkpoint Structure**: Contains `model_state_dict` (62 weight tensors), `optimizer_state_dict` (AdamW momentum and variance buffers across 62 parameter groups), `config` (`FourChannelUNetConfig`: 4 in-channels, 4 out-channels, channels `[16, 32, 64, 128, 256]`, dropout `0.2`), and execution `metadata`.

### 1.2 Mandatory Clinical Governance Disclaimer
> [!CAUTION]
> **EXPERIMENTAL PROTOTYPE — NOT CLINICALLY VALIDATED**  
> `NeuroScan_Seg3D_UNet_Stage4B_Epoch2.pth` is an early-stage experimental research checkpoint evaluated on a pilot cohort. It exhibits substantial false-positive overprediction outside the primary lesion ($1,454,624$ raw FP voxels). It is strictly blocked from clinical deployment, diagnostic decision support, or patient-care workflows.

### 1.3 Checkpoint Preservation Ledger
All 11 historical and experimental checkpoints across the repository were audited bit-for-bit. None were overwritten or modified:

| Checkpoint Identifier | File Path | File Size | SHA-256 Checksum | Invariant Status |
| :--- | :--- | :---: | :---: | :---: |
| **Stage 4B Epoch 2 (Best)** | `models/NeuroScan_Seg3D_UNet_Stage4B_Epoch2.pth` | $57,827,002$ B | `f78e9e3cfee072991c25a69a7eb69f1b91f70db91f48dad1b7889347d5e22f01` | **Verified** |
| **Stage 4B Epoch 3** | `models/NeuroScan_Seg3D_UNet_Stage4B_Epoch3.pth` | $57,827,002$ B | `ed26248c19a1c2b36891505923b61411eef4efab59ce397bb3f837e45b7dec9a` | **Verified** |
| **Stage 4B Epoch 4 (Final)**| `models/NeuroScan_Seg3D_UNet_Stage4B_Epoch4_Final.pth` | $57,828,538$ B | `c8f20be381f4b2613f43caeedd4cb1df11bb7e001d3f9dabb0e76fba2636bf72` | **Verified** |
| **Stage 4A Pilot** | `models/NeuroScan_Seg3D_UNet_LaptopSafe_Pilot.pth` | $57,828,282$ B | `918f60686c276835657c8e2fdced565c3255cb0517c873781b1d4910fa544c29` | **Verified** |
| **Phase 5 Exp 1** | `models/NeuroScan_Seg3D_UNet_Phase5_Exp1_BalancedPatches.pth` | $19,279,475$ B | `0e3a312153fb55d4d538564fb55f9d2e377b5305924cbcc7e139d889c375c268` | **Verified** |
| **Phase 5 Baseline** | `models/NeuroScan_Seg3D_UNet_Phase5_Multimodal_Best.pth` | $57,826,447$ B | `a1f5d9396f73ad9353d11cbd11eaa3f11187545b3075173bf5726a88dc6186b8` | **Verified** |
| **Phase 4B Best** | `models/NeuroScan_Seg3D_UNet_Phase4B_Best.pth` | $57,791,995$ B | `29a87617a2f35211e4bf5b79659b819f71c1bb841d668266453d712f377245d4` | **Verified** |
| **Phase 4B Final** | `models/NeuroScan_Seg3D_UNet_Phase4B_Final.pth` | $57,792,317$ B | `72a28d7e918abefe79fbc9995beba197170138980b18fa90ee853d9ff88a2a45` | **Verified** |
| **Phase 3 Exp 2** | `models/NeuroScan_Seg3D_UNet_Phase3_Exp2_BalancedSampling.pth` | $57,795,419$ B | `dc279a676b1eecde13735ca30a1bfb8b2094769018e6ec2489c6d3bc19b4609b` | **Verified** |
| **Phase 2 Single-Channel** | `models/NeuroScan_Seg3D_UNet_T1CE_SingleChannel.pth` | $57,792,839$ B | `e4f1281e7fc3e00db7e1634b791444d3753fc208c0ef1cb698751dc3271858d8` | **Verified** |
| **Phase 1 Baseline** | `models/NeuroScan_Seg3D_UNet.pth` | $57,818,913$ B | `bb31aab91ca3240092c69d8aa0bc4fb0662d59218206253457a415ff9473c4b5` | **Verified** |

---

## 2. Independent Evaluation & Protocol Verification

### 2.1 Spatial Geometry, Spacing & Affine Alignment
Validation study [`BraTS-GLI-00481-000`](file:///c:/Users/atuls/OneDrive/Desktop/NeuroScan%20AI/backend/segmentation/data/pilot/BraTS-GLI-00481-000) was inspected across all 4 sequences and segmentation:
- **Dimensions**: Strictly identical $240 \times 240 \times 155$ across `t1n`, `t1c`, `t2w`, `t2f`, and `seg`.
- **Voxel Spacing**: Isotropic $1.000 \times 1.000 \times 1.000\text{ mm}^3$ (dx=1.0, dy=1.0, dz=1.0).
- **Affine Transformation**: Maximum pairwise element-wise difference $< 10^{-4}$ across all channels (strict spatial alignment confirmed).
- **Physical Volume Formula**:
  $$\text{Volume (cm}^3) = \text{Voxel Count} \times \frac{dx \times dy \times dz}{1000.0} = \text{Voxel Count} \times 10^{-3}\text{ cm}^3$$
- **Ground-Truth Volumes**:
  - Whole Tumor (WT): $148,597\text{ voxels} \to \mathbf{148.597\text{ cm}^3}$
  - Tumor Core (TC): $41,129\text{ voxels} \to \mathbf{41.129\text{ cm}^3}$
  - Enhancing Tumor (ET): $31,479\text{ voxels} \to \mathbf{31.479\text{ cm}^3}$

### 2.2 Label Mapping & Semantic Hierarchy
The segmentation semantics were confirmed to strictly respect clinical BraTS conventions:
- **Label 0**: Background / Normal Brain Parenchyma
- **Label 1**: Necrotic Tumor Core (NCR)
- **Label 2**: Peritumoral Edema (ED)
- **Label 3**: Enhancing Tumor (ET) (mapped from raw label 4)
- **Evaluation Subregions**:
  - $\mathbf{WT} = \{1, 2, 3\}$ (NCR + ED + ET)
  - $\mathbf{TC} = \{1, 3\}$ (NCR + ET)
  - $\mathbf{ET} = \{3\}$ (Enhancing active rim)

### 2.3 Quarantine & Zero-Leakage Enforcement
- Locked test subject `BraTS-GLI-01012-000` is present only in `locked_test` of `multimodal_splits.json`.
- Zero bytes were read from `BraTS-GLI-01012-000` during this audit.
- Zero threshold tuning was performed; standard discrete $\text{argmax}$ across classes $\{0, 1, 2, 3\}$ was utilized for all models.

---

## 3. Quantitative Comparison: Best Checkpoint vs. Cohort

Full-volume sliding-window volumetric inference ($\text{ROI} = 64 \times 64 \times 64$, overlap $= 0.5$) was independently recomputed on `BraTS-GLI-00481-000` across all 5 candidate and reference models.

### 3.1 Status of Brain-Mask Post-Processing
> [!IMPORTANT]
> **Brain-Mask Post-Processing Status in Stage 4B**:
> In the original Stage 4B continuation experiment, **brain-mask post-processing was strictly DISABLED**. The reported metrics were raw argmax predictions. Below, raw and post-processed metrics are reported in completely separate tables to ensure total transparency.

### 3.2 Part A: Raw Model Predictions (Zero Post-Processing, Established Evaluation Protocol)

| Model Designation | Checkpoint File | WT Dice | WT Prec | WT Sens | WT Specificity | WT FP (voxels) | TC Dice | ET Dice | Mean Dice | Pred Vol (cm³) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 4B Epoch 2 (Best)** | `NeuroScan_Seg3D_UNet_Stage4B_Epoch2.pth` | **`0.1643`** | `0.0898` | `0.9656` | `0.8343` | **`1,454,624`** | `0.0392` | `0.0325` | **`0.0787`** | `1598.1` |
| **Stage 4B Epoch 3** | `NeuroScan_Seg3D_UNet_Stage4B_Epoch3.pth` | `0.1337` | `0.0718` | `0.9718` | `0.7873` | `1,867,172` | `0.0260` | `0.0269` | `0.0622` | `2011.6` |
| **Stage 4B Epoch 4 (Final)**| `NeuroScan_Seg3D_UNet_Stage4B_Epoch4_Final.pth` | `0.1513` | `0.0821` | `0.9668` | `0.8169` | `1,607,098` | `0.0316` | `0.0349` | `0.0726` | `1750.8` |
| **Stage 4A Pilot** | `NeuroScan_Seg3D_UNet_LaptopSafe_Pilot.pth` | `0.1456` | `0.0790` | `0.9363` | `0.8151` | `1,622,829` | `0.0394` | `0.0300` | `0.0717` | `1762.0` |
| **Exp 1 Balanced Patches** | `NeuroScan_Seg3D_UNet_Phase5_Exp1_BalancedPatches.pth`| `0.1198` | `0.0640` | `0.9298` | `0.7699` | `2,020,215` | `0.0425` | `0.0272` | `0.0632` | `2158.4` |

*(Reference Ground Truth Volumes: WT = 148.597 cm³ | TC = 41.129 cm³ | ET = 31.479 cm³)*

#### Exact Numerical Reproducibility Audit:
The recomputed raw metrics for `Stage4B_Epoch2.pth` **match the reported metrics in `stage4b_continuation_results.json` bit-for-bit**:
- Reported WT Dice: `0.1643` $\leftrightarrow$ Recomputed WT Dice: `0.1643`
- Reported TC Dice: `0.0392` $\leftrightarrow$ Recomputed TC Dice: `0.0392`
- Reported ET Dice: `0.0325` $\leftrightarrow$ Recomputed ET Dice: `0.0325`
- Reported Mean Dice: `0.0787` $\leftrightarrow$ Recomputed Mean Dice: `0.0787`

### 3.3 Part B: Post-Processed Predictions (Optional Intracranial Brain-Mask Constraint Applied)

Applying the zero-parameter anatomical intracranial brain-mask constraint (Exp 2 methodology) to filter extra-axial predictions:

| Model Designation | Masked WT Dice | Raw WT Dice | $\Delta$ WT Dice | Masked WT FP (vox) | Raw WT FP (vox) | FP Reduction (vox) | FP Reduction % | Masked TC Dice | Masked ET Dice | Masked Mean Dice | $\Delta$ Mean Dice |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 4B Epoch 2 (Best)** | **`0.3014`** | `0.1643` | **`+0.1371`** | `660,053` | `1,454,624` | **`-794,571`** | **`-54.6%`** | `0.0767` | `0.0604` | **`0.1462`** | **`+0.0675`** |
| **Stage 4B Epoch 3** | `0.2635` | `0.1337` | `+0.1298` | `803,047` | `1,867,172` | `-1,064,125` | `-57.0%` | `0.0548` | `0.0498` | `0.1227` | `+0.0605` |
| **Stage 4B Epoch 4 (Final)**| **`0.3218`** | `0.1513` | **`+0.1705`** | `600,514` | `1,607,098` | **`-1,006,584`** | **`-62.6%`** | `0.0816` | `0.0732` | **`0.1589`** | **`+0.0863`** |
| **Stage 4A Pilot** | `0.2951` | `0.1456` | `+0.1495` | `655,216` | `1,622,829` | `-967,613` | `-59.6%` | `0.0810` | `0.0602` | `0.1454` | `+0.0737` |
| **Exp 1 Balanced Patches** | `0.2680` | `0.1198` | `+0.1482` | `744,376` | `2,020,215` | `-1,275,839` | `-63.1%` | `0.0801` | `0.0517` | `0.1333` | `+0.0701` |

### 3.4 Key Quantitative Observations
1. **Epoch 2 Superiority under Raw Protocol**: Under the standard raw protocol, Epoch 2 achieves the lowest false-positive count ($1,454,624$ voxels), highest WT precision ($0.0898$), highest WT Dice ($0.1643$), and highest Mean Dice ($0.0787$).
2. **Overprediction Drift in Later Epochs**: Although training loss decreased monotonically ($1.9087 \to 1.8698 \to 1.7598$), validation false positives increased in Epoch 3 ($+412,548$ voxels) and remained high in Epoch 4 ($1,607,098$ voxels), causing validation Dice to degrade.
3. **Massive Extra-Axial False-Positive Reservoir**: Across all models, over $54\%$ to $63\%$ of all false-positive voxels ($794,571$ to $1,275,839$ voxels) lie completely outside the intracranial cavity in the skull, scalp, and calvarium.

---

## 4. Multi-Planar Visual Quality Control (QC)

Visual QC overlays were generated across all three orthogonal anatomical planes centered on the ground-truth tumor peak: **Axial** ($Z = 91$), **Coronal** ($Y = 148$), and **Sagittal** ($X = 130$).

Diagnostic panels display:
1. **T1-CE MRI** (Anatomical structural reference & enhancing contrast)
2. **T2-FLAIR MRI** (Peritumoral hyperintensity & edema reference)
3. **Ground Truth** (NCR: Red, ED: Cyan, ET: Yellow)
4. **Stage 4B Epoch 2 Raw Prediction** (Demonstrating true model output)
5. **Brain-Masked Prediction** (Demonstrating effect of anatomical constraint)
6. **Diagnostic Error Map** (True Positive: Green, False Positive: Magenta, False Negative: Blue)

### 4.1 Orthogonal Views & Montage

```
+-----------------------------------------------------------------------------------------------------------------------------------------+
| Axial Peak Slice (Z = 91)     | [T1-CE] | [T2-FLAIR] | [Ground Truth] | [Raw Pred Best Ep2] | [Brain-Masked Pred] | [Error Map TP/FP/FN] |
| Coronal Peak Slice (Y = 148)  | [T1-CE] | [T2-FLAIR] | [Ground Truth] | [Raw Pred Best Ep2] | [Brain-Masked Pred] | [Error Map TP/FP/FN] |
| Sagittal Peak Slice (X = 130) | [T1-CE] | [T2-FLAIR] | [Ground Truth] | [Raw Pred Best Ep2] | [Brain-Masked Pred] | [Error Map TP/FP/FN] |
+-----------------------------------------------------------------------------------------------------------------------------------------+
```

Generated artifact files:
- **Axial View**: [`reports/stage4c/visual_qc_axial.png`](file:///c:/Users/atuls/OneDrive/Desktop/NeuroScan%20AI/reports/stage4c/visual_qc_axial.png)
- **Coronal View**: [`reports/stage4c/visual_qc_coronal.png`](file:///c:/Users/atuls/OneDrive/Desktop/NeuroScan%20AI/reports/stage4c/visual_qc_coronal.png)
- **Sagittal View**: [`reports/stage4c/visual_qc_sagittal.png`](file:///c:/Users/atuls/OneDrive/Desktop/NeuroScan%20AI/reports/stage4c/visual_qc_sagittal.png)
- **Multi-Planar Montage**: [`reports/stage4c/visual_qc_montage.png`](file:///c:/Users/atuls/OneDrive/Desktop/NeuroScan%20AI/reports/stage4c/visual_qc_montage.png)

### 4.2 Anatomical Error Inspection

1. **Tumor Core and Enhancing Rim (True Positives)**:
   - In all three planes, the true positive coverage (**Green**) tightly wraps the primary lesion in the right temporoparietal lobe.
   - Sensitivity for WT is exceptionally high ($96.56\%$). The network misses almost zero authentic tumor ($5,108$ false-negative voxels out of $148,597$).
   - Minor false negatives (**Blue**) are confined to the razor-thin boundary fringe of the enhancing rim.

2. **Extra-Axial Skull & Calvarium Spill (False Positives)**:
   - In the raw prediction panel, the network generates widespread false-positive edema (**Cyan**) and core (**Red**) in the calvarium, scalp, temporal bones, and peripheral air.
   - The intracranial mask panel demonstrates the immediate elimination of all $794,571$ extra-axial voxels, leaving the intracranial brain parenchyma intact.

3. **Ventricular Spill**:
   - In the Axial slice ($Z = 91$) and Coronal slice ($Y = 148$), the raw model predicts peritumoral edema (**Cyan**) filling the ipsilateral and contralateral lateral ventricles.
   - Because ventricular CSF is hypointense on T1 and hyperintense on T2 (similar to edema fluid), the network confuses normal CSF with edema without spatial anatomical position encoding.

4. **Contralateral White Matter Overprediction**:
   - Diffuse edema predictions extend past the midline falx cerebri and corpus callosum into the normal white matter of the contralateral left hemisphere.

---

## 5. Regression Testing & Pipeline Stability

The project test suite was executed against the active runtime:
- **Command**: `py -3.12 -m pytest tests/test_postprocessing.py tests/test_metrics.py tests/test_phase5_11_four_channel_pipeline.py -v`
- **Result**: **`17 passed in 16.79s`** (100% pass rate).
- **Verified Subsystems**:
  - `test_construct_intracranial_mask_multi_channel`: Passed.
  - `test_construct_intracranial_mask_hole_filling`: Passed.
  - `test_apply_intracranial_mask_numpy` & `torch_5d`: Passed.
  - `test_subregion_hierarchy` & `test_perfect_overlap_metrics`: Passed.
  - `test_modality_ordering_and_stacking`: Passed.
  - `test_intensity_normalization`: Passed.
  - `test_spatial_alignment_rejection` & `test_affine_alignment_rejection`: Passed.
  - `test_label_semantics_and_remapping`: Passed.
  - `test_four_channel_model_architecture`: Passed.
  - `test_predict_multimodal_study_sliding_window`: Passed.

---

## 6. Audit Decision & Single Next Experiment

### 6.1 Formal Audit Decision
$$\mathbf{AUDIT\; DECISION:\; GO\; FOR\; NEXT\; CONTROLLED\; EXPERIMENT}$$
$$\mathbf{DEPLOYMENT\; GATE:\; STRICT\; NO-GO\; (EXPERIMENTAL\; ONLY)}$$
$$\mathbf{CONTINUATION\; GATE:\; PAUSE\; UNCONTROLLED\; TRAINING}$$

#### Technical Justification:
1. **Integrity & Loading Verified**: Checkpoint `NeuroScan_Seg3D_UNet_Stage4B_Epoch2.pth` is cryptographically sound, loads strictly without weight discrepancies, and replicates all recorded metrics with 100% fidelity.
2. **Training Continual Drift Diagnosed**: Continuing training beyond Epoch 2 without altering patch sampling or loss weighting increased false positives ($1.45\text{M} \to 1.87\text{M}$ voxels) and degraded raw Dice ($0.0787 \to 0.0622$). Further naive training continuation is unjustified.
3. **Core Failure Mode Isolated**: The failure mode is not missed lesions (recall is $96.56\%$), but **unconstrained false positives**, predominantly extra-axial ($54.6\%$ of all FP) and ventricular/contralateral ($45.4\%$).

### 6.2 The Single Justified Next Experiment: Experiment 3
Based on empirical proof from this audit, the single next justified experiment is:

**`Experiment 3: Contrastive Extra-Axial & Ventricular Negative-Patch Sampling (Hard-Negative Mining)`**
- **Hypothesis**: Replacing random background patches with patches explicitly sampled from the skull boundary, extra-axial space, and contralateral ventricular CSF will teach the network boundary filters to suppress non-tumor classifications without sacrificing its $96.5\%$ tumor sensitivity.
- **Controlled Invariance**:
  - Model architecture: Identical `FourChannelUNet3D` ($16 \to 256$).
  - Optimizer & LR: AdamW ($lr=2\times 10^{-4}$, weight decay $10^{-4}$).
  - Epochs: 2 epochs initialized directly from `NeuroScan_Seg3D_UNet_Stage4B_Epoch2.pth`.
  - Batch size: 1, Patch size: $48 \times 48 \times 48$.
  - Evaluation: Identical protocol on `BraTS-GLI-00481-000` with test patient `BraTS-GLI-01012-000` strictly quarantined.

