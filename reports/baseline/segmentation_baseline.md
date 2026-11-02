# NEUROSCAN AI — 3D Volumetric Segmentation Baseline Benchmark Report

**Generated (UTC):** 2026-10-20T08:53:50.965237+00:00  
**Git Commit SHA:** `2567ba22c73235eb581d9844e8599b7a2759ee88`  
**Evaluation Mode:** Full-Volume Sliding-Window Inference (32×32×32, overlap=0.25)  
**Execution Device:** `NVIDIA GeForce RTX 4050 Laptop GPU` (cuda)  

---

## 1. Provenance & Benchmark Artifacts

* **Model Checkpoint:** `test_4ch.pth` (`C:/Users/atuls/AppData/Local/Temp/pytest-of-atuls/pytest-158/test_dynamic_evaluation_metada1/test_4ch.pth`)
* **Checkpoint SHA-256:** `f874fa975b8f7478e8fa14d81d7e9081255a52750d5e2ca01b7b57f84942d58f`
* **Model Architecture:** 3D U-Net (BraTS Benchmark) (in=4, out=4)
* **Input Channels & Semantics:** 4 channel(s) — 4-channel mpMRI (T1, T1-CE, T2, FLAIR) or 4-channel slot alignment
* **Inference Threshold (`min_fg_prob`):** `Default (Argmax / No Threshold)`
* **Intracranial Brain Masking:** `Enabled (Foreground non-zero suppression)`
* **Training Epochs:** 4 epochs (Best Epoch: N/A)
* **Dataset Manifest SHA-256:** `06849dd8abf85fd312022611df70b38099b31b09fd51b589408bc6a8740beec6`
* **Split Partition Evaluated:** `val` (Cohort: 1 cases evaluated)
* **Splits File:** `C:/Users/atuls/AppData/Local/Temp/pytest-of-atuls/pytest-158/test_dynamic_evaluation_metada1/splits.json` (SHA-256: `795224d4465145f2f7f99c4ade84e25d68e72164811a870a22bd3e990993aec5`)
* **PyTorch / MONAI / NiBabel:** `2.6.0+cu124` / `1.4.0` / `5.4.2`

---

## 2. Aggregate Benchmark Performance

**Overall Mean Dice:** **`0.0000`** | **Overall Median Dice:** **`0.0000`** | **Mean Latency:** `0.02s` / case

| Subregion Hierarchy | Dice Mean (±SD) | Dice Median | Dice [Min, Max] | IoU (Jaccard) | Sensitivity (Recall) | Precision (PPV) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Whole Tumor (WT)** | 0.0000 (±0.0000) | 0.0000 | [0.0000, 0.0000] | 0.0000 | 1.0000 | 0.0000 |
| **Tumor Core (TC)** | 0.0000 (±0.0000) | 0.0000 | [0.0000, 0.0000] | 0.0000 | 1.0000 | 0.0000 |
| **Enhancing Tumor (ET)** | 0.0000 (±0.0000) | 0.0000 | [0.0000, 0.0000] | 0.0000 | 1.0000 | 0.0000 |

---

## 3. Per-Patient Quantitative Breakdown

| Case Key | File ID | Dimensions | Latency | WT Dice | TC Dice | ET Dice | Mean Dice | Pred WT Vol | Ref WT Vol | Vol Diff % |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`case_001`** | `BraTS_case_001_T1CE.nii.gz` | 48×48×48 | 0.0s | 0.0000 | 0.0000 | 0.0000 | **0.0000** | 89.16 cm³ | 0.00 cm³ | 100.0% |

---

## 4. Methodological Findings & Honest Assessment

1. **Full-Volume Reconstructed Space vs. Center-Patch Illusion:**
   - During initial training, validation dice was evaluated solely on isolated $64\times64\times64$ central crops (`val_mean_dice: 0.2758`).
   - Full-volume sliding window evaluation tests the real clinical deployment setting across the entire volume ($240\times240\times155$), capturing peripheral false positives and brain margin transitions.
2. **Single-Modality Input Bottleneck:**
   - The current baseline checkpoint receives axial T1-CE duplicated 4 times across input channels to satisfy the 4-channel U-Net contract.
   - In authentic BraTS benchmarks, T2, FLAIR, and native T1 provide orthogonal contrast required to delineate vasogenic edema from core necrosis.
3. **Statistical Sample Size Safeguard:**
   - This baseline evaluation is conducted on the curated 6-case BraTS cohort.
   - These numbers serve as an engineering baseline for regression tracking and pipeline verification, not as proof of general clinical efficacy.
