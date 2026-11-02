# NEUROSCAN AI — Data Provenance & Verification Specification

This document provides a comprehensive audit trail of all medical imaging data, ground-truth annotations, and model checkpoints in **NEUROSCAN AI**.

---

## 1. 3D Volumetric MRI Dataset (ASNR-MICCAI BraTS 2023)

### Official Distribution & Attribution
* **Dataset Name:** Brain Tumor Segmentation Challenge 2023 (BraTS 2023) Adult Glioma
* **Consortium:** Radiological Society of North America (RSNA), American Society of Neuroradiology (ASNR), Medical Image Computing and Computer Assisted Intervention (MICCAI)
* **Authoritative Distribution:** Hugging Face ([MedOtter/brats2023-gli-dataset](https://huggingface.co/datasets/MedOtter/brats2023-gli-dataset))
* **Applicable License:** Creative Commons Attribution-ShareAlike 4.0 International ([CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/))
* **Attribution Citation:**
  > Baid, U., et al. "The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification." *Frontiers in Neuroscience*, 15:704118 (2021).  
  > Menze, B. H., et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)." *IEEE Transactions on Medical Imaging*, 34(10), 1993–2024 (2015).

---

## 2. Independent Real Patient Studies

All studies are genuine human clinical brain MRI volumes in NIfTI-1 format (`.nii.gz`). Every case consists of co-registered axial T1-CE volumetric MRI and expert manual ground-truth segmentation masks.

| Case Key | Patient Study ID | Dimensions | Voxel Spacing | MRI SHA-256 (T1-CE) | Mask SHA-256 (seg) | WT Vol ($\text{cm}^3$) | TC Vol ($\text{cm}^3$) | ET Vol ($\text{cm}^3$) | Initial Focus |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`case_001`** | `BraTS-GLI-00000-000` | $240 \times 240 \times 155$ | $1.0 \times 1.0 \times 1.0\text{ mm}$ | `735c27fd...` | `5f74bef5...` | 57.31 | 44.47 | 32.73 | Z = 71 |
| **`case_002`** | `BraTS-GLI-00002-000` | $240 \times 240 \times 155$ | $1.0 \times 1.0 \times 1.0\text{ mm}$ | `4f49aa26...` | `8de53bb0...` | 190.59 | 34.90 | 23.65 | Z = 79 |
| **`case_003`** | `BraTS-GLI-00003-000` | $240 \times 240 \times 155$ | $1.0 \times 1.0 \times 1.0\text{ mm}$ | `a211cd8c...` | `de321e64...` | 99.24 | 41.47 | 24.25 | Z = 108 |
| **`case_004`** | `BraTS-GLI-00005-000` | $240 \times 240 \times 155$ | $1.0 \times 1.0 \times 1.0\text{ mm}$ | `2e074611...` | `ac15bcaf...` | 124.78 | 48.04 | 23.85 | Z = 104 |
| **`case_005`** | `BraTS-GLI-00006-000` | $240 \times 240 \times 155$ | $1.0 \times 1.0 \times 1.0\text{ mm}$ | `20847fc1...` | `c55d5122...` | 137.91 | 51.95 | 26.45 | Z = 73 |
| **`case_006`** | `BraTS-GLI-00008-000` | $240 \times 240 \times 155$ | $1.0 \times 1.0 \times 1.0\text{ mm}$ | `3b92f907...` | `4c625564...` | 18.46 | 0.33 | 0.31 | Z = 69 |

Full cryptographic hashes, intensity distributions, and geometry statistics are recorded in the machine-readable manifest [`backend/segmentation/data_manifest.json`](backend/segmentation/data_manifest.json). Deterministic patient-level evaluation partitions are governed by [`backend/segmentation/splits.json`](backend/segmentation/splits.json).

### Elimination of Synthetic Control
The previously mirrored synthetic Case 3 control (`BraTS_Case_003_Control_T1CE.nii.gz` and `_seg.nii.gz`) has been completely removed from the demo. It is not counted as an independent real patient case. The repository now contains strictly **6 genuine, independently acquired real patient MRI studies**.

---

## 3. Spatial Affine & Mask Alignment Validation

Every downloaded study was subjected to spatial geometry and voxel alignment checks:
1. **Dimensions:** Verified exact dimension identity between MRI volume and segmentation mask ($240 \times 240 \times 155$).
2. **Affine Co-registration:** Verified $4 \times 4$ affine transform matrix equality (`np.allclose(mri.affine, gt.affine)`), confirming that both volumes share the identical physical origin, coordinate orientation, and voxel scale.
3. **Multi-compartment Label Consistency:** Verified that all segmentation voxels strictly adhere to the standard BraTS label alphabet:
   - `0`: Background / Normal parenchyma
   - `1`: Necrotic and Non-Enhancing Core (NCR/NET)
   - `2`: Peritumoral Vasogenic Edema (ED)
   - `3`: Active Contrast-Enhancing Neoplastic Margin (ET)
   - `4`: Alternative ET label (consistently and explicitly remapped to `3` if encountered)
4. **Volume Integration:** Verified that physical volumes ($\text{cm}^3$) computed via $\sum \text{voxels} \times (dx \times dy \times dz) / 1000$ strictly match the recorded clinical metadata within 0.1%.

---

## 4. Model Checkpoint Provenance vs. Expert Ground Truth

To maintain research integrity and prevent medical AI misrepresentation:
* **Expert Ground Truth (Gold Standard):** Ground truth masks are manual delineations produced by expert board-certified neuro-radiologists from the ASNR-MICCAI BraTS consortium. They are labeled as **Ground Truth (Expert)** across all 2D slices, volumetric tables, and 3D surface meshes.
* **Model Prediction Checkpoint (`NeuroScan_Seg3D_UNet.pth`):**
  - Architecture: 3D U-Net (BraTS Benchmark, native PyTorch/MONAI)
  - SHA-256: `bb31aab91ca32400880a654a963582b69cfbb646013a641b61cf24599473c4b5`
  - Provenance: Initial baseline training (4 epochs, patch size $64 \times 64 \times 64$, initial patch validation Dice 0.2758)
  - Status: Marked as **Prediction (Experimental)** in the workstation interface.
  - Ethical Guardrail: Ground truth is never presented as a model prediction. Predictions and expert ground truth are selectable as distinct operational modes in the user interface.

---

## 5. 2D Classification Dataset Provenance (Preserved)

The five demo images in `frontend/assets/samples/`:
* `glioma.jpg` (Axial T1-CE showing diffuse glioma)
* `glioma2.jpg` (Axial T1-CE showing frontal astrocytoma)
* `meningioma.jpg` (Axial T1-CE showing dural tail meningioma)
* `notumor.jpg` (Axial T1 normal control brain scan)
* `pituitary.jpg` (Coronal/Axial sella turcica adenoma)

These images originate from the established Figshare / Kaggle Brain Tumor Classification dataset (Sartaj et al., 2020) and are strictly preserved for 2D classification and Grad-CAM interpretability testing.
