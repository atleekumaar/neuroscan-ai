# NEUROSCAN AI — Phase 1 Training Pipeline & Validation Audit

**Document:** `docs/PHASE1_TRAINING_AUDIT.md`  
**Auditor:** Senior Medical Imaging ML Engineer / PyTorch & MONAI Specialist  
**Audited File:** `backend/segmentation/train.py`  
**Reference Commit:** Baseline Git Commit `2567ba2`  
**Audit Date:** October 2026  
**Status:** Complete — Forensic Codebase & Integrity Review  

---

## Executive Summary

A comprehensive, non-destructive audit of `backend/segmentation/train.py` was conducted to evaluate clinical validity, spatial data integrity, evaluation methodology, and patient leakage protections. While the training script implements modern constructs (PyTorch 2.x AMP, MONAI/native fallback loss, and patient-level splitting heuristics), multiple critical defects exist that decouple the training validation scores from true clinical full-volume performance. 

This document classifies every finding by priority (**Critical, High, Medium, Low**), details root causes, demonstrates failure modes with code evidence, and provides concrete architectural recommendations for Phase 2.

---

## Summary of Audit Findings

| ID | Category | Issue Description | Priority |
| :--- | :--- | :--- | :---: |
| **AUD-01** | **Evaluation Validity** | Center-Crop Patch Validation Masking Full-Volume Performance | **CRITICAL** |
| **AUD-02** | **Data Leakage Risk** | Path Discovery Duplication on Case-Insensitive Filesystems | **HIGH** |
| **AUD-03** | **Modality Semantics** | Artificial Channel Duplication (T1-CE × 4) Violating BraTS mpMRI Standard | **HIGH** |
| **AUD-04** | **Metrics & Checkpoints** | Checkpoint Selection Based on Unrepresentative Patch Dice | **HIGH** |
| **AUD-05** | **Reproducibility** | Non-Deterministic Foreground Patch Sampling in DataLoader | **MEDIUM** |
| **AUD-06** | **Spatial Geometry** | Voxel Spacing and Physical Spacing Discarded During Training Validation | **MEDIUM** |
| **AUD-07** | **Label Ingestion** | Unchecked Silent Pass-Through of Unexpected Segmentation Labels | **MEDIUM** |
| **AUD-08** | **Hardware / AMP** | CPU Autocast Behavior & Device Memory Safety Handling | **LOW** |

---

## Detailed Forensic Investigations

### AUD-01: Center-Crop Patch Validation Masking Full-Volume Performance [CRITICAL]

* **Location:** `backend/segmentation/train.py:108-115`, `backend/segmentation/train.py:247-258`
* **Observation:**
  During validation (`is_train=False`), the dataset yields only a single central $64 \times 64 \times 64$ patch extracted via integer division:
  ```python
  sh = max(0, (h - ph) // 2)
  sw = max(0, (w - pw) // 2)
  sd = max(0, (d - pd) // 2)
  mri_patch = mri_data[:, sh:sh+ph, sw:sw+pw, sd:sd+pd]
  seg_patch = seg_data[sh:sh+ph, sw:sw+pw, sd:sd+pd]
  ```
  The epoch validation loop passes this single patch through the model and computes `compute_case_subregion_metrics(v_pred, v_mask)`.
* **Clinical & Technical Defect:**
  1. The full brain volume is $240 \times 240 \times 155$ voxels ($8.928 \times 10^6$ voxels). A $64 \times 64 \times 64$ patch represents only $262,144$ voxels—merely **2.93% of the total volume**.
  2. Glioblastomas and astrocytomas frequently present in lateral, cortical, or anterior frontal/temporal poles (e.g., Case 001 focus at slice 71, Case 006 at slice 69). A central crop completely misses peripheral lesions or clips enhancing rims.
  3. Crucially, patch validation cannot measure **false positive rates in normal brain parenchyma or skull-stripped boundary regions**. A model can produce severe artifactual predictions across the periphery while scoring high on a cropped patch containing only tumor center.
* **Remediation Plan (Phase 2):**
  Replace single center-crop validation with sliding-window inference (`run_sliding_window_inference`) across the complete reconstructed volume, or evaluate full volumes at validation epochs using the validated inference engine.

---

### AUD-02: Path Discovery Duplication on Case-Insensitive Filesystems [HIGH]

* **Location:** `backend/segmentation/train.py:141`
* **Observation:**
  ```python
  files = list(data_dir.glob("*_T1CE.nii*")) + list(data_dir.glob("*_t1ce.nii*"))
  ```
* **Clinical & Technical Defect:**
  On Windows systems (the host OS), the filesystem is case-insensitive. Both glob patterns match the same files, doubling the detected case list from 6 to 12. When partitioned by arbitrary shuffled indices:
  ```python
  indices = np.arange(len(pairs))
  rng.shuffle(indices)
  ```
  Identical cases are assigned simultaneously to both `train_pairs` and `val_pairs`, causing **severe cross-split patient data leakage**.
* **Status:** Resolved in `backend/segmentation/audit_dataset.py` by path-resolution deduplication (`{p.resolve(): p for p in ...}`). Must be ported into `train.py` before any retraining run.

---

### AUD-03: Artificial Channel Duplication Violating BraTS mpMRI Standard [HIGH]

* **Location:** `backend/segmentation/train.py:78-87`
* **Observation:**
  ```python
  if len(channels) == 1:
      mri_data = np.repeat(channels[0][np.newaxis, ...], 4, axis=0)
  ```
* **Clinical & Technical Defect:**
  1. Standard BraTS multi-parametric MRI benchmarks require 4 distinct, complementary pulse sequences:
     - **T1-native:** Anatomical boundary baseline.
     - **T1-CE (Post-contrast):** Delineates blood-brain barrier breakdown (Enhancing Tumor ET).
     - **T2-weighted:** Captures hyperintense fluid and edema.
     - **T2-FLAIR:** Suppresses free CSF to isolate infiltrative peritumoral vasogenic edema (ED).
  2. The current dataset on disk contains only T1-CE acquisitions (`BraTS_Case_00X_T1CE.nii.gz`). Replicating T1-CE across all 4 channels results in a rank-1 channel tensor.
  3. The model cannot learn the cross-spectral feature representations required to distinguish non-enhancing tumor core (NCR) from vasogenic edema (ED).
* **Remediation Plan (Phase 2):**
  Acquire the full 4-sequence bundles (T1, T1-CE, T2, FLAIR) for each patient study from BraTS 2023, or explicitly retrain a native 1-channel 3D U-Net specifically optimized for single-sequence T1-CE contrast.

---

### AUD-04: Checkpoint Selection Based on Unrepresentative Patch Dice [HIGH]

* **Location:** `backend/segmentation/train.py:270-288`
* **Observation:**
  ```python
  if val_mean_dice > best_val_dice or epoch == epochs:
      best_val_dice = val_mean_dice
      torch.save(...)
  ```
* **Clinical & Technical Defect:**
  Because `val_mean_dice` is calculated solely from the central $64^3$ patch (see AUD-01), the model checkpoint saved to disk (`models/NeuroScan_Seg3D_UNet.pth`) was selected based on patch overlap rather than full-volume whole-tumor accuracy. This explains why the baseline checkpoint achieved `final_val_dice: 0.2758` during training but exhibits distinct behavior on full-volume inference.
* **Remediation Plan (Phase 2):**
  Compute model selection checkpoints using full-volume macro-averaged Dice across held-out patient cases.

---

### AUD-05: Non-Deterministic Foreground Patch Sampling in DataLoader [MEDIUM]

* **Location:** `backend/segmentation/train.py:97-107`
* **Observation:**
  ```python
  tumor_indices = np.argwhere(seg_data > 0)
  if len(tumor_indices) > 0 and np.random.rand() > 0.4:
      center = tumor_indices[np.random.choice(len(tumor_indices))]
      ...
  else:
      sh = np.random.randint(0, h - ph + 1)
  ```
* **Clinical & Technical Defect:**
  `np.random` calls the global NumPy RNG without worker-specific seeding or epoch-dependent generator state. In multi-process data loading (`num_workers > 0`), workers produce identical pseudo-random sequences or unrepeatable sampling trajectories.
* **Remediation Plan (Phase 2):**
  Implement PyTorch `torch.Generator` and explicit `worker_init_fn` for deterministic patch sampling.

---

### AUD-06: Voxel Spacing Discarded During Validation Metrics [MEDIUM]

* **Location:** `backend/segmentation/train.py:256`
* **Observation:**
  ```python
  case_m = compute_case_subregion_metrics(v_pred, v_mask)
  ```
* **Clinical & Technical Defect:**
  `compute_case_subregion_metrics` accepts `voxel_spacing: Optional[Tuple_like]`. In `train.py`, this parameter is omitted, causing physical tumor volume calculations ($\text{cm}^3$) to assume unit isotropic $1.0\text{ mm}$ spacing regardless of actual volume header zooms.

---

### AUD-07: Unchecked Silent Pass-Through of Unexpected Labels [MEDIUM]

* **Location:** `backend/segmentation/train.py:90`
* **Observation:**
  ```python
  seg_data = np.where(seg_data == 4, 3, seg_data).astype(np.int64)
  ```
* **Clinical & Technical Defect:**
  If an unexpected label (e.g., label 5 or negative value) were present in corrupted data, it would not be caught. In `NativeDiceCELoss`, passing a class index $\ge 4$ into `nn.functional.one_hot(..., num_classes=4)` triggers an unhandled CUDA index error.
* **Remediation Plan (Phase 2):**
  Add strict label validation: assert `set(np.unique(seg_data)).issubset({0, 1, 2, 3})`.

---

### AUD-08: Mixed-Precision (AMP) & Device Selection Safety [LOW]

* **Location:** `backend/segmentation/train.py:209, 231, 253`
* **Observation:**
  ```python
  with torch.amp.autocast(device_type=device.type if device.type in ["cuda", "cpu"] else "cpu", enabled=(device.type == "cuda")):
  ```
* **Finding:**
  Device selection and AMP scaling are implemented safely with `enabled=(device.type == "cuda")`. Memory clearing (`torch.cuda.empty_cache()`) is recommended between epochs to prevent fragmentation on 6 GB VRAM GPUs.

---

## Actionable Recommendations for Phase 2

1. **Implement Sliding-Window Validation Loop:** Replace central patch sampling in `train.py` with sliding window inference every $N$ epochs.
2. **Transition to Native 1-Channel or Acquire Complete mpMRI:** Either explicitly train a single-channel T1-CE architecture or ingest the official co-registered T1/T2/FLAIR modalities from the BraTS repository.
3. **Deterministic DataLoader Workers:** Integrate worker seed initializers with `torch.utils.data.DataLoader(..., worker_init_fn=seed_worker)`.
4. **Synchronize Manifest with Splits:** Enforce that `train.py` ingests `backend/segmentation/splits.json` directly instead of executing internal on-the-fly partitioning.
