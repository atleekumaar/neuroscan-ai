# NEUROSCAN AI — Strategic Roadmap & Next Engineering Steps

**Date**: 2026-10-20  
**Context**: Following Phase 5 Completion, Phase 5.18 Quality Audit, and Exp 1 Controlled Improvement  
**Active Baseline**: `models/NeuroScan_Seg3D_UNet_Phase5_Multimodal_Best.pth` (Baseline)  
**Retained Improved Candidate**: `models/NeuroScan_Seg3D_UNet_Phase5_Exp1_BalancedPatches.pth` (+10% WT Dice, +32% TC Dice, -345k FP voxels)

---

## 1. Key Lessons from the Controlled Improvement Experiment

The single-variable intervention (rebalancing training patch sampling from a 2:1 foreground bias to a 1:2 background representation) proved that **false positives in normal brain tissue are directly responsive to the frequency of non-tumorous negative context during training**:
- Supplying 4 background patches per volume eliminated over **$345,000$ false-positive voxels**, raised Whole Tumor precision by $+10.9\%$, and boosted Tumor Core Dice by $+32.0\%$, while maintaining an impressive $93.0\%$ Whole Tumor recall.
- This confirms that the model architecture and 4-channel input representation are structurally sound.

---

## 2. Immediate Next Experiments (Short-Term / Single-Variable)

Before full cohort expansion, the following targeted interventions are recommended for evaluation on the pilot cohort:

### Step 2.1: Intracranial Brain-Mask Inference Constraint
- **Hypothesis**: A significant fraction of peripheral false positives occur near the skull or extra-axial tissue.
- **Intervention**: Apply an automated non-zero intracranial brain mask (union of non-zero voxels across all 4 sequences) as a hard post-processing gate, setting all predictions outside the brain parenchyma to class 0.
- **Anticipated Impact**: Instant elimination of extracranial false positives without modifying model weights.

### Step 2.2: Compound Loss Tuning (Focal / Weighted Cross-Entropy)
- **Hypothesis**: The standard `DiceCELoss` weights Dice ($1.0$) and Cross-Entropy ($1.0$) equally. Dice loss emphasizes foreground overlap, while Cross-Entropy penalizes pixel-wise errors across the entire grid.
- **Intervention**: Test shifting the compound loss weighting to $w_{\text{ce}} = 1.5, w_{\text{dice}} = 1.0$, or introducing `FocalLoss` ($\gamma = 2.0$) to penalize easy background false positives during gradient updates.

### Step 2.3: Patch Resolution & Multi-Scale Slicing
- **Intervention**: When memory permits, evaluate increasing patch size from $(48, 48, 48)$ to $(64, 64, 64)$ during training, providing larger anatomical context around tumor boundaries.

---

## 3. Phase 6: Full Cohort Scaling & GPU Migration (Medium-Term)

To transition NEUROSCAN AI from an engineering-validated development pipeline to a clinically benchmarkable segmentation engine:

1. **Authorized Synapse Cohort Expansion**:
   - Utilize the verified Synapse acquisition pipeline (`scripts/acquire_brats_pilot.py`) to acquire the full 100+ study cohort from challenge `syn51156910`.
   - Maintain the 7-point gatekeeper to ensure 100% data integrity, finite intensities, and label conformance.
2. **GPU Cloud Migration**:
   - Migrate training from the local 8-thread CPU environment to a dedicated CUDA GPU instance (NVIDIA A100 / RTX 4090 with 24+ GB VRAM).
   - Enable FP16/BF16 mixed-precision training.
3. **Deep Training Regime (100 Epochs)**:
   - Expand training from 5 exploratory epochs to 100 epochs.
   - Introduce spatial data augmentations: random 3D rotations, flipping, scaling ($0.9 - 1.1$), and elastic deformations.
4. **Statistical Multi-Fold Validation**:
   - Implement 5-fold cross-validation across the expanded training cohort.
   - Benchmark on the quarantined locked test set (`BraTS-GLI-01012-000`) with $95\%$ confidence intervals.

---

## 4. Governance & Architectural Safeguards to Maintain

1. **Quarantine Integrity**: Subject `BraTS-GLI-01012-000` must remain strictly locked and quarantined until final model freeze.
2. **Historical Invariance**: Preserve all historical checkpoints (`Phase4B_Best`, `Phase4B_Final`, `Phase3_Exp2`, `Production Seg3D`, and `T1CE_SingleChannel`) without alteration.
3. **Subsystem Isolation**: Retain decoupling of the 2D tumor classifier (`brain_tumor_detector.pth`), Grad-CAM visualization, and historical single-channel API endpoints.
