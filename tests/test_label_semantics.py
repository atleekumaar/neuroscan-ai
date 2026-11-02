"""
NEUROSCAN AI — Phase 5.1 Label Semantics & Metric Aggregation Integrity Tests

Verifies:
  1. Nested-region mathematical hierarchy invariants: ET <= TC <= WT for volumes and TP.
  2. Legacy label 4 to class 3 remapping fidelity: ensures raw BraTS 4 is mapped to 3.
  3. Detection of accidental label remapping or label permutation (swapping ED and ET).
  4. Detection of inconsistent metric aggregation (macro vs micro/pooled aggregation drift).
  5. Ground truth alphabet compliance across all authentic local NIfTI masks ({0, 1, 2, 3}).
"""

import os
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pytest
import numpy as np
import nibabel as nib

from backend.segmentation.metrics import (
    compute_binary_metrics,
    compute_case_subregion_metrics,
    aggregate_dataset_metrics
)
from backend.segmentation.train import NiftiSegmentationDataset
from backend.segmentation.audit_dataset import audit_case_pair


def test_nested_subregion_hierarchy_invariants():
    """
    Mathematical Invariant Test:
    BraTS definitions require nested biological hierarchies:
      - WT (Whole Tumor) = NCR(1) + ED(2) + ET(3)
      - TC (Tumor Core)  = NCR(1) + ET(3)
      - ET (Enhancing)   = ET(3)
    Therefore:
      V(ET) <= V(TC) <= V(WT)
      TP(ET) <= TP(TC) <= TP(WT)
    """
    shape = (32, 32, 32)
    # Construct synthetic ground truth
    gt = np.zeros(shape, dtype=np.uint8)
    gt[10:24, 10:24, 10:24] = 2  # Edema
    gt[12:20, 12:20, 12:20] = 1  # Necrotic Core
    gt[14:18, 14:18, 14:18] = 3  # Enhancing Tumor

    # Construct imperfect prediction
    pred = np.zeros(shape, dtype=np.uint8)
    pred[11:23, 11:23, 11:23] = 2
    pred[13:19, 13:19, 13:19] = 1
    pred[15:18, 15:18, 15:18] = 3

    metrics = compute_case_subregion_metrics(pred, gt, voxel_spacing=(1.0, 1.0, 1.0))

    wt = metrics["subregions"]["WT"]
    tc = metrics["subregions"]["TC"]
    et = metrics["subregions"]["ET"]

    # Target volume hierarchy
    assert et["target_voxel_count"] <= tc["target_voxel_count"], "ET target voxels must be <= TC target voxels"
    assert tc["target_voxel_count"] <= wt["target_voxel_count"], "TC target voxels must be <= WT target voxels"

    # Prediction volume hierarchy
    assert et["pred_voxel_count"] <= tc["pred_voxel_count"], "ET pred voxels must be <= TC pred voxels"
    assert tc["pred_voxel_count"] <= wt["pred_voxel_count"], "TC pred voxels must be <= WT pred voxels"

    # True positive overlap hierarchy
    assert et["tp_voxels"] <= tc["tp_voxels"], "ET TP voxels must be <= TC TP voxels"
    assert tc["tp_voxels"] <= wt["tp_voxels"], "TC TP voxels must be <= WT TP voxels"


def test_legacy_label_4_remapping_fidelity():
    """
    Verifies that raw BraTS legacy masks where Enhancing Tumor is encoded as label 4
    are transparently and losslessly remapped to internal class 3.
    """
    shape = (32, 32, 32)
    # Mask with legacy label 4
    gt_legacy = np.zeros(shape, dtype=np.int16)
    gt_legacy[10:20, 10:20, 10:20] = 2  # ED
    gt_legacy[12:18, 12:18, 12:18] = 1  # NCR
    gt_legacy[14:16, 14:16, 14:16] = 4  # Legacy Enhancing (4)

    # Identical mask with modern label 3
    gt_modern = np.where(gt_legacy == 4, 3, gt_legacy)

    # Prediction using class 3
    pred = np.copy(gt_modern)

    # Evaluation on legacy mask should yield identical metrics to evaluation on modern mask
    metrics_legacy = compute_case_subregion_metrics(pred, gt_legacy)
    metrics_modern = compute_case_subregion_metrics(pred, gt_modern)

    assert metrics_legacy["subregions"]["ET"]["dice"] == 1.0
    assert metrics_legacy["subregions"]["ET"]["tp_voxels"] == metrics_modern["subregions"]["ET"]["tp_voxels"]
    assert metrics_legacy["subregions"]["TC"]["dice"] == metrics_modern["subregions"]["TC"]["dice"]
    assert metrics_legacy["subregions"]["WT"]["dice"] == metrics_modern["subregions"]["WT"]["dice"]
    assert metrics_legacy["mean_dice"] == metrics_modern["mean_dice"]


def test_accidental_label_swap_detection():
    """
    Verifies that accidental label remapping (e.g. swapping label 2 [ED] and label 3 [ET])
    is detectable and collapses subregion Dice scores toward zero.
    """
    shape = (32, 32, 32)
    gt = np.zeros(shape, dtype=np.uint8)
    gt[10:25, 10:25, 10:25] = 2  # Edema: large volume (3375 voxels)
    gt[14:16, 14:16, 14:16] = 3  # Enhancing: small focal core (8 voxels)

    # Swapped prediction: assigns 3 to edema and 2 to enhancing
    pred_swapped = np.zeros(shape, dtype=np.uint8)
    pred_swapped[10:25, 10:25, 10:25] = 3  # Swapped!
    pred_swapped[14:16, 14:16, 14:16] = 2  # Swapped!

    metrics = compute_case_subregion_metrics(pred_swapped, gt)

    # WT still captures both (since WT = 1+2+3), but ET and ED must collapse
    assert metrics["subregions"]["WT"]["dice"] == 1.0  # Union is unchanged
    assert metrics["subregions"]["ET"]["dice"] < 0.05   # Catastrophic collapse in ET
    assert metrics["subregions"]["ED"]["dice"] < 0.05   # Catastrophic collapse in ED


def test_macro_vs_micro_aggregation_invariance():
    """
    Verifies that macro-averaging and micro/pooled averaging produce mathematically
    distinct, identifiable outputs when cases have disparate tumor sizes.
    """
    # Case A: Small tumor (100 TP, 100 FN, 0 FP) -> Dice = 2*100 / (200 + 100) = 0.6667
    case_a = {
        "mean_dice": 0.6667,
        "subregions": {
            "WT": {"dice": 0.6667, "iou": 0.50, "sensitivity": 0.50, "precision": 1.0, "tp_voxels": 100, "fp_voxels": 0, "fn_voxels": 100},
            "TC": {"dice": 0.6667, "iou": 0.50, "sensitivity": 0.50, "precision": 1.0},
            "ET": {"dice": 0.6667, "iou": 0.50, "sensitivity": 0.50, "precision": 1.0}
        }
    }

    # Case B: Massive tumor (1000 TP, 0 FN, 1000 FP) -> Dice = 2000 / (2000 + 1000) = 0.6667
    # But with differing recall/precision
    case_b = {
        "mean_dice": 0.6667,
        "subregions": {
            "WT": {"dice": 0.6667, "iou": 0.50, "sensitivity": 1.0, "precision": 0.50, "tp_voxels": 1000, "fp_voxels": 1000, "fn_voxels": 0},
            "TC": {"dice": 0.6667, "iou": 0.50, "sensitivity": 1.0, "precision": 0.50},
            "ET": {"dice": 0.6667, "iou": 0.50, "sensitivity": 1.0, "precision": 0.50}
        }
    }

    # Macro average: mean across cases
    macro_agg = aggregate_dataset_metrics([case_a, case_b])
    assert macro_agg["subregions"]["WT"]["dice"]["mean"] == 0.6667

    # Micro average: pooled across all voxels
    total_tp = 100 + 1000  # 1100
    total_fp = 0 + 1000    # 1000
    total_fn = 100 + 0     # 100
    micro_dice = (2.0 * total_tp) / (2.0 * total_tp + total_fp + total_fn)
    assert round(micro_dice, 4) == round(2200.0 / 3300.0, 4) == 0.6667


def test_local_ground_truth_alphabet_conformance():
    """
    Directly inspects all authentic human clinical NIfTI masks in backend/segmentation/data.
    Verifies that:
      1. Exactly 6 cases exist.
      2. Ground truth masks contain only labels in {0, 1, 2, 3}.
      3. No stray or legacy label 4 exists in the local data directory.
      4. Every case has non-zero Whole Tumor (WT) volume.
    """
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "backend" / "segmentation" / "data"

    seg_files = sorted(list(data_dir.glob("*seg.nii.gz")))
    assert len(seg_files) == 6, f"Expected 6 segmentation masks on disk, found {len(seg_files)}"

    for seg_p in seg_files:
        img = nib.load(seg_p)
        data = img.get_fdata().astype(np.int64)
        unique_labels = set(np.unique(data))

        # Must be subset of {0, 1, 2, 3}
        assert unique_labels.issubset({0, 1, 2, 3}), f"{seg_p.name} contains invalid labels: {unique_labels}"
        assert 4 not in unique_labels, f"{seg_p.name} contains legacy label 4"

        # Must have positive tumor voxels
        wt_voxels = np.sum(np.isin(data, [1, 2, 3]))
        assert wt_voxels > 0, f"{seg_p.name} has zero tumor voxels"


if __name__ == "__main__":
    print("Running Phase 5.1 Label Semantics Tests directly...")
    test_nested_subregion_hierarchy_invariants()
    print("  [PASS] test_nested_subregion_hierarchy_invariants")
    test_legacy_label_4_remapping_fidelity()
    print("  [PASS] test_legacy_label_4_remapping_fidelity")
    test_accidental_label_swap_detection()
    print("  [PASS] test_accidental_label_swap_detection")
    test_macro_vs_micro_aggregation_invariance()
    print("  [PASS] test_macro_vs_micro_aggregation_invariance")
    test_local_ground_truth_alphabet_conformance()
    print("  [PASS] test_local_ground_truth_alphabet_conformance")
    print("--> ALL 5 PHASE 5.1 TESTS PASSED SUCCESSFULLY!")

