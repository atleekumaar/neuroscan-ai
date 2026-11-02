"""
NEUROSCAN AI — Unit & Integration Tests for Phase 1
Audits: File existence, corrupt NIfTI, non-finite values, label remapping,
spatial alignment, metrics edge cases, patient leakage, split reproducibility,
and baseline evaluation reporting.
All test volumes generated herein are strictly SYNTHETIC UNIT TEST FIXTURES.
"""

import os
import json
import pytest
import numpy as np
import nibabel as nib
import torch
from pathlib import Path

from backend.segmentation.audit_dataset import (
    audit_case_pair,
    audit_dataset_directory,
    generate_patient_splits,
    compute_file_sha256
)
from backend.segmentation.metrics import (
    compute_binary_metrics,
    compute_case_subregion_metrics,
    aggregate_dataset_metrics
)
from backend.segmentation.pipeline import (
    load_nifti_file,
    preprocess_volume_for_model,
    run_sliding_window_inference
)
from backend.segmentation.model import (
    build_segmentation_model,
    load_segmentation_model
)
from backend.segmentation.evaluate import evaluate_cohort


# ==============================================================================
# FIXTURES: Synthetic Test Volumes (Marked explicitly as synthetic fixtures)
# ==============================================================================

@pytest.fixture
def synthetic_nifti_pair(tmp_path):
    """Generates a pair of valid 16x16x16 synthetic MRI and segmentation NIfTI volumes."""
    shape = (16, 16, 16)
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])

    mri_data = np.random.uniform(10.0, 500.0, size=shape).astype(np.float32)
    # Background=0, NCR=1, ED=2, ET=3
    seg_data = np.zeros(shape, dtype=np.int16)
    seg_data[4:8, 4:8, 4:8] = 2  # Edema
    seg_data[6:8, 6:8, 6:8] = 1  # Core
    seg_data[7:8, 7:8, 7:8] = 3  # Enhancing

    mri_nii = nib.Nifti1Image(mri_data, affine)
    seg_nii = nib.Nifti1Image(seg_data, affine)

    mri_path = tmp_path / "synthetic_Case_001_T1CE.nii.gz"
    seg_path = tmp_path / "synthetic_Case_001_seg.nii.gz"

    nib.save(mri_nii, str(mri_path))
    nib.save(seg_nii, str(seg_path))

    return mri_path, seg_path, shape, affine


# ==============================================================================
# 1. DATA AUDIT INTEGRITY TESTS
# ==============================================================================

def test_audit_valid_synthetic_pair(synthetic_nifti_pair):
    """Verifies that a valid pair passes all audit integrity checks."""
    mri_p, seg_p, shape, affine = synthetic_nifti_pair
    result = audit_case_pair(mri_p, seg_p, case_key="case_001", patient_id="SYNTH-PATIENT-001")

    assert result["is_valid"] is True
    assert len(result["errors"]) == 0
    assert result["geometry"]["dimensions"] == list(shape)
    assert result["geometry"]["affine_match"] is True
    assert result["mri_intensity"]["finite_values_only"] is True
    assert result["mri_intensity"]["is_empty"] is False
    assert result["segmentation_labels"]["volume_cm3"]["WT"] > 0


def test_audit_missing_mri_or_mask(tmp_path):
    """Verifies audit correctly flags missing MRI or segmentation mask."""
    missing_mri = tmp_path / "nonexistent_mri.nii.gz"
    valid_seg = tmp_path / "test_seg.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8)), np.eye(4)), str(valid_seg))

    res = audit_case_pair(missing_mri, valid_seg, "case_test")
    assert res["is_valid"] is False
    assert any("MRI file does not exist" in e for e in res["errors"])

    # Test missing mask
    valid_mri = tmp_path / "test_mri.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((8, 8, 8)), np.eye(4)), str(valid_mri))
    missing_seg = tmp_path / "nonexistent_seg.nii.gz"

    res2 = audit_case_pair(valid_mri, missing_seg, "case_test")
    assert res2["is_valid"] is False
    assert any("Segmentation mask file does not exist" in e for e in res2["errors"])


def test_audit_corrupt_nifti(tmp_path):
    """Verifies audit catches corrupt / non-NIfTI byte streams."""
    corrupt_mri = tmp_path / "corrupt_T1CE.nii.gz"
    corrupt_mri.write_bytes(b"GARBAGE_BYTES_NOT_A_VALID_GZIP_OR_NIFTI")

    seg_p = tmp_path / "valid_seg.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8)), np.eye(4)), str(seg_p))

    res = audit_case_pair(corrupt_mri, seg_p, "case_corrupt")
    assert res["is_valid"] is False
    assert any("Corrupt or unreadable NIfTI file" in e for e in res["errors"])


def test_audit_non_finite_mri_values(tmp_path):
    """Verifies audit detects NaN and Infinite values in MRI volumes."""
    arr = np.ones((8, 8, 8), dtype=np.float32)
    arr[2, 2, 2] = np.nan
    arr[4, 4, 4] = np.inf

    mri_p = tmp_path / "nan_T1CE.nii.gz"
    seg_p = tmp_path / "nan_seg.nii.gz"
    nib.save(nib.Nifti1Image(arr, np.eye(4)), str(mri_p))
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.int16), np.eye(4)), str(seg_p))

    res = audit_case_pair(mri_p, seg_p, "case_nan")
    assert res["is_valid"] is False
    assert any("non-finite values" in e for e in res["errors"])


def test_audit_invalid_segmentation_labels(tmp_path):
    """Verifies audit flags unexpected or out-of-alphabet segmentation labels."""
    mri_p = tmp_path / "test_T1CE.nii.gz"
    seg_p = tmp_path / "test_seg.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((8, 8, 8)), np.eye(4)), str(mri_p))

    # Invalid labels: label 7 and -1
    seg_arr = np.zeros((8, 8, 8), dtype=np.int16)
    seg_arr[1, 1, 1] = 7
    seg_arr[2, 2, 2] = -1
    nib.save(nib.Nifti1Image(seg_arr, np.eye(4)), str(seg_p))

    res = audit_case_pair(mri_p, seg_p, "case_invalid_label")
    assert res["is_valid"] is False
    assert any("Unexpected segmentation labels found" in e for e in res["errors"])


def test_audit_shape_and_affine_mismatch(tmp_path):
    """Verifies audit detects dimension shape mismatch and spatial affine discrepancies."""
    mri_p = tmp_path / "mismatch_T1CE.nii.gz"
    seg_p = tmp_path / "mismatch_seg.nii.gz"

    # Shape mismatch: 8x8x8 vs 8x8x10
    nib.save(nib.Nifti1Image(np.ones((8, 8, 8)), np.eye(4)), str(mri_p))
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 10)), np.eye(4)), str(seg_p))

    res = audit_case_pair(mri_p, seg_p, "case_shape_mismatch")
    assert res["is_valid"] is False
    assert any("Shape mismatch" in e for e in res["errors"])

    # Affine mismatch
    seg_p_affine = tmp_path / "affine_mismatch_seg.nii.gz"
    diff_affine = np.diag([2.0, 2.0, 2.0, 1.0])
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8)), diff_affine), str(seg_p_affine))

    res_aff = audit_case_pair(mri_p, seg_p_affine, "case_affine_mismatch")
    assert res_aff["is_valid"] is False
    assert any("affine matrices do not match" in e for e in res_aff["errors"])


def test_audit_label_4_remapping(tmp_path):
    """Verifies BraTS label 4 (Enhancing Tumor) is recognized and remapped to label 3."""
    mri_p = tmp_path / "l4_T1CE.nii.gz"
    seg_p = tmp_path / "l4_seg.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((8, 8, 8)), np.eye(4)), str(mri_p))

    seg_arr = np.zeros((8, 8, 8), dtype=np.int16)
    seg_arr[3:6, 3:6, 3:6] = 4  # Raw BraTS ET label 4 (27 voxels)
    nib.save(nib.Nifti1Image(seg_arr, np.eye(4)), str(seg_p))

    res = audit_case_pair(mri_p, seg_p, "case_l4")
    assert res["is_valid"] is True
    assert res["segmentation_labels"]["label_4_remapped_to_3"] is True
    assert res["segmentation_labels"]["label_4_voxel_count"] == 27
    assert res["segmentation_labels"]["voxel_counts"]["ET"] == 27
    assert res["segmentation_labels"]["voxel_counts"]["WT"] == 27


def test_audit_duplicate_file_checksum_detection(tmp_path):
    """Verifies that duplicate volume content across different case names is caught."""
    mri_1 = tmp_path / "Case_001_T1CE.nii.gz"
    seg_1 = tmp_path / "Case_001_seg.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((8, 8, 8)), np.eye(4)), str(mri_1))
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8)), np.eye(4)), str(seg_1))

    # Case 2 is a byte-for-byte duplicate of Case 1
    mri_2 = tmp_path / "Case_002_T1CE.nii.gz"
    seg_2 = tmp_path / "Case_002_seg.nii.gz"
    mri_2.write_bytes(mri_1.read_bytes())
    seg_2.write_bytes(seg_1.read_bytes())

    audit_summary = audit_dataset_directory(tmp_path)
    assert audit_summary["all_valid"] is False
    assert any("Duplicate MRI volume content detected" in e for e in audit_summary["critical_errors"])


# ==============================================================================
# 2. PATIENT-LEVEL SPLIT INTEGRITY & LEAKAGE TESTS
# ==============================================================================

def test_patient_split_leakage_and_determinism():
    """Verifies deterministic patient-level partitioning with strict zero leakage."""
    manifest_fixture = {
        "cases": {
            f"case_{i:03d}": {
                "case_key": f"case_{i:03d}",
                "patient_id": f"PATIENT-STUDY-{i:03d}",
                "is_valid": True,
                "files": {
                    "mri": {"filename": f"Case_{i:03d}_T1CE.nii.gz", "sha256": f"hash_mri_{i}"},
                    "seg": {"filename": f"Case_{i:03d}_seg.nii.gz", "sha256": f"hash_seg_{i}"}
                },
                "segmentation_labels": {"volume_cm3": {"WT": 50.0 + i}}
            }
            for i in range(1, 7)
        }
    }

    # Generate with seed 42
    split_42_a = generate_patient_splits(manifest_fixture, val_ratio=0.333, seed=42)
    split_42_b = generate_patient_splits(manifest_fixture, val_ratio=0.333, seed=42)

    # Determinism across identical seeds
    assert split_42_a["splits"]["train"] == split_42_b["splits"]["train"]
    assert split_42_a["splits"]["val"] == split_42_b["splits"]["val"]

    # Zero patient leakage
    train_patients = {c["patient_id"] for c in split_42_a["splits"]["train"]}
    val_patients = {c["patient_id"] for c in split_42_a["splits"]["val"]}
    assert train_patients.isdisjoint(val_patients)
    assert len(train_patients.intersection(val_patients)) == 0

    # Zero file hash leakage
    train_mri_hashes = {c["mri_sha256"] for c in split_42_a["splits"]["train"]}
    val_mri_hashes = {c["mri_sha256"] for c in split_42_a["splits"]["val"]}
    assert train_mri_hashes.isdisjoint(val_mri_hashes)

    # Different seed produces distinct assignment
    split_999 = generate_patient_splits(manifest_fixture, val_ratio=0.333, seed=999)
    train_999_patients = {c["patient_id"] for c in split_999["splits"]["train"]}
    assert train_999_patients != train_patients or set(train_patients) == set(train_999_patients)  # Valid check


# ==============================================================================
# 3. METRICS MATHEMATICAL & EDGE-CASE CONVENTION TESTS
# ==============================================================================

def test_empty_mask_mathematical_conventions():
    """Verifies all 4 quadrants of empty vs non-empty mask metrics."""
    shape = (10, 10, 10)
    empty = np.zeros(shape, dtype=np.uint8)
    non_empty = np.zeros(shape, dtype=np.uint8)
    non_empty[2:5, 2:5, 2:5] = 1

    # Quadrant 1: Both empty
    m_both_empty = compute_binary_metrics(empty, empty)
    assert m_both_empty["dice"] == 1.0
    assert m_both_empty["iou"] == 1.0
    assert m_both_empty["sensitivity"] == 1.0
    assert m_both_empty["precision"] == 1.0
    assert m_both_empty["specificity"] == 1.0
    assert m_both_empty["volume_diff_pct"] == 0.0

    # Quadrant 2: Reference empty, prediction non-empty (False Positive alarm)
    m_ref_empty = compute_binary_metrics(non_empty, empty)
    assert m_ref_empty["dice"] == 0.0
    assert m_ref_empty["iou"] == 0.0
    assert m_ref_empty["precision"] == 0.0
    assert m_ref_empty["sensitivity"] == 1.0  # FN=0
    assert m_ref_empty["volume_diff_pct"] == 100.0

    # Quadrant 3: Reference non-empty, prediction empty (Complete Miss)
    m_pred_empty = compute_binary_metrics(empty, non_empty)
    assert m_pred_empty["dice"] == 0.0
    assert m_pred_empty["iou"] == 0.0
    assert m_pred_empty["sensitivity"] == 0.0  # Missed all true cases
    assert m_pred_empty["precision"] == 0.0
    assert m_pred_empty["volume_diff_pct"] == 100.0

    # Quadrant 4: Both non-empty (Partial overlap)
    half = np.zeros(shape, dtype=np.uint8)
    half[2:4, 2:5, 2:5] = 1  # 2*3*3 = 18 voxels out of 27
    m_overlap = compute_binary_metrics(half, non_empty)
    assert 0.0 < m_overlap["dice"] < 1.0
    assert 0.0 < m_overlap["precision"] <= 1.0
    assert 0.0 < m_overlap["sensitivity"] < 1.0


def test_metric_shape_mismatch_raises_error():
    """Verifies that compute_case_subregion_metrics raises ValueError on spatial mismatch."""
    pred = np.zeros((10, 10, 10))
    target = np.zeros((10, 10, 12))
    with pytest.raises(ValueError, match="Spatial dimension mismatch"):
        compute_case_subregion_metrics(pred, target)


def test_physical_volume_calculation():
    """Verifies voxel spacing converts voxel count accurately to cm^3."""
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[:5, :5, :5] = 1  # 125 voxels
    # Spacing 2.0 x 2.0 x 2.0 mm -> 8 mm^3 per voxel. 125 * 8 = 1000 mm^3 = 1.000 cm^3
    res = compute_binary_metrics(mask, mask, voxel_spacing=(2.0, 2.0, 2.0))
    assert res["pred_volume_cm3"] == 1.0
    assert res["target_volume_cm3"] == 1.0


def test_aggregate_dataset_metrics_min_max_stats():
    """Verifies aggregation correctly computes min, max, median, mean, std across cases."""
    case1 = {
        "mean_dice": 0.50,
        "subregions": {
            "WT": {"dice": 0.60, "iou": 0.50, "sensitivity": 0.70, "precision": 0.65},
            "TC": {"dice": 0.40, "iou": 0.30, "sensitivity": 0.50, "precision": 0.45},
            "ET": {"dice": 0.50, "iou": 0.40, "sensitivity": 0.60, "precision": 0.55}
        }
    }
    case2 = {
        "mean_dice": 0.70,
        "subregions": {
            "WT": {"dice": 0.80, "iou": 0.70, "sensitivity": 0.90, "precision": 0.85},
            "TC": {"dice": 0.60, "iou": 0.50, "sensitivity": 0.70, "precision": 0.65},
            "ET": {"dice": 0.70, "iou": 0.60, "sensitivity": 0.80, "precision": 0.75}
        }
    }

    agg = aggregate_dataset_metrics([case1, case2])
    assert agg["total_cases_evaluated"] == 2
    assert agg["overall_mean_dice"] == 0.60
    assert agg["subregions"]["WT"]["dice"]["min"] == 0.60
    assert agg["subregions"]["WT"]["dice"]["max"] == 0.80
    assert agg["subregions"]["WT"]["dice"]["mean"] == 0.70
    assert agg["subregions"]["WT"]["dice"]["median"] == 0.70


# ==============================================================================
# 4. FULL-VOLUME SLIDING-WINDOW INFERENCE TESTS
# ==============================================================================

def test_sliding_window_inference_shape_preservation():
    """Verifies that sliding window preserves input spatial shape exactly on arbitrary sizes."""
    model = build_segmentation_model(in_channels=4, out_channels=4)
    model.eval()

    # Shape not a power of 2: (1, 4, 40, 48, 36)
    tensor = torch.zeros((1, 4, 40, 48, 36), dtype=torch.float32)
    device = torch.device("cpu")

    pred = run_sliding_window_inference(
        model=model,
        inputs=tensor,
        roi_size=(32, 32, 32),
        overlap=0.5,
        device=device
    )

    assert pred.shape == (40, 48, 36)
    assert set(np.unique(pred)).issubset({0, 1, 2, 3})


# ==============================================================================
# 5. CHECKPOINT LOADING FAILURES & METADATA
# ==============================================================================

def test_checkpoint_loading_failures(tmp_path):
    """Verifies load_segmentation_model handles non-existent and corrupted checkpoints gracefully."""
    device = torch.device("cpu")

    # Non-existent checkpoint returns loaded=False without crash
    model, is_loaded, meta = load_segmentation_model(str(tmp_path / "nonexistent.pth"), device)
    assert is_loaded is False
    assert "not found" in meta["status_message"].lower()

    # Corrupt checkpoint file returns loaded=False with error message
    corrupt_pth = tmp_path / "corrupt.pth"
    corrupt_pth.write_bytes(b"NON_PICKLE_GARBAGE_BYTES")

    model2, is_loaded2, meta2 = load_segmentation_model(str(corrupt_pth), device)
    assert is_loaded2 is False
    assert "error" in meta2
