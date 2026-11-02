"""
NEUROSCAN AI — Phase 5.2 Unit Tests for Multimodal Ingestion Validator
Uses temporary synthetic NIfTI fixtures strictly isolated in pytest tmp_path.
Never touches or alters real training data.

Tests:
  1. Valid synthetic 5-tuple case passes all 7 gatekeeper checks.
  2. Missing modality detection (e.g. missing FLAIR or SEG).
  3. Corrupt/empty NIfTI stream detection.
  4. Non-finite voxel values (NaN/Inf) rejection.
  5. Spatial geometry mismatch (dimension discrepancy).
  6. Rigid registration mismatch (affine matrix discrepancy).
  7. Out-of-alphabet segmentation label detection.
  8. Legacy label 4 detection and reporting.
  9. Duplicate file hash / identity collision detection.
  10. Leakage-free triple-split manifest generation.
"""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pytest
import numpy as np
import nibabel as nib

from backend.segmentation.ingestion import (
    validate_patient_multimodal_study,
    validate_nifti_file_integrity,
    generate_multimodal_split_manifest,
    compute_file_sha256
)


def create_synthetic_volume(
    shape=(16, 16, 16),
    affine=None,
    data_type=np.float32,
    is_seg=False,
    inject_nan=False,
    inject_inf=False,
    invalid_label=False,
    legacy_label_4=False
) -> nib.Nifti1Image:
    """Helper creating small synthetic NIfTI volumes for unit testing."""
    if affine is None:
        affine = np.diag([-1.0, -1.0, 1.0, 1.0])

    if is_seg:
        arr = np.zeros(shape, dtype=np.int16)
        arr[4:12, 4:12, 4:12] = 2  # ED
        arr[6:10, 6:10, 6:10] = 1  # NCR
        if legacy_label_4:
            arr[7:9, 7:9, 7:9] = 4  # Legacy Enhancing
        else:
            arr[7:9, 7:9, 7:9] = 3  # Enhancing
        if invalid_label:
            arr[2, 2, 2] = 99  # Invalid label
    else:
        arr = np.random.uniform(10.0, 500.0, size=shape).astype(data_type)
        if inject_nan:
            arr[5, 5, 5] = np.nan
        if inject_inf:
            arr[6, 6, 6] = np.inf

    return nib.Nifti1Image(arr, affine)


@pytest.fixture
def valid_study_fixture(tmp_path):
    """Generates a complete, valid synthetic 5-tuple patient study."""
    p_dir = tmp_path / "BraTS-GLI-TEST001-000"
    p_dir.mkdir(parents=True)

    file_map = {}
    modalities = ["t1n", "t1c", "t2w", "t2f"]
    for m in modalities:
        img = create_synthetic_volume()
        fpath = p_dir / f"BraTS-GLI-TEST001-000-{m}.nii.gz"
        nib.save(img, str(fpath))
        file_map[m] = fpath

    seg_img = create_synthetic_volume(is_seg=True)
    seg_path = p_dir / "BraTS-GLI-TEST001-000-seg.nii.gz"
    nib.save(seg_img, str(seg_path))
    file_map["seg"] = seg_path

    return "BraTS-GLI-TEST001-000", file_map


def test_valid_multimodal_study_passes_all_checks(valid_study_fixture):
    """Check 1: Valid 5-tuple passes all 7 gatekeeper checks."""
    pid, file_map = valid_study_fixture
    res = validate_patient_multimodal_study(pid, file_map)

    assert res["is_eligible"] is True
    assert res["status"] == "ELIGIBLE"
    assert len(res["errors"]) == 0
    assert all(res["gatekeeper_checks"].values()) is True
    assert res["geometry"]["dimensions"] == [16, 16, 16]
    assert res["label_semantics"]["voxel_counts"]["WT"] > 0


def test_missing_modality_rejected(valid_study_fixture):
    """Check 2: Missing FLAIR sequence causes immediate rejection."""
    pid, file_map = valid_study_fixture
    # Remove FLAIR
    del file_map["t2f"]

    res = validate_patient_multimodal_study(pid, file_map)
    assert res["is_eligible"] is False
    assert res["status"] == "REJECTED"
    assert res["gatekeeper_checks"]["modality_completeness"] is False
    assert any("Missing required modalities" in e for e in res["errors"])


def test_missing_segmentation_mask_rejected(valid_study_fixture):
    """Check 3: Missing segmentation mask causes rejection."""
    pid, file_map = valid_study_fixture
    del file_map["seg"]

    res = validate_patient_multimodal_study(pid, file_map)
    assert res["is_eligible"] is False
    assert res["gatekeeper_checks"]["modality_completeness"] is False


def test_corrupted_byte_stream_rejected(tmp_path, valid_study_fixture):
    """Check 4: Corrupt non-NIfTI byte stream is caught and flagged."""
    pid, file_map = valid_study_fixture
    corrupt_file = tmp_path / "corrupt-t1n.nii.gz"
    corrupt_file.write_bytes(b"NON_GZIP_CORRUPT_BYTES_DATA")
    file_map["t1n"] = corrupt_file

    res = validate_patient_multimodal_study(pid, file_map)
    assert res["is_eligible"] is False
    assert res["gatekeeper_checks"]["file_integrity"] is False
    assert any("Corrupted NIfTI" in e for e in res["errors"])


def test_empty_zero_byte_file_rejected(tmp_path, valid_study_fixture):
    """Check 5: 0-byte truncated file is flagged."""
    pid, file_map = valid_study_fixture
    empty_file = tmp_path / "empty-t2w.nii.gz"
    empty_file.write_bytes(b"")
    file_map["t2w"] = empty_file

    res = validate_patient_multimodal_study(pid, file_map)
    assert res["is_eligible"] is False
    assert res["gatekeeper_checks"]["file_integrity"] is False
    assert any("File is empty" in e for e in res["errors"])


def test_non_finite_intensities_rejected(tmp_path, valid_study_fixture):
    """Check 6: NaN or Inf voxels in MRI volumes are rejected."""
    pid, file_map = valid_study_fixture
    nan_img = create_synthetic_volume(inject_nan=True)
    nan_path = tmp_path / "nan_t1c.nii.gz"
    nib.save(nan_img, str(nan_path))
    file_map["t1c"] = nan_path

    res = validate_patient_multimodal_study(pid, file_map)
    assert res["is_eligible"] is False
    assert res["gatekeeper_checks"]["file_integrity"] is False
    assert any("Non-finite intensities" in e for e in res["errors"])


def test_dimension_mismatch_rejected(tmp_path, valid_study_fixture):
    """Check 7: Discrepant dimensions between modalities are rejected."""
    pid, file_map = valid_study_fixture
    wrong_shape_img = create_synthetic_volume(shape=(16, 16, 20))
    wrong_path = tmp_path / "wrong_shape_t2w.nii.gz"
    nib.save(wrong_shape_img, str(wrong_path))
    file_map["t2w"] = wrong_path

    res = validate_patient_multimodal_study(pid, file_map)
    assert res["is_eligible"] is False
    assert res["gatekeeper_checks"]["spatial_geometry"] is False
    assert any("Shape mismatch" in e for e in res["errors"])


def test_registration_affine_mismatch_rejected(tmp_path, valid_study_fixture):
    """Check 8: Non-co-registered affine coordinate matrices are rejected."""
    pid, file_map = valid_study_fixture
    misaligned_affine = np.diag([-2.0, -2.0, 2.0, 1.0])  # Discrepant 2mm zoom
    misaligned_img = create_synthetic_volume(affine=misaligned_affine)
    misaligned_path = tmp_path / "misaligned_t2f.nii.gz"
    nib.save(misaligned_img, str(misaligned_path))
    file_map["t2f"] = misaligned_path

    res = validate_patient_multimodal_study(pid, file_map)
    assert res["is_eligible"] is False
    assert res["gatekeeper_checks"]["rigid_registration"] is False
    assert any("Affine co-registration mismatch" in e for e in res["errors"])


def test_invalid_segmentation_labels_rejected(tmp_path, valid_study_fixture):
    """Check 9: Out-of-alphabet segmentation label values are rejected."""
    pid, file_map = valid_study_fixture
    bad_seg = create_synthetic_volume(is_seg=True, invalid_label=True)
    bad_seg_path = tmp_path / "bad_seg.nii.gz"
    nib.save(bad_seg, str(bad_seg_path))
    file_map["seg"] = bad_seg_path

    res = validate_patient_multimodal_study(pid, file_map)
    assert res["is_eligible"] is False
    assert res["gatekeeper_checks"]["label_alphabet"] is False
    assert any("Invalid/unexpected segmentation labels" in e for e in res["errors"])


def test_legacy_label_4_warning_reporting(tmp_path, valid_study_fixture):
    """Check 10: Legacy label 4 is detected and documented with warning."""
    pid, file_map = valid_study_fixture
    legacy_seg = create_synthetic_volume(is_seg=True, legacy_label_4=True)
    legacy_path = tmp_path / "legacy_seg.nii.gz"
    nib.save(legacy_seg, str(legacy_path))
    file_map["seg"] = legacy_path

    res = validate_patient_multimodal_study(pid, file_map, allow_legacy_label_4=True)
    assert res["is_eligible"] is True
    assert res["label_semantics"]["contains_legacy_label_4"] is True
    assert any("legacy BraTS label 4" in w for w in res["warnings"])


def test_duplicate_file_hash_detection(valid_study_fixture):
    """Check 11: Collision with previously seen file hash is detected."""
    pid, file_map = valid_study_fixture
    # Compute hash of t1n
    t1n_hash = compute_file_sha256(file_map["t1n"])
    res = validate_patient_multimodal_study(pid, file_map, known_hashes={t1n_hash})

    assert res["is_eligible"] is False
    assert res["gatekeeper_checks"]["deduplication"] is False
    assert any("Duplicate file hash detected" in e for e in res["errors"])


def test_split_manifest_generation_zero_leakage():
    """Check 12: Triple-split generator creates strictly disjoint patient sets."""
    # Mock 20 validated studies
    mock_studies = []
    for i in range(20):
        mock_studies.append({
            "patient_id": f"BraTS-GLI-{i:05d}-000",
            "is_eligible": True,
            "status": "ELIGIBLE",
            "file_paths": {"t1n": f"p{i}_t1n.nii.gz"},
            "file_sha256": {"t1n": f"sha_{i}"}
        })
    # Add 2 rejected studies
    mock_studies.append({"patient_id": "BraTS-GLI-BAD01-000", "is_eligible": False, "status": "REJECTED"})
    mock_studies.append({"patient_id": "BraTS-GLI-BAD02-000", "is_eligible": False, "status": "REJECTED"})

    manifest = generate_multimodal_split_manifest(mock_studies, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42)

    train_pids = {s["patient_id"] for s in manifest["splits"]["train"]}
    val_pids = {s["patient_id"] for s in manifest["splits"]["val"]}
    test_pids = {s["patient_id"] for s in manifest["splits"]["locked_test"]}

    assert len(train_pids) == 14
    assert len(val_pids) == 3
    assert len(test_pids) == 3
    assert len(manifest["rejected_cases"]) == 2

    # Zero leakage
    assert len(train_pids.intersection(val_pids)) == 0
    assert len(train_pids.intersection(test_pids)) == 0
    assert len(val_pids.intersection(test_pids)) == 0


if __name__ == "__main__":
    print("Running Phase 5.2 Ingestion Validator Unit Tests directly...")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        pid, fmap = "TEST-DIRECT-001", {}
        for m in ["t1n", "t1c", "t2w", "t2f"]:
            im = create_synthetic_volume()
            fp = p / f"sub-{m}.nii.gz"
            nib.save(im, str(fp))
            fmap[m] = fp
        sm = create_synthetic_volume(is_seg=True)
        sp = p / "sub-seg.nii.gz"
        nib.save(sm, str(sp))
        fmap["seg"] = sp

        test_valid_multimodal_study_passes_all_checks((pid, fmap))
        print("  [PASS] test_valid_multimodal_study_passes_all_checks")
        test_missing_modality_rejected((pid, dict(fmap)))
        print("  [PASS] test_missing_modality_rejected")
        test_missing_segmentation_mask_rejected((pid, dict(fmap)))
        print("  [PASS] test_missing_segmentation_mask_rejected")
        test_corrupted_byte_stream_rejected(p, (pid, dict(fmap)))
        print("  [PASS] test_corrupted_byte_stream_rejected")
        test_empty_zero_byte_file_rejected(p, (pid, dict(fmap)))
        print("  [PASS] test_empty_zero_byte_file_rejected")
        test_non_finite_intensities_rejected(p, (pid, dict(fmap)))
        print("  [PASS] test_non_finite_intensities_rejected")
        test_dimension_mismatch_rejected(p, (pid, dict(fmap)))
        print("  [PASS] test_dimension_mismatch_rejected")
        test_registration_affine_mismatch_rejected(p, (pid, dict(fmap)))
        print("  [PASS] test_registration_affine_mismatch_rejected")
        test_invalid_segmentation_labels_rejected(p, (pid, dict(fmap)))
        print("  [PASS] test_invalid_segmentation_labels_rejected")
        test_legacy_label_4_warning_reporting(p, (pid, dict(fmap)))
        print("  [PASS] test_legacy_label_4_warning_reporting")
        test_duplicate_file_hash_detection((pid, dict(fmap)))
        print("  [PASS] test_duplicate_file_hash_detection")
        test_split_manifest_generation_zero_leakage()
        print("  [PASS] test_split_manifest_generation_zero_leakage")
    print("--> ALL 12 PHASE 5.2 INGESTION TESTS PASSED SUCCESSFULLY!")
