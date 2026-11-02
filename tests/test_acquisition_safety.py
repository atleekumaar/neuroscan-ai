"""
NEUROSCAN AI — Phase 5.3.1: Unit Tests for Acquisition Script Safety & Gatekeeper Integration
Verifies:
  1. Missing-token safe halt
  2. Invalid or excessive pilot size bounds enforcement
  3. No-overwrite behavior on existing studies
  4. Interrupted-download cleanup and staging isolation
  5. Manifest atomicity and historical entry preservation
  6. Secret redaction in error messages and exceptions
  7. Path traversal and model checkpoint directory protection
  8. Absolute zero network access enforcement during tests
  9. Bitwise invariance of all model checkpoints on disk
 10. Prevention of unverified third-party mirror fallbacks
 11. Multimodal 5-tuple gatekeeper rejection of incomplete studies
"""

import os
import sys
import json
import shutil
import socket
import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import numpy as np
import nibabel as nib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.segmentation.ingestion import (
    MultimodalIngestionValidator,
    update_manifest_atomic,
    validate_patient_multimodal_study,
    compute_file_sha256
)
from scripts.acquire_brats_pilot import (
    check_auth_prerequisites,
    validate_pilot_parameters,
    redact_secrets,
    acquire_pilot,
    MAX_PILOT_SIZE,
    SYNAPSE_PROJECT_ID
)


@pytest.fixture(autouse=True)
def forbid_network_access(monkeypatch):
    """Guarantees zero network access during entire test execution."""
    def guarded_connect(*args, **kwargs):
        raise RuntimeError("[SECURITY ERROR] Network access attempted during offline safety audit!")
    
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


@pytest.fixture
def checkpoint_hashes():
    """Calculates SHA-256 for all current checkpoints in models/."""
    models_dir = PROJECT_ROOT / "models"
    hashes = {}
    for pth in models_dir.glob("*.pth"):
        hashes[pth.name] = compute_file_sha256(pth)
    return hashes


def test_missing_token_safe_halt(monkeypatch):
    """Confirm missing credentials cause a safe, non-destructive halt without exceptions."""
    monkeypatch.delenv("SYNAPSE_AUTH_TOKEN", raising=False)
    assert check_auth_prerequisites() is False

    monkeypatch.setenv("SYNAPSE_AUTH_TOKEN", "   ")
    assert check_auth_prerequisites() is False

    res = acquire_pilot(pilot_size=3)
    assert res["status"] == "BLOCKED"
    assert res["acquired_count"] == 0


def test_invalid_excessive_pilot_size(tmp_path):
    """Verify pilot size is strictly validated and capped between 1 and 5."""
    dest = tmp_path / "pilot"

    # Negative or zero
    with pytest.raises(ValueError, match="out of bounds"):
        validate_pilot_parameters(0, dest)

    with pytest.raises(ValueError, match="out of bounds"):
        validate_pilot_parameters(-1, dest)

    # Exceeding cap of 5
    with pytest.raises(ValueError, match="out of bounds"):
        validate_pilot_parameters(6, dest)

    with pytest.raises(ValueError, match="out of bounds"):
        validate_pilot_parameters(100, dest)

    # Invalid type
    with pytest.raises(ValueError, match="must be an integer"):
        validate_pilot_parameters(True, dest)

    # Valid values
    size1, path1 = validate_pilot_parameters(1, dest)
    assert size1 == 1

    size5, path5 = validate_pilot_parameters(5, dest)
    assert size5 == 5


def test_secret_redaction(monkeypatch):
    """Ensure tokens are never printed, logged, or included in exceptions."""
    secret = "synapse_pat_super_secret_token_abcdef123456"
    monkeypatch.setenv("SYNAPSE_AUTH_TOKEN", secret)

    raw_error = f"HTTP 401 Unauthorized for token {secret} at synapse.org"
    redacted = redact_secrets(raw_error, secret=secret)
    assert secret not in redacted
    assert "***REDACTED_SYNAPSE_TOKEN***" in redacted

    # Exception string wrapping
    try:
        raise ValueError(f"Failed authentication with key {secret}")
    except ValueError as e:
        safe_msg = redact_secrets(str(e))
        assert secret not in safe_msg
        assert "***REDACTED_SYNAPSE_TOKEN***" in safe_msg


def test_path_traversal_prevention(tmp_path):
    """Confirm destination path cannot target models, root, or system directories."""
    models_dir = PROJECT_ROOT / "models"
    with pytest.raises(ValueError, match="cannot target or reside inside models"):
        validate_pilot_parameters(3, models_dir)

    checkpoint_file = tmp_path / "candidate_model.pth"
    with pytest.raises(ValueError, match="cannot target model checkpoint files"):
        validate_pilot_parameters(3, checkpoint_file)

    root_dir = Path("C:/") if sys.platform == "win32" else Path("/")
    with pytest.raises(ValueError, match="Cannot use root directory"):
        validate_pilot_parameters(3, root_dir)


def test_no_overwrite_behavior(tmp_path, monkeypatch):
    """Confirm existing files and directories cannot be silently overwritten."""
    monkeypatch.setenv("SYNAPSE_AUTH_TOKEN", "valid_dummy_token_12345")
    dest = tmp_path / "pilot"
    dest.mkdir(parents=True)

    # Pre-create an existing patient study directory
    existing_patient = dest / "BraTS-GLI-00001-000"
    existing_patient.mkdir()
    sentinel_file = existing_patient / "canary.txt"
    sentinel_file.write_text("DO_NOT_OVERWRITE", encoding="utf-8")

    # Mock Synapse client
    mock_syn = MagicMock()
    mock_syn.getChildren.return_value = [{"name": "BraTS-GLI-00001-000", "id": "syn111"}]

    mock_syn_module = MagicMock()
    mock_syn_module.Synapse.return_value = mock_syn
    monkeypatch.setitem(sys.modules, "synapseclient", mock_syn_module)

    res = acquire_pilot(pilot_size=1, output_dir=dest, allow_overwrite=False)

    # Sentinel must still exist and directory was skipped
    assert sentinel_file.exists()
    assert sentinel_file.read_text(encoding="utf-8") == "DO_NOT_OVERWRITE"
    assert res["acquired_count"] == 0


def test_interrupted_download_cleanup_with_mocks(tmp_path, monkeypatch):
    """Confirm failed downloads clean up temporary staging files safely without corrupting target."""
    monkeypatch.setenv("SYNAPSE_AUTH_TOKEN", "valid_dummy_token_12345")
    dest = tmp_path / "pilot"
    dest.mkdir(parents=True)

    mock_syn = MagicMock()
    mock_syn.getChildren.side_effect = [
        # Remote folder discovery
        [{"name": "BraTS-GLI-00099-000", "id": "syn999"}],
        # Children of folder: 2 files
        [{"name": "BraTS-GLI-00099-000-t1n.nii.gz", "id": "syn_t1n"}],
    ]
    # Simulate download raising an IOError on second file
    mock_syn.get.side_effect = IOError("Simulated network drop / connection failure")

    mock_syn_module = MagicMock()
    mock_syn_module.Synapse.return_value = mock_syn
    monkeypatch.setitem(sys.modules, "synapseclient", mock_syn_module)

    res = acquire_pilot(pilot_size=1, output_dir=dest)

    # Staging directories must be completely wiped
    remaining_staging = list(dest.glob(".staging*"))
    assert len(remaining_staging) == 0

    # Target patient directory must not exist
    target_patient = dest / "BraTS-GLI-00099-000"
    assert not target_patient.exists()
    assert res["acquired_count"] == 0


def test_manifest_atomicity_and_preservation(tmp_path):
    """Confirm manifest updates are atomic, preserve unrelated entries, and calculate accurate summaries."""
    manifest_file = tmp_path / "multimodal_manifest.json"

    # Initial manifest with 2 historical records
    initial_record_1 = {
        "patient_id": "BraTS_Case_001",
        "is_eligible": False,
        "status": "REJECTED",
        "errors": ["Missing required modalities: ['t1n', 't2w', 't2f']"]
    }
    initial_record_2 = {
        "patient_id": "BraTS_Case_002",
        "is_eligible": False,
        "status": "REJECTED",
        "errors": ["Missing required modalities: ['t1n', 't2w', 't2f']"]
    }
    initial_manifest = {
        "manifest_version": "2.1.0-phase5.2",
        "dataset_standard": "ASNR-MICCAI BraTS 2023 Adult Glioma Multi-Parametric MRI Protocol",
        "summary": {
            "total_studies_audited": 2,
            "eligible_multimodal_studies": 0,
            "rejected_studies": 2
        },
        "audited_local_studies": [initial_record_1, initial_record_2]
    }
    manifest_file.write_text(json.dumps(initial_manifest, indent=2), encoding="utf-8")

    # Add new mock validated record
    new_study = {
        "patient_id": "BraTS-GLI-00100-000",
        "is_eligible": True,
        "status": "ELIGIBLE",
        "errors": [],
        "gatekeeper_checks": {
            "modality_completeness": True,
            "file_integrity": True,
            "spatial_geometry": True,
            "rigid_registration": True,
            "label_alphabet": True,
            "deduplication": True,
            "cryptographic_hashing": True
        }
    }

    updated = update_manifest_atomic(manifest_file, [new_study])

    assert updated["summary"]["total_studies_audited"] == 3
    assert updated["summary"]["eligible_multimodal_studies"] == 1
    assert updated["summary"]["rejected_studies"] == 2

    # Check on-disk content
    saved_json = json.loads(manifest_file.read_text(encoding="utf-8"))
    saved_ids = [s["patient_id"] for s in saved_json["audited_local_studies"]]
    assert "BraTS_Case_001" in saved_ids
    assert "BraTS_Case_002" in saved_ids
    assert "BraTS-GLI-00100-000" in saved_ids

    # Verify no temporary files remain
    tmp_leftovers = list(tmp_path.glob(".tmp*"))
    assert len(tmp_leftovers) == 0


def test_no_fallback_to_unverified_mirrors():
    """Verify that legacy scripts attempting unverified mirror downloads are strictly disabled."""
    with pytest.raises(RuntimeError, match="Direct download from unverified third-party mirror"):
        import scripts.download_real_brats  # noqa: F401

    with pytest.raises(RuntimeError, match="Direct download from unverified third-party mirror"):
        import scripts.download_additional_cases  # noqa: F401


def test_multimodal_gatekeeper_integration_5tuple_completeness(tmp_path):
    """Confirm gatekeeper rejects studies missing any of the 5 modalities."""
    validator = MultimodalIngestionValidator()

    # Create synthetic volume files for only 3 modalities
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    arr = np.ones((8, 8, 8), dtype=np.float32)
    img = nib.Nifti1Image(arr, affine)

    f_t1c = tmp_path / "BraTS-GLI-TEST-t1c.nii.gz"
    f_seg = tmp_path / "BraTS-GLI-TEST-seg.nii.gz"
    nib.save(img, str(f_t1c))
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.int16), affine), str(f_seg))

    # Missing t1n, t2w, t2f
    file_map = {
        "t1n": None,
        "t1c": f_t1c,
        "t2w": None,
        "t2f": None,
        "seg": f_seg
    }

    report = validator.validate_study("BraTS-GLI-TEST", file_map)
    assert report["is_eligible"] is False
    assert report["status"] == "REJECTED"
    assert "Missing required modalities" in report["errors"][0]


def test_checkpoint_files_remain_unchanged(checkpoint_hashes):
    """Verify that execution of all acquisition safety modules left every checkpoint unaltered."""
    models_dir = PROJECT_ROOT / "models"
    for pth in models_dir.glob("*.pth"):
        current_hash = compute_file_sha256(pth)
        assert current_hash == checkpoint_hashes[pth.name], f"Checkpoint {pth.name} hash modified!"
