"""
NEUROSCAN AI — Phase 2 Regression & Verification Test Suite
Tests for:
1. Dynamic single-modality (in_channels=1) vs multimodal (in_channels=4) preprocessing and modeling
2. Intracranial brain masking suppressing exterior air false positives
3. Checkpoint auto-detection of input channels
4. Windows case-insensitivity deduplication and splits.json ingestion with 0% patient leakage
"""

import json
import pytest
import numpy as np
import torch
import nibabel as nib
from pathlib import Path

from backend.segmentation.pipeline import (
    NiftiVolume,
    preprocess_volume_for_model,
    normalize_intensity,
    run_sliding_window_inference,
    native_sliding_window_inference
)
from backend.segmentation.model import (
    build_segmentation_model,
    load_segmentation_model
)
from backend.segmentation.train import (
    partition_cases_by_patient,
    NiftiSegmentationDataset
)


def test_single_channel_preprocessing_and_model_forward():
    """Verifies single-channel T1-CE processing and 3D U-Net forward pass."""
    data = np.random.uniform(0, 100, size=(1, 32, 32, 32)).astype(np.float32)
    vol = NiftiVolume(
        data=data,
        affine=np.eye(4),
        voxel_spacing=(1.0, 1.0, 1.0),
        shape=(1, 32, 32, 32),
        modality_channels=1
    )

    tensor_1ch = preprocess_volume_for_model(vol, target_channels=1)
    assert tensor_1ch.shape == (1, 1, 32, 32, 32)

    model = build_segmentation_model(in_channels=1, out_channels=4)
    model.eval()
    with torch.no_grad():
        out = model(tensor_1ch)
    assert out.shape == (1, 4, 32, 32, 32)


def test_multimodal_backward_compatibility():
    """Verifies target_channels=4 replicates 1-channel data for backward compatibility."""
    data = np.ones((1, 16, 16, 16), dtype=np.float32)
    vol = NiftiVolume(
        data=data,
        affine=np.eye(4),
        voxel_spacing=(1.0, 1.0, 1.0),
        shape=(1, 16, 16, 16),
        modality_channels=1
    )

    tensor_4ch = preprocess_volume_for_model(vol, target_channels=4)
    assert tensor_4ch.shape == (1, 4, 16, 16, 16)
    # Check channels 0 and 1 are identical copies
    np.testing.assert_allclose(tensor_4ch[0, 0].numpy(), tensor_4ch[0, 1].numpy())


def test_intracranial_brain_mask_suppresses_air_false_positives():
    """Verifies that brain_mask forces exterior air predictions to background class 0."""
    model = build_segmentation_model(in_channels=1, out_channels=4)
    model.eval()

    # Create synthetic volume with brain in center and air in periphery
    shape = (1, 1, 48, 48, 48)
    inputs = torch.randn(shape)
    device = torch.device("cpu")

    # Define brain mask (only center 24x24x24)
    brain_mask = np.zeros((48, 48, 48), dtype=bool)
    brain_mask[12:36, 12:36, 12:36] = True

    pred_masked = run_sliding_window_inference(
        model=model,
        inputs=inputs,
        roi_size=(32, 32, 32),
        overlap=0.5,
        device=device,
        brain_mask=brain_mask
    )

    # Outside the brain mask, every single voxel must be strictly 0 (background)
    air_preds = pred_masked[~brain_mask]
    assert np.all(air_preds == 0), "Exterior air voxels contained non-zero tumor predictions!"


def test_partition_cases_ingests_splits_json(tmp_path):
    """Verifies partition_cases_by_patient ingests splits.json with zero leakage."""
    # Create fake files
    mri_files = []
    seg_files = []
    for i in range(1, 5):
        mri = tmp_path / f"BraTS_Case_{i:03d}_T1CE.nii.gz"
        seg = tmp_path / f"BraTS_Case_{i:03d}_seg.nii.gz"
        mri.write_bytes(b"dummy")
        seg.write_bytes(b"dummy")
        mri_files.append(mri)
        seg_files.append(seg)

    splits_data = {
        "splits": {
            "train": [
                {"case_key": "case_001", "mri_filename": mri_files[0].name, "seg_filename": seg_files[0].name},
                {"case_key": "case_002", "mri_filename": mri_files[1].name, "seg_filename": seg_files[1].name}
            ],
            "val": [
                {"case_key": "case_003", "mri_filename": mri_files[2].name, "seg_filename": seg_files[2].name},
                {"case_key": "case_004", "mri_filename": mri_files[3].name, "seg_filename": seg_files[3].name}
            ]
        }
    }
    splits_file = tmp_path / "splits.json"
    with open(splits_file, "w") as f:
        json.dump(splits_data, f)

    train_p, val_p = partition_cases_by_patient(tmp_path, splits_path=splits_file)

    train_names = {p[0].name for p in train_p}
    val_names = {p[0].name for p in val_p}

    assert train_names == {mri_files[0].name, mri_files[1].name}
    assert val_names == {mri_files[2].name, mri_files[3].name}
    assert train_names.isdisjoint(val_names)


def test_load_segmentation_model_auto_detects_channels(tmp_path):
    """Verifies load_segmentation_model detects in_channels=1 from checkpoint metadata."""
    ckpt_path = tmp_path / "test_1ch.pth"
    model_1ch = build_segmentation_model(in_channels=1, out_channels=4)

    torch.save({
        "model_state_dict": model_1ch.state_dict(),
        "metadata": {
            "architecture": "3D U-Net",
            "in_channels": 1,
            "out_channels": 4
        }
    }, str(ckpt_path))

    device = torch.device("cpu")
    loaded_model, is_loaded, meta = load_segmentation_model(str(ckpt_path), device)

    assert is_loaded is True
    assert meta["in_channels"] == 1
    assert loaded_model.in_channels == 1
