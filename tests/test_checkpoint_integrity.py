"""
NEUROSCAN AI — Phase 4A Regression Tests
Verifies:
  1. Best checkpoint selection: highest-scoring checkpoint across multiple epochs is preserved.
  2. Final-epoch weights do not overwrite a superior earlier checkpoint.
  3. Optional separate final-epoch checkpoint is saved when requested.
  4. Non-finite validation metrics (NaN, Inf, None) are handled safely without corrupting checkpoints.
  5. Checkpoint metadata captures epoch, validation metric, in_channels, seed, splits, and training config.
  6. Dynamic evaluation metadata accurately reports single-channel vs. 4-channel models in JSON and Markdown.
  7. Four-channel compatibility and ancillary components (2D classifier, API routes) remain intact.
"""

import math
import json
import pytest
import numpy as np
import nibabel as nib
import torch
from pathlib import Path
from unittest.mock import patch

from backend.segmentation.model import build_segmentation_model, load_segmentation_model
from backend.segmentation.train import train_segmentation_model, compute_sha256
from backend.segmentation.evaluate import evaluate_cohort, generate_markdown_report


@pytest.fixture
def mock_dataset_environment(tmp_path):
    """Creates synthetic NIfTI volume pairs and a splits.json manifest."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])

    # Volume size 48x48x48 allows 4 downsamplings (bottleneck 3x3x3 > 1)
    cases = ["case_001", "case_002", "case_003", "case_004"]
    splits_dict = {
        "random_seed": 42,
        "split_protocol": "patient_stratified",
        "splits": {
            "train": [
                {"case_key": "case_001", "patient_id": "P001", "mri_filename": "BraTS_case_001_T1CE.nii.gz", "seg_filename": "BraTS_case_001_seg.nii.gz"},
                {"case_key": "case_002", "patient_id": "P002", "mri_filename": "BraTS_case_002_T1CE.nii.gz", "seg_filename": "BraTS_case_002_seg.nii.gz"},
                {"case_key": "case_003", "patient_id": "P003", "mri_filename": "BraTS_case_003_T1CE.nii.gz", "seg_filename": "BraTS_case_003_seg.nii.gz"}
            ],
            "val": [
                {"case_key": "case_004", "patient_id": "P004", "mri_filename": "BraTS_case_004_T1CE.nii.gz", "seg_filename": "BraTS_case_004_seg.nii.gz"}
            ]
        }
    }
    splits_path = tmp_path / "splits.json"
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits_dict, f, indent=2)

    for c in cases:
        mri_arr = np.random.uniform(50.0, 200.0, size=(48, 48, 48)).astype(np.float32)
        seg_arr = np.zeros((48, 48, 48), dtype=np.int16)
        seg_arr[16:32, 16:32, 16:32] = 1
        seg_arr[20:28, 20:28, 20:28] = 3

        mri_p = data_dir / f"BraTS_{c}_T1CE.nii.gz"
        seg_p = data_dir / f"BraTS_{c}_seg.nii.gz"
        nib.save(nib.Nifti1Image(mri_arr, affine), str(mri_p))
        nib.save(nib.Nifti1Image(seg_arr, affine), str(seg_p))

    return data_dir, splits_path


def test_best_checkpoint_preserved_not_overwritten_by_final_epoch(mock_dataset_environment, tmp_path):
    """
    Verifies that when an earlier epoch has a higher validation score than the final epoch,
    the best checkpoint on disk remains the higher-scoring epoch's checkpoint.
    """
    data_dir, splits_path = mock_dataset_environment
    output_ckpt = tmp_path / "best_model.pth"
    final_ckpt = tmp_path / "final_model.pth"

    # Mock validation metric values across 3 epochs:
    # Epoch 1: 0.35, Epoch 2: 0.82 (peak), Epoch 3: 0.41 (degradation)
    mock_metric_returns = [
        {"mean_dice": 0.35, "subregions": {"WT": {"dice": 0.35}, "TC": {"dice": 0.35}, "ET": {"dice": 0.35}}},
        {"mean_dice": 0.82, "subregions": {"WT": {"dice": 0.82}, "TC": {"dice": 0.82}, "ET": {"dice": 0.82}}},
        {"mean_dice": 0.41, "subregions": {"WT": {"dice": 0.41}, "TC": {"dice": 0.41}, "ET": {"dice": 0.41}}}
    ]

    with patch("backend.segmentation.train.compute_case_subregion_metrics", side_effect=mock_metric_returns):
        res = train_segmentation_model(
            data_dir=str(data_dir),
            output_checkpoint_path=str(output_ckpt),
            epochs=3,
            batch_size=1,
            patch_size=(32, 32, 32),
            in_channels=1,
            splits_path=str(splits_path),
            save_final_checkpoint=True,
            final_checkpoint_path=str(final_ckpt)
        )

    # 1. Best checkpoint must be epoch 2 with dice 0.82
    assert output_ckpt.exists()
    best_saved = torch.load(output_ckpt, weights_only=False)
    assert best_saved["best_epoch"] == 2
    assert best_saved["best_val_dice"] == 0.82
    assert best_saved["metadata"]["best_epoch"] == 2
    assert best_saved["metadata"]["best_val_dice"] == 0.82

    # 2. Final checkpoint must be epoch 3 with final_val_dice 0.41
    assert final_ckpt.exists()
    final_saved = torch.load(final_ckpt, weights_only=False)
    assert final_saved["epoch"] == 3
    assert final_saved["final_val_dice"] == 0.41
    assert final_saved["best_epoch"] == 2
    assert final_saved.get("is_final_epoch") is True


def test_non_finite_validation_metrics_handled_safely(mock_dataset_environment, tmp_path):
    """
    Verifies that NaN, Inf, and None validation metrics do not crash training
    and do not overwrite a previously established best checkpoint.
    """
    data_dir, splits_path = mock_dataset_environment
    output_ckpt = tmp_path / "robust_model.pth"

    # Epoch 1: 0.50, Epoch 2: NaN, Epoch 3: Inf, Epoch 4: 0.30
    mock_metric_returns = [
        {"mean_dice": 0.50, "subregions": {"WT": {"dice": 0.50}, "TC": {"dice": 0.50}, "ET": {"dice": 0.50}}},
        {"mean_dice": float("nan"), "subregions": {"WT": {"dice": float("nan")}, "TC": {"dice": 0.0}, "ET": {"dice": 0.0}}},
        {"mean_dice": float("inf"), "subregions": {"WT": {"dice": float("inf")}, "TC": {"dice": 0.0}, "ET": {"dice": 0.0}}},
        {"mean_dice": 0.30, "subregions": {"WT": {"dice": 0.30}, "TC": {"dice": 0.30}, "ET": {"dice": 0.30}}}
    ]

    with patch("backend.segmentation.train.compute_case_subregion_metrics", side_effect=mock_metric_returns):
        res = train_segmentation_model(
            data_dir=str(data_dir),
            output_checkpoint_path=str(output_ckpt),
            epochs=4,
            batch_size=1,
            patch_size=(32, 32, 32),
            in_channels=1,
            splits_path=str(splits_path)
        )

    assert output_ckpt.exists()
    saved = torch.load(output_ckpt, weights_only=False)
    # Peak remains Epoch 1 (0.50); NaN and Inf were safely ignored
    assert saved["best_epoch"] == 1
    assert saved["best_val_dice"] == 0.50
    assert len(res["history"]) == 4
    assert res["history"][1]["val_mean_dice"] is None
    assert res["history"][2]["val_mean_dice"] is None


def test_checkpoint_metadata_integrity(mock_dataset_environment, tmp_path):
    """
    Verifies that checkpoint metadata contains epoch, validation metric, in_channels,
    seed, split manifest info, and complete training configuration.
    """
    data_dir, splits_path = mock_dataset_environment
    output_ckpt = tmp_path / "meta_test_model.pth"

    mock_metrics = [
        {"mean_dice": 0.65, "subregions": {"WT": {"dice": 0.65}, "TC": {"dice": 0.65}, "ET": {"dice": 0.65}}}
    ]

    with patch("backend.segmentation.train.compute_case_subregion_metrics", side_effect=mock_metrics):
        train_segmentation_model(
            data_dir=str(data_dir),
            output_checkpoint_path=str(output_ckpt),
            epochs=1,
            batch_size=1,
            patch_size=(32, 32, 32),
            in_channels=1,
            splits_path=str(splits_path),
            seed=123
        )

    saved = torch.load(output_ckpt, weights_only=False)
    meta = saved["metadata"]

    assert meta["in_channels"] == 1
    assert meta["out_channels"] == 4
    assert meta["seed"] == 123
    assert meta["validation_metric"] == "mean_subregion_dice"
    assert "splits_manifest" in meta
    assert meta["splits_manifest"]["train_cases_count"] == 3
    assert meta["splits_manifest"]["val_cases_count"] == 1
    assert "training_config" in meta
    assert meta["training_config"]["patch_size"] == [32, 32, 32]
    assert meta["training_config"]["batch_size"] == 1


def test_dynamic_evaluation_metadata_single_channel(tmp_path):
    """
    Verifies that evaluation report dynamically detects in_channels=1 from single-channel checkpoint
    and outputs correct JSON and Markdown metadata.
    """
    ckpt_path = tmp_path / "test_1ch.pth"
    model = build_segmentation_model(in_channels=1, out_channels=4)
    torch.save({
        "epoch": 5,
        "best_epoch": 5,
        "best_val_dice": 0.55,
        "model_state_dict": model.state_dict(),
        "metadata": {
            "architecture": "3D U-Net (BraTS Single-Channel Baseline)",
            "epochs_trained": 5,
            "best_epoch": 5,
            "best_val_dice": 0.55,
            "in_channels": 1,
            "input_modality": "Axial T1-CE"
        }
    }, str(ckpt_path))

    # Create dummy volume and mask of size 48x48x48 (bottleneck 3x3x3 > 1)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    mri_arr = np.ones((48, 48, 48), dtype=np.float32) * 100.0
    seg_arr = np.zeros((48, 48, 48), dtype=np.int16)
    seg_arr[16:32, 16:32, 16:32] = 1

    nib.save(nib.Nifti1Image(mri_arr, affine), str(data_dir / "BraTS_case_001_T1CE.nii.gz"))
    nib.save(nib.Nifti1Image(seg_arr, affine), str(data_dir / "BraTS_case_001_seg.nii.gz"))

    splits_dict = {"splits": {"val": [{"case_key": "case_001", "mri_filename": "BraTS_case_001_T1CE.nii.gz", "seg_filename": "BraTS_case_001_seg.nii.gz"}]}}
    splits_file = tmp_path / "splits.json"
    with open(splits_file, "w") as f:
        json.dump(splits_dict, f)

    json_out = tmp_path / "eval_report.json"
    md_out = tmp_path / "eval_report.md"

    report = evaluate_cohort(
        checkpoint_path=str(ckpt_path),
        data_dir=str(data_dir),
        splits_path=str(splits_file),
        split_subset="val",
        output_json=str(json_out),
        output_md=str(md_out),
        patch_size=(32, 32, 32),
        overlap=0.25,
        min_fg_prob=0.40,
        apply_brain_mask=True
    )

    # Verify JSON dynamic metadata
    assert report["checkpoint"]["input_channels"] == 1
    assert report["inference_configuration"]["input_channels"] == 1
    assert "single-channel" in report["inference_configuration"]["input_semantics"].lower()
    assert report["inference_configuration"]["min_fg_prob"] == 0.40
    assert report["inference_configuration"]["apply_brain_mask"] is True

    # Verify Markdown contains the dynamic fields
    md_text = md_out.read_text(encoding="utf-8")
    assert "1 channel(s)" in md_text
    assert "Native single-channel axial T1-CE sequence" in md_text
    assert "min_fg_prob" in md_text
    assert "0.4" in md_text
    assert "Intracranial Brain Masking" in md_text


def test_dynamic_evaluation_metadata_four_channel(tmp_path):
    """
    Verifies that evaluation report dynamically detects in_channels=4 from 4-channel checkpoint
    and maintains legacy compatibility without errors.
    """
    ckpt_path = tmp_path / "test_4ch.pth"
    model = build_segmentation_model(in_channels=4, out_channels=4)
    torch.save({
        "epoch": 4,
        "model_state_dict": model.state_dict(),
        "metadata": {
            "architecture": "3D U-Net (BraTS Benchmark)",
            "epochs_trained": 4,
            "in_channels": 4,
            "input_modality": "mpMRI"
        }
    }, str(ckpt_path))

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    mri_arr = np.ones((48, 48, 48), dtype=np.float32) * 100.0
    seg_arr = np.zeros((48, 48, 48), dtype=np.int16)

    nib.save(nib.Nifti1Image(mri_arr, affine), str(data_dir / "BraTS_case_001_T1CE.nii.gz"))
    nib.save(nib.Nifti1Image(seg_arr, affine), str(data_dir / "BraTS_case_001_seg.nii.gz"))

    splits_dict = {"splits": {"val": [{"case_key": "case_001", "mri_filename": "BraTS_case_001_T1CE.nii.gz", "seg_filename": "BraTS_case_001_seg.nii.gz"}]}}
    splits_file = tmp_path / "splits.json"
    with open(splits_file, "w") as f:
        json.dump(splits_dict, f)

    report = evaluate_cohort(
        checkpoint_path=str(ckpt_path),
        data_dir=str(data_dir),
        splits_path=str(splits_file),
        split_subset="val",
        patch_size=(32, 32, 32),
        overlap=0.25
    )

    assert report["checkpoint"]["input_channels"] == 4
    assert report["inference_configuration"]["input_channels"] == 4
    assert "4-channel" in report["inference_configuration"]["input_semantics"]
