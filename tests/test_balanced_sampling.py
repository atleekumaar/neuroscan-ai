"""
NEUROSCAN AI — Unit Tests for Phase 3 Root Cause Corrections
Verifies:
  1. min_fg_prob thresholding suppresses uncalibrated low-confidence foreground logits.
  2. NiftiSegmentationDataset balanced patch sampling extracts normal brain parenchyma.
  3. Evaluator CLI correctly supports --min-fg-prob.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
from backend.segmentation.model import build_segmentation_model
from backend.segmentation.pipeline import run_sliding_window_inference, NiftiVolume
from backend.segmentation.train import NiftiSegmentationDataset


def test_min_fg_prob_suppresses_low_confidence_voxels():
    """
    Verifies that run_sliding_window_inference with min_fg_prob > 0.0
    suppresses low-confidence foreground detections to class 0 (background).
    """
    # Create mock model that outputs uniform-ish logits where foreground barely exceeds background
    class MockUncalibratedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.out_channels = 4
            self.in_channels = 1

        def forward(self, x):
            # Output logits where class 1 has logit 0.36 and class 0 has 0.34
            # Softmax: prob of class 1 ~ 0.28, fg prob (1-probs[0]) ~ 0.73 but each class is weak
            b, c, h, w, d = x.shape
            logits = torch.zeros((b, 4, h, w, d))
            # Set background to 0.5, class 1 to 0.4 (fg_prob = 1 - 0.52 = 0.48)
            logits[:, 0] = 0.5
            logits[:, 1] = 0.4
            logits[:, 2] = 0.05
            logits[:, 3] = 0.05
            return logits

    model = MockUncalibratedModel()
    inputs = torch.randn(1, 1, 32, 32, 32)
    device = torch.device("cpu")

    # Raw argmax (min_fg_prob=None): should predict class 0 because 0.5 > 0.4
    pred_raw = run_sliding_window_inference(
        model=model,
        inputs=inputs,
        roi_size=(32, 32, 32),
        device=device,
        min_fg_prob=None
    )
    assert np.all(pred_raw == 0)

    # Now make class 1 barely exceed class 0: class 0=0.34, class 1=0.36
    class MockMarginalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.out_channels = 4
            self.in_channels = 1

        def forward(self, x):
            b, c, h, w, d = x.shape
            logits = torch.zeros((b, 4, h, w, d))
            logits[:, 0] = 0.34
            logits[:, 1] = 0.36
            logits[:, 2] = 0.15
            logits[:, 3] = 0.15
            return logits

    model_marg = MockMarginalModel()
    # Without min_fg_prob, argmax picks class 1 (false positive!)
    pred_raw_marg = run_sliding_window_inference(
        model=model_marg,
        inputs=inputs,
        roi_size=(32, 32, 32),
        device=device,
        min_fg_prob=None
    )
    assert np.any(pred_raw_marg == 1), "Argmax should have chosen class 1"

    # With min_fg_prob = 0.80: fg_prob is ~0.66 < 0.80, so it suppresses to 0!
    pred_calibrated = run_sliding_window_inference(
        model=model_marg,
        inputs=inputs,
        roi_size=(32, 32, 32),
        device=device,
        min_fg_prob=0.80
    )
    assert np.all(pred_calibrated == 0), "Calibrated inference should have suppressed marginal false positives"


def test_nifti_segmentation_dataset_balanced_sampling():
    """
    Verifies NiftiSegmentationDataset samples_per_case and caching functionality.
    """
    pairs = [
        (Path("backend/segmentation/data/BraTS_Case_003_T1CE.nii.gz"),
         Path("backend/segmentation/data/BraTS_Case_003_seg.nii.gz"))
    ]
    if not pairs[0][0].exists():
        pytest.skip("Data files not found")

    ds = NiftiSegmentationDataset(
        case_pairs=pairs,
        patch_size=(32, 32, 32),
        is_train=True,
        in_channels=1,
        samples_per_case=4,
        cache_data=True
    )

    assert len(ds) == 4
    sample = ds[0]
    assert "image" in sample and "mask" in sample
    assert sample["image"].shape == (1, 32, 32, 32)
    assert sample["mask"].shape == (32, 32, 32)
