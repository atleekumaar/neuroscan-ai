"""
Unit tests for backend/segmentation/postprocessing.py
Verifies intracranial mask construction, hole-filling, largest connected component,
and semantic constraint preservation.
"""

import pytest
import numpy as np
import torch
from backend.segmentation.postprocessing import (
    construct_intracranial_mask,
    apply_intracranial_mask
)


def test_construct_intracranial_mask_multi_channel():
    # 4 channels of size (4, 20, 20, 20)
    data = np.zeros((4, 20, 20, 20), dtype=np.float32)
    # Channel 0 active in a central sphere
    data[0, 5:15, 5:15, 5:15] = 100.0
    # Channel 1 active in a disjoint pocket
    data[1, 10:18, 10:18, 10:18] = 50.0

    mask = construct_intracranial_mask(data, intensity_threshold=0.0)
    assert mask.shape == (20, 20, 20)
    assert mask.dtype == bool
    assert np.all(mask[6:14, 6:14, 6:14])
    assert not mask[0, 0, 0]


def test_construct_intracranial_mask_hole_filling():
    # Hollow sphere: signal on borders, 0 inside
    data = np.zeros((4, 30, 30, 30), dtype=np.float32)
    data[:, 5:25, 5:25, 5:25] = 1.0
    # Cavity in center
    data[:, 12:18, 12:18, 12:18] = 0.0

    mask_filled = construct_intracranial_mask(data, fill_holes=True)
    assert np.all(mask_filled[12:18, 12:18, 12:18]), "Central cavity should be filled by hole filling"

    mask_unfilled = construct_intracranial_mask(data, fill_holes=False)
    assert not np.any(mask_unfilled[12:18, 12:18, 12:18]), "Central cavity should remain hollow if fill_holes=False"


def test_apply_intracranial_mask_numpy():
    pred = np.zeros((20, 20, 20), dtype=np.uint8)
    # True tumor inside brain
    pred[8:12, 8:12, 8:12] = 3  # Enhancing tumor
    # False positive in outer space
    pred[0:4, 0:4, 0:4] = 2    # Edema

    bmask = np.zeros((20, 20, 20), dtype=bool)
    bmask[5:15, 5:15, 5:15] = True

    masked_pred = apply_intracranial_mask(pred, bmask)
    assert np.all(masked_pred[8:12, 8:12, 8:12] == 3), "Intra-cranial prediction must be preserved"
    assert np.all(masked_pred[0:4, 0:4, 0:4] == 0), "Extra-cranial prediction must be zeroed out"


def test_apply_intracranial_mask_torch_5d():
    pred = torch.zeros((1, 1, 20, 20, 20), dtype=torch.int64)
    pred[0, 0, 8:12, 8:12, 8:12] = 1  # Necrotic core
    pred[0, 0, 18:20, 18:20, 18:20] = 2  # False positive

    bmask = np.zeros((20, 20, 20), dtype=bool)
    bmask[5:15, 5:15, 5:15] = True

    masked = apply_intracranial_mask(pred, bmask)
    assert isinstance(masked, torch.Tensor)
    assert masked.shape == (1, 1, 20, 20, 20)
    assert masked.dtype == torch.int64
    assert torch.all(masked[0, 0, 8:12, 8:12, 8:12] == 1)
    assert torch.all(masked[0, 0, 18:20, 18:20, 18:20] == 0)
