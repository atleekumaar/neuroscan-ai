"""
NEUROSCAN AI — Unit Tests for Hard-Negative Patch Sampler (Stage 5 / Exp 3)
Verifies anatomical candidate region derivation, zero-tumor contamination in negative patches,
reproducibility across seeds, and strict split isolation.
"""

import pytest
import numpy as np
import torch

from backend.segmentation.sampler import (
    HardNegativeCategory,
    HardNegativeSamplerConfig,
    HardNegativeSampler,
    PatchMetadata
)


def create_synthetic_brain_volume(shape=(128, 128, 128)):
    """Creates a synthetic multi-channel brain volume with an intracranial sphere and tumor."""
    d, h, w = shape
    unnorm_img = np.zeros((4, d, h, w), dtype=np.float32)
    seg = np.zeros((1, 1, d, h, w), dtype=np.int64)

    # Spherical brain mask
    cz, cy, cx = d // 2, h // 2, w // 2
    r_brain = 45
    zz, yy, xx = np.ogrid[:d, :h, :w]
    dist_brain = np.sqrt((zz - cz)**2 + (yy - cy)**2 + (xx - cx)**2)
    brain_voxels = dist_brain <= r_brain

    # Background tissue intensities
    unnorm_img[0, brain_voxels] = 100.0   # T1n: grey matter
    unnorm_img[1, brain_voxels] = 110.0   # T1c
    unnorm_img[2, brain_voxels] = 90.0    # T2w
    unnorm_img[3, brain_voxels] = 95.0    # T2f

    # Ventricles (central CSF: low T1, high T2)
    r_ventricle = 10
    ventricle_voxels = dist_brain <= r_ventricle
    unnorm_img[0, ventricle_voxels] = 20.0   # T1 hypointense
    unnorm_img[2, ventricle_voxels] = 220.0  # T2 hyperintense

    # Tumor in right hemisphere (e.g. z < cz - 15)
    r_tumor = 12
    tz, ty, tx = cz - 22, cy + 5, cx + 5
    dist_tumor = np.sqrt((zz - tz)**2 + (yy - ty)**2 + (xx - tx)**2)
    tumor_voxels = dist_tumor <= r_tumor

    # Label 1: NCR, Label 2: ED, Label 3: ET
    seg[0, 0, tumor_voxels] = 2
    seg[0, 0, dist_tumor <= 6] = 1
    seg[0, 0, dist_tumor <= 3] = 3

    unnorm_img[1, tumor_voxels] = 250.0  # T1c enhancing

    norm_img = unnorm_img.copy()
    for c in range(4):
        m = norm_img[c][brain_voxels].mean()
        s = norm_img[c][brain_voxels].std() + 1e-5
        norm_img[c] = (norm_img[c] - m) / s

    return (
        torch.from_numpy(norm_img).unsqueeze(0),
        torch.from_numpy(seg),
        torch.from_numpy(unnorm_img).unsqueeze(0)
    )


def test_sampler_config_defaults():
    cfg = HardNegativeSamplerConfig()
    assert cfg.patch_size == (48, 48, 48)
    assert cfg.n_foreground == 2
    assert cfg.n_extra_axial == 1
    assert cfg.n_ventricular == 1
    assert cfg.n_contralateral == 1
    assert cfg.n_general_background == 1
    assert cfg.total_patches_per_volume == 6


def test_derive_candidate_regions_synthetic():
    norm_img, seg, unnorm_img = create_synthetic_brain_volume()
    sampler = HardNegativeSampler(HardNegativeSamplerConfig(patch_size=(32, 32, 32)))

    regions = sampler.derive_candidate_regions(unnorm_img[0].numpy(), seg[0].numpy())

    # Verify all 5 categories are present
    for cat in HardNegativeCategory:
        assert cat in regions
        assert isinstance(regions[cat], np.ndarray)
        assert regions[cat].dtype == bool

    # Verify foreground contains true tumor voxels
    tumor_gt = (seg[0, 0] > 0).numpy()
    assert np.array_equal(regions[HardNegativeCategory.TUMOR_FOREGROUND], tumor_gt)

    # Verify negative regions are strictly disjoint from raw tumor
    assert not np.any(regions[HardNegativeCategory.EXTRA_AXIAL_BOUNDARY] & tumor_gt)
    assert not np.any(regions[HardNegativeCategory.VENTRICULAR_CSF] & tumor_gt)
    assert not np.any(regions[HardNegativeCategory.CONTRALATERAL_PARENCHYMA] & tumor_gt)
    assert not np.any(regions[HardNegativeCategory.GENERAL_BACKGROUND] & tumor_gt)


def test_sample_patches_shapes_and_counts():
    norm_img, seg, unnorm_img = create_synthetic_brain_volume()
    sampler = HardNegativeSampler(HardNegativeSamplerConfig(patch_size=(32, 32, 32)))

    patches_x, patches_y, metas = sampler.sample_patches_for_study(
        norm_image=norm_img,
        seg=seg,
        unnorm_image=unnorm_img,
        patient_id="SYNTHETIC-TRAIN-001",
        seed=100
    )

    assert patches_x.shape == (6, 4, 32, 32, 32)
    assert patches_y.shape == (6, 1, 32, 32, 32)
    assert len(metas) == 6

    # Verify categories extracted
    categories = [m.category for m in metas]
    assert categories.count(HardNegativeCategory.TUMOR_FOREGROUND.value) == 2
    assert categories.count(HardNegativeCategory.EXTRA_AXIAL_BOUNDARY.value) == 1
    assert categories.count(HardNegativeCategory.VENTRICULAR_CSF.value) == 1
    assert categories.count(HardNegativeCategory.CONTRALATERAL_PARENCHYMA.value) == 1
    assert categories.count(HardNegativeCategory.GENERAL_BACKGROUND.value) == 1


def test_zero_tumor_contamination_in_negative_patches():
    norm_img, seg, unnorm_img = create_synthetic_brain_volume()
    sampler = HardNegativeSampler(HardNegativeSamplerConfig(patch_size=(32, 32, 32)))

    patches_x, patches_y, metas = sampler.sample_patches_for_study(
        norm_image=norm_img,
        seg=seg,
        unnorm_image=unnorm_img,
        patient_id="SYNTHETIC-TRAIN-001",
        seed=123
    )

    for i, meta in enumerate(metas):
        py = patches_y[i, 0]
        if meta.category != HardNegativeCategory.TUMOR_FOREGROUND.value:
            assert py.sum().item() == 0, f"Negative patch {i} ({meta.category}) contains tumor voxels!"
            assert meta.has_tumor is False
            assert meta.tumor_voxel_count == 0
        else:
            assert py.sum().item() > 0, f"Foreground patch {i} should contain tumor voxels"
            assert meta.has_tumor is True


def test_deterministic_reproducibility():
    norm_img, seg, unnorm_img = create_synthetic_brain_volume()
    sampler = HardNegativeSampler(HardNegativeSamplerConfig(patch_size=(32, 32, 32)))

    px1, py1, m1 = sampler.sample_patches_for_study(norm_img, seg, unnorm_img, "P01", seed=42)
    px2, py2, m2 = sampler.sample_patches_for_study(norm_img, seg, unnorm_img, "P01", seed=42)

    assert torch.allclose(px1, px2), "Patches with same seed must be identical"
    assert torch.equal(py1, py2), "Seg masks with same seed must be identical"
    assert [m.center_coord for m in m1] == [m.center_coord for m in m2]


def test_different_seeds_produce_distinct_patches():
    norm_img, seg, unnorm_img = create_synthetic_brain_volume()
    sampler = HardNegativeSampler(HardNegativeSamplerConfig(patch_size=(32, 32, 32)))

    _, _, m1 = sampler.sample_patches_for_study(norm_img, seg, unnorm_img, "P01", seed=42)
    _, _, m2 = sampler.sample_patches_for_study(norm_img, seg, unnorm_img, "P01", seed=999)

    coords1 = [m.center_coord for m in m1]
    coords2 = [m.center_coord for m in m2]
    assert coords1 != coords2, "Different seeds should sample different coordinates"

