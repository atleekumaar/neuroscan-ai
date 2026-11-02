"""
NEUROSCAN AI — Phase 5.11: Unit & Integration Tests for Four-Channel Multimodal Pipeline
Covers:
  1. Strict modality ordering: ("t1n", "t1c", "t2w", "t2f") in channels 0..3
  2. Non-zero z-score intensity normalization
  3. Spatial alignment and dimension mismatch rejection
  4. Segmentation label semantics & legacy label 4->3 remapping
  5. 3D spatial patch extraction and patch batch shapes
  6. Model 4-channel construction and forward pass tensor shape
  7. Full sliding-window inference output shape (1, 1, H, W, D) and discrete label classes
  8. Integration test loading a real physical pilot study on disk
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import torch
import nibabel as nib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.segmentation.model import (
    build_four_channel_unet,
    FourChannelUNetConfig,
    build_segmentation_model
)
from backend.segmentation.pipeline import (
    MODALITY_ORDER,
    load_multimodal_study,
    normalize_intensity,
    extract_3d_patches,
    predict_multimodal_study
)


def create_synthetic_nifti(
    tmp_path: Path,
    filename: str,
    shape=(64, 64, 32),
    affine=np.eye(4),
    data=None
) -> Path:
    fpath = tmp_path / filename
    if data is None:
        data = np.random.uniform(10, 100, shape).astype(np.float32)
    img = nib.Nifti1Image(data, affine, dtype=data.dtype)
    nib.save(img, str(fpath))
    return fpath


def test_modality_ordering_and_stacking(tmp_path):
    """Verify that t1n, t1c, t2w, t2f are strictly assigned to channels 0, 1, 2, 3."""
    assert MODALITY_ORDER == ("t1n", "t1c", "t2w", "t2f")

    # Create distinct constant signals per modality
    signals = {"t1n": 10.0, "t1c": 20.0, "t2w": 30.0, "t2f": 40.0}
    file_map = {}
    for m, val in signals.items():
        arr = np.full((32, 32, 16), val, dtype=np.float32)
        # Leave a zero background so non-zero normalization maintains relative constants
        arr[0, 0, 0] = 0.0
        file_map[m] = create_synthetic_nifti(tmp_path, f"study_{m}.nii.gz", shape=(32, 32, 16), data=arr)

    # Load unnormalized to verify channel content
    image_tensor, seg_tensor, meta = load_multimodal_study(file_map, normalize=False)

    assert image_tensor.shape == (1, 4, 32, 32, 16)
    assert seg_tensor is None
    assert meta["modality_order"] == ["t1n", "t1c", "t2w", "t2f"]

    # Verify each channel matches its modality constant
    assert float(image_tensor[0, 0, 10, 10, 10]) == 10.0  # t1n -> Ch 0
    assert float(image_tensor[0, 1, 10, 10, 10]) == 20.0  # t1c -> Ch 1
    assert float(image_tensor[0, 2, 10, 10, 10]) == 30.0  # t2w -> Ch 2
    assert float(image_tensor[0, 3, 10, 10, 10]) == 40.0  # t2f -> Ch 3


def test_intensity_normalization():
    """Verify non-zero intracranial z-score normalization."""
    arr = np.zeros((20, 20, 20), dtype=np.float32)
    arr[5:15, 5:15, 5:15] = np.random.normal(loc=100.0, scale=15.0, size=(10, 10, 10)).astype(np.float32)

    normed = normalize_intensity(arr)

    # Zeros must remain strictly zero
    assert np.all(normed[arr == 0] == 0.0)

    # Non-zero voxels must have zero mean and unit variance
    nz_vals = normed[arr > 0]
    assert np.isclose(np.mean(nz_vals), 0.0, atol=1e-2)
    assert np.isclose(np.std(nz_vals), 1.0, atol=1e-2)


def test_spatial_alignment_rejection(tmp_path):
    """Verify dimension or affine mismatches between modalities are strictly rejected."""
    # Create matching t1n, t1c, t2w
    file_map = {
        "t1n": create_synthetic_nifti(tmp_path, "s_t1n.nii.gz", shape=(32, 32, 16)),
        "t1c": create_synthetic_nifti(tmp_path, "s_t1c.nii.gz", shape=(32, 32, 16)),
        "t2w": create_synthetic_nifti(tmp_path, "s_t2w.nii.gz", shape=(32, 32, 16)),
    }
    # Create mismatched t2f (shape 32, 32, 20)
    file_map["t2f"] = create_synthetic_nifti(tmp_path, "s_t2f.nii.gz", shape=(32, 32, 20))

    with pytest.raises(ValueError, match="Dimension mismatch"):
        load_multimodal_study(file_map)


def test_affine_alignment_rejection(tmp_path):
    """Verify affine orientation differences between modalities are rejected."""
    affine1 = np.eye(4)
    affine2 = np.diag([-1, 1, 1, 1])

    file_map = {
        "t1n": create_synthetic_nifti(tmp_path, "a_t1n.nii.gz", shape=(16, 16, 16), affine=affine1),
        "t1c": create_synthetic_nifti(tmp_path, "a_t1c.nii.gz", shape=(16, 16, 16), affine=affine1),
        "t2w": create_synthetic_nifti(tmp_path, "a_t2w.nii.gz", shape=(16, 16, 16), affine=affine1),
        "t2f": create_synthetic_nifti(tmp_path, "a_t2f.nii.gz", shape=(16, 16, 16), affine=affine2),
    }

    with pytest.raises(ValueError, match="Affine alignment mismatch"):
        load_multimodal_study(file_map)


def test_label_semantics_and_remapping(tmp_path):
    """Verify ground truth segmentation loads with shape (1, 1, H, W, D) and remaps legacy 4->3."""
    file_map = {}
    for m in MODALITY_ORDER:
        file_map[m] = create_synthetic_nifti(tmp_path, f"lbl_{m}.nii.gz", shape=(32, 32, 16))

    # Create segmentation mask containing labels 0, 1, 2, and legacy 4
    seg_arr = np.zeros((32, 32, 16), dtype=np.int16)
    seg_arr[5:10, 5:10, 5:10] = 1   # Necrotic
    seg_arr[10:15, 10:15, 5:10] = 2  # Edema
    seg_arr[15:20, 15:20, 5:10] = 4  # Legacy Enhancing Tumor
    file_map["seg"] = create_synthetic_nifti(tmp_path, "lbl_seg.nii.gz", shape=(32, 32, 16), data=seg_arr)

    image_tensor, seg_tensor, meta = load_multimodal_study(file_map)

    assert seg_tensor is not None
    assert seg_tensor.shape == (1, 1, 32, 32, 16)
    assert seg_tensor.dtype == torch.int64

    # Verify label 4 was converted to 3
    unique_labels = torch.unique(seg_tensor).tolist()
    assert 4 not in unique_labels
    assert set(unique_labels) == {0, 1, 2, 3}


def test_patch_extraction():
    """Verify 3D patch extraction produces expected batch dimensions."""
    img = torch.randn(1, 4, 96, 96, 64)
    seg = torch.randint(0, 4, (1, 1, 96, 96, 64))

    patches_img, patches_seg = extract_3d_patches(img, seg, patch_size=(32, 32, 32), max_patches=4)

    assert patches_img.shape == (4, 4, 32, 32, 32)
    assert patches_seg.shape == (4, 1, 32, 32, 32)


def test_four_channel_model_architecture():
    """Verify 4-channel model instantiation and parameterization."""
    config = FourChannelUNetConfig(
        in_channels=4,
        out_channels=4,
        channels=(8, 16, 32, 64, 128)
    )
    model = build_four_channel_unet(config)

    assert model.in_channels == 4
    assert model.out_channels == 4

    # Dummy input
    dummy_input = torch.randn(2, 4, 32, 32, 32)
    with torch.no_grad():
        out = model(dummy_input)

    assert out.shape == (2, 4, 32, 32, 32)


def test_predict_multimodal_study_sliding_window():
    """Verify sliding-window prediction produces discrete mask (1, 1, H, W, D) with {0, 1, 2, 3}."""
    config = FourChannelUNetConfig(
        in_channels=4,
        out_channels=4,
        channels=(8, 16, 32, 64, 128)
    )
    model = build_four_channel_unet(config)

    # Input tensor of full shape
    img = torch.randn(1, 4, 64, 64, 32)
    pred_mask = predict_multimodal_study(model, img, roi_size=(32, 32, 32), overlap=0.5)

    assert pred_mask.shape == (1, 1, 64, 64, 32)
    assert pred_mask.dtype == torch.int64

    # All predicted classes must be in {0, 1, 2, 3}
    pred_classes = set(torch.unique(pred_mask).tolist())
    assert pred_classes.issubset({0, 1, 2, 3})


def test_real_pilot_study_loading():
    """Integration test: Load a genuine physical pilot study from disk."""
    pilot_dir = PROJECT_ROOT / "backend" / "segmentation" / "data" / "pilot" / "BraTS-GLI-00324-000"
    if not pilot_dir.exists():
        pytest.skip("Pilot directory BraTS-GLI-00324-000 not present on disk.")

    image_tensor, seg_tensor, meta = load_multimodal_study(pilot_dir)

    assert image_tensor.shape == (1, 4, 240, 240, 155)
    assert seg_tensor is not None
    assert seg_tensor.shape == (1, 1, 240, 240, 155)
    assert meta["modality_order"] == ["t1n", "t1c", "t2w", "t2f"]
    assert meta["dimensions"] == [240, 240, 155]
    assert set(meta["seg_unique_labels"]) == {0, 1, 2, 3}
