"""
NEUROSCAN AI — Intracranial Post-Processing and Masking Module
Provides separate, optional post-processing utilities to constrain 3D volumetric
predictions within anatomical intracranial boundaries.
Never modifies model weights, training pipelines, or baseline inference behavior.
"""

from typing import Union, Tuple, Optional
import numpy as np
import torch
from scipy import ndimage


def construct_intracranial_mask(
    image: Union[np.ndarray, torch.Tensor],
    intensity_threshold: float = 0.0,
    fill_holes: bool = True,
    largest_component_only: bool = True
) -> np.ndarray:
    """
    Constructs an anatomically grounded 3D intracranial brain mask from multi-parametric MRI sequences.
    
    Robustness guarantees:
      1. Aggregates signal across all available modality channels (e.g. T1n, T1c, T2w, T2f).
         Regions hypointense on T1 (e.g., CSF, ventricles) are covered by hyperintense T2/FLAIR.
      2. 3D binary hole filling ensures internal necrotic cores, cysts, and deep ventricles
         are not hollowed out or excluded from the valid brain parenchyma.
      3. Largest connected component extraction removes disconnected noise or extra-axial artifacts.

    Parameters:
      image: Multi-channel volume array of shape (C, H, W, D) or (1, C, H, W, D).
      intensity_threshold: Minimum intensity to consider as foreground tissue signal.
                           Default is 0.0 (exact zero-padded background boundary).
      fill_holes: Whether to execute 3D binary hole filling.
      largest_component_only: Whether to retain only the largest connected 3D anatomical volume.

    Returns:
      Boolean numpy array of shape (H, W, D) where True indicates intracranial tissue.
    """
    if isinstance(image, torch.Tensor):
        img_np = image.detach().cpu().numpy()
    else:
        img_np = np.asarray(image)

    # Squeeze batch dimension if present
    if img_np.ndim == 5:
        if img_np.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got shape {img_np.shape}")
        img_np = img_np[0]

    if img_np.ndim != 4:
        raise ValueError(f"Expected 4D array (C, H, W, D), got shape {img_np.shape}")

    c, h, w, d = img_np.shape

    # Union of active signal across all modality channels
    # In clinical BraTS NIfTI volumes, voxels outside the skull are strictly zero-padded.
    active_union = np.zeros((h, w, d), dtype=bool)
    for ch in range(c):
        active_union |= (img_np[ch] > intensity_threshold)

    mask = active_union

    # 3D binary hole filling to preserve ventricles and hypo-intense necrotic centers
    if fill_holes and np.any(mask):
        mask = ndimage.binary_fill_holes(mask)

    # Largest connected component to eliminate stray voxels in peripheral background
    if largest_component_only and np.any(mask):
        labeled, num_features = ndimage.label(mask)
        if num_features > 1:
            component_sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
            largest_label = int(np.argmax(component_sizes) + 1)
            mask = (labeled == largest_label)

    return mask.astype(bool)


def apply_intracranial_mask(
    pred_mask: Union[np.ndarray, torch.Tensor],
    brain_mask: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Applies an intracranial mask as a zero-parameter post-processing constraint.
    Zeroes out any predicted positive class residing outside the brain boundary.
    Preserves exact multi-class label semantics {0: BG, 1: NCR, 2: ED, 3: ET}
    and canonical spatial orientation.

    Parameters:
      pred_mask: Discrete prediction mask (H, W, D) or (1, 1, H, W, D) with integer classes.
      brain_mask: Intracranial tissue mask of matching spatial dimensions (H, W, D).

    Returns:
      Masked prediction array/tensor of identical shape, dtype, and memory device.
    """
    is_torch = isinstance(pred_mask, torch.Tensor)
    if is_torch:
        pred_tensor = pred_mask
        pred_np = pred_tensor.detach().cpu().numpy()
        device = pred_tensor.device
        dtype = pred_tensor.dtype
    else:
        pred_np = np.asarray(pred_mask)

    if isinstance(brain_mask, torch.Tensor):
        bmask_np = brain_mask.detach().cpu().numpy().astype(bool)
    else:
        bmask_np = np.asarray(brain_mask, dtype=bool)

    # Match spatial dimensions
    orig_shape = pred_np.shape
    if pred_np.ndim == 5 and pred_np.shape[:2] == (1, 1):
        p_3d = pred_np[0, 0]
        was_5d = True
    elif pred_np.ndim == 3:
        p_3d = pred_np
        was_5d = False
    else:
        raise ValueError(f"Unsupported prediction dimensions: {pred_np.shape}. Expected (H, W, D) or (1, 1, H, W, D).")

    if bmask_np.ndim == 5 and bmask_np.shape[:2] == (1, 1):
        bmask_np = bmask_np[0, 0]

    if p_3d.shape != bmask_np.shape:
        raise ValueError(f"Shape mismatch: prediction {p_3d.shape} vs brain mask {bmask_np.shape}")

    # Gated output: keep predicted class inside brain mask, set to 0 (background) outside
    constrained_3d = np.where(bmask_np, p_3d, 0).astype(p_3d.dtype)

    if was_5d:
        result_np = constrained_3d[np.newaxis, np.newaxis, ...]
    else:
        result_np = constrained_3d

    if is_torch:
        return torch.from_numpy(result_np).to(dtype=dtype, device=device)
    return result_np
