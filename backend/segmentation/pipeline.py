"""
NEUROSCAN AI — NIfTI Processing, Spatial Alignment & Inference Pipeline
Handles .nii/.nii.gz volumes, non-zero intensity normalization,
patch-based sliding window inference, and multi-planar slice generation.
"""

import os
import io
import base64
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union
import numpy as np
import torch
from PIL import Image

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False

try:
    from monai.inferers import sliding_window_inference as monai_sw_inference
    MONAI_INFERER_AVAILABLE = True
except ImportError:
    MONAI_INFERER_AVAILABLE = False


class NiftiVolume:
    """Encapsulates a loaded NIfTI 3D/4D volume with spatial geometry and sanitized headers."""
    def __init__(
        self,
        data: np.ndarray,
        affine: np.ndarray,
        voxel_spacing: Tuple[float, float, float],
        shape: Tuple[int, ...],
        modality_channels: int
    ):
        self.data = data  # Shape (C, H, W, D)
        self.affine = affine
        self.voxel_spacing = voxel_spacing  # (dx, dy, dz) in mm
        self.shape = shape
        self.modality_channels = modality_channels


def load_nifti_file(file_path: Union[str, Path]) -> NiftiVolume:
    """
    Safely loads a .nii or .nii.gz file, extracting spatial affine,
    voxel zooms, and channel structure while anonymizing patient metadata.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {path}")

    if not (path.name.endswith(".nii") or path.name.endswith(".nii.gz")):
        raise ValueError(f"Unsupported file format. Must be .nii or .nii.gz: {path.name}")

    if not NIBABEL_AVAILABLE:
        raise RuntimeError("nibabel is required to load NIfTI volumes. Please ensure it is installed.")

    img = nib.load(str(path))
    # Conform volume to canonical RAS+ spatial coordinate system
    img = nib.as_closest_canonical(img)
    # Conform data to float32
    raw_data = np.asarray(img.dataobj, dtype=np.float32)
    affine = np.asarray(img.affine, dtype=np.float32)

    # Voxel spacing from affine or zooms
    try:
        zooms = img.header.get_zooms()
        dx = float(zooms[0]) if len(zooms) > 0 else 1.0
        dy = float(zooms[1]) if len(zooms) > 1 else 1.0
        dz = float(zooms[2]) if len(zooms) > 2 else 1.0
    except Exception:
        # Compute from affine
        dx = float(np.sqrt(np.sum(affine[:3, 0] ** 2)))
        dy = float(np.sqrt(np.sum(affine[:3, 1] ** 2)))
        dz = float(np.sqrt(np.sum(affine[:3, 2] ** 2)))

    voxel_spacing = (round(dx, 3), round(dy, 3), round(dz, 3))

    # Standardize shape to (C, H, W, D)
    if raw_data.ndim == 3:
        # Single modality (e.g. T1ce or FLAIR)
        data = raw_data[np.newaxis, ...]
        modality_channels = 1
    elif raw_data.ndim == 4:
        # Could be (H, W, D, C) or (C, H, W, D)
        if raw_data.shape[3] in [1, 2, 3, 4] and raw_data.shape[0] > 10:
            # Transpose (H, W, D, C) -> (C, H, W, D)
            data = np.transpose(raw_data, (3, 0, 1, 2))
        else:
            data = raw_data
        modality_channels = data.shape[0]
    else:
        raise ValueError(f"Invalid volume dimensions: {raw_data.ndim}D. Expected 3D or 4D NIfTI.")

    return NiftiVolume(
        data=data,
        affine=affine,
        voxel_spacing=voxel_spacing,
        shape=data.shape,
        modality_channels=modality_channels
    )


def normalize_intensity(channel_data: np.ndarray) -> np.ndarray:
    """
    Z-score intensity normalization restricted to non-zero intracranial voxels.
    Prevents background zero-padding from skewing brain tissue contrast.
    """
    normalized = np.zeros_like(channel_data, dtype=np.float32)
    non_zero_mask = channel_data > 0

    if np.any(non_zero_mask):
        mean = float(np.mean(channel_data[non_zero_mask]))
        std = float(np.std(channel_data[non_zero_mask]))
        if std > 1e-5:
            normalized[non_zero_mask] = (channel_data[non_zero_mask] - mean) / std
        else:
            normalized[non_zero_mask] = channel_data[non_zero_mask] - mean
    return normalized


def preprocess_volume_for_model(
    nifti_vol: NiftiVolume,
    target_channels: int = 4
) -> torch.Tensor:
    """
    Applies per-channel normalization and conforms channels to target_channels.
    If input has 1 channel and target is 4, replicates across modalities.
    Returns torch.Tensor of shape (1, C, H, W, D).
    """
    channels_data = []
    for c in range(nifti_vol.modality_channels):
        norm_c = normalize_intensity(nifti_vol.data[c])
        channels_data.append(norm_c)

    # Replicate or slice to match target_channels
    if len(channels_data) == 1 and target_channels == 4:
        processed_data = np.repeat(channels_data[0][np.newaxis, ...], 4, axis=0)
    elif len(channels_data) >= target_channels:
        processed_data = np.stack(channels_data[:target_channels], axis=0)
    else:
        # Pad remaining channels with zeros
        padded = np.zeros((target_channels, *nifti_vol.data.shape[1:]), dtype=np.float32)
        for i, c_data in enumerate(channels_data):
            padded[i] = c_data
        processed_data = padded

    tensor = torch.from_numpy(processed_data).float().unsqueeze(0)  # (1, C, H, W, D)
    return tensor


def native_sliding_window_inference(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    roi_size: Tuple[int, int, int] = (64, 64, 64),
    overlap: float = 0.5,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Native sliding-window 3D patch inference with overlap blending.
    Allows large 3D NIfTI inference on constrained memory hardware without CUDA OOM.
    """
    b, c, h, w, d = inputs.shape
    rh, rw, rd = roi_size

    # Ensure inputs fit roi_size with symmetric padding if volume is smaller
    pad_h = max(0, rh - h)
    pad_w = max(0, rw - w)
    pad_d = max(0, rd - d)
    if pad_h > 0 or pad_w > 0 or pad_d > 0:
        inputs = torch.nn.functional.pad(
            inputs,
            (0, pad_d, 0, pad_w, 0, pad_h),
            mode="constant",
            value=0
        )
        _, _, h, w, d = inputs.shape

    step_h = max(1, int(rh * (1 - overlap)))
    step_w = max(1, int(rw * (1 - overlap)))
    step_d = max(1, int(rd * (1 - overlap)))

    h_starts = list(range(0, max(1, h - rh + 1), step_h))
    if h_starts[-1] + rh < h:
        h_starts.append(h - rh)

    w_starts = list(range(0, max(1, w - rw + 1), step_w))
    if w_starts[-1] + rw < w:
        w_starts.append(w - rw)

    d_starts = list(range(0, max(1, d - rd + 1), step_d))
    if d_starts[-1] + rd < d:
        d_starts.append(d - rd)

    # Accumulator tensors
    out_channels = getattr(model, "out_channels", 4)
    output_accum = torch.zeros((b, out_channels, h, w, d), dtype=torch.float32, device=device)
    count_accum = torch.zeros((b, 1, h, w, d), dtype=torch.float32, device=device)

    # Cosine/Hanning patch weight window to prevent seam artifacts
    window_h = torch.hann_window(rh, periodic=False).view(rh, 1, 1)
    window_w = torch.hann_window(rw, periodic=False).view(1, rw, 1)
    window_d = torch.hann_window(rd, periodic=False).view(1, 1, rd)
    patch_weight = (window_h * window_w * window_d).to(device)
    patch_weight = torch.clamp(patch_weight, min=1e-3).unsqueeze(0).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        for hs in h_starts:
            for ws in w_starts:
                for ds in d_starts:
                    patch = inputs[:, :, hs:hs+rh, ws:ws+rw, ds:ds+rd].to(device)
                    # Forward pass
                    with torch.amp.autocast(device_type=device.type if device.type in ["cuda", "cpu"] else "cpu", enabled=(device.type == "cuda")):
                        logits = model(patch)
                    # Accumulate
                    output_accum[:, :, hs:hs+rh, ws:ws+rw, ds:ds+rd] += logits * patch_weight
                    count_accum[:, :, hs:hs+rh, ws:ws+rw, ds:ds+rd] += patch_weight

    output = output_accum / torch.clamp(count_accum, min=1e-5)
    # Crop back if padded
    orig_h = h - pad_h
    orig_w = w - pad_w
    orig_d = d - pad_d
    return output[:, :, :orig_h, :orig_w, :orig_d]


def run_sliding_window_inference(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    roi_size: Tuple[int, int, int] = (64, 64, 64),
    overlap: float = 0.5,
    device: torch.device = torch.device("cpu"),
    brain_mask: Optional[np.ndarray] = None,
    min_fg_prob: Optional[float] = None
) -> np.ndarray:
    """
    Executes sliding window inference using MONAI when available or native patch engine.
    Returns argmax discrete mask numpy array of shape (H, W, D).
    If brain_mask is provided, suppresses false positives outside the intracranial parenchyma.
    If min_fg_prob is provided, suppresses predictions where total foreground probability is below threshold.
    """
    model.eval()
    inputs = inputs.to(device)

    if MONAI_INFERER_AVAILABLE:
        try:
            with torch.no_grad():
                with torch.amp.autocast(device_type=device.type if device.type == "cuda" else "cpu", enabled=(device.type == "cuda")):
                    logits = monai_sw_inference(
                        inputs=inputs,
                        roi_size=roi_size,
                        sw_batch_size=2,
                        predictor=model,
                        overlap=overlap,
                        mode="gaussian"
                    )
        except Exception:
            logits = native_sliding_window_inference(model, inputs, roi_size, overlap, device)
    else:
        logits = native_sliding_window_inference(model, inputs, roi_size, overlap, device)

    if min_fg_prob is not None and min_fg_prob > 0.0:
        probs = torch.softmax(logits, dim=1).squeeze(0)  # (4, H, W, D)
        fg_prob = (1.0 - probs[0]).cpu().numpy()
        argmax_classes = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        pred_mask = np.where(fg_prob >= min_fg_prob, argmax_classes, 0).astype(np.uint8)
    else:
        pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    if brain_mask is not None:
        # Zero out predictions outside intracranial tissue mask
        mask_bool = np.asarray(brain_mask > 0)
        if mask_bool.shape == pred_mask.shape:
            pred_mask = np.where(mask_bool, pred_mask, 0).astype(np.uint8)

    return pred_mask


def extract_slice_rgba(
    mri_slice: np.ndarray,
    mask_slice: Optional[np.ndarray] = None,
    color_map: Optional[Dict[int, List[int]]] = None
) -> Tuple[str, Optional[str]]:
    """
    Converts a single 2D MRI slice (and optional segmentation mask)
    into base64 encoded PNG data URLs for fast frontend canvas rendering.
    """
    # Robust percentile-based intensity normalization for authentic clinical MRI contrast
    non_zero = mri_slice[mri_slice > 0]
    if len(non_zero) > 10:
        p_min = float(np.percentile(non_zero, 1.0))
        p_max = float(np.percentile(non_zero, 99.5))
        if p_max - p_min > 1e-3:
            scaled = np.clip((mri_slice - p_min) / (p_max - p_min) * 255.0, 0, 255).astype(np.uint8)
        else:
            scaled = np.zeros_like(mri_slice, dtype=np.uint8)
    else:
        p_min = float(np.min(mri_slice))
        p_max = float(np.max(mri_slice))
        if p_max - p_min > 1e-4:
            scaled = np.clip((mri_slice - p_min) / (p_max - p_min) * 255.0, 0, 255).astype(np.uint8)
        else:
            scaled = np.zeros_like(mri_slice, dtype=np.uint8)

    img_pil = Image.fromarray(scaled, mode="L").convert("RGB")
    buf_mri = io.BytesIO()
    img_pil.save(buf_mri, format="PNG")
    mri_b64 = "data:image/png;base64," + base64.b64encode(buf_mri.getvalue()).decode("utf-8")

    mask_b64 = None
    if mask_slice is not None:
        h, w = mask_slice.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        default_colors = {
            1: [255, 69, 58, 210],    # NCR (Red)
            2: [57, 213, 255, 190],   # ED (Cyan)
            3: [255, 214, 10, 225],   # ET (Yellow)
            4: [255, 214, 10, 225]    # ET alternative
        }
        cmap = color_map or default_colors
        for label, rgba_val in cmap.items():
            locs = (mask_slice == label)
            if np.any(locs):
                rgba[locs] = rgba_val

        mask_pil = Image.fromarray(rgba, mode="RGBA")
        buf_mask = io.BytesIO()
        mask_pil.save(buf_mask, format="PNG")
        mask_b64 = "data:image/png;base64," + base64.b64encode(buf_mask.getvalue()).decode("utf-8")

    return mri_b64, mask_b64


def generate_volume_slice_pack(
    mri_vol: np.ndarray,
    pred_mask: Optional[np.ndarray] = None,
    gt_mask: Optional[np.ndarray] = None,
    plane: str = "axial"  # "axial" (Z), "coronal" (Y), "sagittal" (X)
) -> Dict[str, Any]:
    """
    Generates structured slice metadata and image representations along specified plane
    under canonical RAS+ radiological coordinate convention.
    """
    # Assuming mri_vol shape (X, Y, Z)
    if plane == "axial":
        total_slices = mri_vol.shape[2]
        get_mri_slice = lambda idx: np.rot90(mri_vol[:, :, idx])
        get_pred_slice = (lambda idx: np.rot90(pred_mask[:, :, idx])) if pred_mask is not None else None
        get_gt_slice = (lambda idx: np.rot90(gt_mask[:, :, idx])) if gt_mask is not None else None
    elif plane == "coronal":
        total_slices = mri_vol.shape[1]
        get_mri_slice = lambda idx: np.rot90(mri_vol[:, idx, :])
        get_pred_slice = (lambda idx: np.rot90(pred_mask[:, idx, :])) if pred_mask is not None else None
        get_gt_slice = (lambda idx: np.rot90(gt_mask[:, idx, :])) if gt_mask is not None else None
    else:  # sagittal
        total_slices = mri_vol.shape[0]
        get_mri_slice = lambda idx: np.rot90(mri_vol[idx, :, :])
        get_pred_slice = (lambda idx: np.rot90(pred_mask[idx, :, :])) if pred_mask is not None else None
        get_gt_slice = (lambda idx: np.rot90(gt_mask[idx, :, :])) if gt_mask is not None else None

    # Identify slice with largest tumor cross-section as initial focus slice
    focus_slice = total_slices // 2
    max_tumor_pixels = 0
    active_eval_slice = get_gt_slice if get_gt_slice is not None else get_pred_slice
    if active_eval_slice is not None:
        for idx in range(total_slices):
            slice_m = active_eval_slice(idx)
            count = int(np.sum(slice_m > 0))
            if count > max_tumor_pixels:
                max_tumor_pixels = count
                focus_slice = idx

    return {
        "plane": plane,
        "total_slices": total_slices,
        "initial_focus_slice": focus_slice,
        "has_tumor_in_plane": max_tumor_pixels > 0
    }


# =========================================================================
# Four-Channel Multimodal 3D Pipeline Extensions (Phase 5.11)
# =========================================================================

MODALITY_ORDER: Tuple[str, ...] = ("t1n", "t1c", "t2w", "t2f")


def load_multimodal_study(
    study_path_or_dict: Union[str, Path, Dict[str, Union[str, Path]]],
    modality_order: Tuple[str, ...] = MODALITY_ORDER,
    normalize: bool = True
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
    """
    Loads and assembles a 4-channel multiparametric 3D MRI study into a standardized PyTorch tensor.
    Consumes modalities strictly in the documented canonical order: t1n, t1c, t2w, t2f.
    Validates spatial dimension, affine alignment, and finite intensities across all sequences.

    Returns:
      - image_tensor: Float32 Tensor of shape (1, 4, H, W, D)
      - seg_tensor: Optional Int64 Tensor of shape (1, 1, H, W, D) if 'seg' is available
      - metadata: Dict with affine, zooms, shapes, and modality paths
    """
    if isinstance(study_path_or_dict, (str, Path)):
        study_dir = Path(study_path_or_dict)
        if not study_dir.is_dir():
            raise NotADirectoryError(f"Study path is not a directory: {study_dir}")
        files = list(study_dir.glob("*.nii*"))
        file_map: Dict[str, Path] = {}
        for f in files:
            name_lower = f.name.lower()
            for m in list(modality_order) + ["seg"]:
                if f"-{m}." in name_lower or f"_{m}." in name_lower or f"_{m.upper()}." in f.name:
                    file_map[m] = f
    elif isinstance(study_path_or_dict, dict):
        file_map = {k: Path(v) if v else None for k, v in study_path_or_dict.items()}
    else:
        raise TypeError(f"Expected directory path or dict of modality paths, got {type(study_path_or_dict).__name__}")

    # Verify presence of all required 4 channels
    missing = [m for m in modality_order if m not in file_map or file_map[m] is None or not file_map[m].exists()]
    if missing:
        raise ValueError(f"Missing required modalities in study: {missing} (required order: {modality_order})")

    ref_shape = None
    ref_affine = None
    ref_zooms = None
    channels_data: List[np.ndarray] = []

    for mod in modality_order:
        fpath = file_map[mod]
        vol = load_nifti_file(fpath)
        # vol.data shape: (C, H, W, D) where C=1
        channel_slice = vol.data[0]

        if not np.all(np.isfinite(channel_slice)):
            raise ValueError(f"Non-finite intensities detected in sequence {mod} at {fpath}")

        if ref_shape is None:
            ref_shape = channel_slice.shape
            ref_affine = vol.affine
            ref_zooms = vol.voxel_spacing
        else:
            if channel_slice.shape != ref_shape:
                raise ValueError(f"Dimension mismatch in {mod}: {channel_slice.shape} != reference {ref_shape}")
            if not np.allclose(vol.affine, ref_affine, atol=1e-3):
                raise ValueError(f"Affine alignment mismatch in sequence {mod}")

        if normalize:
            channel_slice = normalize_intensity(channel_slice)

        channels_data.append(channel_slice)

    # Stack strictly in modality_order -> shape (4, H, W, D)
    stacked_image = np.stack(channels_data, axis=0).astype(np.float32)
    image_tensor = torch.from_numpy(stacked_image).unsqueeze(0)  # Shape (1, 4, H, W, D)

    # Load ground truth segmentation if present
    seg_tensor = None
    seg_labels = None
    if "seg" in file_map and file_map["seg"] and file_map["seg"].exists():
        seg_vol = load_nifti_file(file_map["seg"])
        seg_data = seg_vol.data[0].astype(np.int64)
        if seg_data.shape != ref_shape:
            raise ValueError(f"Segmentation shape mismatch: {seg_data.shape} != image shape {ref_shape}")
        # Remap legacy BraTS label 4 to 3 (Enhancing Tumor) if present
        seg_data[seg_data == 4] = 3
        seg_labels = sorted(list(np.unique(seg_data)))
        seg_tensor = torch.from_numpy(seg_data).unsqueeze(0).unsqueeze(0)  # Shape (1, 1, H, W, D)

    metadata = {
        "modality_order": list(modality_order),
        "dimensions": list(ref_shape),
        "voxel_spacing_mm": list(ref_zooms),
        "affine": ref_affine.tolist(),
        "file_paths": {k: str(v.as_posix()) if v else None for k, v in file_map.items()},
        "seg_unique_labels": seg_labels
    }

    return image_tensor, seg_tensor, metadata


def extract_3d_patches(
    image: torch.Tensor,
    seg: Optional[torch.Tensor] = None,
    patch_size: Tuple[int, int, int] = (64, 64, 64),
    stride: Optional[Tuple[int, int, int]] = None,
    max_patches: Optional[int] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Extracts regular 3D spatial patches from volume tensors for batch training or evaluation.
    Inputs:
      - image: Tensor of shape (B, C, H, W, D)
      - seg: Optional Tensor of shape (B, 1, H, W, D)
    Returns:
      - image_patches: Tensor of shape (N, C, ph, pw, pd)
      - seg_patches: Optional Tensor of shape (N, 1, ph, pw, pd)
    """
    b, c, h, w, d = image.shape
    ph, pw, pd = patch_size

    if stride is None:
        stride = (ph // 2, pw // 2, pd // 2)

    sh, sw, sd = stride
    h_starts = list(range(0, max(1, h - ph + 1), max(1, sh)))
    if h_starts[-1] + ph < h:
        h_starts.append(h - ph)

    w_starts = list(range(0, max(1, w - pw + 1), max(1, sw)))
    if w_starts[-1] + pw < w:
        w_starts.append(w - pw)

    d_starts = list(range(0, max(1, d - pd + 1), max(1, sd)))
    if d_starts[-1] + pd < d:
        d_starts.append(d - pd)

    img_patch_list: List[torch.Tensor] = []
    seg_patch_list: List[torch.Tensor] = []

    for bi in range(b):
        for hs in h_starts:
            for ws in w_starts:
                for ds in d_starts:
                    img_p = image[bi:bi+1, :, hs:hs+ph, ws:ws+pw, ds:ds+pd]
                    img_patch_list.append(img_p)
                    if seg is not None:
                        seg_p = seg[bi:bi+1, :, hs:hs+ph, ws:ws+pw, ds:ds+pd]
                        seg_patch_list.append(seg_p)

                    if max_patches and len(img_patch_list) >= max_patches:
                        break
                if max_patches and len(img_patch_list) >= max_patches:
                    break
            if max_patches and len(img_patch_list) >= max_patches:
                break
        if max_patches and len(img_patch_list) >= max_patches:
            break

    image_patches = torch.cat(img_patch_list, dim=0) if img_patch_list else torch.empty((0, c, ph, pw, pd))
    seg_patches = torch.cat(seg_patch_list, dim=0) if (seg is not None and seg_patch_list) else None

    return image_patches, seg_patches


def predict_multimodal_study(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    roi_size: Tuple[int, int, int] = (64, 64, 64),
    overlap: float = 0.5,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Executes sliding-window 3D volumetric inference on a 4-channel input tensor.
    Input image_tensor: (1, 4, H, W, D)
    Returns discrete predicted segmentation mask: (1, 1, H, W, D) with classes in {0, 1, 2, 3}.
    """
    model.eval()
    if image_tensor.ndim != 5 or image_tensor.shape[1] != 4:
        raise ValueError(f"Expected 4-channel 5D tensor (1, 4, H, W, D), got {image_tensor.shape}")

    with torch.no_grad():
        inputs = image_tensor.to(device)
        logits = native_sliding_window_inference(
            model=model,
            inputs=inputs,
            roi_size=roi_size,
            overlap=overlap,
            device=device
        )
        pred_classes = torch.argmax(logits, dim=1, keepdim=True).to(torch.int64)

    return pred_classes

