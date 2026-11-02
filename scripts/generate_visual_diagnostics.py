"""
NEUROSCAN AI — Phase 3 Visual Diagnostic Suite
Generates high-resolution multi-planar diagnostic panels for held-out validation cases:
- Case 001 and Case 002
- Axial, Coronal, and Sagittal orthogonal views
- Both Tumor-Positive slices (peak ground-truth tumor area) and Representative Tumor-Negative slices (normal parenchyma)
- Panels: [1] T1-CE MRI, [2] Ground-Truth Mask, [3] Model Prediction, [4] Spatial Error Map (TP=Green, FP=Red, FN=Blue)
Saves structured metadata and images to reports/figures/
"""

import sys
import json
from pathlib import Path
import numpy as np
import torch
import nibabel as nib
from PIL import Image, ImageDraw, ImageFont

# Ensure repo root is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.segmentation.model import load_segmentation_model
from backend.segmentation.pipeline import (
    load_nifti_file,
    preprocess_volume_for_model,
    run_sliding_window_inference
)


# Color mapping: 0=Black, 1=Necrotic (Red), 2=Edema (Cyan), 3=Enhancing (Yellow)
CLASS_COLORS = {
    0: (0, 0, 0),
    1: (255, 60, 60),    # Red: NCR/NET
    2: (50, 200, 255),   # Cyan: ED
    3: (255, 220, 20)    # Yellow: ET
}


def colorize_mask(mask_slice: np.ndarray) -> np.ndarray:
    """Converts discrete label slice to RGB."""
    h, w = mask_slice.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for lbl, col in CLASS_COLORS.items():
        if lbl > 0:
            rgb[mask_slice == lbl] = col
    return rgb


def normalize_mri_slice(slice_data: np.ndarray) -> np.ndarray:
    """Robust 1-99% percentile normalization to 0-255 uint8."""
    non_zero = slice_data[slice_data > 0]
    if len(non_zero) > 10:
        p1 = np.percentile(non_zero, 1.0)
        p99 = np.percentile(non_zero, 99.0)
        if p99 > p1:
            scaled = np.clip((slice_data - p1) / (p99 - p1) * 255.0, 0, 255)
        else:
            scaled = np.zeros_like(slice_data)
    else:
        scaled = np.zeros_like(slice_data)
    return scaled.astype(np.uint8)


def create_error_map(mri_uint8: np.ndarray, gt_slice: np.ndarray, pred_slice: np.ndarray) -> np.ndarray:
    """
    Creates clinical error map:
    - Green (0, 255, 0): True Positive (concordant overlap)
    - Red (255, 50, 50): False Positive (model hallucination / over-prediction)
    - Blue (50, 100, 255): False Negative (missed tumor margin)
    - Background: Grayscale MRI
    """
    rgb = np.stack([mri_uint8, mri_uint8, mri_uint8], axis=-1)

    gt_fg = gt_slice > 0
    pred_fg = pred_slice > 0

    tp = gt_fg & pred_fg
    fp = (~gt_fg) & pred_fg
    fn = gt_fg & (~pred_fg)

    # Blend overlays
    rgb[tp] = [0, 255, 50]     # True Positive
    rgb[fp] = [255, 50, 50]    # False Positive
    rgb[fn] = [50, 120, 255]   # False Negative
    return rgb


def render_panel(
    mri_slice: np.ndarray,
    gt_slice: np.ndarray,
    pred_slice: np.ndarray,
    title_text: str
) -> Image.Image:
    """
    Renders 4-column diagnostic strip:
    [T1-CE MRI] | [Ground Truth] | [Model Prediction] | [Error Map: TP/FP/FN]
    """
    mri_u8 = normalize_mri_slice(mri_slice)
    mri_rgb = np.stack([mri_u8, mri_u8, mri_u8], axis=-1)
    
    # Overlay GT on MRI
    gt_rgb = mri_rgb.copy()
    gt_color = colorize_mask(gt_slice)
    gt_fg = gt_slice > 0
    gt_rgb[gt_fg] = (gt_rgb[gt_fg] * 0.4 + gt_color[gt_fg] * 0.6).astype(np.uint8)

    # Overlay Pred on MRI
    pred_rgb = mri_rgb.copy()
    pred_color = colorize_mask(pred_slice)
    pred_fg = pred_slice > 0
    pred_rgb[pred_fg] = (pred_rgb[pred_fg] * 0.4 + pred_color[pred_fg] * 0.6).astype(np.uint8)

    error_map = create_error_map(mri_u8, gt_slice, pred_slice)

    # Concat horizontally
    strip = np.concatenate([mri_rgb, gt_rgb, pred_rgb, error_map], axis=1)
    img = Image.fromarray(strip)

    # Add header banner with titles
    banner_height = 36
    total_w, total_h = img.width, img.height + banner_height
    composite = Image.new("RGB", (total_w, total_h), color=(18, 22, 28))
    composite.paste(img, (0, banner_height))

    draw = ImageDraw.Draw(composite)
    col_w = mri_slice.shape[1]

    # Labels
    headers = [
        ("T1-CE Contrast MRI", (200, 200, 200)),
        ("Ground Truth (Expert)", (50, 255, 120)),
        ("Model Prediction", (255, 180, 50)),
        ("Error Map (Green=TP, Red=FP, Blue=FN)", (255, 100, 100))
    ]

    for idx, (hdr, col) in enumerate(headers):
        x = idx * col_w + 10
        draw.text((x, 8), hdr, fill=col)

    return composite


def main():
    print("=" * 70)
    print("NEUROSCAN AI — Phase 3 Visual Diagnostic Generator")
    print("=" * 70)

    data_dir = root_dir / "backend" / "segmentation" / "data"
    out_dir = root_dir / "reports" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = root_dir / "models" / "NeuroScan_Seg3D_UNet_T1CE_SingleChannel.pth"
    device = torch.device("cpu")
    model, is_loaded, _ = load_segmentation_model(str(ckpt_path), device)
    if not is_loaded:
        print(f"[ERROR] Could not load checkpoint from {ckpt_path}")
        sys.exit(1)

    cases = [
        ("case_001", data_dir / "BraTS_Case_001_T1CE.nii.gz", data_dir / "BraTS_Case_001_seg.nii.gz"),
        ("case_002", data_dir / "BraTS_Case_002_T1CE.nii.gz", data_dir / "BraTS_Case_002_seg.nii.gz")
    ]

    manifest = {"generated_figures": []}

    for case_key, mri_p, seg_p in cases:
        print(f"\n--> Processing {case_key}...")
        mri_vol = load_nifti_file(mri_p)
        gt_vol = load_nifti_file(seg_p)

        mri_arr = mri_vol.data[0] # (H, W, D) = (240, 240, 155)
        gt_arr = gt_vol.data[0] if gt_vol.data.ndim == 4 else gt_vol.data
        gt_arr = np.where(gt_arr == 4, 3, gt_arr).astype(np.uint8)

        # Run inference with brain mask
        inputs = preprocess_volume_for_model(mri_vol, target_channels=1)
        brain_mask = mri_arr > 0
        pred_mask = run_sliding_window_inference(
            model=model,
            inputs=inputs,
            roi_size=(64, 64, 64),
            overlap=0.5,
            device=device,
            brain_mask=brain_mask
        )

        # Determine Peak Tumor Slices
        # Axial: axis 2 (z)
        z_tumor = np.sum(gt_arr > 0, axis=(0, 1))
        best_z = int(np.argmax(z_tumor))
        
        # Coronal: axis 1 (y)
        y_tumor = np.sum(gt_arr > 0, axis=(0, 2))
        best_y = int(np.argmax(y_tumor))

        # Sagittal: axis 0 (x)
        x_tumor = np.sum(gt_arr > 0, axis=(1, 2))
        best_x = int(np.argmax(x_tumor))

        # Determine Representative Negative Slices (normal brain parenchyma, zero GT tumor)
        z_brain = np.sum(brain_mask, axis=(0, 1))
        z_normal_candidates = [z for z in range(mri_arr.shape[2]) if z_tumor[z] == 0 and z_brain[z] > 5000]
        neg_z = z_normal_candidates[len(z_normal_candidates) // 2] if z_normal_candidates else 25

        y_brain = np.sum(brain_mask, axis=(0, 2))
        y_normal_candidates = [y for y in range(mri_arr.shape[1]) if y_tumor[y] == 0 and y_brain[y] > 5000]
        neg_y = y_normal_candidates[len(y_normal_candidates) // 2] if y_normal_candidates else 40

        x_brain = np.sum(brain_mask, axis=(1, 2))
        x_normal_candidates = [x for x in range(mri_arr.shape[0]) if x_tumor[x] == 0 and x_brain[x] > 5000]
        neg_x = x_normal_candidates[len(x_normal_candidates) // 2] if x_normal_candidates else 40

        slice_specs = [
            ("axial_positive", "Axial (Z)", best_z, mri_arr[:, :, best_z], gt_arr[:, :, best_z], pred_mask[:, :, best_z]),
            ("axial_negative", "Axial (Z)", neg_z, mri_arr[:, :, neg_z], gt_arr[:, :, neg_z], pred_mask[:, :, neg_z]),
            ("coronal_positive", "Coronal (Y)", best_y, mri_arr[:, best_y, :], gt_arr[:, best_y, :], pred_mask[:, best_y, :]),
            ("coronal_negative", "Coronal (Y)", neg_y, mri_arr[:, neg_y, :], gt_arr[:, neg_y, :], pred_mask[:, neg_y, :]),
            ("sagittal_positive", "Sagittal (X)", best_x, mri_arr[best_x, :, :], gt_arr[best_x, :, :], pred_mask[best_x, :, :]),
            ("sagittal_negative", "Sagittal (X)", neg_x, mri_arr[neg_x, :, :], gt_arr[neg_x, :, :], pred_mask[neg_x, :, :]),
        ]

        for view_key, plane_name, s_idx, mri_s, gt_s, pred_s in slice_specs:
            fname = f"{case_key}_{view_key}_slice_{s_idx}.png"
            fpath = out_dir / fname
            
            panel_img = render_panel(
                mri_slice=mri_s,
                gt_slice=gt_s,
                pred_slice=pred_s,
                title_text=f"{case_key} | {plane_name} Slice {s_idx}"
            )
            panel_img.save(str(fpath))
            print(f"  Saved: {fname} (Shape={mri_s.shape}, Slice={s_idx})")

            gt_vox = int(np.sum(gt_s > 0))
            pred_vox = int(np.sum(pred_s > 0))
            fp_vox = int(np.sum((gt_s == 0) & (pred_s > 0)))
            fn_vox = int(np.sum((gt_s > 0) & (pred_s == 0)))

            manifest["generated_figures"].append({
                "case_key": case_key,
                "view": view_key,
                "plane": plane_name,
                "slice_index": s_idx,
                "geometry": list(mri_s.shape),
                "file_path": str(fpath.relative_to(root_dir)),
                "gt_tumor_voxels": gt_vox,
                "pred_tumor_voxels": pred_vox,
                "false_positive_voxels": fp_vox,
                "false_negative_voxels": fn_vox
            })

    meta_path = out_dir / "diagnostics_manifest.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[SUCCESS] Generated all 12 multi-planar diagnostic panels into {out_dir}")


if __name__ == "__main__":
    main()
