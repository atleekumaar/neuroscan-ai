"""
NEUROSCAN AI — Experiment: Intracranial Brain-Mask Constraint (Exp 2)
Applies a separate, optional intracranial brain-mask post-processing filter to
candidate model predictions on validation study BraTS-GLI-00481-000.

Guarantees:
  1. Candidate checkpoint weights are strictly frozen and unaltered.
  2. Training pipelines and baseline models are 100% untouched.
  3. Locked test subject BraTS-GLI-01012-000 is strictly quarantined.
  4. Spatial orientation, NIfTI affine, and label semantics {0, 1, 2, 3} are preserved.
  5. Metrics are evaluated with identical subregion formulas.
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.segmentation.model import (
    FourChannelUNetConfig,
    build_four_channel_unet,
    SUBREGION_HIERARCHY,
    LABEL_DEFINITIONS
)
from backend.segmentation.pipeline import (
    load_multimodal_study,
    predict_multimodal_study
)
from backend.segmentation.postprocessing import (
    construct_intracranial_mask,
    apply_intracranial_mask
)

CLASS_COLORS = {
    0: (0, 0, 0),
    1: (255, 60, 60),    # Red: Necrotic Core (NCR)
    2: (50, 200, 255),   # Cyan: Peritumoral Edema (ED)
    3: (255, 220, 20)    # Yellow: Enhancing Tumor (ET)
}


def compute_subregion_metrics(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> Dict[str, Any]:
    pred = pred_mask.squeeze().cpu().numpy()
    gt = gt_mask.squeeze().cpu().numpy()

    subregions = {
        "WT": (np.isin(pred, [1, 2, 3]), np.isin(gt, [1, 2, 3])),
        "TC": (np.isin(pred, [1, 3]), np.isin(gt, [1, 3])),
        "ET": (pred == 3, gt == 3)
    }

    metrics = {}
    for name, (p_bin, g_bin) in subregions.items():
        tp = float(np.logical_and(p_bin, g_bin).sum())
        fp = float(np.logical_and(p_bin, np.logical_not(g_bin)).sum())
        fn = float(np.logical_and(np.logical_not(p_bin), g_bin).sum())
        tn = float(np.logical_and(np.logical_not(p_bin), np.logical_not(g_bin)).sum())

        dice = (2.0 * tp + 1e-5) / (2.0 * tp + fp + fn + 1e-5)
        sens = (tp + 1e-5) / (tp + fn + 1e-5)
        spec = (tn + 1e-5) / (tn + fp + 1e-5)
        prec = (tp + 1e-5) / (tp + fp + 1e-5)

        pred_vol_cm3 = float(p_bin.sum()) / 1000.0
        gt_vol_cm3 = float(g_bin.sum()) / 1000.0

        metrics[name] = {
            "dice": round(float(dice), 4),
            "sensitivity": round(float(sens), 4),
            "specificity": round(float(spec), 4),
            "precision": round(float(prec), 4),
            "pred_volume_cm3": round(pred_vol_cm3, 3),
            "gt_volume_cm3": round(gt_vol_cm3, 3),
            "tp_voxels": int(tp),
            "fp_voxels": int(fp),
            "fn_voxels": int(fn),
            "tn_voxels": int(tn)
        }

    mean_dice = (metrics["WT"]["dice"] + metrics["TC"]["dice"] + metrics["ET"]["dice"]) / 3.0
    metrics["mean_dice"] = round(float(mean_dice), 4)
    return metrics


def normalize_slice_for_display(slice_data: np.ndarray) -> np.ndarray:
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


def colorize_mask(mask_slice: np.ndarray) -> np.ndarray:
    h, w = mask_slice.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for lbl, col in CLASS_COLORS.items():
        if lbl > 0:
            rgb[mask_slice == lbl] = col
    return rgb


def build_comparison_overlay(
    t1c_slice: np.ndarray,
    gt_slice: np.ndarray,
    unmasked_slice: np.ndarray,
    masked_slice: np.ndarray,
    title: str
) -> Image.Image:
    """
    Renders 5 side-by-side diagnostic panels:
      1. T1-CE MRI
      2. Ground Truth
      3. Unmasked Candidate Pred
      4. Masked Candidate Pred
      5. Removed False-Positive Map (diff)
    """
    mri_rgb = np.stack([normalize_slice_for_display(t1c_slice)] * 3, axis=-1)
    gt_rgb = colorize_mask(gt_slice)
    unmasked_rgb = colorize_mask(unmasked_slice)
    masked_rgb = colorize_mask(masked_slice)

    # Difference: voxels active in unmasked but zeroed in masked (i.e. removed false positives)
    removed_fp = (unmasked_slice > 0) & (masked_slice == 0)
    diff_rgb = np.zeros_like(mri_rgb)
    diff_rgb[removed_fp] = [255, 50, 150]  # Magenta: removed extra-axial voxels
    # Green: preserved true positives
    tp_preserved = (masked_slice > 0) & (gt_slice > 0)
    diff_rgb[tp_preserved] = [0, 220, 0]

    panels = [mri_rgb, gt_rgb, unmasked_rgb, masked_rgb, diff_rgb]
    labels = ["T1-CE MRI", "Ground Truth", "Unmasked Candidate", "Masked Candidate", "Suppressed FP (Magenta)"]

    h, w, _ = mri_rgb.shape
    banner_h = 32
    label_h = 20
    panel_w = w
    total_w = panel_w * 5 + 24
    total_h = h + banner_h + label_h + 10

    canvas = Image.new("RGB", (total_w, total_h), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)

    # Banner
    draw.rectangle([(0, 0), (total_w, banner_h)], fill=(15, 23, 42))
    draw.text((12, 8), title, fill=(255, 255, 255))

    for idx, (p_img, lbl) in enumerate(zip(panels, labels)):
        x_off = 4 + idx * (panel_w + 4)
        draw.text((x_off + 4, banner_h + 2), lbl, fill=(200, 200, 200))
        p_pil = Image.fromarray(p_img)
        canvas.paste(p_pil, (x_off, banner_h + label_h + 4))

    return canvas


def main():
    print("=" * 80)
    print("NEUROSCAN AI — Experiment: Intracranial Brain-Mask Constraint (Exp 2)")
    print("=" * 80)

    val_pid = "BraTS-GLI-00481-000"
    val_dir = PROJECT_ROOT / "backend" / "segmentation" / "data" / "pilot" / val_pid
    candidate_ckpt_path = PROJECT_ROOT / "models" / "NeuroScan_Seg3D_UNet_Phase5_Exp1_BalancedPatches.pth"

    if not candidate_ckpt_path.exists():
        raise FileNotFoundError(f"Candidate checkpoint missing: {candidate_ckpt_path}")
    if not val_dir.is_dir():
        raise FileNotFoundError(f"Validation study missing: {val_dir}")

    # 1. Load validation study
    print(f"\n[INFO] Loading validation study: {val_pid}...")
    val_img, val_seg, val_meta = load_multimodal_study(val_dir, normalize=True)
    unnorm_img, _, _ = load_multimodal_study(val_dir, normalize=False)
    print(f"  Image shape: {val_img.shape}, GT shape: {val_seg.shape}")

    # 2. Construct Intracranial Brain Mask
    print("\n[INFO] Constructing anatomically grounded intracranial brain mask...")
    brain_mask = construct_intracranial_mask(unnorm_img, intensity_threshold=0.0, fill_holes=True, largest_component_only=True)
    brain_voxels = int(np.sum(brain_mask))
    total_vol_voxels = int(brain_mask.size)
    print(f"  Intracranial mask voxels: {brain_voxels:,} ({100.0 * brain_voxels / total_vol_voxels:.2f}% of bounding box)")

    # Verify zero ground truth exclusion
    gt_np = val_seg.squeeze().cpu().numpy()
    gt_tumor_voxels = int(np.sum(gt_np > 0))
    excluded_gt = int(np.sum((gt_np > 0) & (~brain_mask)))
    print(f"  Ground truth tumor voxels: {gt_tumor_voxels:,}")
    print(f"  GT tumor voxels outside brain mask: {excluded_gt} (100% preservation verified)")
    assert excluded_gt == 0, f"Critical error: brain mask excluded {excluded_gt} true tumor voxels!"

    # 3. Load Candidate Model
    print(f"\n[INFO] Loading candidate checkpoint: {candidate_ckpt_path.name}...")
    config = FourChannelUNetConfig(
        in_channels=4,
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        dropout=0.2
    )
    model = build_four_channel_unet(config)
    ckpt_payload = torch.load(candidate_ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt_payload["model_state_dict"])
    model.eval()

    # 4. Perform Minimum Inference Needed
    print("\n[INFO] Running sliding-window inference on validation case...")
    t0 = time.time()
    with torch.no_grad():
        unmasked_pred = predict_multimodal_study(model, val_img, roi_size=(64, 64, 64), overlap=0.5)
    inf_time = time.time() - t0
    print(f"  Inference completed in {inf_time:.2f}s.")

    # 5. Compute Unmasked Candidate Metrics (Verification)
    unmasked_metrics = compute_subregion_metrics(unmasked_pred, val_seg)
    print("\n[VERIFIED] UNMASKED CANDIDATE METRICS:")
    print(f"  WT | Dice: {unmasked_metrics['WT']['dice']:.4f} | Recall: {unmasked_metrics['WT']['sensitivity']:.4f} | Prec: {unmasked_metrics['WT']['precision']:.4f} | FP: {unmasked_metrics['WT']['fp_voxels']:,} | Pred Vol: {unmasked_metrics['WT']['pred_volume_cm3']} cm³")
    print(f"  TC | Dice: {unmasked_metrics['TC']['dice']:.4f} | Recall: {unmasked_metrics['TC']['sensitivity']:.4f} | Prec: {unmasked_metrics['TC']['precision']:.4f} | FP: {unmasked_metrics['TC']['fp_voxels']:,} | Pred Vol: {unmasked_metrics['TC']['pred_volume_cm3']} cm³")
    print(f"  ET | Dice: {unmasked_metrics['ET']['dice']:.4f} | Recall: {unmasked_metrics['ET']['sensitivity']:.4f} | Prec: {unmasked_metrics['ET']['precision']:.4f} | FP: {unmasked_metrics['ET']['fp_voxels']:,} | Pred Vol: {unmasked_metrics['ET']['pred_volume_cm3']} cm³")
    print(f"  Mean Dice: {unmasked_metrics['mean_dice']:.4f}")

    # 6. Apply Intracranial Mask Constraint
    print("\n[INFO] Applying intracranial brain-mask constraint...")
    masked_pred = apply_intracranial_mask(unmasked_pred, brain_mask)

    # 7. Compute Masked Candidate Metrics
    masked_metrics = compute_subregion_metrics(masked_pred, val_seg)
    print("\n[RESULT] MASKED CANDIDATE METRICS (Exp 2):")
    print(f"  WT | Dice: {masked_metrics['WT']['dice']:.4f} | Recall: {masked_metrics['WT']['sensitivity']:.4f} | Prec: {masked_metrics['WT']['precision']:.4f} | FP: {masked_metrics['WT']['fp_voxels']:,} | Pred Vol: {masked_metrics['WT']['pred_volume_cm3']} cm³")
    print(f"  TC | Dice: {masked_metrics['TC']['dice']:.4f} | Recall: {masked_metrics['TC']['sensitivity']:.4f} | Prec: {masked_metrics['TC']['precision']:.4f} | FP: {masked_metrics['TC']['fp_voxels']:,} | Pred Vol: {masked_metrics['TC']['pred_volume_cm3']} cm³")
    print(f"  ET | Dice: {masked_metrics['ET']['dice']:.4f} | Recall: {masked_metrics['ET']['sensitivity']:.4f} | Prec: {masked_metrics['ET']['precision']:.4f} | FP: {masked_metrics['ET']['fp_voxels']:,} | Pred Vol: {masked_metrics['ET']['pred_volume_cm3']} cm³")
    print(f"  Mean Dice: {masked_metrics['mean_dice']:.4f}")

    # 8. Success Criteria Evaluation
    wt_fp_unmasked = unmasked_metrics["WT"]["fp_voxels"]
    wt_fp_masked = masked_metrics["WT"]["fp_voxels"]
    fp_reduction_abs = wt_fp_unmasked - wt_fp_masked
    fp_reduction_pct = 100.0 * fp_reduction_abs / wt_fp_unmasked

    dice_gain_abs = masked_metrics["WT"]["dice"] - unmasked_metrics["WT"]["dice"]
    dice_gain_pct = 100.0 * dice_gain_abs / unmasked_metrics["WT"]["dice"]

    print("\n" + "=" * 80)
    print("SUCCESS CRITERIA EVALUATION (Predeclared in NEUROSCAN_NEXT_MILESTONE.md)")
    print("=" * 80)
    print(f"  Criterion 1 (FP Reduction >= 15.0%): {fp_reduction_pct:.2f}% (Reduction: {fp_reduction_abs:,} voxels)")
    crit1_pass = fp_reduction_pct >= 15.0

    print(f"  Criterion 2 (WT Dice >= 0.1500): {masked_metrics['WT']['dice']:.4f} (Delta: {dice_gain_abs:+.4f} / {dice_gain_pct:+.2f}%)")
    crit2_pass = masked_metrics["WT"]["dice"] >= 0.1500

    print(f"  Criterion 3 (WT Sensitivity >= 90.0%): {masked_metrics['WT']['sensitivity'] * 100:.2f}%")
    crit3_pass = masked_metrics["WT"]["sensitivity"] >= 0.90

    stopping_triggered = fp_reduction_pct < 5.0
    print(f"  Stopping Condition Triggered (< 5.0% FP reduction): {stopping_triggered}")

    all_criteria_met = crit1_pass and crit2_pass and crit3_pass
    hypothesis_status = "SUPPORTED" if all_criteria_met else ("PARTIALLY_SUPPORTED" if (crit1_pass and crit3_pass) else "REJECTED")
    print(f"\nFinal Hypothesis Status: {hypothesis_status}")

    # 9. Render Multi-Planar Comparison Overlays
    print("\n[INFO] Rendering multi-planar comparison overlays...")
    figures_dir = PROJECT_ROOT / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    t1c_raw = unnorm_img[0, 1].cpu().numpy()
    unmasked_np = unmasked_pred.squeeze().cpu().numpy()
    masked_np = masked_pred.squeeze().cpu().numpy()

    # Cardinal slices
    axial_sums = (gt_np > 0).sum(axis=(0, 1))
    coronal_sums = (gt_np > 0).sum(axis=(0, 2))
    sagittal_sums = (gt_np > 0).sum(axis=(1, 2))
    max_z = int(np.argmax(axial_sums))
    max_y = int(np.argmax(coronal_sums))
    max_x = int(np.argmax(sagittal_sums))

    # Render Axial Peak
    img_ax = build_comparison_overlay(
        np.rot90(t1c_raw[:, :, max_z]),
        np.rot90(gt_np[:, :, max_z]),
        np.rot90(unmasked_np[:, :, max_z]),
        np.rot90(masked_np[:, :, max_z]),
        f"Validation Study {val_pid} — Axial View (Z={max_z})"
    )
    ax_path = figures_dir / "exp2_brain_mask_axial_z91.png"
    img_ax.save(ax_path)
    print(f"  Saved {ax_path.name}")

    # Render Coronal Peak
    img_cor = build_comparison_overlay(
        np.rot90(t1c_raw[:, max_y, :]),
        np.rot90(gt_np[:, max_y, :]),
        np.rot90(unmasked_np[:, max_y, :]),
        np.rot90(masked_np[:, max_y, :]),
        f"Validation Study {val_pid} — Coronal View (Y={max_y})"
    )
    cor_path = figures_dir / "exp2_brain_mask_coronal_y148.png"
    img_cor.save(cor_path)
    print(f"  Saved {cor_path.name}")

    # Combined figure
    total_w = img_ax.width
    total_h = img_ax.height + img_cor.height + 10
    montage = Image.new("RGB", (total_w, total_h), (25, 25, 25))
    montage.paste(img_ax, (0, 0))
    montage.paste(img_cor, (0, img_ax.height + 10))
    montage_path = figures_dir / "exp2_brain_mask_multiview_comparison.png"
    montage.save(montage_path)
    print(f"  Saved multi-view overlay: {montage_path.name}")

    # 10. Save Results JSON
    res_payload = {
        "experiment_id": "Exp2_Intracranial_BrainMask_Constraint",
        "validation_patient_id": val_pid,
        "inference_wall_time_sec": round(inf_time, 2),
        "intracranial_mask": {
            "total_brain_voxels": brain_voxels,
            "bounding_box_voxels": total_vol_voxels,
            "brain_volume_pct": round(100.0 * brain_voxels / total_vol_voxels, 2),
            "gt_tumor_voxels_total": gt_tumor_voxels,
            "gt_tumor_voxels_excluded": excluded_gt,
            "gt_tumor_preserved_pct": 100.0
        },
        "unmasked_candidate_metrics": unmasked_metrics,
        "masked_candidate_metrics": masked_metrics,
        "deltas": {
            "WT_fp_reduction_voxels": fp_reduction_abs,
            "WT_fp_reduction_pct": round(fp_reduction_pct, 2),
            "WT_dice_delta": round(dice_gain_abs, 4),
            "WT_dice_pct_change": round(dice_gain_pct, 2),
            "WT_recall_delta": round(masked_metrics["WT"]["sensitivity"] - unmasked_metrics["WT"]["sensitivity"], 4),
            "WT_precision_delta": round(masked_metrics["WT"]["precision"] - unmasked_metrics["WT"]["precision"], 4),
            "WT_pred_volume_cm3_delta": round(masked_metrics["WT"]["pred_volume_cm3"] - unmasked_metrics["WT"]["pred_volume_cm3"], 3),
            "TC_dice_delta": round(masked_metrics["TC"]["dice"] - unmasked_metrics["TC"]["dice"], 4),
            "TC_fp_reduction_voxels": unmasked_metrics["TC"]["fp_voxels"] - masked_metrics["TC"]["fp_voxels"],
            "ET_dice_delta": round(masked_metrics["ET"]["dice"] - unmasked_metrics["ET"]["dice"], 4),
            "ET_fp_reduction_voxels": unmasked_metrics["ET"]["fp_voxels"] - masked_metrics["ET"]["fp_voxels"],
            "mean_dice_delta": round(masked_metrics["mean_dice"] - unmasked_metrics["mean_dice"], 4)
        },
        "success_criteria_audit": {
            "criterion_1_fp_reduction_ge_15pct": {
                "threshold": 15.0,
                "observed": round(fp_reduction_pct, 2),
                "passed": crit1_pass
            },
            "criterion_2_wt_dice_ge_0_1500": {
                "threshold": 0.1500,
                "observed": masked_metrics["WT"]["dice"],
                "passed": crit2_pass
            },
            "criterion_3_recall_ge_90pct": {
                "threshold": 0.90,
                "observed": masked_metrics["WT"]["sensitivity"],
                "passed": crit3_pass
            },
            "stopping_condition_triggered": stopping_triggered,
            "final_verdict": hypothesis_status
        }
    }

    out_res_path = PROJECT_ROOT / "reports" / "experiment_brain_mask_constraint_results.json"
    with open(out_res_path, "w") as f:
        json.dump(res_payload, f, indent=2)
    print(f"\n[SAVED] Structured Experiment Results: {out_res_path.name}")
    print("=" * 80)


if __name__ == "__main__":
    main()
