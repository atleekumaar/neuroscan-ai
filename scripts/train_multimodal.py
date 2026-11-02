"""
NEUROSCAN AI — Phase 5.14: Controlled Multi-Parametric 3D U-Net Training
Trains 4-channel segmentation model on the 3 verified training studies:
  - BraTS-GLI-00324-000
  - BraTS-GLI-00442-000
  - BraTS-GLI-00456-000
Evaluates strictly on the validation study:
  - BraTS-GLI-00481-000
Quarantines locked-test study:
  - BraTS-GLI-01012-000
Saves weights to:
  models/NeuroScan_Seg3D_UNet_Phase5_Multimodal_Best.pth
"""

import os
import sys
import time
import json
import psutil
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.segmentation.model import (
    FourChannelUNetConfig,
    build_four_channel_unet,
    SUBREGION_HIERARCHY
)
from backend.segmentation.pipeline import (
    MODALITY_ORDER,
    load_multimodal_study,
    predict_multimodal_study
)

try:
    from monai.losses import DiceCELoss
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False


def compute_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def extract_balanced_patches(
    image: torch.Tensor,
    seg: torch.Tensor,
    patch_size: Tuple[int, int, int] = (48, 48, 48),
    n_foreground: int = 4,
    n_background: int = 2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts balanced patches from a 4-channel image (1, 4, H, W, D)
    and integer segmentation mask (1, 1, H, W, D).
    """
    _, _, H, W, D = image.shape
    pd, ph, pw = patch_size

    # Find foreground coordinates
    fg_indices = (seg[0, 0] > 0).nonzero(as_tuple=False)

    patches_x = []
    patches_y = []

    # 1. Foreground-centered patches
    if len(fg_indices) > 0:
        for _ in range(n_foreground):
            idx = torch.randint(0, len(fg_indices), (1,)).item()
            cz, cy, cx = fg_indices[idx].tolist()

            # Bounding box with clamping
            z_start = max(0, min(cz - pd // 2, H - pd))
            y_start = max(0, min(cy - ph // 2, W - ph))
            x_start = max(0, min(cx - pw // 2, D - pw))

            patch_img = image[:, :, z_start:z_start + pd, y_start:y_start + ph, x_start:x_start + pw]
            patch_seg = seg[:, :, z_start:z_start + pd, y_start:y_start + ph, x_start:x_start + pw]

            patches_x.append(patch_img)
            patches_y.append(patch_seg)
    else:
        n_background += n_foreground

    # 2. Random background patches
    for _ in range(n_background):
        z_start = torch.randint(0, max(1, H - pd), (1,)).item() if H > pd else 0
        y_start = torch.randint(0, max(1, W - ph), (1,)).item() if W > ph else 0
        x_start = torch.randint(0, max(1, D - pw), (1,)).item() if D > pw else 0

        patch_img = image[:, :, z_start:z_start + pd, y_start:y_start + ph, x_start:x_start + pw]
        patch_seg = seg[:, :, z_start:z_start + pd, y_start:y_start + ph, x_start:x_start + pw]

        patches_x.append(patch_img)
        patches_y.append(patch_seg)

    return torch.cat(patches_x, dim=0), torch.cat(patches_y, dim=0)


def compute_subregion_metrics(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> Dict[str, Any]:
    """Computes Dice, Sensitivity, Specificity, and Volumes for WT, TC, and ET."""
    pred = pred_mask.squeeze().cpu().numpy()
    gt = gt_mask.squeeze().cpu().numpy()

    # Subregion mappings
    # WT: 1, 2, 3
    # TC: 1, 3
    # ET: 3
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

        pred_vol_cm3 = float(p_bin.sum()) / 1000.0  # 1 voxel = 1 mm^3 = 0.001 cm^3
        gt_vol_cm3 = float(g_bin.sum()) / 1000.0

        metrics[name] = {
            "dice": round(float(dice), 4),
            "sensitivity": round(float(sens), 4),
            "specificity": round(float(spec), 4),
            "pred_volume_cm3": round(pred_vol_cm3, 3),
            "gt_volume_cm3": round(gt_vol_cm3, 3),
            "tp_voxels": int(tp),
            "fp_voxels": int(fp),
            "fn_voxels": int(fn)
        }

    mean_dice = (metrics["WT"]["dice"] + metrics["TC"]["dice"] + metrics["ET"]["dice"]) / 3.0
    metrics["mean_dice"] = round(float(mean_dice), 4)
    return metrics


def main():
    print("=" * 80)
    print("NEUROSCAN AI — Phase 5.14 Controlled Multimodal 3D U-Net Training")
    print("=" * 80)

    torch.manual_seed(42)
    np.random.seed(42)

    pilot_dir = PROJECT_ROOT / "backend" / "segmentation" / "data" / "pilot"
    splits_file = PROJECT_ROOT / "backend" / "segmentation" / "multimodal_splits.json"
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    out_checkpoint = models_dir / "NeuroScan_Seg3D_UNet_Phase5_Multimodal_Best.pth"

    with open(splits_file, "r") as f:
        split_meta = json.load(f)

    train_ids = [s["patient_id"] for s in split_meta["splits"]["train"]]
    val_ids = [s["patient_id"] for s in split_meta["splits"]["val"]]
    locked_ids = [s["patient_id"] for s in split_meta["splits"]["locked_test"]]

    print(f"Train Cohort: {train_ids}")
    print(f"Validation Cohort: {val_ids}")
    print(f"Locked Test Cohort (Quarantined): {locked_ids}")

    # Load training studies into memory
    print("\n[INFO] Pre-loading and normalizing training studies...")
    train_data = []
    for pid in train_ids:
        sdir = pilot_dir / pid
        img, seg, meta = load_multimodal_study(sdir, normalize=True)
        train_data.append((pid, img, seg))
        print(f"  Loaded {pid}: img={tuple(img.shape)}, seg={tuple(seg.shape)}")

    # Load validation study
    print("\n[INFO] Loading validation study...")
    val_pid = val_ids[0]
    val_img, val_seg, val_meta = load_multimodal_study(pilot_dir / val_pid, normalize=True)
    print(f"  Loaded Val {val_pid}: img={tuple(val_img.shape)}, seg={tuple(val_seg.shape)}")

    # Instantiate Model
    config = FourChannelUNetConfig(
        in_channels=4,
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        dropout=0.2
    )
    model = build_four_channel_unet(config)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)

    epochs = 5
    batch_size = 2
    patch_size = (48, 48, 48)

    training_log = []
    start_total_time = time.time()

    print("\n[INFO] Starting Bounded Experimental Training (5 Epochs)...")
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()

        # Extract balanced patches for this epoch
        epoch_patches_x = []
        epoch_patches_y = []
        for pid, img, seg in train_data:
            px, py = extract_balanced_patches(img, seg, patch_size=patch_size, n_foreground=4, n_background=2)
            epoch_patches_x.append(px)
            epoch_patches_y.append(py)

        all_x = torch.cat(epoch_patches_x, dim=0)
        all_y = torch.cat(epoch_patches_y, dim=0)

        # Shuffle patches
        perm = torch.randperm(all_x.shape[0])
        all_x = all_x[perm]
        all_y = all_y[perm]

        n_batches = int(np.ceil(all_x.shape[0] / batch_size))
        epoch_losses = []
        max_grad_norm = 0.0

        for b in range(n_batches):
            bx = all_x[b * batch_size:(b + 1) * batch_size]
            by = all_y[b * batch_size:(b + 1) * batch_size]

            optimizer.zero_grad()
            out = model(bx)
            loss = loss_fn(out, by)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss encountered at epoch {epoch}, batch {b}: {loss.item()}")

            loss.backward()

            # Monitor gradients
            total_norm = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2
            ).item()
            max_grad_norm = max(max_grad_norm, total_norm)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()
        epoch_duration = time.time() - epoch_start
        mean_loss = float(np.mean(epoch_losses))
        current_ram_gb = psutil.virtual_memory().used / (1024**3)

        epoch_rec = {
            "epoch": epoch,
            "mean_loss": round(mean_loss, 4),
            "max_grad_norm": round(max_grad_norm, 4),
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "duration_sec": round(epoch_duration, 2),
            "ram_used_gb": round(current_ram_gb, 2)
        }
        training_log.append(epoch_rec)
        print(f"  Epoch {epoch}/{epochs} | Loss: {mean_loss:.4f} | Max Grad Norm: {max_grad_norm:.4f} | RAM: {current_ram_gb:.2f} GB | Time: {epoch_duration:.2f}s")

    total_train_time = time.time() - start_total_time
    print(f"\n[SUCCESS] Experimental training completed in {total_train_time:.2f}s.")

    # Validation Phase
    print("\n[INFO] Evaluating on Validation Study (BraTS-GLI-00481-000) using Sliding Window Inference...")
    model.eval()
    val_start = time.time()
    with torch.no_grad():
        val_pred = predict_multimodal_study(model, val_img, roi_size=(64, 64, 64), overlap=0.5)
    val_duration = time.time() - val_start

    val_metrics = compute_subregion_metrics(val_pred, val_seg)
    print(f"Validation Duration: {val_duration:.2f}s")
    print(f"Validation Metrics on {val_pid}:")
    print(f"  Whole Tumor (WT)  | Dice: {val_metrics['WT']['dice']:.4f} | Sens: {val_metrics['WT']['sensitivity']:.4f} | Spec: {val_metrics['WT']['specificity']:.4f}")
    print(f"  Tumor Core (TC)   | Dice: {val_metrics['TC']['dice']:.4f} | Sens: {val_metrics['TC']['sensitivity']:.4f} | Spec: {val_metrics['TC']['specificity']:.4f}")
    print(f"  Enhancing (ET)    | Dice: {val_metrics['ET']['dice']:.4f} | Sens: {val_metrics['ET']['sensitivity']:.4f} | Spec: {val_metrics['ET']['specificity']:.4f}")
    print(f"  Mean Validation Dice: {val_metrics['mean_dice']:.4f}")

    # Serialize Best Checkpoint
    checkpoint_payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            "architecture": "FourChannelUNet3D",
            "in_channels": 4,
            "out_channels": 4,
            "channels": (16, 32, 64, 128, 256),
            "modality_order": list(MODALITY_ORDER)
        },
        "metadata": {
            "phase": "5.14",
            "model_name": "NeuroScan 3D Multimodal U-Net (BraTS 2023 Pilot)",
            "seed": 42,
            "epochs_completed": epochs,
            "train_patients": train_ids,
            "val_patient": val_pid,
            "locked_test_patient_excluded": locked_ids[0],
            "val_metrics": val_metrics,
            "training_history": training_log,
            "training_time_sec": round(total_train_time, 2)
        }
    }

    torch.save(checkpoint_payload, str(out_checkpoint))
    ckpt_hash = compute_file_sha256(out_checkpoint)
    ckpt_size = out_checkpoint.stat().st_size
    print(f"\n[SAVED] Checkpoint: {out_checkpoint}")
    print(f"  Size: {ckpt_size:,} bytes")
    print(f"  SHA-256: {ckpt_hash}")

    # Save results json for report generation
    results_json = PROJECT_ROOT / "reports" / "phase5_14_training_results.json"
    results_data = {
        "training_log": training_log,
        "total_train_time_sec": round(total_train_time, 2),
        "val_duration_sec": round(val_duration, 2),
        "val_metrics": val_metrics,
        "checkpoint_file": str(out_checkpoint.name),
        "checkpoint_size_bytes": ckpt_size,
        "checkpoint_sha256": ckpt_hash,
        "train_patients": train_ids,
        "val_patient": val_pid,
        "locked_test_patient_quarantined": locked_ids[0]
    }
    with open(results_json, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"[SAVED] Results data: {results_json}")


if __name__ == "__main__":
    main()
