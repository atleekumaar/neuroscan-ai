"""
NEUROSCAN AI — Controlled Experiment: Patch Sampling Rebalancing (Exp 1)
Intervention:
  Shift patch sampling from 4 foreground / 2 background (2:1 foreground bias)
  to 2 foreground / 4 background (1:2 balanced background representation).
All other variables held strictly constant:
  - Cohort: 3 train, 1 val, locked test quarantined
  - Epochs: 5
  - Patches per epoch: 18 (6 per volume x 3 volumes)
  - Batch size: 2 (9 steps per epoch)
  - Optimizer: AdamW (lr=2e-4, weight_decay=1e-4), CosineAnnealingLR (T_max=5)
  - Loss: DiceCELoss (to_onehot_y=True, softmax=True)
  - Patch resolution: (48, 48, 48)
  - Random seed: 42
Saves candidate checkpoint to:
  models/NeuroScan_Seg3D_UNet_Phase5_Exp1_BalancedPatches.pth
"""

import os
import sys
import time
import json
import psutil
import hashlib
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.segmentation.model import (
    FourChannelUNetConfig,
    build_four_channel_unet
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
    n_foreground: int = 2,  # INTERVENTION: reduced from 4 to 2
    n_background: int = 4   # INTERVENTION: increased from 2 to 4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts patches with rebalanced background representation to suppress
    normal parenchymal false positives.
    """
    _, _, H, W, D = image.shape
    pd, ph, pw = patch_size

    fg_indices = (seg[0, 0] > 0).nonzero(as_tuple=False)

    patches_x = []
    patches_y = []

    # 1. Foreground patches (2 per volume)
    if len(fg_indices) > 0:
        for _ in range(n_foreground):
            idx = torch.randint(0, len(fg_indices), (1,)).item()
            cz, cy, cx = fg_indices[idx].tolist()

            z_start = max(0, min(cz - pd // 2, H - pd))
            y_start = max(0, min(cy - ph // 2, W - ph))
            x_start = max(0, min(cx - pw // 2, D - pw))

            patch_img = image[:, :, z_start:z_start + pd, y_start:y_start + ph, x_start:x_start + pw]
            patch_seg = seg[:, :, z_start:z_start + pd, y_start:y_start + ph, x_start:x_start + pw]

            patches_x.append(patch_img)
            patches_y.append(patch_seg)
    else:
        n_background += n_foreground

    # 2. Background / non-tumor patches (4 per volume)
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


def main():
    print("=" * 80)
    print("NEUROSCAN AI — Controlled Segmentation Experiment (Intervention: 1:2 Patch Balance)")
    print("=" * 80)

    torch.manual_seed(42)
    np.random.seed(42)

    pilot_dir = PROJECT_ROOT / "backend" / "segmentation" / "data" / "pilot"
    splits_file = PROJECT_ROOT / "backend" / "segmentation" / "multimodal_splits.json"
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    out_checkpoint = models_dir / "NeuroScan_Seg3D_UNet_Phase5_Exp1_BalancedPatches.pth"

    with open(splits_file, "r") as f:
        split_meta = json.load(f)

    train_ids = [s["patient_id"] for s in split_meta["splits"]["train"]]
    val_ids = [s["patient_id"] for s in split_meta["splits"]["val"]]
    locked_ids = [s["patient_id"] for s in split_meta["splits"]["locked_test"]]

    print(f"Train Cohort: {train_ids}")
    print(f"Validation Cohort: {val_ids}")
    print(f"Locked Test Cohort (QUARANTINED): {locked_ids}")

    # Load training data
    print("\n[INFO] Pre-loading training studies...")
    train_data = []
    for pid in train_ids:
        sdir = pilot_dir / pid
        img, seg, meta = load_multimodal_study(sdir, normalize=True)
        train_data.append((pid, img, seg))
        print(f"  Loaded {pid}")

    # Load validation study
    val_pid = val_ids[0]
    val_img, val_seg, val_meta = load_multimodal_study(pilot_dir / val_pid, normalize=True)
    print(f"  Loaded Validation Study: {val_pid}")

    # Model definition (Identical to baseline)
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
    start_time = time.time()

    print("\n[INFO] Starting Controlled Training (Intervention: 2 FG / 4 BG patches)...")
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()

        epoch_patches_x = []
        epoch_patches_y = []
        for pid, img, seg in train_data:
            px, py = extract_balanced_patches(img, seg, patch_size=patch_size, n_foreground=2, n_background=4)
            epoch_patches_x.append(px)
            epoch_patches_y.append(py)

        all_x = torch.cat(epoch_patches_x, dim=0)
        all_y = torch.cat(epoch_patches_y, dim=0)

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
                raise RuntimeError(f"Non-finite loss at epoch {epoch}: {loss.item()}")

            loss.backward()

            total_norm = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2
            ).item()
            max_grad_norm = max(max_grad_norm, total_norm)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()
        epoch_dur = time.time() - epoch_start
        mean_loss = float(np.mean(epoch_losses))
        current_ram = psutil.virtual_memory().used / (1024**3)

        training_log.append({
            "epoch": epoch,
            "loss": round(mean_loss, 4),
            "grad_norm": round(max_grad_norm, 4),
            "time_sec": round(epoch_dur, 2)
        })
        print(f"  Epoch {epoch}/5 | Loss: {mean_loss:.4f} | Grad Norm: {max_grad_norm:.4f} | Time: {epoch_dur:.2f}s")

    total_train_time = time.time() - start_time
    print(f"\n[INFO] Candidate training finished in {total_train_time:.2f}s.")

    # Validation Phase
    print(f"\n[INFO] Evaluating Candidate on {val_pid} via sliding window...")
    model.eval()
    val_start = time.time()
    with torch.no_grad():
        val_pred = predict_multimodal_study(model, val_img, roi_size=(64, 64, 64), overlap=0.5)
    val_time = time.time() - val_start

    candidate_metrics = compute_subregion_metrics(val_pred, val_seg)
    print(f"Validation completed in {val_time:.2f}s.")
    print(f"\nCANDIDATE RESULTS ON {val_pid}:")
    print(f"  WT | Dice: {candidate_metrics['WT']['dice']:.4f} | Recall: {candidate_metrics['WT']['sensitivity']:.4f} | Prec: {candidate_metrics['WT']['precision']:.4f} | Pred Vol: {candidate_metrics['WT']['pred_volume_cm3']} cm³")
    print(f"  TC | Dice: {candidate_metrics['TC']['dice']:.4f} | Recall: {candidate_metrics['TC']['sensitivity']:.4f} | Prec: {candidate_metrics['TC']['precision']:.4f} | Pred Vol: {candidate_metrics['TC']['pred_volume_cm3']} cm³")
    print(f"  ET | Dice: {candidate_metrics['ET']['dice']:.4f} | Recall: {candidate_metrics['ET']['sensitivity']:.4f} | Prec: {candidate_metrics['ET']['precision']:.4f} | Pred Vol: {candidate_metrics['ET']['pred_volume_cm3']} cm³")
    print(f"  Mean Dice: {candidate_metrics['mean_dice']:.4f}")

    # Baseline for immediate direct comparison
    baseline_metrics = {
        "WT": {"dice": 0.1089, "sensitivity": 0.9739, "precision": 0.0577, "pred_volume_cm3": 2510.167, "fp_voxels": 2365455},
        "TC": {"dice": 0.0322, "sensitivity": 0.3682, "precision": 0.0168, "pred_volume_cm3": 898.650, "fp_voxels": 883508},
        "ET": {"dice": 0.0282, "sensitivity": 0.1989, "precision": 0.0152, "pred_volume_cm3": 412.057, "fp_voxels": 405795},
        "mean_dice": 0.0564
    }

    # Save Candidate Checkpoint
    ckpt_payload = {
        "model_state_dict": model.state_dict(),
        "config": {
            "architecture": "FourChannelUNet3D",
            "in_channels": 4,
            "out_channels": 4,
            "channels": (16, 32, 64, 128, 256),
            "modality_order": list(MODALITY_ORDER)
        },
        "metadata": {
            "experiment": "Exp1_BalancedPatches_1to2",
            "intervention": "Rebalanced patch sampling (2 foreground / 4 background)",
            "seed": 42,
            "epochs": 5,
            "candidate_metrics": candidate_metrics,
            "baseline_metrics": baseline_metrics,
            "training_history": training_log,
            "training_time_sec": round(total_train_time, 2)
        }
    }

    torch.save(ckpt_payload, str(out_checkpoint))
    ckpt_hash = compute_file_sha256(out_checkpoint)
    print(f"\n[SAVED] Unique Candidate Checkpoint: {out_checkpoint}")
    print(f"  Size: {out_checkpoint.stat().st_size:,} bytes")
    print(f"  SHA-256: {ckpt_hash}")

    # Save structured results
    res_path = PROJECT_ROOT / "reports" / "experiment_phase5_balanced_patches_results.json"
    with open(res_path, "w") as f:
        json.dump({
            "experiment_id": "Exp1_BalancedPatches_1to2",
            "candidate_checkpoint": str(out_checkpoint.name),
            "candidate_sha256": ckpt_hash,
            "candidate_metrics": candidate_metrics,
            "baseline_metrics": baseline_metrics,
            "training_log": training_log,
            "training_time_sec": round(total_train_time, 2),
            "val_time_sec": round(val_time, 2)
        }, f, indent=2)
    print(f"[SAVED] Structured Comparison Data: {res_path}")


if __name__ == "__main__":
    main()
