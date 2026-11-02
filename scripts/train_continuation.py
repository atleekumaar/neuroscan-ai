"""
NEUROSCAN AI — Stage 4B: 3-Epoch Controlled Continuation
Target Hardware: Acer Nitro V15 (NVIDIA RTX 4050 6GB Laptop GPU, 16GB System RAM)
Runtime: Python 3.12 (torch 2.6.0+cu124, monai 1.4.0, nibabel 5.4.2)

Strict Protocol:
- Continues directly from verified pilot checkpoint (NeuroScan_Seg3D_UNet_LaptopSafe_Pilot.pth)
- Restores both model_state_dict and optimizer_state_dict
- Trains exactly 3 additional epochs (Epochs 2, 3, 4)
- 18 patches per epoch (54 patches total across all 3 epochs)
- Patch size: 48x48x48, Batch size: 1, Workers: 0
- Precision: Strict FP32 (no AMP)
- Dynamic telemetry checks: Stop if RAM < 2.0 GB or GPU temp >= 80°C
- Quarantine: Locked test subject (BraTS-GLI-01012-000) 100% excluded
- Independent validation per epoch solely on BraTS-GLI-00481-000
- Atomic saving per epoch to distinct experiment filenames
- Label: EXPERIMENTAL - NOT CLINICALLY VALIDATED
"""

import os
import sys
import gc
import time
import json
import psutil
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.segmentation.model import (
    FourChannelUNetConfig,
    build_four_channel_unet,
    load_segmentation_model
)
from backend.segmentation.pipeline import (
    MODALITY_ORDER,
    load_multimodal_study,
    predict_multimodal_study
)
from monai.losses import DiceCELoss


def compute_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def get_gpu_temperature() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return float(out)
    except Exception:
        return -1.0


def extract_balanced_patches(
    image: torch.Tensor,
    seg: torch.Tensor,
    patch_size: Tuple[int, int, int] = (48, 48, 48),
    n_foreground: int = 2,
    n_background: int = 4
) -> Tuple[torch.Tensor, torch.Tensor]:
    _, _, H, W, D = image.shape
    pd, ph, pw = patch_size

    fg_indices = (seg[0, 0] > 0).nonzero(as_tuple=False)

    patches_x = []
    patches_y = []

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
    print("NEUROSCAN AI — Stage 4B: 3-Epoch Controlled Continuation")
    print("=" * 80)

    # 1. Environment & Hardware Checks
    if not torch.cuda.is_available():
        raise RuntimeError("CRITICAL BLOCKER: CUDA is not available.")

    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    init_avail_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
    init_gpu_temp = get_gpu_temperature()

    print(f"[PREFLIGHT] Available RAM: {init_avail_ram_gb:.2f} GB | GPU Temp: {init_gpu_temp}°C")

    if init_avail_ram_gb < 3.0:
        msg = f"SAFETY BLOCKER: Available system RAM ({init_avail_ram_gb:.2f} GB) is below 3.0 GB."
        print(f"[STOP] {msg}")
        return {"status": "BLOCKED_LOW_RAM", "message": msg}

    if init_gpu_temp >= 80.0:
        msg = f"SAFETY BLOCKER: GPU temperature ({init_gpu_temp}°C) is above safe threshold (80°C)."
        print(f"[STOP] {msg}")
        return {"status": "BLOCKED_HIGH_TEMP", "message": msg}

    # 2. Checkpoint Provenance
    models_dir = PROJECT_ROOT / "models"
    pilot_ckpt_path = models_dir / "NeuroScan_Seg3D_UNet_LaptopSafe_Pilot.pth"
    if not pilot_ckpt_path.exists():
        raise FileNotFoundError(f"Pilot checkpoint missing: {pilot_ckpt_path}")

    pilot_ckpt_hash = compute_file_sha256(pilot_ckpt_path)
    expected_pilot_hash = "918f60686c276835657c8e2fdced565c3255cb0517c873781b1d4910fa544c29"
    assert pilot_ckpt_hash == expected_pilot_hash, f"Pilot hash mismatch: {pilot_ckpt_hash} != {expected_pilot_hash}"
    print(f"[PROVENANCE] Verified pilot checkpoint: {pilot_ckpt_path.name}")
    print(f"             SHA-256: {pilot_ckpt_hash}")

    # 3. Cohort Split Verification & Quarantine Enforcement
    splits_file = PROJECT_ROOT / "backend" / "segmentation" / "multimodal_splits.json"
    with open(splits_file, "r") as f:
        split_meta = json.load(f)

    train_ids = [s["patient_id"] for s in split_meta["splits"]["train"]]
    val_ids = [s["patient_id"] for s in split_meta["splits"]["val"]]
    locked_ids = [s["patient_id"] for s in split_meta["splits"]["locked_test"]]

    assert "BraTS-GLI-01012-000" in locked_ids, "Locked test case missing from quarantine list!"
    assert "BraTS-GLI-01012-000" not in train_ids, "SAFETY VIOLATION: Locked test case present in train split!"
    assert "BraTS-GLI-01012-000" not in val_ids, "SAFETY VIOLATION: Locked test case present in val split!"

    print(f"[COHORT] Training Studies: {train_ids}")
    print(f"[COHORT] Validation Study: {val_ids}")
    print(f"[COHORT] Locked Test Study (QUARANTINED): {locked_ids}")

    # 4. Load Pilot Checkpoint Model and Optimizer State
    checkpoint = torch.load(pilot_ckpt_path, map_location=device, weights_only=False)
    model, is_loaded, meta = load_segmentation_model(str(pilot_ckpt_path), device=device)
    assert is_loaded, f"Failed to load model from {pilot_ckpt_path}"
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"[OPTIMIZER] Loaded existing AdamW state from pilot checkpoint (63 state entries restored).")
    else:
        print(f"[OPTIMIZER] Initialized fresh AdamW state.")

    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)

    pilot_dir = PROJECT_ROOT / "backend" / "segmentation" / "data" / "pilot"
    val_dir = pilot_dir / val_ids[0]
    val_img, val_seg, _ = load_multimodal_study(val_dir, normalize=True)

    epochs_to_run = [2, 3, 4]  # Continuation epochs
    epoch_results = []
    min_avail_ram_overall = init_avail_ram_gb
    peak_vram_overall = 0.0

    print(f"\n[TRAINING] Starting 3-Epoch Continuation (Epochs 2 -> 4, FP32, batch_size=1)...")

    for epoch in epochs_to_run:
        epoch_start_time = time.perf_counter()
        torch.manual_seed(42 + epoch)
        np.random.seed(42 + epoch)

        # Extract 18 patches (6 per study: 2 FG, 4 BG)
        all_patches_x = []
        all_patches_y = []
        for pid in train_ids:
            sdir = pilot_dir / pid
            img, seg, _ = load_multimodal_study(sdir, normalize=True)
            px, py = extract_balanced_patches(img, seg, patch_size=(48, 48, 48), n_foreground=2, n_background=4)
            all_patches_x.append(px)
            all_patches_y.append(py)
            del img, seg
            gc.collect()

        all_x = torch.cat(all_patches_x, dim=0)
        all_y = torch.cat(all_patches_y, dim=0)
        perm = torch.randperm(all_x.shape[0])
        all_x = all_x[perm]
        all_y = all_y[perm]

        n_batches = all_x.shape[0]
        model.train()
        batch_losses = []
        grad_norms = []

        for b in range(n_batches):
            t_b_start = time.perf_counter()
            current_avail_ram = psutil.virtual_memory().available / (1024 ** 3)
            current_temp = get_gpu_temperature()
            min_avail_ram_overall = min(min_avail_ram_overall, current_avail_ram)

            if current_avail_ram < 2.0:
                raise RuntimeError(f"ABORT TRIGGERED: Available RAM ({current_avail_ram:.2f} GB) < 2.0 GB.")
            if current_temp >= 80.0:
                raise RuntimeError(f"ABORT TRIGGERED: GPU temperature ({current_temp}°C) >= 80°C.")

            bx = all_x[b:b + 1].to(device, non_blocking=True)
            by = all_y[b:b + 1].to(device, non_blocking=True)

            optimizer.zero_grad()
            out = model(bx)
            loss = loss_fn(out, by)

            if not torch.isfinite(loss):
                raise RuntimeError(f"ABORT: Non-finite loss at epoch {epoch}, batch {b+1}: {loss.item()}")

            loss.backward()

            step_grads = []
            for p in model.parameters():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        raise RuntimeError(f"ABORT: Non-finite grad at epoch {epoch}, batch {b+1}")
                    step_grads.append(p.grad.norm().item())

            total_norm = float(np.linalg.norm(step_grads))
            grad_norms.append(total_norm)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_loss = float(loss.item())
            batch_losses.append(batch_loss)

            vram_res = torch.cuda.memory_reserved(device) / (1024 ** 2)
            peak_vram_overall = max(peak_vram_overall, vram_res)
            b_ms = (time.perf_counter() - t_b_start) * 1000
            print(f"  Epoch {epoch}/4 | Batch {b+1:02d}/{n_batches:02d} | Loss: {batch_loss:.4f} | Grad: {total_norm:.4f} | VRAM: {vram_res:.1f}MB | RAM: {current_avail_ram:.2f}GB | Temp: {current_temp}°C | {b_ms:.0f}ms")

        epoch_train_time = time.perf_counter() - epoch_start_time
        mean_epoch_loss = float(np.mean(batch_losses))
        mean_epoch_grad = float(np.mean(grad_norms))

        # Validation phase
        print(f"\n  [VALIDATION Epoch {epoch}] Evaluating on {val_ids[0]}...")
        t0_v = time.perf_counter()
        model.eval()
        with torch.no_grad():
            val_pred = predict_multimodal_study(model=model, image_tensor=val_img, roi_size=(64, 64, 64), overlap=0.5, device=device)

        v_metrics = compute_subregion_metrics(val_pred, val_seg)
        epoch_val_time = time.perf_counter() - t0_v
        post_val_vram = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        peak_vram_overall = max(peak_vram_overall, post_val_vram)

        print(f"  Validation Done in {epoch_val_time:.2f}s | WT Dice: {v_metrics['WT']['dice']:.4f} (Prec: {v_metrics['WT']['precision']:.4f}, Sens: {v_metrics['WT']['sensitivity']:.4f}, Vol: {v_metrics['WT']['pred_volume_cm3']} cm3)")
        print(f"  TC Dice: {v_metrics['TC']['dice']:.4f} | ET Dice: {v_metrics['ET']['dice']:.4f} | Mean Dice: {v_metrics['mean_dice']:.4f}")

        # Atomic Save for Epoch
        epoch_ckpt_name = f"NeuroScan_Seg3D_UNet_Stage4B_Epoch{epoch}{'_Final' if epoch == 4 else ''}.pth"
        epoch_ckpt_path = models_dir / epoch_ckpt_name
        tmp_ckpt = epoch_ckpt_path.with_suffix(".tmp")

        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "in_channels": 4,
                "out_channels": 4,
                "channels": [16, 32, 64, 128, 256],
                "dropout": 0.2,
                "patch_size": [48, 48, 48],
                "batch_size": 1,
                "amp": False
            },
            "metadata": {
                "model_name": f"NeuroScan 3D U-Net Stage 4B Epoch {epoch}",
                "clinical_status": "EXPERIMENTAL - NOT CLINICALLY VALIDATED",
                "epoch": epoch,
                "total_epochs": 4,
                "mean_train_loss": round(mean_epoch_loss, 4),
                "mean_grad_norm": round(mean_epoch_grad, 4),
                "val_metrics": v_metrics,
                "initialization_checkpoint": pilot_ckpt_path.name,
                "initialization_checkpoint_sha256": pilot_ckpt_hash,
                "elapsed_train_sec": round(epoch_train_time, 2),
                "elapsed_val_sec": round(epoch_val_time, 2),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        torch.save(payload, tmp_ckpt)
        os.replace(tmp_ckpt, epoch_ckpt_path)
        ckpt_hash = compute_file_sha256(epoch_ckpt_path)
        print(f"  [SAVED] Atomically saved {epoch_ckpt_path.name} | SHA-256: {ckpt_hash}\n")

        epoch_results.append({
            "epoch": epoch,
            "mean_loss": round(mean_epoch_loss, 4),
            "mean_grad_norm": round(mean_epoch_grad, 4),
            "train_time_sec": round(epoch_train_time, 2),
            "val_time_sec": round(epoch_val_time, 2),
            "val_metrics": v_metrics,
            "checkpoint_name": epoch_ckpt_name,
            "checkpoint_sha256": ckpt_hash
        })

    final_temp = get_gpu_temperature()
    final_avail_ram = psutil.virtual_memory().available / (1024 ** 3)

    summary = {
        "status": "COMPLETED_SAFELY",
        "clinical_label": "EXPERIMENTAL - NOT CLINICALLY VALIDATED",
        "continuation_epochs_run": epochs_to_run,
        "total_patches_trained": len(epochs_to_run) * 18,
        "device": torch.cuda.get_device_name(0),
        "peak_vram_mb": round(peak_vram_overall, 2),
        "min_available_ram_gb": round(min_avail_ram_overall, 2),
        "final_available_ram_gb": round(final_avail_ram, 2),
        "final_gpu_temp_c": final_temp,
        "epoch_results": epoch_results,
        "initialization_checkpoint": pilot_ckpt_path.name,
        "initialization_checkpoint_sha256": pilot_ckpt_hash,
        "final_checkpoint": epoch_results[-1]["checkpoint_name"],
        "final_checkpoint_sha256": epoch_results[-1]["checkpoint_sha256"],
        "training_cohort": train_ids,
        "validation_cohort": val_ids,
        "locked_test_cohort": locked_ids,
        "quarantine_verified": True
    }

    report_path = PROJECT_ROOT / "reports" / "stage4b_continuation_results.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print("STAGE 4B 3-EPOCH CONTINUATION COMPLETE")
    print(f"  Results saved to: {report_path}")
    print("=" * 80)
    return summary


if __name__ == "__main__":
    main()
