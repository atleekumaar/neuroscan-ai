"""
NEUROSCAN AI — Stage 4A: One-Epoch Laptop-Safe Pilot Training
Target Hardware: Acer Nitro V15 (NVIDIA RTX 4050 6GB Laptop GPU, 16GB System RAM)
Runtime: Python 3.12 (torch 2.6.0+cu124, monai 1.4.0, nibabel 5.4.2)

Strict Protocol:
- Exactly 1 bounded epoch (18 patches total: 6 per training study across 3 studies)
- Patch size: 48x48x48, Batch size: 1, DataLoader workers: 0
- AMP (Automatic Mixed Precision) with GradScaler
- Continuous telemetry safeguards: abort if RAM < 2.0 GB or GPU temp >= 80°C
- Initial RAM check: must be >= 3.0 GB available to start
- Locked test study (BraTS-GLI-01012-000) strictly quarantined
- Independent validation solely on BraTS-GLI-00481-000
- Atomic checkpoint saving to: models/NeuroScan_Seg3D_UNet_LaptopSafe_Pilot.pth
- Never overwrite historical checkpoints
- Label checkpoint: EXPERIMENTAL - NOT CLINICALLY VALIDATED
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
    """Reads GPU temperature via nvidia-smi."""
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
    """
    Extracts 2 foreground and 4 background patches (1:2 ratio) as established in Exp 1.
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

    # 2. Background patches (4 per volume)
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
    print("NEUROSCAN AI — Stage 4A: One-Epoch Laptop-Safe Pilot Training")
    print("=" * 80)

    # 1. Environment & Hardware Preflight Checks
    if not torch.cuda.is_available():
        raise RuntimeError("CRITICAL BLOCKER: CUDA is not available on this PyTorch installation.")

    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Preflight RAM check
    init_avail_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
    init_total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    init_gpu_temp = get_gpu_temperature()

    print(f"[PREFLIGHT] Available RAM: {init_avail_ram_gb:.2f} GB / {init_total_ram_gb:.2f} GB")
    print(f"[PREFLIGHT] GPU: {torch.cuda.get_device_name(0)} | Initial Temperature: {init_gpu_temp}°C")

    # Safety Gate: Must have >= 3.0 GB available RAM
    if init_avail_ram_gb < 3.0:
        msg = f"SAFETY BLOCKER: Available system RAM ({init_avail_ram_gb:.2f} GB) is below required 3.0 GB threshold."
        print(f"[STOP] {msg}")
        return {"status": "BLOCKED_LOW_RAM", "message": msg}

    # Safety Gate: GPU Temperature must be nominal
    if init_gpu_temp >= 80.0:
        msg = f"SAFETY BLOCKER: GPU temperature ({init_gpu_temp}°C) is above safe threshold (80°C)."
        print(f"[STOP] {msg}")
        return {"status": "BLOCKED_HIGH_TEMP", "message": msg}

    # 2. Checkpoint Provenance & Destination Verification
    models_dir = PROJECT_ROOT / "models"
    out_checkpoint = models_dir / "NeuroScan_Seg3D_UNet_LaptopSafe_Pilot.pth"

    protected_checkpoints = [
        "NeuroScan_Seg3D_UNet_Phase5_Multimodal_Best.pth",
        "NeuroScan_Seg3D_UNet_Phase5_Exp1_BalancedPatches.pth",
        "NeuroScan_Seg3D_UNet_Phase4B_Best.pth",
        "NeuroScan_Seg3D_UNet_Phase4B_Final.pth",
        "NeuroScan_Seg3D_UNet_Phase3_Exp2_BalancedSampling.pth",
        "NeuroScan_Seg3D_UNet.pth",
        "NeuroScan_Seg3D_UNet_T1CE_SingleChannel.pth"
    ]
    assert out_checkpoint.name not in protected_checkpoints, f"CRITICAL: Destination {out_checkpoint.name} matches protected file!"

    # Verify initialization checkpoint
    init_ckpt_path = models_dir / "NeuroScan_Seg3D_UNet_Phase5_Exp1_BalancedPatches.pth"
    if not init_ckpt_path.exists():
        raise FileNotFoundError(f"Initialization checkpoint missing: {init_ckpt_path}")

    init_ckpt_hash = compute_file_sha256(init_ckpt_path)
    expected_init_hash = "0e3a312153fb55d4d538564fb55f9d2e377b5305924cbcc7e139d889c375c268"
    assert init_ckpt_hash == expected_init_hash, f"Init checkpoint hash mismatch: {init_ckpt_hash} != {expected_init_hash}"
    print(f"[PROVENANCE] Verified init checkpoint: {init_ckpt_path.name}")
    print(f"             SHA-256: {init_ckpt_hash}")

    # 3. Cohort Split Verification & Quarantine Enforcement
    splits_file = PROJECT_ROOT / "backend" / "segmentation" / "multimodal_splits.json"
    with open(splits_file, "r") as f:
        split_meta = json.load(f)

    train_ids = [s["patient_id"] for s in split_meta["splits"]["train"]]
    val_ids = [s["patient_id"] for s in split_meta["splits"]["val"]]
    locked_ids = [s["patient_id"] for s in split_meta["splits"]["locked_test"]]

    assert len(train_ids) == 3, f"Expected 3 training studies, got {len(train_ids)}"
    assert "BraTS-GLI-01012-000" in locked_ids, "Locked test case missing from quarantine list!"
    assert "BraTS-GLI-01012-000" not in train_ids, "SAFETY VIOLATION: Locked test case present in train split!"
    assert "BraTS-GLI-01012-000" not in val_ids, "SAFETY VIOLATION: Locked test case present in val split!"

    print(f"[COHORT] Training Studies: {train_ids}")
    print(f"[COHORT] Validation Study: {val_ids}")
    print(f"[COHORT] Locked Test Study (QUARANTINED): {locked_ids}")

    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # 4. Extract Patches Sequentially (Memory-Conscious)
    pilot_dir = PROJECT_ROOT / "backend" / "segmentation" / "data" / "pilot"
    patch_size = (48, 48, 48)
    n_fg_per_vol = 2
    n_bg_per_vol = 4
    total_expected_patches = len(train_ids) * (n_fg_per_vol + n_bg_per_vol)  # 18

    print(f"\n[DATA] Extracting {total_expected_patches} patches total ({n_fg_per_vol} FG, {n_bg_per_vol} BG per study)...")
    all_patches_x = []
    all_patches_y = []

    for pid in train_ids:
        sdir = pilot_dir / pid
        img, seg, meta = load_multimodal_study(sdir, normalize=True)
        px, py = extract_balanced_patches(
            img, seg,
            patch_size=patch_size,
            n_foreground=n_fg_per_vol,
            n_background=n_bg_per_vol
        )
        all_patches_x.append(px)
        all_patches_y.append(py)
        print(f"  Extracted from {pid}: {px.shape[0]} patches")
        del img, seg
        gc.collect()

    all_x = torch.cat(all_patches_x, dim=0)
    all_y = torch.cat(all_patches_y, dim=0)
    assert all_x.shape[0] == total_expected_patches, f"Expected {total_expected_patches} patches, got {all_x.shape[0]}"
    print(f"[DATA] Total training batch tensor assembled: {all_x.shape} (image), {all_y.shape} (label)")

    # Shuffle patches deterministically
    perm = torch.randperm(all_x.shape[0])
    all_x = all_x[perm]
    all_y = all_y[perm]

    # 5. Build Model and Load Initialization Checkpoint
    config = FourChannelUNetConfig(
        in_channels=4,
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        dropout=0.2
    )
    model, is_loaded, meta = load_segmentation_model(str(init_ckpt_path), device=device)
    if not is_loaded:
        raise RuntimeError(f"Failed to load initialization weights from {init_ckpt_path}")
    print(f"[MODEL] Model architecture initialized on {device} from {init_ckpt_path.name}")
    print(f"        Loaded status: {meta.get('status_message')}")

    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    scaler = torch.amp.GradScaler('cuda')

    use_amp = False  # Set to False because AMP failed numerical stability test (non-finite gradient at batch 13)
    batch_size = 1  # Strictly 1
    n_batches = all_x.shape[0] // batch_size
    print(f"\n[TRAINING] Starting 1 Pilot Epoch ({n_batches} batches, batch_size={batch_size}, AMP={use_amp})...")

    batch_losses = []
    grad_norms = []
    min_avail_ram_gb = init_avail_ram_gb
    peak_vram_mb = 0.0
    t0_train = time.perf_counter()

    for b in range(n_batches):
        t_batch_start = time.perf_counter()

        # Telemetry safeguard checks before each batch
        current_avail_ram = psutil.virtual_memory().available / (1024 ** 3)
        current_temp = get_gpu_temperature()
        min_avail_ram_gb = min(min_avail_ram_gb, current_avail_ram)

        if current_avail_ram < 2.0:
            raise RuntimeError(f"ABORT TRIGGERED: Available RAM dropped to {current_avail_ram:.2f} GB (below 2.0 GB limit).")
        if current_temp >= 80.0:
            raise RuntimeError(f"ABORT TRIGGERED: GPU temperature reached {current_temp}°C (at or above 80°C threshold).")

        bx = all_x[b:b + 1].to(device, non_blocking=True)
        by = all_y[b:b + 1].to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                out = model(bx)
                loss = loss_fn(out, by)

            if not torch.isfinite(loss):
                raise RuntimeError(f"ABORT TRIGGERED: Non-finite loss at batch {b+1}: {loss.item()}")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            out = model(bx)
            loss = loss_fn(out, by)

            if not torch.isfinite(loss):
                raise RuntimeError(f"ABORT TRIGGERED: Non-finite loss at batch {b+1}: {loss.item()}")

            loss.backward()

        # Check gradients
        step_grads = []
        for p in model.parameters():
            if p.grad is not None:
                if not torch.isfinite(p.grad).all():
                    raise RuntimeError(f"ABORT TRIGGERED: Non-finite gradient at batch {b+1}")
                step_grads.append(p.grad.norm().item())

        total_grad_norm = float(np.linalg.norm(step_grads))
        grad_norms.append(total_grad_norm)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        batch_loss = float(loss.item())
        batch_losses.append(batch_loss)

        vram_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        peak_vram_mb = max(peak_vram_mb, vram_reserved)
        batch_ms = (time.perf_counter() - t_batch_start) * 1000

        print(f"  Batch {b+1:02d}/{n_batches:02d} | Loss: {batch_loss:.4f} | GradNorm: {total_grad_norm:.4f} | VRAM: {vram_reserved:.1f}MB | RAM Avail: {current_avail_ram:.2f}GB | Temp: {current_temp}°C | Time: {batch_ms:.0f}ms")

    elapsed_train_sec = time.perf_counter() - t0_train
    mean_train_loss = float(np.mean(batch_losses))
    mean_grad_norm = float(np.mean(grad_norms))
    print(f"\n[TRAINING COMPLETE] 1 Epoch Elapsed Time: {elapsed_train_sec:.2f}s | Mean Loss: {mean_train_loss:.4f} | Peak VRAM: {peak_vram_mb:.1f} MB")

    # 6. Independent Validation on BraTS-GLI-00481-000
    val_pid = val_ids[0]
    print(f"\n[VALIDATION] Evaluating strictly on independent validation subject: {val_pid}...")
    t0_val = time.perf_counter()

    val_dir = pilot_dir / val_pid
    val_img, val_seg, val_meta = load_multimodal_study(val_dir, normalize=True)
    model.eval()

    with torch.no_grad():
        val_pred = predict_multimodal_study(
            model=model,
            image_tensor=val_img,
            roi_size=(64, 64, 64),
            overlap=0.5,
            device=device
        )

    val_metrics = compute_subregion_metrics(val_pred, val_seg)
    elapsed_val_sec = time.perf_counter() - t0_val
    post_val_vram_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    peak_vram_mb = max(peak_vram_mb, post_val_vram_mb)

    print(f"[VALIDATION COMPLETE] Elapsed: {elapsed_val_sec:.2f}s")
    print(f"  WT Dice: {val_metrics['WT']['dice']} (Sens: {val_metrics['WT']['sensitivity']}, Prec: {val_metrics['WT']['precision']}, PredVol: {val_metrics['WT']['pred_volume_cm3']} cm3)")
    print(f"  TC Dice: {val_metrics['TC']['dice']} (Sens: {val_metrics['TC']['sensitivity']}, Prec: {val_metrics['TC']['precision']}, PredVol: {val_metrics['TC']['pred_volume_cm3']} cm3)")
    print(f"  ET Dice: {val_metrics['ET']['dice']} (Sens: {val_metrics['ET']['sensitivity']}, Prec: {val_metrics['ET']['precision']}, PredVol: {val_metrics['ET']['pred_volume_cm3']} cm3)")
    print(f"  Mean Dice: {val_metrics['mean_dice']}")

    # 7. Atomic Checkpoint Save
    print("\n[CHECKPOINT] Saving pilot checkpoint atomically...")
    tmp_ckpt_path = models_dir / "NeuroScan_Seg3D_UNet_LaptopSafe_Pilot.pth.tmp"

    checkpoint_payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            "in_channels": 4,
            "out_channels": 4,
            "channels": [16, 32, 64, 128, 256],
            "dropout": 0.2,
            "patch_size": [48, 48, 48],
            "batch_size": 1,
            "amp": use_amp
        },
        "metadata": {
            "model_name": "NeuroScan 3D U-Net Laptop-Safe Pilot (Stage 4A)",
            "clinical_status": "EXPERIMENTAL - NOT CLINICALLY VALIDATED",
            "training_type": "Laptop-Safe Bounded Pilot (1 Epoch, 18 Patches)",
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0),
            "initialization_checkpoint": init_ckpt_path.name,
            "initialization_checkpoint_sha256": init_ckpt_hash,
            "training_cohort": train_ids,
            "validation_cohort": val_ids,
            "locked_test_quarantined": locked_ids,
            "train_loss_mean": round(mean_train_loss, 4),
            "val_metrics": val_metrics,
            "elapsed_train_sec": round(elapsed_train_sec, 2),
            "elapsed_val_sec": round(elapsed_val_sec, 2),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "min_avail_ram_gb": round(min_avail_ram_gb, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }

    torch.save(checkpoint_payload, tmp_ckpt_path)
    os.replace(tmp_ckpt_path, out_checkpoint)

    final_ckpt_hash = compute_file_sha256(out_checkpoint)
    final_ckpt_size_mb = out_checkpoint.stat().st_size / (1024 ** 2)
    print(f"  Atomically saved to: {out_checkpoint.name} ({final_ckpt_size_mb:.2f} MB)")
    print(f"  Final Checkpoint SHA-256: {final_ckpt_hash}")

    # Reload check to confirm integrity
    reloaded_model, is_reloaded, reload_meta = load_segmentation_model(str(out_checkpoint), device=device)
    assert is_reloaded, "Integrity check failed: Unable to reload saved pilot checkpoint!"
    print(f"  Reload integrity check: PASSED ({reload_meta.get('status_message')})")

    # Final telemetry
    final_temp = get_gpu_temperature()
    final_avail_ram = psutil.virtual_memory().available / (1024 ** 3)

    results = {
        "status": "COMPLETED_SAFELY",
        "clinical_label": "EXPERIMENTAL - NOT CLINICALLY VALIDATED",
        "device": torch.cuda.get_device_name(0),
        "total_vram_mb": round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "min_available_ram_gb": round(min_avail_ram_gb, 2),
        "final_available_ram_gb": round(final_avail_ram, 2),
        "final_gpu_temp_c": final_temp,
        "elapsed_train_sec": round(elapsed_train_sec, 2),
        "elapsed_val_sec": round(elapsed_val_sec, 2),
        "mean_train_loss": round(mean_train_loss, 4),
        "mean_grad_norm": round(mean_grad_norm, 4),
        "validation_metrics": val_metrics,
        "checkpoint_path": str(out_checkpoint.relative_to(PROJECT_ROOT).as_posix()),
        "checkpoint_sha256": final_ckpt_hash,
        "checkpoint_size_mb": round(final_ckpt_size_mb, 2),
        "initialization_checkpoint": init_ckpt_path.name,
        "initialization_checkpoint_sha256": init_ckpt_hash,
        "training_cohort": train_ids,
        "validation_cohort": val_ids,
        "locked_test_cohort": locked_ids,
        "quarantine_verified": True
    }

    report_path = PROJECT_ROOT / "reports" / "laptop_safe_pilot_results.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("STAGE 4A PILOT RUN COMPLETED SAFELY")
    print(f"  Checkpoint: {out_checkpoint.name} ({final_ckpt_hash[:16]}...)")
    print(f"  Label: EXPERIMENTAL - NOT CLINICALLY VALIDATED")
    print(f"  Summary JSON saved to: {report_path}")
    print("=" * 80)
    return results


if __name__ == "__main__":
    main()
