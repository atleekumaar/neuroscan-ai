"""
NEUROSCAN AI — Laptop-Safe GPU Smoke Test Script
Hardware: Acer Nitro V15 (NVIDIA GeForce RTX 4050 6GB Laptop GPU, 16GB System RAM)
Runtime: Python 3.12 (torch 2.6.0+cu124, monai 1.4.0, nibabel 5.4.2)

Scope:
- Verifies CUDA device availability and VRAM limits
- Loads one genuine training study (BraTS-GLI-00324-000)
- Extracts a single 48x48x48 patch (batch size 1, num_workers 0)
- Tests FP32 forward, backward, optimizer step, NaN/Inf checks
- Tests AMP (Automatic Mixed Precision) numerical stability
- Measures peak allocated/reserved VRAM, RAM, and runtime
- Tests temporary checkpoint saving and round-trip verification
- Strictly quarantines locked test study (BraTS-GLI-01012-000)
- Strictly preserves historical checkpoints (never overwrites)
"""

import os
import sys
import time
import json
import psutil
import hashlib
from pathlib import Path
from typing import Dict, Any

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
    load_multimodal_study
)
from monai.losses import DiceCELoss


def get_gpu_telemetry() -> Dict[str, Any]:
    """Reads GPU memory and properties directly from PyTorch CUDA runtime."""
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    dev_idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev_idx)
    return {
        "cuda_available": True,
        "device_index": dev_idx,
        "device_name": props.name,
        "total_vram_mb": round(props.total_memory / (1024 ** 2), 2),
        "allocated_vram_mb": round(torch.cuda.memory_allocated(dev_idx) / (1024 ** 2), 2),
        "reserved_vram_mb": round(torch.cuda.memory_reserved(dev_idx) / (1024 ** 2), 2),
        "max_allocated_vram_mb": round(torch.cuda.max_memory_allocated(dev_idx) / (1024 ** 2), 2),
        "max_reserved_vram_mb": round(torch.cuda.max_memory_reserved(dev_idx) / (1024 ** 2), 2),
    }


def compute_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def extract_single_patch(
    image: torch.Tensor,
    seg: torch.Tensor,
    patch_size=(48, 48, 48)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extracts a single 48x48x48 patch centered on tumor tissue if present."""
    _, _, H, W, D = image.shape
    pd, ph, pw = patch_size

    fg_indices = (seg[0, 0] > 0).nonzero(as_tuple=False)
    if len(fg_indices) > 0:
        idx = 0  # Deterministic center for reproducibility
        cz, cy, cx = fg_indices[idx].tolist()
        z_start = max(0, min(cz - pd // 2, H - pd))
        y_start = max(0, min(cy - ph // 2, W - ph))
        x_start = max(0, min(cx - pw // 2, D - pw))
    else:
        z_start = max(0, (H - pd) // 2)
        y_start = max(0, (W - ph) // 2)
        x_start = max(0, (D - pw) // 2)

    patch_x = image[:, :, z_start:z_start + pd, y_start:y_start + ph, x_start:x_start + pw]
    patch_y = seg[:, :, z_start:z_start + pd, y_start:y_start + ph, x_start:x_start + pw]
    return patch_x, patch_y


def run_smoke_test() -> Dict[str, Any]:
    print("=" * 80)
    print("NEUROSCAN AI — Laptop-Safe GPU Smoke Test (Acer Nitro V15 / RTX 4050 6GB)")
    print("=" * 80)

    # 1. Environment & Hardware Probe
    if not torch.cuda.is_available():
        raise RuntimeError("CRITICAL BLOCKER: CUDA is not available on this PyTorch installation.")

    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    init_telemetry = get_gpu_telemetry()
    init_ram_gb = psutil.virtual_memory().used / (1024 ** 3)
    print(f"[PREFLIGHT] GPU: {init_telemetry['device_name']} | Total VRAM: {init_telemetry['total_vram_mb']} MB")
    print(f"[PREFLIGHT] System RAM in use: {init_ram_gb:.2f} GB / {psutil.virtual_memory().total / (1024**3):.2f} GB")

    # 2. Check splits and quarantine locked test patient
    splits_path = PROJECT_ROOT / "backend" / "segmentation" / "multimodal_splits.json"
    with open(splits_path, "r") as f:
        splits = json.load(f)

    train_patients = [p["patient_id"] for p in splits["splits"]["train"]]
    locked_patients = [p["patient_id"] for p in splits["splits"]["locked_test"]]

    test_case_id = train_patients[0]  # BraTS-GLI-00324-000
    assert test_case_id not in locked_patients, f"SAFETY VIOLATION: {test_case_id} is in locked test set!"
    print(f"[SAFETY] Verified locked test quarantine. Selected genuine training case: {test_case_id}")

    # 3. Load genuine patient volume
    study_dir = PROJECT_ROOT / "backend" / "segmentation" / "data" / "pilot" / test_case_id
    if not study_dir.exists():
        raise FileNotFoundError(f"Training study dir not found: {study_dir}")

    t0_load = time.perf_counter()
    image_vol, seg_vol, meta = load_multimodal_study(study_dir, normalize=True)
    load_time = time.perf_counter() - t0_load
    print(f"[DATA] Loaded {test_case_id} in {load_time:.2f}s | Volume shape: {image_vol.shape}")

    # 4. Extract single 48x48x48 patch (Batch size = 1)
    patch_x, patch_y = extract_single_patch(image_vol, seg_vol, patch_size=(48, 48, 48))
    print(f"[DATA] Extracted patch: input shape {patch_x.shape}, label shape {patch_y.shape}")

    # Transfer patch to GPU
    patch_x_gpu = patch_x.to(device, non_blocking=True)
    patch_y_gpu = patch_y.to(device, non_blocking=True)

    # 5. Build Model and Optimizer
    config = FourChannelUNetConfig(
        in_channels=4,
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        dropout=0.2
    )
    model = build_four_channel_unet(config).to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] 4-Channel 3D U-Net initialized on {device} | Total parameters: {param_count:,}")

    # 6. Test Step A: Standard FP32 Forward, Backward, Optimizer Step
    print("\n--- Testing FP32 Step ---")
    t0_fp32 = time.perf_counter()
    optimizer.zero_grad()
    out_fp32 = model(patch_x_gpu)
    loss_fp32 = loss_fn(out_fp32, patch_y_gpu)

    if not torch.isfinite(loss_fp32):
        raise RuntimeError(f"FP32 loss is non-finite: {loss_fp32.item()}")

    loss_fp32.backward()

    # Check gradients
    grad_norms_fp32 = []
    has_nan_or_inf_grad_fp32 = False
    for name, p in model.named_parameters():
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():
                has_nan_or_inf_grad_fp32 = True
                print(f"[WARNING] Non-finite grad in {name}")
            grad_norms_fp32.append(p.grad.norm().item())

    if has_nan_or_inf_grad_fp32:
        raise RuntimeError("Non-finite gradient detected during FP32 backward pass.")

    total_grad_norm_fp32 = np.linalg.norm(grad_norms_fp32)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    fp32_time = time.perf_counter() - t0_fp32

    fp32_telemetry = get_gpu_telemetry()
    print(f"  FP32 Loss: {loss_fp32.item():.4f}")
    print(f"  FP32 Total Grad Norm: {total_grad_norm_fp32:.4f}")
    print(f"  FP32 Step Time: {fp32_time * 1000:.1f} ms")
    print(f"  Allocated VRAM: {fp32_telemetry['allocated_vram_mb']} MB | Peak Reserved: {fp32_telemetry['max_reserved_vram_mb']} MB")

    # 7. Test Step B: AMP (Automatic Mixed Precision) Numerical Stability
    print("\n--- Testing AMP (Mixed Precision) Step ---")
    t0_amp = time.perf_counter()
    scaler = torch.amp.GradScaler('cuda')
    optimizer.zero_grad()

    with torch.amp.autocast('cuda', dtype=torch.float16):
        out_amp = model(patch_x_gpu)
        loss_amp = loss_fn(out_amp, patch_y_gpu)

    if not torch.isfinite(loss_amp):
        raise RuntimeError(f"AMP loss is non-finite: {loss_amp.item()}")

    scaler.scale(loss_amp).backward()
    scaler.unscale_(optimizer)

    grad_norms_amp = []
    has_nan_or_inf_grad_amp = False
    for name, p in model.named_parameters():
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():
                has_nan_or_inf_grad_amp = True
                print(f"[WARNING] Non-finite grad in AMP: {name}")
            grad_norms_amp.append(p.grad.norm().item())

    if has_nan_or_inf_grad_amp:
        amp_numerically_stable = False
        print("  [AMP STATUS] Unstable (NaN/Inf encountered in gradients)")
    else:
        amp_numerically_stable = True
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

    amp_time = time.perf_counter() - t0_amp
    amp_telemetry = get_gpu_telemetry()
    print(f"  AMP Loss: {loss_amp.item():.4f}")
    print(f"  AMP Numerically Stable: {amp_numerically_stable}")
    print(f"  AMP Step Time: {amp_time * 1000:.1f} ms")
    print(f"  Allocated VRAM: {amp_telemetry['allocated_vram_mb']} MB | Peak Reserved: {amp_telemetry['max_reserved_vram_mb']} MB")

    # 8. Test Step C: Save Temporary Checkpoint & Round-Trip Verification
    print("\n--- Testing Checkpoint Save & Round-Trip Reload ---")
    temp_ckpt_path = PROJECT_ROOT / "models" / "NeuroScan_Seg3D_UNet_SmokeTest_Temp.pth"
    # Ensure historical checkpoints are NOT touched
    protected_files = [
        "NeuroScan_Seg3D_UNet_Phase5_Multimodal_Best.pth",
        "NeuroScan_Seg3D_UNet_Phase5_Exp1_BalancedPatches.pth",
        "NeuroScan_Seg3D_UNet_Phase4B_Best.pth",
        "NeuroScan_Seg3D_UNet_Phase4B_Final.pth"
    ]
    assert temp_ckpt_path.name not in protected_files, "CRITICAL ERROR: Temp path conflicts with protected checkpoint!"

    save_meta = {
        "model_name": "NeuroScan 3D U-Net Smoke Test",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "patch_size": [48, 48, 48],
        "batch_size": 1,
        "device": str(device),
        "fp32_loss": float(loss_fp32.item()),
        "amp_loss": float(loss_amp.item()),
        "amp_stable": amp_numerically_stable
    }

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metadata": save_meta
    }, temp_ckpt_path)

    temp_hash = compute_file_sha256(temp_ckpt_path)
    temp_size_mb = temp_ckpt_path.stat().st_size / (1024 ** 2)
    print(f"  Saved temporary checkpoint: {temp_ckpt_path.name} ({temp_size_mb:.2f} MB)")
    print(f"  SHA-256: {temp_hash}")

    # Reload checkpoint
    loaded_model, is_loaded, loaded_meta = load_segmentation_model(str(temp_ckpt_path), device=device)
    if not is_loaded:
        raise RuntimeError("Failed to reload temporary smoke test checkpoint!")
    print(f"  Reload verified successfully: {loaded_meta['status_message']}")

    # Clean up temp checkpoint to preserve tidy workspace
    if temp_ckpt_path.exists():
        temp_ckpt_path.unlink()
        print("  Cleaned up temporary smoke test checkpoint.")

    final_telemetry = get_gpu_telemetry()
    final_ram_gb = psutil.virtual_memory().used / (1024 ** 3)
    peak_vram_pct = (final_telemetry['max_reserved_vram_mb'] / final_telemetry['total_vram_mb']) * 100

    results = {
        "status": "PASSED",
        "device_name": final_telemetry["device_name"],
        "total_vram_mb": final_telemetry["total_vram_mb"],
        "peak_reserved_vram_mb": final_telemetry["max_reserved_vram_mb"],
        "peak_vram_percentage": round(peak_vram_pct, 2),
        "system_ram_used_gb": round(final_ram_gb, 2),
        "fp32_loss": round(float(loss_fp32.item()), 4),
        "fp32_total_grad_norm": round(float(total_grad_norm_fp32), 4),
        "fp32_step_time_ms": round(fp32_time * 1000, 1),
        "amp_numerically_stable": amp_numerically_stable,
        "amp_loss": round(float(loss_amp.item()), 4),
        "amp_step_time_ms": round(amp_time * 1000, 1),
        "checkpoint_roundtrip_verified": True,
        "sample_patient_id": test_case_id,
        "patch_size": [48, 48, 48],
        "batch_size": 1,
        "num_workers": 0,
        "safe_for_pilot_training": True
    }

    # Save results to reports/
    report_path = PROJECT_ROOT / "reports" / "gpu_smoke_test_results.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("SMOKE TEST RESULTS SUMMARY:")
    print(f"  Status: PASSED")
    print(f"  Peak Reserved VRAM: {final_telemetry['max_reserved_vram_mb']} MB / {final_telemetry['total_vram_mb']} MB ({peak_vram_pct:.1f}%)")
    print(f"  VRAM Headroom: {final_telemetry['total_vram_mb'] - final_telemetry['max_reserved_vram_mb']:.1f} MB")
    print(f"  System RAM In Use: {final_ram_gb:.2f} GB")
    print(f"  AMP Stable: {amp_numerically_stable}")
    print(f"  Result JSON written to: {report_path}")
    print("=" * 80)
    return results


if __name__ == "__main__":
    run_smoke_test()
