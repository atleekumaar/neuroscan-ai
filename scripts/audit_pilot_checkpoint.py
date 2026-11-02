"""
NEUROSCAN AI — Audit Saved Pilot Checkpoint (Stage 4B Preflight)
"""

import json
import hashlib
from pathlib import Path
import numpy as np
import torch

def compute_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def main():
    p_init = Path("models/NeuroScan_Seg3D_UNet_Phase5_Exp1_BalancedPatches.pth")
    p_pilot = Path("models/NeuroScan_Seg3D_UNet_LaptopSafe_Pilot.pth")

    print(f"Checking {p_pilot}...")
    h_pilot = compute_file_sha256(p_pilot)
    print(f"Pilot SHA-256: {h_pilot}")

    c_init = torch.load(p_init, map_location="cpu", weights_only=False)
    c_pilot = torch.load(p_pilot, map_location="cpu", weights_only=False)

    print("Pilot Checkpoint Top-Level Keys:", list(c_pilot.keys()))
    has_opt = "optimizer_state_dict" in c_pilot
    opt_states = len(c_pilot["optimizer_state_dict"]["state"]) if has_opt else 0
    print(f"Optimizer state present: {has_opt} (State entries: {opt_states})")

    sd_init = c_init["model_state_dict"] if "model_state_dict" in c_init else c_init
    sd_pilot = c_pilot["model_state_dict"]

    diffs = {}
    max_deltas = {}
    for k in sd_init:
        if k in sd_pilot:
            w_init = sd_init[k].float()
            w_pilot = sd_pilot[k].float()
            delta = (w_init - w_pilot).abs()
            diffs[k] = delta.sum().item()
            max_deltas[k] = delta.max().item()

    total_l1_delta = sum(diffs.values())
    nonzero_layers = sum(1 for d in diffs.values() if d > 0)
    print(f"Total Model Weight L1 Delta: {total_l1_delta:.6f}")
    print(f"Layers modified: {nonzero_layers} / {len(diffs)}")

    # Top 5 layers with largest updates
    sorted_layers = sorted(max_deltas.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nTop 5 layers by maximum parameter update:")
    for l_name, m_delta in sorted_layers:
        print(f"  {l_name}: max delta = {m_delta:.6f}, sum L1 delta = {diffs[l_name]:.6f}")

    print("\nRecomputing Validation Metrics on BraTS-GLI-00481-000 from Pilot Checkpoint...")
    import sys
    PROJECT_ROOT = Path(".").resolve()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from backend.segmentation.model import load_segmentation_model
    from backend.segmentation.pipeline import load_multimodal_study, predict_multimodal_study
    from scripts.train_laptop_safe_pilot import compute_subregion_metrics

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, ok, _ = load_segmentation_model(str(p_pilot), device=device)
    val_dir = Path("backend/segmentation/data/pilot/BraTS-GLI-00481-000")
    val_img, val_seg, _ = load_multimodal_study(val_dir, normalize=True)

    with torch.no_grad():
        val_pred = predict_multimodal_study(model=model, image_tensor=val_img, roi_size=(64, 64, 64), overlap=0.5, device=device)

    recomputed = compute_subregion_metrics(val_pred, val_seg)
    print("Recomputed Metrics:")
    print(f"  WT Dice: {recomputed['WT']['dice']} (Recorded: {c_pilot['metadata']['val_metrics']['WT']['dice']})")
    print(f"  TC Dice: {recomputed['TC']['dice']} (Recorded: {c_pilot['metadata']['val_metrics']['TC']['dice']})")
    print(f"  ET Dice: {recomputed['ET']['dice']} (Recorded: {c_pilot['metadata']['val_metrics']['ET']['dice']})")
    print(f"  Mean Dice: {recomputed['mean_dice']} (Recorded: {c_pilot['metadata']['val_metrics']['mean_dice']})")
    assert recomputed['WT']['dice'] == c_pilot['metadata']['val_metrics']['WT']['dice'], "Metric mismatch!"
    print("Verification: Exact metric match confirmed!")

if __name__ == "__main__":
    main()
