"""
Diagnostic script to test foreground probability thresholding on the Phase 2 checkpoint.
"""
import sys
from pathlib import Path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import torch
import numpy as np
from backend.segmentation.model import load_segmentation_model
from backend.segmentation.pipeline import load_nifti_file, normalize_intensity
from backend.segmentation.metrics import compute_case_subregion_metrics
from monai.inferers import sliding_window_inference

checkpoint_path = _repo_root / "models" / "NeuroScan_Seg3D_UNet_T1CE_SingleChannel.pth"
val_cases = [
    ("case_001", _repo_root / "backend" / "segmentation" / "data" / "BraTS_Case_001_T1CE.nii.gz",
                 _repo_root / "backend" / "segmentation" / "data" / "BraTS_Case_001_seg.nii.gz"),
    ("case_002", _repo_root / "backend" / "segmentation" / "data" / "BraTS_Case_002_T1CE.nii.gz",
                 _repo_root / "backend" / "segmentation" / "data" / "BraTS_Case_002_seg.nii.gz"),
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading {checkpoint_path.name} on {device}...")
model, is_loaded, meta = load_segmentation_model(str(checkpoint_path), device=str(device))
assert is_loaded, "Model failed to load"
model.eval()

for case_id, mri_p, seg_p in val_cases:
    mri_vol = load_nifti_file(mri_p)
    seg_vol = load_nifti_file(seg_p)
    seg_data = seg_vol.data[0] if seg_vol.data.ndim == 4 else seg_vol.data
    seg_data = np.where(seg_data == 4, 3, seg_data).astype(np.int64)

    norm_img = normalize_intensity(mri_vol.data[0])
    inp_tensor = torch.from_numpy(norm_img[np.newaxis, np.newaxis, ...]).float().to(device)
    brain_mask = (norm_img > 0)

    with torch.no_grad():
        logits = sliding_window_inference(
            inputs=inp_tensor,
            roi_size=(64, 64, 64),
            sw_batch_size=2,
            predictor=model,
            overlap=0.5,
            mode="gaussian"
        )
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu() # (4, H, W, D)
    fg_prob = (1.0 - probs[0]).numpy()
    argmax_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    argmax_mask = np.where(brain_mask, argmax_mask, 0)

    gt_wt = np.sum(seg_data > 0)
    print(f"\n--- {case_id} (GT WT voxels: {gt_wt}) ---")
    
    for thresh in [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        if thresh == 0.0:
            pred_mask = argmax_mask
        else:
            pred_mask = np.where((fg_prob >= thresh) & brain_mask, argmax_mask, 0)
        
        m = compute_case_subregion_metrics(pred_mask, seg_data)
        pred_wt = np.sum(pred_mask > 0)
        sub = m["subregions"]
        print(f"Thresh {thresh:0.2f} -> WT Dice: {sub['WT']['dice']:.4f} | TC Dice: {sub['TC']['dice']:.4f} | ET Dice: {sub['ET']['dice']:.4f} | Mean Dice: {m['mean_dice']:.4f} | Pred WT: {pred_wt} (Ratio: {pred_wt/max(1, gt_wt):.2f}x)")
