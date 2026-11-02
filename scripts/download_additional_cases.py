import os
import sys
import shutil
from pathlib import Path

# BLOCKED: Third-party unverified mirrors (Hugging Face MedOtter) are prohibited.
# Acquisition must strictly proceed via official Synapse project syn51156910 under BraTS DUA.
raise RuntimeError(
    "[BLOCKED] Direct download from unverified third-party mirror 'MedOtter/brats2023-gli-dataset' is disabled. "
    "Use official acquisition script: python scripts/acquire_brats_pilot.py with valid SYNAPSE_AUTH_TOKEN."
)

# We already have:
# Case 1: BraTS-GLI-00000-000 (BraTS_Case_001)
# Case 2: BraTS-GLI-00002-000 (BraTS_Case_002)

# Candidate IDs to check
candidates = [f"BraTS-GLI-{i:05d}-000" for i in range(3, 30)]

downloaded_cases = []

for cid in candidates:
    if len(downloaded_cases) >= 4:  # Get 4 more genuine cases -> 6 real cases total
        break
    remote_t1c = f"ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/{cid}/{cid}-t1c.nii.gz"
    remote_seg = f"ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/{cid}/{cid}-seg.nii.gz"
    
    print(f"Testing {cid}...", flush=True)
    try:
        t1c_path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=remote_t1c)
        seg_path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=remote_seg)
        
        # Verify valid NIfTI
        img = nib.load(t1c_path)
        seg = nib.load(seg_path)
        img_data = img.get_fdata()
        seg_data = seg.get_fdata()
        
        active_voxels = int(np.sum(seg_data > 0))
        unique_labels = list(np.unique(seg_data))
        
        print(f"  [SUCCESS] {cid}: shape={img_data.shape}, active_voxels={active_voxels}, labels={unique_labels}", flush=True)
        downloaded_cases.append({
            "id": cid,
            "t1c_path": t1c_path,
            "seg_path": seg_path,
            "shape": list(img_data.shape),
            "zooms": [round(float(z), 2) for z in img.header.get_zooms()[:3]],
            "active_voxels": active_voxels,
            "unique_labels": unique_labels
        })
    except EntryNotFoundError:
        print(f"  [SKIP] {cid} not in training set", flush=True)
    except Exception as e:
        print(f"  [ERROR] {cid}: {e}", flush=True)

print(f"\nTotal new cases successfully downloaded: {len(downloaded_cases)}")
for i, c in enumerate(downloaded_cases, start=3):
    local_mri = DEST_DIR / f"BraTS_Case_{i:03d}_T1CE.nii.gz"
    local_seg = DEST_DIR / f"BraTS_Case_{i:03d}_seg.nii.gz"
    shutil.copyfile(c["t1c_path"], local_mri)
    shutil.copyfile(c["seg_path"], local_seg)
    print(f"Saved {c['id']} as {local_mri.name} and {local_seg.name}")
