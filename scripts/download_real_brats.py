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

REPO_ID = "MedOtter/brats2023-gli-dataset"

# Cases to fetch:
# 1. BraTS-GLI-00000-000 (Authentic Adult Glioblastoma with WT, TC, ET)
# 2. BraTS-GLI-00002-000 (Second distinct real Glioma case)

CASES = [
    {
        "patient_id": "BraTS-GLI-00000-000",
        "remote_t1c": "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t1c.nii.gz",
        "remote_t2f": "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t2f.nii.gz",
        "remote_seg": "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00000-000/BraTS-GLI-00000-000-seg.nii.gz",
        "local_mri": "BraTS_Case_001_T1CE.nii.gz",
        "local_seg": "BraTS_Case_001_seg.nii.gz"
    },
    {
        "patient_id": "BraTS-GLI-00002-000",
        "remote_t1c": "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00002-000/BraTS-GLI-00002-000-t1c.nii.gz",
        "remote_t2f": "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00002-000/BraTS-GLI-00002-000-t2f.nii.gz",
        "remote_seg": "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00002-000/BraTS-GLI-00002-000-seg.nii.gz",
        "local_mri": "BraTS_Case_002_T1CE.nii.gz",
        "local_seg": "BraTS_Case_002_seg.nii.gz"
    }
]

print("Starting download of authentic BraTS 2023 NIfTI volumes...")

for case in CASES:
    print(f"\nDownloading {case['patient_id']}...")
    downloaded_t1c = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=case["remote_t1c"]
    )
    downloaded_seg = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=case["remote_seg"]
    )

    dest_mri = DEST_DIR / case["local_mri"]
    dest_seg = DEST_DIR / case["local_seg"]

    shutil.copyfile(downloaded_t1c, dest_mri)
    shutil.copyfile(downloaded_seg, dest_seg)

    # Verify NIfTI integrity
    img = nib.load(str(dest_mri))
    seg = nib.load(str(dest_seg))
    img_data = img.get_fdata()
    seg_data = seg.get_fdata()

    print(f"Verified {case['local_mri']}: shape={img_data.shape}, dtype={img_data.dtype}, zooms={img.header.get_zooms()}")
    print(f"Verified {case['local_seg']}: shape={seg_data.shape}, unique labels={np.unique(seg_data)}, active voxels={np.sum(seg_data > 0)}")

print("\nSuccessfully downloaded and verified authentic BraTS 2023 datasets!")
