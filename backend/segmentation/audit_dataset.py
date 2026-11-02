"""
NEUROSCAN AI — Reproducible 3D MRI Dataset Integrity Audit & Patient Splitting
Rigorously inspects volumetric NIfTI datasets, validates spatial geometry,
verifies affine/voxel alignment, checks label compliance, and generates versioned
data manifests and patient-level evaluation splits.
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timezone
import numpy as np

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False


# Canonical BraTS Label Definitions
# 0: Background
# 1: Necrotic & Non-Enhancing Tumor Core (NCR/NET)
# 2: Peritumoral Vasogenic Edema (ED)
# 3: Active Contrast-Enhancing Neoplastic Margin (ET)
# 4: ET in older/alternate BraTS releases (remapped to 3)
EXPECTED_LABELS = {0, 1, 2, 3}
PERMISSIBLE_RAW_LABELS = {0, 1, 2, 3, 4}

# Ground truth study ID mapping for genuine BraTS 2023 cases
PATIENT_STUDY_IDS = {
    "case_001": "BraTS-GLI-00000-000",
    "case_002": "BraTS-GLI-00002-000",
    "case_003": "BraTS-GLI-00003-000",
    "case_004": "BraTS-GLI-00005-000",
    "case_005": "BraTS-GLI-00006-000",
    "case_006": "BraTS-GLI-00008-000",
}


def compute_file_sha256(filepath: Path, chunk_size: int = 1048576) -> str:
    """Calculates SHA-256 cryptographic checksum of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            sha.update(chunk)
    return sha.hexdigest()


def audit_case_pair(
    mri_path: Path,
    seg_path: Path,
    case_key: str,
    patient_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Rigorously validates a single MRI/segmentation NIfTI pair.
    Checks existence, integrity, physical geometry, intensity stats, and labels.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # 1. File existence & basic attributes
    if not mri_path.exists():
        errors.append(f"MRI file does not exist: {mri_path}")
    if not seg_path.exists():
        errors.append(f"Segmentation mask file does not exist: {seg_path}")

    if errors:
        return {
            "case_key": case_key,
            "patient_id": patient_id or "UNKNOWN",
            "is_valid": False,
            "errors": errors,
            "warnings": warnings,
            "mri_path": str(mri_path),
            "seg_path": str(seg_path)
        }

    # 2. File size and SHA-256 checksums
    mri_size = mri_path.stat().st_size
    seg_size = seg_path.stat().st_size

    if mri_size == 0:
        errors.append(f"MRI file is empty (0 bytes): {mri_path.name}")
    if seg_size == 0:
        errors.append(f"Segmentation mask is empty (0 bytes): {seg_path.name}")

    try:
        mri_sha256 = compute_file_sha256(mri_path)
        seg_sha256 = compute_file_sha256(seg_path)
    except Exception as e:
        errors.append(f"Failed to compute SHA-256 checksum: {e}")
        return {
            "case_key": case_key,
            "patient_id": patient_id or "UNKNOWN",
            "is_valid": False,
            "errors": errors,
            "warnings": warnings
        }

    if not NIBABEL_AVAILABLE:
        errors.append("nibabel library is not installed; cannot parse NIfTI headers")
        return {
            "case_key": case_key,
            "patient_id": patient_id or "UNKNOWN",
            "is_valid": False,
            "mri_sha256": mri_sha256,
            "seg_sha256": seg_sha256,
            "errors": errors,
            "warnings": warnings
        }

    # 3. Read NIfTI headers and data arrays
    try:
        mri_img = nib.load(str(mri_path))
        seg_img = nib.load(str(seg_path))
    except Exception as e:
        errors.append(f"Corrupt or unreadable NIfTI file: {e}")
        return {
            "case_key": case_key,
            "patient_id": patient_id or "UNKNOWN",
            "is_valid": False,
            "errors": errors,
            "warnings": warnings
        }

    try:
        mri_data = np.asarray(mri_img.dataobj, dtype=np.float32)
        seg_data = np.asarray(seg_img.dataobj)
    except Exception as e:
        errors.append(f"Failed to read volume array data: {e}")
        return {
            "case_key": case_key,
            "patient_id": patient_id or "UNKNOWN",
            "is_valid": False,
            "errors": errors,
            "warnings": warnings
        }

    # 4. Dimensionality and Shape Compatibility
    mri_ndim = mri_data.ndim
    seg_ndim = seg_data.ndim
    mri_shape = list(mri_data.shape)
    seg_shape = list(seg_data.shape)

    if mri_shape != seg_shape:
        errors.append(f"Shape mismatch: MRI shape {mri_shape} != Seg shape {seg_shape}")

    # Standard 3D volume verification
    if mri_ndim != 3:
        warnings.append(f"MRI volume is {mri_ndim}D (shape: {mri_shape}), expected 3D (H, W, D)")
    if seg_ndim != 3:
        warnings.append(f"Seg volume is {seg_ndim}D (shape: {seg_shape}), expected 3D (H, W, D)")

    # 5. Spatial Geometry: Spacing, Affine, Orientation
    mri_affine = mri_img.affine
    seg_affine = seg_img.affine
    mri_zooms = tuple(float(z) for z in mri_img.header.get_zooms()[:3])
    seg_zooms = tuple(float(z) for z in seg_img.header.get_zooms()[:3])

    if not np.allclose(mri_zooms, seg_zooms, atol=1e-3):
        errors.append(f"Voxel zoom mismatch: MRI {mri_zooms} != Seg {seg_zooms}")

    # Physical spacing
    voxel_spacing = [round(mri_zooms[0], 4), round(mri_zooms[1], 4), round(mri_zooms[2], 4)]
    voxel_volume_mm3 = float(voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2])

    # Affine co-registration check
    affine_match = bool(np.allclose(mri_affine, seg_affine, atol=1e-3))
    if not affine_match:
        errors.append("MRI and ground-truth segmentation affine matrices do not match (misaligned spatial grid)")

    mri_orientation = "".join(nib.aff2axcodes(mri_affine))
    seg_orientation = "".join(nib.aff2axcodes(seg_affine))
    if mri_orientation != seg_orientation:
        errors.append(f"Orientation mismatch: MRI={mri_orientation} vs Seg={seg_orientation}")

    physical_dims_mm = [
        round(mri_shape[i] * voxel_spacing[i], 2) for i in range(min(3, len(mri_shape)))
    ]

    # 6. MRI Intensity Statistics and Finite Checks
    is_finite = bool(np.all(np.isfinite(mri_data)))
    if not is_finite:
        errors.append("MRI volume contains non-finite values (NaN or Inf)")

    non_zero_mri = mri_data[mri_data > 0]
    is_empty_mri = bool(len(non_zero_mri) == 0)
    if is_empty_mri:
        errors.append("MRI volume is empty (all voxels <= 0)")

    mri_stats = {
        "min": round(float(np.min(mri_data)), 2) if is_finite else None,
        "max": round(float(np.max(mri_data)), 2) if is_finite else None,
        "mean": round(float(np.mean(mri_data)), 2) if is_finite else None,
        "std": round(float(np.std(mri_data)), 2) if is_finite else None,
        "p01": round(float(np.percentile(non_zero_mri, 1.0)), 2) if len(non_zero_mri) > 0 else 0.0,
        "p99": round(float(np.percentile(non_zero_mri, 99.0)), 2) if len(non_zero_mri) > 0 else 0.0,
        "finite_values_only": is_finite,
        "is_empty": is_empty_mri
    }

    # 7. Segmentation Label Audit & Remapping
    unique_labels = sorted([int(x) for x in np.unique(seg_data)])
    label_counts: Dict[int, int] = {
        lbl: int(np.sum(seg_data == lbl)) for lbl in unique_labels
    }

    # Check for invalid labels
    invalid_labels = [l for l in unique_labels if l not in PERMISSIBLE_RAW_LABELS]
    if invalid_labels:
        errors.append(f"Unexpected segmentation labels found: {invalid_labels}. Expected subset of {PERMISSIBLE_RAW_LABELS}")

    # Check label 4 presence and remapping requirement
    has_label_4 = 4 in unique_labels
    remapped_count_label_4 = label_counts.get(4, 0)
    if has_label_4:
        warnings.append(
            f"Label 4 detected ({remapped_count_label_4} voxels). Explicitly remapped to Label 3 (Enhancing Tumor) for BraTS compliance."
        )

    # Cleaned label array for region calculations: remap 4 -> 3
    seg_clean = np.where(seg_data == 4, 3, seg_data)

    # Subregion voxel counts
    ncr_voxels = int(np.sum(seg_clean == 1))
    ed_voxels = int(np.sum(seg_clean == 2))
    et_voxels = int(np.sum(seg_clean == 3))
    wt_voxels = int(np.sum(np.isin(seg_clean, [1, 2, 3])))
    tc_voxels = int(np.sum(np.isin(seg_clean, [1, 3])))
    bg_voxels = int(np.sum(seg_clean == 0))

    voxel_counts = {
        "BG": bg_voxels,
        "NCR": ncr_voxels,
        "ED": ed_voxels,
        "ET": et_voxels,
        "WT": wt_voxels,
        "TC": tc_voxels
    }

    # Physical volumes (cm^3 = mm^3 / 1000)
    volume_cm3 = {
        "NCR": round((ncr_voxels * voxel_volume_mm3) / 1000.0, 3),
        "ED": round((ed_voxels * voxel_volume_mm3) / 1000.0, 3),
        "ET": round((et_voxels * voxel_volume_mm3) / 1000.0, 3),
        "WT": round((wt_voxels * voxel_volume_mm3) / 1000.0, 3),
        "TC": round((tc_voxels * voxel_volume_mm3) / 1000.0, 3)
    }

    is_mask_empty = (wt_voxels == 0)
    if is_mask_empty:
        warnings.append("Ground-truth mask contains zero foreground tumor voxels (empty mask)")

    is_valid = len(errors) == 0

    return {
        "case_key": case_key,
        "patient_id": patient_id or PATIENT_STUDY_IDS.get(case_key, f"BraTS-UNKNOWN-{case_key}"),
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "files": {
            "mri": {
                "path": str(mri_path.as_posix()),
                "filename": mri_path.name,
                "size_bytes": mri_size,
                "sha256": mri_sha256
            },
            "seg": {
                "path": str(seg_path.as_posix()),
                "filename": seg_path.name,
                "size_bytes": seg_size,
                "sha256": seg_sha256
            }
        },
        "geometry": {
            "dimensions": mri_shape,
            "voxel_spacing_mm": voxel_spacing,
            "physical_dimensions_mm": physical_dims_mm,
            "voxel_volume_mm3": round(voxel_volume_mm3, 4),
            "orientation": mri_orientation,
            "affine_match": affine_match,
            "affine_matrix": mri_affine.tolist()
        },
        "mri_intensity": mri_stats,
        "segmentation_labels": {
            "raw_labels_present": unique_labels,
            "raw_label_counts": label_counts,
            "label_4_remapped_to_3": has_label_4,
            "label_4_voxel_count": remapped_count_label_4,
            "voxel_counts": voxel_counts,
            "volume_cm3": volume_cm3,
            "is_empty": is_mask_empty
        }
    }


def audit_dataset_directory(
    data_dir: Path,
    expected_patient_map: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Audits all NIfTI volume pairs in data_dir.
    Detects duplicate file hashes, duplicate case IDs, and spatial errors.
    """
    patient_map = expected_patient_map or PATIENT_STUDY_IDS

    # Discover T1CE volumes without Windows case-insensitive duplication
    raw_files = list(data_dir.glob("*.nii*"))
    unique_paths = {p.resolve(): p for p in raw_files}
    t1ce_files = sorted(
        [p for p in unique_paths.values() if "t1ce" in p.name.lower() and "seg" not in p.name.lower()],
        key=lambda x: x.name.lower()
    )
    if not t1ce_files:
        t1ce_files = sorted(
            [p for p in unique_paths.values() if "seg" not in p.name.lower()],
            key=lambda x: x.name.lower()
        )

    cases_audited: Dict[str, Any] = {}
    mri_checksums: Dict[str, str] = {}
    seg_checksums: Dict[str, str] = {}
    patient_ids_seen: Dict[str, str] = {}
    global_errors: List[str] = []

    for idx, mri_p in enumerate(t1ce_files, start=1):
        # Extract case key from filename (e.g., BraTS_Case_001_T1CE -> case_001)
        name_lower = mri_p.name.lower()
        if "case_" in name_lower:
            parts = name_lower.split("case_")[1].split("_")[0]
            try:
                case_key = f"case_{int(parts):03d}"
            except ValueError:
                case_key = f"case_{idx:03d}"
        else:
            case_key = f"case_{idx:03d}"

        # Determine paired seg file
        seg_candidates = [
            mri_p.parent / mri_p.name.replace("_T1CE", "_seg").replace("_t1ce", "_seg"),
            mri_p.parent / mri_p.name.replace("_T1CE.nii.gz", "_seg.nii.gz").replace("_t1ce.nii.gz", "_seg.nii.gz")
        ]
        seg_p = None
        for cand in seg_candidates:
            if cand.exists():
                seg_p = cand
                break
        if seg_p is None:
            # Fallback search by case key
            for p in unique_paths.values():
                if "seg" in p.name.lower() and case_key in p.name.lower():
                    seg_p = p
                    break
        if seg_p is None:
            seg_p = mri_p.parent / mri_p.name.replace("_T1CE", "_seg")

        patient_id = patient_map.get(case_key, f"BraTS-GLI-{idx:05d}-000")

        case_audit = audit_case_pair(
            mri_path=mri_p,
            seg_path=seg_p,
            case_key=case_key,
            patient_id=patient_id
        )

        cases_audited[case_key] = case_audit

        if case_audit["is_valid"]:
            mri_sha = case_audit["files"]["mri"]["sha256"]
            seg_sha = case_audit["files"]["seg"]["sha256"]
            pid = case_audit["patient_id"]

            # Duplicate content detection
            if mri_sha in mri_checksums:
                err = f"Duplicate MRI volume content detected: {case_key} has identical SHA-256 to {mri_checksums[mri_sha]}"
                case_audit["errors"].append(err)
                case_audit["is_valid"] = False
                global_errors.append(err)
            else:
                mri_checksums[mri_sha] = case_key

            if seg_sha in seg_checksums:
                err = f"Duplicate segmentation mask content detected: {case_key} has identical SHA-256 to {seg_checksums[seg_sha]}"
                case_audit["errors"].append(err)
                case_audit["is_valid"] = False
                global_errors.append(err)
            else:
                seg_checksums[seg_sha] = case_key

            # Duplicate patient ID detection
            if pid in patient_ids_seen:
                err = f"Duplicate patient study ID detected: {pid} is used by both {case_key} and {patient_ids_seen[pid]}"
                case_audit["errors"].append(err)
                case_audit["is_valid"] = False
                global_errors.append(err)
            else:
                patient_ids_seen[pid] = case_key

    total_cases = len(cases_audited)
    valid_cases = sum(1 for c in cases_audited.values() if c["is_valid"])
    invalid_cases = total_cases - valid_cases

    audit_summary = {
        "audit_version": "1.0.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_directory": str(data_dir.resolve().as_posix()),
        "total_cases_found": total_cases,
        "valid_cases": valid_cases,
        "invalid_cases": invalid_cases,
        "critical_errors": global_errors,
        "all_valid": (total_cases > 0 and invalid_cases == 0 and len(global_errors) == 0),
        "cases": cases_audited
    }

    return audit_summary


def generate_patient_splits(
    manifest_data: Dict[str, Any],
    val_ratio: float = 0.333,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Generates deterministic, patient-level train/validation splits.
    Guarantees zero patient leakage, records SHA-256 hashes, and documents
    sample-size limitations.
    """
    valid_cases = [c for c in manifest_data["cases"].values() if c["is_valid"]]
    # Sort strictly by patient_id for deterministic order independent of filesystem
    valid_cases.sort(key=lambda x: x["patient_id"])

    patient_ids = [c["patient_id"] for c in valid_cases]
    unique_patient_ids = sorted(list(set(patient_ids)))

    rng = np.random.RandomState(seed)
    shuffled_pids = list(unique_patient_ids)
    rng.shuffle(shuffled_pids)

    n_val = max(1, int(round(len(shuffled_pids) * val_ratio)))
    val_pids = set(shuffled_pids[:n_val])
    train_pids = set(shuffled_pids[n_val:])

    # Strictly verify zero overlap
    overlap = train_pids.intersection(val_pids)
    assert len(overlap) == 0, f"Critical patient leakage: {overlap}"

    train_cases = []
    val_cases = []

    for c in valid_cases:
        item = {
            "case_key": c["case_key"],
            "patient_id": c["patient_id"],
            "mri_filename": c["files"]["mri"]["filename"],
            "mri_sha256": c["files"]["mri"]["sha256"],
            "seg_filename": c["files"]["seg"]["filename"],
            "seg_sha256": c["files"]["seg"]["sha256"],
            "wt_volume_cm3": c["segmentation_labels"]["volume_cm3"]["WT"]
        }
        if c["patient_id"] in val_pids:
            val_cases.append(item)
        else:
            train_cases.append(item)

    splits_manifest = {
        "manifest_version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "split_protocol": "development_and_held_out_validation",
        "statistical_safeguard": {
            "cohort_size": len(valid_cases),
            "limitation": (
                "The available dataset consists of 6 genuine BraTS cases. "
                "This cohort is suitable solely for engineering pipeline verification, "
                "baseline benchmarking, and smoke testing. It does NOT establish clinical "
                "generalization to unseen clinical populations or diverse imaging hardware."
            ),
            "leakage_prevention": "Strict patient-level partitioning verified. Zero patient ID or volume hash overlap."
        },
        "split_summary": {
            "total_cases": len(valid_cases),
            "train_cases_count": len(train_cases),
            "val_cases_count": len(val_cases),
            "train_ratio": round(len(train_cases) / max(1, len(valid_cases)), 3),
            "val_ratio": round(len(val_cases) / max(1, len(valid_cases)), 3)
        },
        "splits": {
            "train": train_cases,
            "val": val_cases
        }
    }

    return splits_manifest


def main():
    parser = argparse.ArgumentParser(description="NeuroScan AI — 3D Dataset Integrity Audit & Patient Splitting")
    parser.add_argument("--data-dir", type=str, default="backend/segmentation/data", help="Path to dataset directory")
    parser.add_argument("--output-manifest", type=str, default="backend/segmentation/data_manifest.json", help="Path to save audit manifest")
    parser.add_argument("--output-splits", type=str, default="backend/segmentation/splits.json", help="Path to save splits manifest")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for patient splitting")
    parser.add_argument("--val-ratio", type=float, default=0.333, help="Fraction of cases for validation")
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"[CRITICAL] Dataset directory does not exist: {data_path}", file=sys.stderr)
        sys.exit(1)

    print("=================================================================")
    print("NEUROSCAN AI — Reproducible Medical Imaging Data Integrity Audit")
    print(f"Directory: {data_path.resolve()}")
    print("=================================================================\n")

    audit_result = audit_dataset_directory(data_path)

    print(f"Total Cases Found: {audit_result['total_cases_found']}")
    print(f"Valid Cases:       {audit_result['valid_cases']}")
    print(f"Invalid Cases:     {audit_result['invalid_cases']}\n")

    for case_key, c in audit_result["cases"].items():
        status_str = "PASS" if c["is_valid"] else "FAIL"
        wt_vol = c.get("segmentation_labels", {}).get("volume_cm3", {}).get("WT", "N/A")
        print(f"[{status_str}] {case_key} ({c['patient_id']}): WT Vol={wt_vol} cm3 | Dims={c.get('geometry', {}).get('dimensions')}")
        if c["errors"]:
            for err in c["errors"]:
                print(f"       ERROR: {err}")
        if c["warnings"]:
            for w in c["warnings"]:
                print(f"       WARN:  {w}")

    # Save data manifest
    out_manifest = Path(args.output_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(audit_result, f, indent=2)
    print(f"\n[INFO] Saved complete audit data manifest to: {out_manifest.resolve()}")

    # Generate patient splits
    splits_result = generate_patient_splits(audit_result, val_ratio=args.val_ratio, seed=args.seed)
    out_splits = Path(args.output_splits)
    out_splits.parent.mkdir(parents=True, exist_ok=True)
    with open(out_splits, "w", encoding="utf-8") as f:
        json.dump(splits_result, f, indent=2)
    print(f"[INFO] Saved reproducible patient splits to: {out_splits.resolve()}")
    print(f"       Train Cases ({splits_result['split_summary']['train_cases_count']}): {[c['case_key'] for c in splits_result['splits']['train']]}")
    print(f"       Val Cases   ({splits_result['split_summary']['val_cases_count']}): {[c['case_key'] for c in splits_result['splits']['val']]}")

    if not audit_result["all_valid"]:
        print(f"\n[CRITICAL] Dataset audit FAILED with {audit_result['invalid_cases']} invalid case(s)!", file=sys.stderr)
        sys.exit(1)

    print("\n[SUCCESS] Dataset integrity audit PASSED. All geometry, checksums, and labels verified.")
    sys.exit(0)


if __name__ == "__main__":
    main()
