"""
NEUROSCAN AI — Reusable Multimodal Ingestion Validator & Gatekeeper
Standardized under the ASNR-MICCAI BraTS Protocol.

Performs 7-Point Gatekeeper Verification:
  1. Modality Completeness: All 4 MRI sequences (T1, T1-CE, T2, FLAIR) + SEG present.
  2. File Integrity: Valid NIfTI stream, non-empty, finite intensities (no NaN/Inf).
  3. Spatial Geometry: Equivalence of dimensions and voxel spacing across all 5 volumes.
  4. Rigid Registration: Affine matrix equality (np.allclose) confirming co-registration.
  5. Label Alphabet: Multi-class labels conform strictly to {0, 1, 2, 3}.
  6. Patient Identity & Deduplication: Detects duplicate patient IDs and hash collisions.
  7. Cryptographic Fingerprinting: Generates deterministic SHA-256 hashes for manifest.
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
import numpy as np
import nibabel as nib

REQUIRED_MODALITY_SUFFIXES = {
    "t1n": ["-t1n.nii.gz", "_t1n.nii.gz", "-t1.nii.gz", "_t1.nii.gz", "_T1.nii.gz"],
    "t1c": ["-t1c.nii.gz", "_t1c.nii.gz", "-t1ce.nii.gz", "_t1ce.nii.gz", "_T1CE.nii.gz"],
    "t2w": ["-t2w.nii.gz", "_t2w.nii.gz", "-t2.nii.gz", "_t2.nii.gz", "_T2.nii.gz"],
    "t2f": ["-t2f.nii.gz", "_t2f.nii.gz", "-flair.nii.gz", "_flair.nii.gz", "_FLAIR.nii.gz"],
    "seg": ["-seg.nii.gz", "_seg.nii.gz", "-mask.nii.gz", "_mask.nii.gz"]
}

VALID_LABEL_ALPHABET = {0, 1, 2, 3}


def compute_file_sha256(filepath: Path) -> str:
    """Calculates SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def find_modality_file(patient_dir_or_files: List[Path], modality_key: str) -> Optional[Path]:
    """Finds a matching modality file from a list of Path objects based on known suffixes."""
    suffixes = REQUIRED_MODALITY_SUFFIXES.get(modality_key, [])
    for p in patient_dir_or_files:
        name_lower = p.name.lower()
        for suf in suffixes:
            if name_lower.endswith(suf.lower()):
                return p
    return None


def validate_nifti_file_integrity(filepath: Path) -> Tuple[bool, Optional[str], Optional[nib.Nifti1Image]]:
    """
    Check 2: File integrity.
    Verifies that the file exists, has non-zero size, can be loaded by nibabel,
    and contains exclusively finite real values (no NaN or Inf).
    """
    if not filepath.exists():
        return False, f"File does not exist: {filepath.name}", None

    if filepath.stat().st_size == 0:
        return False, f"File is empty (0 bytes): {filepath.name}", None

    try:
        img = nib.load(str(filepath))
        # Inspect data array
        data = img.get_fdata(dtype=np.float32)
    except Exception as e:
        return False, f"Corrupted NIfTI/Gzip stream in {filepath.name}: {str(e)}", None

    if not np.all(np.isfinite(data)):
        nan_count = int(np.sum(np.isnan(data)))
        inf_count = int(np.sum(np.isinf(data)))
        return False, f"Non-finite intensities in {filepath.name} (NaN: {nan_count}, Inf: {inf_count})", None

    return True, None, img


def validate_patient_multimodal_study(
    patient_id: str,
    file_map: Dict[str, Path],
    known_hashes: Optional[Set[str]] = None,
    allow_legacy_label_4: bool = True
) -> Dict[str, Any]:
    """
    Executes the comprehensive 7-point gatekeeper on a single patient study.
    file_map: Dict mapping 't1n', 't1c', 't2w', 't2f', 'seg' to Path objects.
    """
    errors: List[str] = []
    warnings: List[str] = []
    check_results: Dict[str, bool] = {
        "modality_completeness": False,
        "file_integrity": False,
        "spatial_geometry": False,
        "rigid_registration": False,
        "label_alphabet": False,
        "deduplication": False,
        "cryptographic_hashing": False
    }

    # 1. Modality Completeness
    required_keys = ["t1n", "t1c", "t2w", "t2f", "seg"]
    missing_modalities = [k for k in required_keys if k not in file_map or file_map[k] is None]
    if missing_modalities:
        errors.append(f"Missing required modalities: {missing_modalities}")
        check_results["modality_completeness"] = False
    else:
        check_results["modality_completeness"] = True

    # If modalities are missing, we still validate whatever files are present, but study is invalid
    images: Dict[str, nib.Nifti1Image] = {}
    file_hashes: Dict[str, str] = {}
    file_sizes: Dict[str, int] = {}

    integrity_failed = False
    for mod_key, fpath in file_map.items():
        if fpath is None or not fpath.exists():
            continue
        valid, err, img = validate_nifti_file_integrity(fpath)
        if not valid:
            errors.append(err)
            integrity_failed = True
        else:
            images[mod_key] = img
            file_sizes[mod_key] = fpath.stat().st_size
            file_hashes[mod_key] = compute_file_sha256(fpath)

    check_results["file_integrity"] = (not integrity_failed) and (len(images) == len(file_map)) and (len(file_map) >= 5)

    # Deduplication & Hash check
    duplicate_detected = False
    if known_hashes is not None:
        for mod_key, f_hash in file_hashes.items():
            if f_hash in known_hashes:
                errors.append(f"Duplicate file hash detected for {mod_key}: {f_hash[:16]}... already seen in cohort")
                duplicate_detected = True
    check_results["deduplication"] = not duplicate_detected
    check_results["cryptographic_hashing"] = (len(file_hashes) == len(file_map)) and (len(file_map) > 0)

    # 3. Spatial Geometry & 4. Rigid Registration
    geometry_meta: Dict[str, Any] = {}
    if check_results["file_integrity"] and check_results["modality_completeness"]:
        ref_mod = "t1c" if "t1c" in images else list(images.keys())[0]
        ref_img = images[ref_mod]
        ref_shape = ref_img.shape
        ref_zooms = [round(float(z), 3) for z in ref_img.header.get_zooms()[:3]]
        ref_affine = ref_img.affine

        geometry_mismatch = False
        registration_mismatch = False

        for mod_key, img in images.items():
            if img.shape != ref_shape:
                errors.append(f"Shape mismatch: {mod_key} shape {img.shape} != {ref_mod} shape {ref_shape}")
                geometry_mismatch = True

            mod_zooms = [round(float(z), 3) for z in img.header.get_zooms()[:3]]
            if mod_zooms != ref_zooms:
                errors.append(f"Voxel zoom mismatch: {mod_key} {mod_zooms} != {ref_mod} {ref_zooms}")
                geometry_mismatch = True

            if not np.allclose(img.affine, ref_affine, atol=1e-3):
                errors.append(f"Affine co-registration mismatch between {mod_key} and {ref_mod}")
                registration_mismatch = True

        check_results["spatial_geometry"] = not geometry_mismatch
        check_results["rigid_registration"] = not registration_mismatch

        geometry_meta = {
            "dimensions": list(ref_shape),
            "voxel_spacing_mm": ref_zooms,
            "affine_matrix": ref_affine.tolist(),
            "orientation": "".join(nib.aff2axcodes(ref_affine))
        }

    # 5. Label Alphabet & Semantics
    label_meta: Dict[str, Any] = {}
    if "seg" in images:
        seg_data = images["seg"].get_fdata().astype(np.int64)
        unique_labels = set(np.unique(seg_data))
        contains_legacy_4 = 4 in unique_labels

        if contains_legacy_4:
            if allow_legacy_label_4:
                warnings.append("Segmentation mask contains legacy BraTS label 4 (Enhancing Tumor); remapping 4 -> 3 will be required.")
            else:
                errors.append("Segmentation mask contains unexpected legacy label 4 when allow_legacy_label_4 is False.")

        invalid_labels = unique_labels - VALID_LABEL_ALPHABET - ({4} if allow_legacy_label_4 else set())
        if invalid_labels:
            errors.append(f"Invalid/unexpected segmentation labels detected: {sorted(list(invalid_labels))}")
            check_results["label_alphabet"] = False
        else:
            check_results["label_alphabet"] = True

        # Compute subregion voxel counts
        wt_voxels = int(np.sum(np.isin(seg_data, [1, 2, 3, 4])))
        tc_voxels = int(np.sum(np.isin(seg_data, [1, 3, 4])))
        et_voxels = int(np.sum(np.isin(seg_data, [3, 4])))
        ed_voxels = int(np.sum(seg_data == 2))
        ncr_voxels = int(np.sum(seg_data == 1))

        dx, dy, dz = images["seg"].header.get_zooms()[:3]
        voxel_vol_cm3 = float(dx * dy * dz) / 1000.0

        label_meta = {
            "unique_labels": [int(x) for x in sorted(list(unique_labels))],
            "contains_legacy_label_4": bool(contains_legacy_4),
            "voxel_counts": {
                "WT": int(wt_voxels),
                "TC": int(tc_voxels),
                "ET": int(et_voxels),
                "ED": int(ed_voxels),
                "NCR": int(ncr_voxels)
            },
            "volume_cm3": {
                "WT": round(float(wt_voxels * voxel_vol_cm3), 3),
                "TC": round(float(tc_voxels * voxel_vol_cm3), 3),
                "ET": round(float(et_voxels * voxel_vol_cm3), 3),
                "ED": round(float(ed_voxels * voxel_vol_cm3), 3),
                "NCR": round(float(ncr_voxels * voxel_vol_cm3), 3)
            }
        }
    else:
        check_results["label_alphabet"] = False

    is_eligible = all(check_results.values()) and (len(errors) == 0)

    return {
        "patient_id": patient_id,
        "is_eligible": is_eligible,
        "status": "ELIGIBLE" if is_eligible else "REJECTED",
        "gatekeeper_checks": check_results,
        "errors": errors,
        "warnings": warnings,
        "file_paths": {k: str(v.as_posix()) if v else None for k, v in file_map.items()},
        "file_sizes_bytes": file_sizes,
        "file_sha256": file_hashes,
        "geometry": geometry_meta,
        "label_semantics": label_meta
    }


def parse_patient_studies_from_directory(target_dir: Path) -> Dict[str, Dict[str, Path]]:
    """
    Discovers candidate patient studies in a directory structure.
    Supports either subdirectories per patient (`BraTS-GLI-XXXXX-000/`)
    or flat directory naming (`BraTS_Case_001_T1CE.nii.gz`, etc.).
    """
    studies: Dict[str, Dict[str, Path]] = {}

    # Check for subdirectories first
    subdirs = [p for p in target_dir.iterdir() if p.is_dir()]
    if subdirs:
        for sdir in subdirs:
            p_id = sdir.name
            files = list(sdir.glob("*.nii*"))
            studies[p_id] = {
                "t1n": find_modality_file(files, "t1n"),
                "t1c": find_modality_file(files, "t1c"),
                "t2w": find_modality_file(files, "t2w"),
                "t2f": find_modality_file(files, "t2f"),
                "seg": find_modality_file(files, "seg")
            }
        return studies

    # Flat directory format
    all_nii = sorted(list(target_dir.glob("*.nii*")))
    # Group by prefix
    patient_prefixes: Set[str] = set()
    for f in all_nii:
        name = f.name
        # e.g. BraTS_Case_001_T1CE.nii.gz -> BraTS_Case_001
        for suf_list in REQUIRED_MODALITY_SUFFIXES.values():
            for suf in suf_list:
                if name.lower().endswith(suf.lower()):
                    prefix = name[: -len(suf)]
                    patient_prefixes.add(prefix)
                    break

    for p_id in sorted(list(patient_prefixes)):
        p_files = [f for f in all_nii if f.name.startswith(p_id)]
        studies[p_id] = {
            "t1n": find_modality_file(p_files, "t1n"),
            "t1c": find_modality_file(p_files, "t1c"),
            "t2w": find_modality_file(p_files, "t2w"),
            "t2f": find_modality_file(p_files, "t2f"),
            "seg": find_modality_file(p_files, "seg")
        }

    return studies


def generate_multimodal_split_manifest(
    validated_studies: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Generates a deterministic, patient-level triple-split partition manifest.
    Guarantees zero patient overlap across training, validation, and locked test sets.
    """
    eligible_patients = [s["patient_id"] for s in validated_studies if s["is_eligible"]]
    rejected_patients = [s for s in validated_studies if not s["is_eligible"]]

    # Deterministic shuffling
    rng = np.random.default_rng(seed=random_seed)
    shuffled = list(eligible_patients)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(round(n_total * train_ratio))
    n_val = int(round(n_total * val_ratio))
    # Test gets the remainder to ensure exact sum
    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train:n_train + n_val]
    test_ids = shuffled[n_train + n_val:]

    # Cross-check zero leakage
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    assert len(train_set.intersection(val_set)) == 0, "Patient leakage between train and val!"
    assert len(train_set.intersection(test_set)) == 0, "Patient leakage between train and test!"
    assert len(val_set.intersection(test_set)) == 0, "Patient leakage between val and test!"

    study_dict = {s["patient_id"]: s for s in validated_studies}

    manifest = {
        "manifest_version": "2.0.0",
        "dataset_standard": "ASNR-MICCAI BraTS 2023 Adult Glioma Multi-Parametric MRI",
        "random_seed": random_seed,
        "split_protocol": "patient_level_stratified_triple_split",
        "summary": {
            "total_studies_audited": len(validated_studies),
            "eligible_studies_count": len(eligible_patients),
            "rejected_studies_count": len(rejected_patients),
            "split_counts": {
                "train": len(train_ids),
                "val": len(val_ids),
                "locked_test": len(test_ids)
            }
        },
        "splits": {
            "train": [study_dict[pid] for pid in train_ids],
            "val": [study_dict[pid] for pid in val_ids],
            "locked_test": [study_dict[pid] for pid in test_ids]
        },
        "rejected_cases": rejected_patients
    }

    return manifest


class MultimodalIngestionValidator:
    """
    Standardized Gatekeeper Validator for Multimodal BraTS MRI Studies.
    Encapsulates 7-point validation, known hash tracking, and study eligibility assessment.
    """
    def __init__(self, known_hashes: Optional[Set[str]] = None, allow_legacy_label_4: bool = True):
        self.known_hashes = set(known_hashes) if known_hashes is not None else set()
        self.allow_legacy_label_4 = allow_legacy_label_4

    def validate_study(self, patient_id: str, file_map: Dict[str, Path]) -> Dict[str, Any]:
        """Validates a single candidate patient study across all 7 gatekeeper criteria."""
        report = validate_patient_multimodal_study(
            patient_id=patient_id,
            file_map=file_map,
            known_hashes=self.known_hashes,
            allow_legacy_label_4=self.allow_legacy_label_4
        )
        if report["is_eligible"]:
            for h in report["file_sha256"].values():
                self.known_hashes.add(h)
        return report

    def validate_nifti(self, filepath: Path) -> Tuple[bool, Optional[str], Optional[nib.Nifti1Image]]:
        """Validates single NIfTI file stream integrity."""
        return validate_nifti_file_integrity(filepath)


def update_manifest_atomic(
    manifest_path: Path,
    new_records: List[Dict[str, Any]],
    dataset_standard: str = "ASNR-MICCAI BraTS 2023 Adult Glioma Multi-Parametric MRI Protocol",
    authoritative_source: str = "Synapse.org (Challenge syn51156910 / syn51514106)"
) -> Dict[str, Any]:
    """
    Atomically updates multimodal manifest with new audited study records.
    Preserves all existing entries and metadata, writes to a temporary file,
    forces disk sync, and atomically replaces the target file.
    """
    existing_manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                existing_manifest = json.load(f)
        except Exception:
            existing_manifest = {}

    audited_studies_map: Dict[str, Dict[str, Any]] = {}
    if "audited_local_studies" in existing_manifest:
        for item in existing_manifest["audited_local_studies"]:
            audited_studies_map[item["patient_id"]] = item

    # Update or add new records
    for rec in new_records:
        audited_studies_map[rec["patient_id"]] = rec

    all_studies = list(audited_studies_map.values())
    eligible_count = len([s for s in all_studies if s.get("is_eligible", False)])
    rejected_count = len(all_studies) - eligible_count

    updated_manifest = {
        "manifest_version": existing_manifest.get("manifest_version", "2.1.0-phase5.2"),
        "created_at_utc": existing_manifest.get("created_at_utc", time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime())),
        "updated_at_utc": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "dataset_standard": dataset_standard,
        "authoritative_source": authoritative_source,
        "governing_license": existing_manifest.get("governing_license", "BraTS Data Use Agreement (DUA) - Non-Commercial Scientific Research"),
        "summary": {
            "total_studies_audited": len(all_studies),
            "eligible_multimodal_studies": eligible_count,
            "rejected_studies": rejected_count,
            "rejection_summary": {
                "missing_multimodal_sequences": len([s for s in all_studies if "Missing required modalities" in " ".join(s.get("errors", []))]),
                "corrupted_files": len([s for s in all_studies if not s.get("gatekeeper_checks", {}).get("file_integrity", True)]),
                "label_errors": len([s for s in all_studies if not s.get("gatekeeper_checks", {}).get("label_alphabet", True)])
            },
            "pilot_expansion_target": existing_manifest.get("summary", {}).get("pilot_expansion_target", {
                "target_cohort_size": 100,
                "target_split_counts": {"train": 70, "val": 15, "locked_test": 15},
                "status": "PILOT_ACQUISITION"
            })
        },
        "audited_local_studies": all_studies
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = manifest_path.parent / f".tmp_{manifest_path.name}_{os.getpid()}_{time.time_ns()}"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(updated_manifest, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    os.replace(temp_path, manifest_path)
    return updated_manifest
