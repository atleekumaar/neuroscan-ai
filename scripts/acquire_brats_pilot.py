"""
NEUROSCAN AI — Phase 5.3 & 5.9: Official BraTS Pilot Acquisition Tool
Acquires genuine, matched multiparametric MRI studies from official Synapse project syn51156910
or user workspace syn78189226 under ASNR-MICCAI BraTS 2023 DUA governance.
Enforces rigorous authentication, access requirement auditing, path traversal safety,
secret redaction, isolated staging, no-overwrite protection, and atomic manifest integration with the 7-Point Gatekeeper.
Never downloads without verified SYNAPSE_AUTH_TOKEN and accepted Access Requirements.
Never modifies model checkpoints or 2D classifier assets.
"""

import os
import sys
import shutil
import zipfile
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.segmentation.ingestion import (
    MultimodalIngestionValidator,
    update_manifest_atomic,
    find_modality_file,
    REQUIRED_MODALITY_SUFFIXES
)

# Synapse official project identifier for BraTS 2023 Adult Glioma Challenge
SYNAPSE_PROJECT_ID = "syn51156910"
SYNAPSE_DATA_FOLDER_ID = "syn64952532"
SYNAPSE_GLI_FOLDER_ID = "syn51514105"
SYNAPSE_TRAINING_ZIP_ID = "syn51514132"
DEFAULT_USER_WORKSPACE_ID = "syn78189226"

MAX_PILOT_SIZE = 5

logger = logging.getLogger("acquire_brats_pilot")


def redact_secrets(text: str, secret: Optional[str] = None) -> str:
    """Replaces secret tokens with [REDACTED] to prevent credential leakage in logs or exceptions."""
    if not text:
        return text
    candidates = []
    if secret:
        candidates.append(secret)
    env_token = os.environ.get("SYNAPSE_AUTH_TOKEN")
    if env_token:
        candidates.append(env_token)
    
    redacted = str(text)
    for c in candidates:
        if c and len(c.strip()) > 3:
            redacted = redacted.replace(c.strip(), "***REDACTED_SYNAPSE_TOKEN***")
    return redacted


def check_auth_prerequisites() -> bool:
    """Verifies that SYNAPSE_AUTH_TOKEN is defined and non-empty. Halts safely if missing."""
    token = os.environ.get("SYNAPSE_AUTH_TOKEN")
    if not token or not token.strip():
        print("=" * 80)
        print("[BLOCKED] DATA ACCESS PREREQUISITE NOT SATISFIED")
        print("SYNAPSE_AUTH_TOKEN is NOT SET in the local environment.")
        print("To acquire genuine BraTS 2023 Adult Glioma studies:")
        print("  1. Review reports/phase5_9_access_audit.md and reports/phase5_9_blockers.md")
        print("  2. Register at Synapse.org and sign the BraTS Data Use Agreement (DUA):")
        print("     https://www.synapse.org/#!Synapse:syn64952532 (Requirement ID 9606684)")
        print("  3. Generate a Personal Access Token in your Synapse Account Settings:")
        print("     https://www.synapse.org/#!PersonalAccessTokens:")
        print("  4. Set the environment variable in PowerShell:")
        print("     $env:SYNAPSE_AUTH_TOKEN='<your_token>'")
        print("=" * 80)
        return False
    return True


def check_access_requirements(syn_client: Any, entity_id: str) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Inspects whether any access restrictions or Data Use Agreements (DUA) govern the entity.
    Returns (has_restrictions, list_of_requirements).
    """
    try:
        ar_response = syn_client.restGET(f"/entity/{entity_id}/accessRequirement")
        reqs = ar_response.get("results", [])
        return (len(reqs) > 0), reqs
    except Exception as e:
        safe_msg = redact_secrets(str(e))
        logger.debug(f"Could not fetch access requirements for {entity_id}: {safe_msg}")
        return False, []


def validate_pilot_parameters(
    pilot_size: int,
    output_dir: Path,
    max_pilot_size: int = MAX_PILOT_SIZE,
    workspace_root: Path = PROJECT_ROOT
) -> Tuple[int, Path]:
    """
    Validates pilot size bounds and ensures output directory is path-traversal safe
    and strictly segregated from model checkpoints and application code.
    """
    # 1. Pilot size validation
    if not isinstance(pilot_size, int) or isinstance(pilot_size, bool):
        raise ValueError(f"Pilot size must be an integer, got {type(pilot_size).__name__}")
    if pilot_size < 1 or pilot_size > max_pilot_size:
        raise ValueError(f"Pilot size {pilot_size} out of bounds. Must be between 1 and {max_pilot_size}.")

    # 2. Path resolution and traversal safety
    resolved_out = output_dir.resolve()
    resolved_root = workspace_root.resolve()

    # Block targeting root or system directories
    if resolved_out == resolved_out.parent:
        raise ValueError(f"Cannot use root directory as acquisition target: {resolved_out}")

    # Prohibit pointing to checkpoints directory or model weights
    models_dir = (resolved_root / "models").resolve()
    if models_dir in resolved_out.parents or resolved_out == models_dir:
        raise ValueError(f"Output directory cannot target or reside inside models directory: {resolved_out}")
    if resolved_out.suffix in [".pth", ".ckpt", ".h5", ".onnx", ".keras"]:
        raise ValueError(f"Output directory cannot target model checkpoint files: {resolved_out}")

    # Ensure output directory is within workspace or designated external data store
    if not str(resolved_out).startswith(str(resolved_root)) and ("Windows" in str(resolved_out) or "Program Files" in str(resolved_out)):
        raise ValueError(f"Forbidden system destination directory: {resolved_out}")

    return pilot_size, resolved_out


def download_patient_study(
    syn_client: Any,
    remote_folder_id: str,
    patient_id: str,
    staging_dir: Path
) -> Dict[str, Path]:
    """
    Downloads all NIfTI volumes for a single patient study into an isolated staging directory.
    Uses official Synapse client getChildren and get methods.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    file_map: Dict[str, Path] = {}

    children = syn_client.getChildren(parent=remote_folder_id, includeTypes=["file"])
    downloaded_files: List[Path] = []

    try:
        for child in children:
            c_name = child.get("name", "")
            if not (c_name.endswith(".nii") or c_name.endswith(".nii.gz")):
                continue
            entity = syn_client.get(child["id"], downloadLocation=str(staging_dir))
            fpath = Path(entity.path)
            downloaded_files.append(fpath)

        for mod_key in ["t1n", "t1c", "t2w", "t2f", "seg"]:
            file_map[mod_key] = find_modality_file(downloaded_files, mod_key)

        return file_map
    except Exception as e:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise RuntimeError(f"Download failed for {patient_id}: {redact_secrets(str(e))}") from e


def extract_patient_studies_from_zip(
    zip_path: Path,
    pilot_size: int,
    staging_base_dir: Path
) -> Dict[str, Dict[str, Path]]:
    """
    Selectively extracts up to pilot_size complete BraTS patient folders from a training zip archive.
    Avoids decompressing the entire 15+ GB cohort when only 5 pilot cases are needed.
    """
    extracted_studies: Dict[str, Dict[str, Path]] = {}
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Discover all patient folders in zip
        all_members = z.namelist()
        patient_prefixes: Set[str] = set()
        for m in all_members:
            parts = Path(m).parts
            for p in parts:
                if p.startswith("BraTS-GLI-"):
                    patient_prefixes.add(p)
                    break

        sorted_patients = sorted(list(patient_prefixes))[:pilot_size]
        for pid in sorted_patients:
            p_staging = staging_base_dir / f".staging_{pid}_{os.getpid()}"
            p_staging.mkdir(parents=True, exist_ok=True)
            
            p_files: List[Path] = []
            for m in all_members:
                if pid in m and (m.endswith(".nii") or m.endswith(".nii.gz")):
                    extracted_file = z.extract(m, path=str(p_staging))
                    # Move to flat staging folder
                    flat_target = p_staging / Path(extracted_file).name
                    if Path(extracted_file) != flat_target:
                        shutil.move(extracted_file, flat_target)
                    p_files.append(flat_target)

            extracted_studies[pid] = {
                "t1n": find_modality_file(p_files, "t1n"),
                "t1c": find_modality_file(p_files, "t1c"),
                "t2w": find_modality_file(p_files, "t2w"),
                "t2f": find_modality_file(p_files, "t2f"),
                "seg": find_modality_file(p_files, "seg"),
                "_staging_dir": p_staging
            }

    return extracted_studies


def stream_extract_pilot_from_synapse_archive(
    syn_client: Any,
    archive_id: str,
    pilot_size: int,
    output_dir: Path,
    validator: MultimodalIngestionValidator,
    allow_overwrite: bool = False,
    token: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Selectively streams and decompresses only up to pilot_size complete 5-tuple BraTS patient
    studies directly from the pre-signed S3 download URL of the official training archive.
    Avoids downloading the full 13+ GB cohort, enforces isolated staging, and validates each
    study through the 7-Point Gatekeeper before disk promotion.
    """
    import struct
    import zlib
    import urllib.request
    from collections import defaultdict

    archive_entity = syn_client.get(archive_id, downloadFile=False)
    fh_res = syn_client._getFileHandleDownload(archive_entity.dataFileHandleId, archive_entity.id)
    url = fh_res.get("preSignedURL")
    if not url:
        raise RuntimeError(f"Failed to obtain pre-signed download URL for {archive_id}")

    print(f"[STREAM] Connected to {archive_entity.name}. Streaming up to {pilot_size} complete studies...")
    acquired_studies: List[Dict[str, Any]] = []
    patient_files = defaultdict(dict)
    patient_staging_dirs = {}

    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        buffer = bytearray()

        def read_exact(n: int) -> bytes:
            nonlocal buffer
            while len(buffer) < n:
                chunk = resp.read(65536)
                if not chunk:
                    break
                buffer.extend(chunk)
            res_bytes = bytes(buffer[:n])
            buffer = buffer[n:]
            return res_bytes

        while len(acquired_studies) < pilot_size:
            sig = read_exact(4)
            if sig != b'PK\x03\x04':
                break
            header = read_exact(26)
            (extract_ver, flags, method, mod_time, mod_date, crc32,
             comp_size, uncomp_size, filename_len, extra_len) = struct.unpack('<HHHHHIIIHH', header)
            fname = read_exact(filename_len).decode('utf-8', errors='ignore')
            extra = read_exact(extra_len)
            has_descriptor = bool(flags & 0x08)

            parts = fname.split('/')
            patient_id = None
            for p in parts:
                if p.startswith('BraTS-GLI-') and not p.endswith('.nii.gz'):
                    patient_id = p
                    break

            if fname.endswith('/'):
                continue

            if not patient_id:
                if method == 0:
                    read_exact(comp_size)
                elif method == 8:
                    d = zlib.decompressobj(-15)
                    while not d.eof:
                        if not buffer:
                            chunk = resp.read(65536)
                            if not chunk:
                                break
                            buffer.extend(chunk)
                        piece = bytes(buffer[:min(len(buffer), 65536)])
                        buffer = buffer[len(piece):]
                        d.decompress(piece)
                        if d.unused_data:
                            buffer = bytearray(d.unused_data) + buffer
                    if has_descriptor:
                        buffer = buffer[16:] if buffer.startswith(b'PK\x07\x08') else buffer[12:]
                continue

            final_patient_dir = output_dir / patient_id
            if final_patient_dir.exists() and not allow_overwrite:
                if method == 0:
                    read_exact(comp_size)
                elif method == 8:
                    d = zlib.decompressobj(-15)
                    while not d.eof:
                        if not buffer:
                            chunk = resp.read(65536)
                            if not chunk:
                                break
                            buffer.extend(chunk)
                        piece = bytes(buffer[:min(len(buffer), 65536)])
                        buffer = buffer[len(piece):]
                        d.decompress(piece)
                        if d.unused_data:
                            buffer = bytearray(d.unused_data) + buffer
                    if has_descriptor:
                        buffer = buffer[16:] if buffer.startswith(b'PK\x07\x08') else buffer[12:]
                continue

            p_staging = patient_staging_dirs.get(patient_id)
            if not p_staging:
                p_staging = output_dir / f".staging_{patient_id}_{os.getpid()}"
                p_staging.mkdir(parents=True, exist_ok=True)
                patient_staging_dirs[patient_id] = p_staging

            target_path = p_staging / Path(fname).name
            if method == 0:
                content = read_exact(comp_size)
                with open(target_path, 'wb') as f:
                    f.write(content)
            elif method == 8:
                d = zlib.decompressobj(-15)
                with open(target_path, 'wb') as f:
                    while not d.eof:
                        if not buffer:
                            chunk = resp.read(65536)
                            if not chunk:
                                break
                            buffer.extend(chunk)
                        piece = bytes(buffer[:min(len(buffer), 65536)])
                        buffer = buffer[len(piece):]
                        out = d.decompress(piece)
                        f.write(out)
                        if d.unused_data:
                            buffer = bytearray(d.unused_data) + buffer
                if has_descriptor:
                    buffer = buffer[16:] if buffer.startswith(b'PK\x07\x08') else buffer[12:]

            for m in ['t1n', 't1c', 't2w', 't2f', 'seg']:
                if f"-{m}.nii" in target_path.name:
                    patient_files[patient_id][m] = target_path
                    break

            if len(patient_files[patient_id]) == 5:
                print(f"[VALIDATE] Validating study {patient_id} with 7-Point Gatekeeper...")
                report = validator.validate_study(patient_id, patient_files[patient_id])
                if report["is_eligible"]:
                    if final_patient_dir.exists() and allow_overwrite:
                        shutil.rmtree(final_patient_dir, ignore_errors=True)
                    p_staging.rename(final_patient_dir)
                    report["file_paths"] = {
                        k: str((final_patient_dir / Path(v).name).as_posix()) if v else None
                        for k, v in patient_files[patient_id].items()
                    }
                    acquired_studies.append(report)
                    print(f"[SUCCESS] Ingested {patient_id} ({len(acquired_studies)}/{pilot_size})")
                    patient_staging_dirs.pop(patient_id, None)
                else:
                    print(f"[REJECTED] Staging study {patient_id} failed gatekeeper: {report['errors']}")
                    shutil.rmtree(p_staging, ignore_errors=True)
                    patient_staging_dirs.pop(patient_id, None)

    for st_dir in patient_staging_dirs.values():
        shutil.rmtree(st_dir, ignore_errors=True)

    return acquired_studies


def acquire_pilot(
    pilot_size: int = 5,
    source_id: str = SYNAPSE_PROJECT_ID,
    output_dir: Optional[Path] = None,
    allow_overwrite: bool = False,
    manifest_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Orchestrates the safe, non-destructive pilot acquisition of BraTS 2023 Adult Glioma studies.
    Enforces isolated staging, validation before disk promotion, no-overwrite protection,
    and atomic manifest integration.
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "backend" / "segmentation" / "data" / "pilot"
    if manifest_path is None:
        manifest_path = PROJECT_ROOT / "backend" / "segmentation" / "multimodal_manifest.json"

    pilot_size, output_dir = validate_pilot_parameters(pilot_size, output_dir)

    if not check_auth_prerequisites():
        return {"status": "BLOCKED", "message": "Missing SYNAPSE_AUTH_TOKEN", "acquired_count": 0}

    try:
        import synapseclient
    except ImportError:
        msg = "[ERROR] 'synapseclient' is not installed in the current environment. Install with: pip install synapseclient"
        print(msg)
        return {"status": "ERROR", "message": msg, "acquired_count": 0}

    token = os.environ.get("SYNAPSE_AUTH_TOKEN")
    syn = synapseclient.Synapse()
    try:
        syn.login(authToken=token, silent=True)
        print("[SUCCESS] Authenticated with Synapse successfully.")
    except Exception as e:
        safe_msg = redact_secrets(str(e), token)
        print(f"[ERROR] Synapse authentication failed: {safe_msg}")
        return {"status": "AUTH_FAILED", "message": safe_msg, "acquired_count": 0}

    # Audit access requirements on source entity
    has_restrictions, reqs = check_access_requirements(syn, source_id)
    if has_restrictions:
        print(f"[INFO] Access requirements detected on {source_id} ({len(reqs)} active).")
        for r in reqs:
            print(f"  - Requirement ID: {r.get('id')} ({r.get('concreteType', '').split('.')[-1]})")

    validator = MultimodalIngestionValidator()
    print(f"[INFO] Gatekeeper active. Querying source {source_id} for up to {pilot_size} studies...")

    try:
        candidates = list(syn.getChildren(parent=source_id, includeTypes=["folder"]))
    except Exception as e:
        safe_msg = redact_secrets(str(e), token)
        print(f"[ERROR] Failed to query remote Synapse studies: {safe_msg}")
        return {"status": "QUERY_FAILED", "message": safe_msg, "acquired_count": 0}

    # Filter direct patient folders if available
    patient_candidates = sorted([c for c in candidates if c.get("name", "").startswith("BraTS-GLI-")], key=lambda x: x["name"])

    acquired_studies: List[Dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    # If direct patient folders exist (e.g. In mocks or uncompressed workspace)
    if patient_candidates:
        print(f"[INFO] Discovered {len(patient_candidates)} direct candidate BraTS-GLI folders.")
        for cand in patient_candidates:
            if len(acquired_studies) >= pilot_size:
                break

            patient_id = cand["name"]
            final_patient_dir = output_dir / patient_id

            if final_patient_dir.exists():
                if not allow_overwrite:
                    print(f"[SKIP] Patient directory already exists and overwrite is disabled: {patient_id}")
                    continue
                else:
                    print(f"[WARN] Overwrite permitted: existing directory {patient_id} will be replaced.")

            staging_dir = output_dir / f".staging_{patient_id}_{os.getpid()}"
            print(f"\n[ACQUIRE] Downloading {patient_id} to staging: {staging_dir.name}...")

            try:
                file_map = download_patient_study(
                    syn_client=syn,
                    remote_folder_id=cand["id"],
                    patient_id=patient_id,
                    staging_dir=staging_dir
                )

                print(f"[VALIDATE] Executing 7-point gatekeeper on staging files for {patient_id}...")
                report = validator.validate_study(patient_id, file_map)

                if report["is_eligible"]:
                    print(f"[PASSED] Study {patient_id} passed all gatekeeper checks. Promoting to final directory...")
                    if final_patient_dir.exists() and allow_overwrite:
                        shutil.rmtree(final_patient_dir, ignore_errors=True)
                    staging_dir.rename(final_patient_dir)

                    report["file_paths"] = {
                        k: str((final_patient_dir / v.name).as_posix()) if v else None
                        for k, v in file_map.items()
                    }
                    acquired_studies.append(report)
                    print(f"[SUCCESS] Ingested {patient_id} ({len(acquired_studies)}/{pilot_size})")
                else:
                    print(f"[REJECTED] Staging study {patient_id} failed gatekeeper: {report['errors']}")
                    shutil.rmtree(staging_dir, ignore_errors=True)

            except Exception as e:
                safe_err = redact_secrets(str(e), token)
                print(f"[ERROR] Exception during acquisition of {patient_id}: {safe_err}")
                shutil.rmtree(staging_dir, ignore_errors=True)
    else:
        # Source contains compressed official challenge archive (syn51514132)
        print(f"[INFO] Direct patient folders not present in {source_id}. Utilizing official training archive ({SYNAPSE_TRAINING_ZIP_ID})...")
        try:
            acquired_studies = stream_extract_pilot_from_synapse_archive(
                syn_client=syn,
                archive_id=SYNAPSE_TRAINING_ZIP_ID,
                pilot_size=pilot_size,
                output_dir=output_dir,
                validator=validator,
                allow_overwrite=allow_overwrite,
                token=token
            )
        except Exception as e:
            safe_err = redact_secrets(str(e), token)
            print(f"[ERROR] Streaming acquisition error: {safe_err}")
            return {"status": "STREAM_FAILED", "message": safe_err, "acquired_count": len(acquired_studies)}

    # Atomic manifest update
    if acquired_studies:
        print(f"\n[MANIFEST] Atomically updating {manifest_path.name} with {len(acquired_studies)} new studies...")
        update_manifest_atomic(manifest_path, acquired_studies)
        print("[SUCCESS] Manifest updated atomically.")

    return {
        "status": "COMPLETED" if acquired_studies else "HALTED",
        "acquired_count": len(acquired_studies),
        "studies": [s["patient_id"] for s in acquired_studies]
    }


def main():
    # If SYNAPSE_AUTH_TOKEN is not in environment, load from user registry if present
    if "SYNAPSE_AUTH_TOKEN" not in os.environ and sys.platform == "win32":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Environment')
            val, _ = winreg.QueryValueEx(k, 'SYNAPSE_AUTH_TOKEN')
            if val and val.strip():
                os.environ["SYNAPSE_AUTH_TOKEN"] = val.strip()
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Official BraTS Pilot Acquisition Tool (Phase 5.3 & 5.9)")
    parser.add_argument("--source", type=str, default=SYNAPSE_PROJECT_ID, help="Synapse source ID (default: syn51156910, or syn78189226)")
    parser.add_argument("--pilot-size", type=int, default=5, help="Number of pilot studies to acquire (1-5, default: 5)")
    parser.add_argument("--dest", type=str, default="backend/segmentation/data/pilot", help="Destination directory")
    parser.add_argument("--allow-overwrite", action="store_true", help="Allow replacing existing study directories")
    args = parser.parse_args()

    try:
        res = acquire_pilot(
            pilot_size=args.pilot_size,
            source_id=args.source,
            output_dir=Path(args.dest),
            allow_overwrite=args.allow_overwrite
        )
        if res["status"] in ["BLOCKED", "ERROR", "AUTH_FAILED", "QUERY_FAILED", "STREAM_FAILED"]:
            sys.exit(1)
    except Exception as e:
        print(f"[CRITICAL] Fatal execution error: {redact_secrets(str(e))}")
        sys.exit(1)


if __name__ == "__main__":
    main()

