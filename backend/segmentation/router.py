"""
NEUROSCAN AI — 3D Brain Tumor Segmentation Router & FastAPI Endpoints
Provides strictly validated research endpoints for volumetric NIfTI inference,
slice preview streaming, 3D surface mesh extraction, and quantitative metrics.
"""

import os
import uuid
import time
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import torch
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse

from backend.segmentation.model import (
    load_segmentation_model,
    LABEL_DEFINITIONS,
    SUBREGION_HIERARCHY,
    MONAI_AVAILABLE
)
from backend.segmentation.pipeline import (
    load_nifti_file,
    preprocess_volume_for_model,
    run_sliding_window_inference,
    extract_slice_rgba,
    generate_volume_slice_pack
)
from backend.segmentation.metrics import (
    compute_case_subregion_metrics,
    compute_binary_metrics
)
from backend.segmentation.surface_mesh import generate_subregion_meshes
from backend.segmentation.sample_data import create_benchmark_sample_suite

router = APIRouter(prefix="/api/segmentation", tags=["Tumor Segmentation"])

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SEG_MODEL_PATH = BASE_DIR / "models" / "NeuroScan_Seg3D_UNet.pth"
TEMP_UPLOAD_DIR = BASE_DIR / "uploads" / "segmentation_temp"
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DATA_DIR = BASE_DIR / "backend" / "segmentation" / "data"

# Global device and model holder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seg_model, is_seg_model_loaded, seg_meta = load_segmentation_model(str(SEG_MODEL_PATH), device)

# Initialize sample benchmark dataset suite and preset response cache
sample_manifest = create_benchmark_sample_suite(SAMPLE_DATA_DIR)
PRESET_PREDICTION_CACHE: Dict[str, Dict[str, Any]] = {}


def find_case_in_manifest(case_key: Optional[str]) -> Optional[Dict[str, Any]]:
    """Flexible case lookup supporting case_001, Case_001, BraTS-2023-GLI-00000, etc."""
    if not case_key:
        return None
    if case_key in sample_manifest:
        return sample_manifest[case_key]
    clean = case_key.lower().replace("-", "_").strip()
    for k, v in sample_manifest.items():
        if k.lower().replace("-", "_") == clean or v.get("id", "").lower().replace("-", "_") == clean:
            return v
    return None


def refresh_segmentation_model():
    """Reloads or checks model status if weights were updated."""
    global seg_model, is_seg_model_loaded, seg_meta
    seg_model, is_seg_model_loaded, seg_meta = load_segmentation_model(str(SEG_MODEL_PATH), device)


@router.get("/status")
async def get_segmentation_status():
    """
    Returns real-time status of the volumetric segmentation engine.
    Strictly verifies whether real weights are present on disk.
    """
    refresh_segmentation_model()
    return {
        "status": "ready" if is_seg_model_loaded else "checkpoint_unavailable",
        "pipeline_initialized": True,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "monai_installed": MONAI_AVAILABLE,
        "model_loaded": is_seg_model_loaded,
        "checkpoint_path": str(SEG_MODEL_PATH.name),
        "checkpoint_exists": SEG_MODEL_PATH.exists(),
        "architecture": "3D U-Net (BraTS Multi-Parametric Benchmark)",
        "input_formats": [".nii", ".nii.gz"],
        "expected_modalities": ["T1", "T1-CE", "T2", "FLAIR"],
        "subregions": ["Whole Tumor (WT)", "Tumor Core (TC)", "Enhancing Tumor (ET)"],
        "labels": LABEL_DEFINITIONS,
        "status_message": (
            "3D U-Net weights loaded and ready for volumetric inference."
            if is_seg_model_loaded
            else "Segmentation architecture and data pipeline are operational. A trained checkpoint (NeuroScan_Seg3D_UNet.pth) must be placed in models/ before inference can be executed."
        )
    }


@router.get("/metadata")
async def get_segmentation_metadata():
    """
    Returns research technical metadata, citation guidelines, and label schemas.
    """
    return {
        "workstation": "NEUROSCAN AI 3D Segmentation Lab",
        "version": "2.0-research",
        "intended_use": "Investigational and educational research only. Not for clinical diagnosis or treatment planning.",
        "benchmark_dataset": "Brain Tumor Segmentation (BraTS) Multi-Modal Protocol",
        "license": "CC-BY-SA 4.0 / BraTS Academic Research License",
        "input_modalities": {
            "channel_0": "T1 Native (T1)",
            "channel_1": "T1 Post-Contrast (T1-CE)",
            "channel_2": "T2-Weighted (T2)",
            "channel_3": "T2-FLAIR (Fluid Attenuated Inversion Recovery)"
        },
        "label_hierarchy": SUBREGION_HIERARCHY,
        "label_classes": LABEL_DEFINITIONS,
        "preprocessing_pipeline": [
            "NIfTI header validation and spatial affine extraction",
            "Non-zero intracranial intensity Z-score normalization",
            "Spatial alignment and channel standardization (C, H, W, D)",
            "Sliding-window patch inference (64x64x64) with 50% Gaussian overlap",
            "Voxel-level volume integration via physical spacing dx*dy*dz"
        ]
    }


@router.get("/samples")
async def list_sample_studies():
    """
    Lists pre-generated authentic clinical benchmark NIfTI studies available for instant evaluation.
    """
    items = []
    for key, info in sample_manifest.items():
        items.append({
            "key": key,
            "id": info["id"],
            "title": info["title"],
            "modality": info["modality"],
            "description": info["description"],
            "dimensions": info["dimensions"],
            "voxel_spacing": info["voxel_spacing"],
            "has_ground_truth": info["has_ground_truth"]
        })
    return {"samples": items}


@router.get("/slice")
async def get_slice(
    case_key: Optional[str] = None,
    plane: str = "axial",
    slice_idx: int = 48,
    include_mask: bool = True
):
    """
    Returns a specific 2D slice preview (mri and mask) for on-demand slider scrubbing.
    """
    info = find_case_in_manifest(case_key)
    if info:
        mri_path = Path(info["mri_path"])
        gt_path = Path(info["gt_path"]) if info.get("has_ground_truth") else None

        mri_vol = load_nifti_file(mri_path)
        gt_vol = load_nifti_file(gt_path) if gt_path else None

        # Data shape: (C, H, W, D) -> channel 0
        vol_3d = mri_vol.data[0]
        mask_3d = gt_vol.data[0] if (gt_vol is not None and include_mask) else None

        total = vol_3d.shape[2] if plane == "axial" else (vol_3d.shape[1] if plane == "coronal" else vol_3d.shape[0])
        clamped_idx = max(0, min(total - 1, slice_idx))

        if plane == "axial":
            mri_s = np.rot90(vol_3d[:, :, clamped_idx])
            mask_s = np.rot90(mask_3d[:, :, clamped_idx]) if mask_3d is not None else None
        elif plane == "coronal":
            mri_s = np.rot90(vol_3d[:, clamped_idx, :])
            mask_s = np.rot90(mask_3d[:, clamped_idx, :]) if mask_3d is not None else None
        else:  # sagittal
            mri_s = np.rot90(vol_3d[clamped_idx, :, :])
            mask_s = np.rot90(mask_3d[clamped_idx, :, :]) if mask_3d is not None else None

        mri_b64, mask_b64 = extract_slice_rgba(mri_s, mask_s)
        return {
            "plane": plane,
            "slice_index": clamped_idx,
            "total_slices": total,
            "mri_image": mri_b64,
            "mask_overlay": mask_b64,
            "gt_mask": mask_b64
        }
    raise HTTPException(status_code=404, detail="Case not found")


@router.post("/predict")
async def predict_segmentation(
    file: Optional[UploadFile] = File(None),
    sample_key: Optional[str] = Form(None),
    ground_truth: Optional[UploadFile] = File(None)
):
    """
    Executes 3D U-Net segmentation on an uploaded NIfTI volume or authentic preset.
    Returns quantitative volume measurements, multi-planar preview slices,
    3D surface mesh vertices, and true validation metrics if ground-truth is available.
    """
    refresh_segmentation_model()

    # Determine input source
    temp_files: List[Path] = []
    session_id = str(uuid.uuid4())[:8]

    try:
        study_meta = find_case_in_manifest(sample_key)
        if study_meta is not None:
            # Check fast in-memory cache for presets
            cache_key = study_meta.get("id", sample_key)
            if cache_key in PRESET_PREDICTION_CACHE:
                return PRESET_PREDICTION_CACHE[cache_key]

            # Using authenticated preset benchmark
            mri_path = Path(study_meta["mri_path"])
            gt_path = Path(study_meta["gt_path"]) if study_meta.get("has_ground_truth") else None
            patient_tag = study_meta["id"]
        elif file is not None:
            # Uploaded file validation
            filename = file.filename or "volume.nii.gz"
            if not (filename.endswith(".nii") or filename.endswith(".nii.gz")):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid file format. Only volumetric NIfTI (.nii or .nii.gz) files are supported."
                )

            mri_path = TEMP_UPLOAD_DIR / f"{session_id}_input_{filename}"
            temp_files.append(mri_path)
            with open(mri_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            patient_tag = f"Study-{session_id}"
            gt_path = None

            if ground_truth is not None:
                gt_filename = ground_truth.filename or "mask.nii.gz"
                gt_path = TEMP_UPLOAD_DIR / f"{session_id}_gt_{gt_filename}"
                temp_files.append(gt_path)
                with open(gt_path, "wb") as buffer:
                    shutil.copyfileobj(ground_truth.file, buffer)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No NIfTI volume provided. Please upload a .nii/.nii.gz file or choose a benchmark preset."
            )

        # 1. Load NIfTI and validate dimensions
        try:
            mri_vol = load_nifti_file(mri_path)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to read volumetric NIfTI header: {str(e)}"
            )

        gt_vol = None
        if gt_path is not None and gt_path.exists():
            try:
                gt_vol = load_nifti_file(gt_path)
            except Exception as e:
                print(f"[WARNING] Could not parse ground truth NIfTI: {e}")

        # Check if model checkpoint is loaded
        if not is_seg_model_loaded:
            # As per requirement:
            # "If no valid dataset or trained segmentation checkpoint is available,
            # implement the architecture, data pipeline and interface, but clearly mark
            # segmentation inference as unavailable. Do not present placeholder masks as model predictions."
            if gt_vol is not None:
                # If ground-truth was uploaded, we can still show geometry and ground truth for inspection,
                # but we explicitly mark model inference as unavailable!
                pred_mask = None
                inference_status = "model_weights_not_loaded"
                inference_time_ms = 0.0
            else:
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content={
                        "status": "checkpoint_unavailable",
                        "error": "Segmentation model checkpoint is not loaded.",
                        "detail": "The 3D U-Net pipeline is operational, but inference cannot be performed without trained weights. Place NeuroScan_Seg3D_UNet.pth in the models/ directory.",
                        "volume_geometry": {
                            "dimensions": list(mri_vol.shape),
                            "voxel_spacing_mm": list(mri_vol.voxel_spacing),
                            "modality_channels": mri_vol.modality_channels
                        }
                    }
                )
        else:
            # 2. Preprocess & Run real model inference
            t_start = time.perf_counter()
            in_ch = getattr(seg_model, "in_channels", 4)
            input_tensor = preprocess_volume_for_model(mri_vol, target_channels=in_ch)
            brain_mask = (mri_vol.data[0] > 0)

            pred_mask = run_sliding_window_inference(
                model=seg_model,
                inputs=input_tensor,
                roi_size=(64, 64, 64),
                overlap=0.5,
                device=device,
                brain_mask=brain_mask
            )
            inference_time_ms = round((time.perf_counter() - t_start) * 1000, 2)
            inference_status = "completed"

        # 3. Compute Quantitative Voxel and Physical Volume Measurements
        dx, dy, dz = mri_vol.voxel_spacing
        voxel_vol_mm3 = float(dx * dy * dz)
        gt_data = gt_vol.data[0] if (gt_vol is not None and gt_vol.data.ndim == 4) else (gt_vol.data if gt_vol is not None else None)

        def calc_metrics_for_mask(m, tag):
            if m is None:
                return None
            clean = np.where(m == 4, 3, m)
            total_v = int(np.sum(clean > 0))
            ncr_v = int(np.sum(clean == 1))
            ed_v = int(np.sum(clean == 2))
            et_v = int(np.sum(clean == 3))
            return {
                "source": tag,
                "voxel_spacing_mm": list(mri_vol.voxel_spacing),
                "voxel_unit_volume_mm3": round(voxel_vol_mm3, 3),
                "subregions": {
                    "WT": {
                        "name": "Whole Tumor",
                        "voxel_count": total_v,
                        "volume_cm3": round((total_v * voxel_vol_mm3) / 1000.0, 3),
                        "volume_mm3": round(total_v * voxel_vol_mm3, 1)
                    },
                    "TC": {
                        "name": "Tumor Core",
                        "voxel_count": ncr_v + et_v,
                        "volume_cm3": round(((ncr_v + et_v) * voxel_vol_mm3) / 1000.0, 3),
                        "volume_mm3": round((ncr_v + et_v) * voxel_vol_mm3, 1)
                    },
                    "ET": {
                        "name": "Enhancing Tumor",
                        "voxel_count": et_v,
                        "volume_cm3": round((et_v * voxel_vol_mm3) / 1000.0, 3),
                        "volume_mm3": round(et_v * voxel_vol_mm3, 1)
                    },
                    "ED": {
                        "name": "Peritumoral Edema",
                        "voxel_count": ed_v,
                        "volume_cm3": round((ed_v * voxel_vol_mm3) / 1000.0, 3),
                        "volume_mm3": round(ed_v * voxel_vol_mm3, 1)
                    },
                    "NCR": {
                        "name": "Necrotic Core",
                        "voxel_count": ncr_v,
                        "volume_cm3": round((ncr_v * voxel_vol_mm3) / 1000.0, 3),
                        "volume_mm3": round(ncr_v * voxel_vol_mm3, 1)
                    }
                }
            }

        gt_measurements = calc_metrics_for_mask(gt_data, "expert_ground_truth") if gt_data is not None else None
        pred_measurements = calc_metrics_for_mask(pred_mask, "experimental_prediction") if pred_mask is not None else None
        # Default active measurements: ground truth if present (gold standard), otherwise prediction
        active_measurements = gt_measurements if gt_measurements is not None else pred_measurements

        # 4. Evaluation against Ground Truth (Only when actual ground-truth exists)
        eval_metrics = None
        if pred_mask is not None and gt_data is not None:
            eval_metrics = compute_case_subregion_metrics(
                pred_mask=pred_mask,
                target_mask=gt_data,
                voxel_spacing=mri_vol.voxel_spacing
            )

        # 5. Extract 3D Surface Meshes
        gt_meshes = None
        if gt_data is not None and np.any(gt_data > 0):
            try:
                gt_meshes = generate_subregion_meshes(gt_data, mri_vol.voxel_spacing)
            except Exception as mesh_err:
                print(f"[WARNING] Ground-truth surface mesh generation failed: {mesh_err}")

        pred_meshes = None
        if pred_mask is not None and np.any(pred_mask > 0):
            try:
                pred_meshes = generate_subregion_meshes(pred_mask, mri_vol.voxel_spacing)
            except Exception as mesh_err:
                print(f"[WARNING] Prediction surface mesh generation failed: {mesh_err}")

        # Active mesh defaults to gold-standard ground truth when available
        active_mesh = gt_meshes if gt_meshes is not None else pred_meshes

        # 6. Generate representative Multi-Planar Slice Previews
        # Primary channel 3D array (H, W, D)
        vol_3d = mri_vol.data[0]
        gt_data = gt_vol.data[0] if (gt_vol is not None and gt_vol.data.ndim == 4) else (gt_vol.data if gt_vol is not None else None)

        axial_meta = generate_volume_slice_pack(vol_3d, pred_mask, gt_data, plane="axial")
        coronal_meta = generate_volume_slice_pack(vol_3d, pred_mask, gt_data, plane="coronal")
        sagittal_meta = generate_volume_slice_pack(vol_3d, pred_mask, gt_data, plane="sagittal")

        # Initial representative focus slices
        focus_idx = axial_meta["initial_focus_slice"]
        mri_focus_slice = np.rot90(vol_3d[:, :, focus_idx])
        pred_focus_slice = np.rot90(pred_mask[:, :, focus_idx]) if pred_mask is not None else None
        gt_focus_slice = np.rot90(gt_data[:, :, focus_idx]) if gt_data is not None else None

        mri_b64, pred_b64 = extract_slice_rgba(mri_focus_slice, pred_focus_slice)
        _, gt_b64 = extract_slice_rgba(mri_focus_slice, gt_focus_slice)

        # Attribution metadata for benchmark studies
        attribution = study_meta.get("dataset_name", "Research Study") if (sample_key and sample_key in sample_manifest) else "Uploaded Study"
        license_str = study_meta.get("license", "CC BY-SA 4.0") if (sample_key and sample_key in sample_manifest) else "Research Use Only"

        res_payload = {
            "status": "success",
            "inference_status": inference_status,
            "patient_study_id": patient_tag,
            "dataset_attribution": attribution,
            "license": license_str,
            "inference_time_ms": inference_time_ms,
            "device": str(device),
            "geometry": {
                "dimensions": list(vol_3d.shape),
                "voxel_spacing_mm": list(mri_vol.voxel_spacing),
                "modality_channels": mri_vol.modality_channels
            },
            "planes": {
                "axial": axial_meta,
                "coronal": coronal_meta,
                "sagittal": sagittal_meta
            },
            "initial_view": {
                "plane": "axial",
                "slice_index": focus_idx,
                "total_slices": vol_3d.shape[2],
                "mri_image": mri_b64,
                "pred_mask": pred_b64,
                "gt_mask": gt_b64,
                "has_ground_truth": gt_b64 is not None,
                "active_overlay": "gt" if gt_b64 is not None else "pred"
            },
            "measurements": active_measurements,
            "ground_truth_measurements": gt_measurements,
            "prediction_measurements": pred_measurements,
            "evaluation_metrics": eval_metrics,
            "mesh_3d": active_mesh,
            "ground_truth_mesh_3d": gt_meshes,
            "prediction_mesh_3d": pred_meshes
        }

        if study_meta is not None:
            cache_key = study_meta.get("id", sample_key)
            PRESET_PREDICTION_CACHE[cache_key] = res_payload
            if sample_key:
                PRESET_PREDICTION_CACHE[sample_key] = res_payload

        return res_payload

    finally:
        # Resource cleanup
        for p in temp_files:
            try:
                if p.exists():
                    os.remove(p)
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
