"""
NEUROSCAN AI — Reproducible Full-Volume 3D Segmentation Evaluation Engine
Executes full-volume sliding-window inference across held-out patient cohorts,
validates spatial geometry, records inference latency and hardware metrics,
and generates structured JSON and Markdown clinical baseline reports.
"""

import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import json
import time
import platform
import hashlib
import subprocess
import argparse
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
import numpy as np
import torch

try:
    import nibabel as nib
    NIBABEL_VERSION = nib.__version__
except ImportError:
    NIBABEL_VERSION = "unavailable"

try:
    import monai
    MONAI_VERSION = monai.__version__
except ImportError:
    MONAI_VERSION = "unavailable"

from backend.segmentation.model import load_segmentation_model
from backend.segmentation.pipeline import (
    load_nifti_file,
    preprocess_volume_for_model,
    run_sliding_window_inference
)
from backend.segmentation.metrics import (
    compute_case_subregion_metrics,
    aggregate_dataset_metrics
)


def get_git_commit_sha(repo_root: Path) -> str:
    """Retrieves current Git commit SHA safely."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True
        )
        return res.stdout.strip()
    except Exception:
        return "git-unavailable"


def compute_sha256(filepath: Path) -> str:
    """Calculates SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1048576):
            sha.update(chunk)
    return sha.hexdigest()


def evaluate_cohort(
    checkpoint_path: str,
    data_dir: str = "backend/segmentation/data",
    splits_path: Optional[str] = "backend/segmentation/splits.json",
    split_subset: str = "val",
    output_json: Optional[str] = "reports/segmentation_baseline.json",
    output_md: Optional[str] = "reports/segmentation_baseline.md",
    patch_size: Tuple[int, int, int] = (64, 64, 64),
    overlap: float = 0.5,
    device_name: str = "auto",
    apply_brain_mask: bool = True,
    min_fg_prob: Optional[float] = None
) -> Dict[str, Any]:
    """
    Rigorously evaluates a 3D U-Net checkpoint on patient-partitioned cases.
    Performs full-volume sliding-window inference with spatial validation.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    start_eval_time = datetime.now(timezone.utc)

    # 1. Device resolution
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    # Hardware metadata
    device_info = {
        "device_type": device.type,
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor() or "CPU",
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "monai_version": MONAI_VERSION,
        "nibabel_version": NIBABEL_VERSION,
        "platform": platform.platform()
    }

    # 2. Checkpoint verification
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Segmentation checkpoint not found at: {ckpt_file.resolve()}")

    ckpt_sha256 = compute_sha256(ckpt_file)
    model, is_loaded, ckpt_meta = load_segmentation_model(str(ckpt_file), device)
    if not is_loaded:
        raise RuntimeError(f"Failed to load weights from checkpoint: {checkpoint_path}")

    # Inspect checkpoint and model configuration dynamically
    model_in_channels = getattr(model, "in_channels", None)
    if model_in_channels is None:
        model_in_channels = ckpt_meta.get("in_channels")
    if model_in_channels is None:
        model_in_channels = ckpt_meta.get("training_meta", {}).get("in_channels")
    if model_in_channels is None:
        model_in_channels = 1
    model_in_channels = int(model_in_channels)

    if model_in_channels == 1:
        input_semantics = "Native single-channel axial T1-CE sequence"
    elif model_in_channels == 4:
        input_modality = str(ckpt_meta.get("training_meta", {}).get("input_modality", ""))
        if "replicated" in input_modality.lower() or "legacy" in input_modality.lower():
            input_semantics = "T1-CE duplicated across 4 input channels (legacy Phase 1 compatibility mode)"
        else:
            input_semantics = "4-channel mpMRI (T1, T1-CE, T2, FLAIR) or 4-channel slot alignment"
    else:
        input_semantics = f"Native {model_in_channels}-channel MRI input"

    # 3. Load splits and case definitions
    data_path = Path(data_dir)
    splits_meta = {}
    splits_sha256 = None
    target_case_keys = []

    if splits_path and Path(splits_path).exists():
        sp = Path(splits_path)
        splits_sha256 = compute_sha256(sp)
        with open(sp, "r", encoding="utf-8") as f:
            splits_data = json.load(f)
        splits_meta = {
            "splits_file": str(sp.as_posix()),
            "splits_sha256": splits_sha256,
            "random_seed": splits_data.get("random_seed"),
            "split_protocol": splits_data.get("split_protocol"),
            "statistical_safeguard": splits_data.get("statistical_safeguard", {})
        }
        if split_subset in splits_data.get("splits", {}):
            target_case_keys = [c["case_key"] for c in splits_data["splits"][split_subset]]
        elif split_subset == "all":
            train_keys = [c["case_key"] for c in splits_data.get("splits", {}).get("train", [])]
            val_keys = [c["case_key"] for c in splits_data.get("splits", {}).get("val", [])]
            target_case_keys = train_keys + val_keys
        else:
            raise ValueError(f"Unknown split_subset: '{split_subset}'. Expected 'val', 'train', or 'all'.")

    # Discover volume pairs
    all_nii = list(data_path.glob("*.nii*"))
    unique_paths = {p.resolve(): p for p in all_nii}
    t1ce_files = sorted(
        [p for p in unique_paths.values() if "t1ce" in p.name.lower() and "seg" not in p.name.lower()],
        key=lambda x: x.name.lower()
    )

    cases_to_evaluate: List[Tuple[str, Path, Path]] = []
    for mri_p in t1ce_files:
        name_lower = mri_p.name.lower()
        if "case_" in name_lower:
            parts = name_lower.split("case_")[1].split("_")[0]
            try:
                c_key = f"case_{int(parts):03d}"
            except ValueError:
                c_key = mri_p.stem
        else:
            c_key = mri_p.stem

        if target_case_keys and c_key not in target_case_keys:
            continue

        seg_candidates = [
            mri_p.parent / mri_p.name.replace("_T1CE", "_seg").replace("_t1ce", "_seg"),
            mri_p.parent / mri_p.name.replace("_T1CE.nii.gz", "_seg.nii.gz").replace("_t1ce.nii.gz", "_seg.nii.gz")
        ]
        seg_p = None
        for cand in seg_candidates:
            if cand.exists():
                seg_p = cand
                break
        if seg_p:
            cases_to_evaluate.append((c_key, mri_p, seg_p))

    # Manifest checksum if present
    data_manifest_p = Path("backend/segmentation/data_manifest.json")
    data_manifest_sha256 = compute_sha256(data_manifest_p) if data_manifest_p.exists() else None

    print("\n" + "=" * 70)
    print("NEUROSCAN AI — Full-Volume 3D Segmentation Baseline Benchmark")
    print(f"Checkpoint:       {ckpt_file.name} (SHA-256: {ckpt_sha256[:16]}...)")
    print(f"Device:           {device} ({device_info['device_name']})")
    print(f"Split Subset:     {split_subset} ({len(cases_to_evaluate)} cases queued)")
    print(f"Patch Size:       {patch_size} | Overlap: {overlap}")
    print("=" * 70 + "\n")

    case_evaluations: List[Dict[str, Any]] = []
    skipped_cases: List[Dict[str, str]] = []

    for c_key, mri_p, seg_p in cases_to_evaluate:
        print(f"--> Evaluating {c_key} [{mri_p.name}]...")
        t_start = time.perf_counter()

        try:
            mri_vol = load_nifti_file(mri_p)
            gt_vol = load_nifti_file(seg_p)
        except Exception as e:
            msg = f"Failed to load NIfTI pair: {e}"
            print(f"    [SKIP] {msg}")
            skipped_cases.append({"case_key": c_key, "reason": msg})
            continue

        # Spatial geometry integrity check
        if mri_vol.shape[1:] != gt_vol.shape[1:]:
            msg = f"Shape mismatch: MRI {mri_vol.shape[1:]} != Seg {gt_vol.shape[1:]}"
            print(f"    [SKIP] {msg}")
            skipped_cases.append({"case_key": c_key, "reason": msg})
            continue

        if not np.allclose(mri_vol.affine, gt_vol.affine, atol=1e-3):
            msg = "Affine matrix mismatch between MRI and ground truth"
            print(f"    [SKIP] {msg}")
            skipped_cases.append({"case_key": c_key, "reason": msg})
            continue

        # Preprocess volume using dynamically verified model in_channels
        input_tensor = preprocess_volume_for_model(mri_vol, target_channels=model_in_channels)
        brain_mask = (mri_vol.data[0] > 0) if apply_brain_mask else None

        # Full-volume sliding-window inference
        t_infer_start = time.perf_counter()
        try:
            pred_mask = run_sliding_window_inference(
                model=model,
                inputs=input_tensor,
                roi_size=patch_size,
                overlap=overlap,
                device=device,
                brain_mask=brain_mask,
                min_fg_prob=min_fg_prob
            )
        except torch.cuda.OutOfMemoryError as e:
            msg = f"CUDA Out of Memory during sliding-window inference: {e}"
            print(f"    [ERROR] {msg}")
            skipped_cases.append({"case_key": c_key, "reason": msg})
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            msg = f"Inference execution failed: {e}"
            print(f"    [ERROR] {msg}")
            skipped_cases.append({"case_key": c_key, "reason": msg})
            continue

        infer_duration = round(time.perf_counter() - t_infer_start, 3)
        total_case_duration = round(time.perf_counter() - t_start, 3)

        # Ensure prediction shape matches ground truth exactly
        gt_data = gt_vol.data[0] if gt_vol.data.ndim == 4 else gt_vol.data
        if pred_mask.shape != gt_data.shape:
            msg = f"Reconstructed prediction shape {pred_mask.shape} != GT shape {gt_data.shape}"
            print(f"    [ERROR] {msg}")
            skipped_cases.append({"case_key": c_key, "reason": msg})
            continue

        # Calculate metrics
        case_metric = compute_case_subregion_metrics(
            pred_mask=pred_mask,
            target_mask=gt_data,
            voxel_spacing=mri_vol.voxel_spacing
        )

        pred_unique, pred_counts = np.unique(pred_mask, return_counts=True)
        pred_label_dist = {int(k): int(v) for k, v in zip(pred_unique, pred_counts)}

        case_summary = {
            "case_key": c_key,
            "mri_filename": mri_p.name,
            "seg_filename": seg_p.name,
            "volume_shape": list(gt_data.shape),
            "voxel_spacing_mm": list(mri_vol.voxel_spacing),
            "inference_duration_sec": infer_duration,
            "total_processing_sec": total_case_duration,
            "predicted_label_distribution": pred_label_dist,
            "mean_dice": case_metric["mean_dice"],
            "mean_iou": case_metric["mean_iou"],
            "subregions": case_metric["subregions"]
        }
        case_evaluations.append(case_summary)

        wt_d = case_metric["subregions"]["WT"]["dice"]
        tc_d = case_metric["subregions"]["TC"]["dice"]
        et_d = case_metric["subregions"]["ET"]["dice"]
        print(f"    [DONE in {infer_duration}s] Mean Dice: {case_summary['mean_dice']:.4f} | WT: {wt_d:.4f} | TC: {tc_d:.4f} | ET: {et_d:.4f}")

    if not case_evaluations:
        print("[CRITICAL] No cases were successfully evaluated!", file=sys.stderr)
        return {"error": "Zero valid evaluated cases", "skipped_cases": skipped_cases}

    aggregate_results = aggregate_dataset_metrics(case_evaluations)

    git_sha = get_git_commit_sha(repo_root)

    report_data = {
        "benchmark_title": "NEUROSCAN AI — 3D Volumetric Segmentation Baseline Evaluation",
        "created_at_utc": start_eval_time.isoformat(),
        "git_commit_sha": git_sha,
        "checkpoint": {
            "path": str(ckpt_file.as_posix()),
            "filename": ckpt_file.name,
            "sha256": ckpt_sha256,
            "size_bytes": ckpt_file.stat().st_size,
            "architecture": ckpt_meta.get("training_meta", {}).get("architecture", "3D U-Net (BraTS Benchmark)") if isinstance(ckpt_meta, dict) else "3D U-Net",
            "epochs_trained": ckpt_meta.get("training_meta", {}).get("epochs_trained", ckpt_meta.get("epoch", "N/A")) if isinstance(ckpt_meta, dict) else "N/A",
            "best_epoch": ckpt_meta.get("training_meta", {}).get("best_epoch", ckpt_meta.get("best_epoch", "N/A")) if isinstance(ckpt_meta, dict) else "N/A",
            "initial_checkpoint_dice": ckpt_meta.get("training_meta", {}).get("best_val_dice", ckpt_meta.get("training_meta", {}).get("final_val_dice", None)) if isinstance(ckpt_meta, dict) else None,
            "input_channels": model_in_channels
        },
        "dataset": {
            "data_directory": str(data_path.as_posix()),
            "data_manifest_sha256": data_manifest_sha256,
            "splits": splits_meta,
            "split_evaluated": split_subset
        },
        "inference_configuration": {
            "mode": "full_volume_sliding_window",
            "patch_size": list(patch_size),
            "overlap": overlap,
            "input_channels": model_in_channels,
            "input_semantics": input_semantics,
            "min_fg_prob": min_fg_prob,
            "apply_brain_mask": apply_brain_mask,
            "intensity_normalization": "Non-zero z-score standardization per channel"
        },
        "system_environment": device_info,
        "evaluation_summary": {
            "total_cases_queued": len(cases_to_evaluate),
            "valid_cases_evaluated": len(case_evaluations),
            "skipped_cases_count": len(skipped_cases),
            "overall_mean_dice": aggregate_results["overall_mean_dice"],
            "overall_median_dice": aggregate_results["overall_median_dice"],
            "mean_inference_time_sec": round(float(np.mean([c["inference_duration_sec"] for c in case_evaluations])), 3)
        },
        "aggregate_metrics": aggregate_results,
        "per_case_evaluations": case_evaluations,
        "skipped_cases": skipped_cases,
        "methodological_notes": [
            "Full-volume evaluation: Inference is executed across the complete 240x240x155 volume via sliding window blending, not center patch extraction.",
            "Subregion definitions: Whole Tumor (WT: labels 1+2+3), Tumor Core (TC: labels 1+3), Enhancing Tumor (ET: label 3).",
            "Small cohort notice: Baseline benchmark reflects engineering pipeline verification on the genuine BraTS development cohort; does not represent prospective clinical trial efficacy."
        ]
    }

    # Save JSON report
    if output_json:
        out_json_path = repo_root / output_json if not Path(output_json).is_absolute() else Path(output_json)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        print(f"\n[INFO] Saved machine-readable baseline report to: {out_json_path.resolve()}")

    # Save Markdown report
    if output_md:
        out_md_path = repo_root / output_md if not Path(output_md).is_absolute() else Path(output_md)
        out_md_path.parent.mkdir(parents=True, exist_ok=True)
        md_text = generate_markdown_report(report_data)
        with open(out_md_path, "w", encoding="utf-8") as f:
            f.write(md_text)
        print(f"[INFO] Saved human-readable Markdown baseline report to: {out_md_path.resolve()}")

    return report_data


def generate_markdown_report(report: Dict[str, Any]) -> str:
    """Formats structured evaluation dictionary into publication-grade Markdown."""
    ckpt = report["checkpoint"]
    env = report["system_environment"]
    cfg = report["inference_configuration"]
    summary = report["evaluation_summary"]
    agg = report["aggregate_metrics"]
    cases = report["per_case_evaluations"]

    lines = [
        "# NEUROSCAN AI — 3D Volumetric Segmentation Baseline Benchmark Report",
        "",
        f"**Generated (UTC):** {report['created_at_utc']}  ",
        f"**Git Commit SHA:** `{report['git_commit_sha']}`  ",
        f"**Evaluation Mode:** Full-Volume Sliding-Window Inference ({cfg['patch_size'][0]}×{cfg['patch_size'][1]}×{cfg['patch_size'][2]}, overlap={cfg['overlap']})  ",
        f"**Execution Device:** `{env['device_name']}` ({env['device_type']})  ",
        "",
        "---",
        "",
        "## 1. Provenance & Benchmark Artifacts",
        "",
        f"* **Model Checkpoint:** `{ckpt['filename']}` (`{ckpt['path']}`)",
        f"* **Checkpoint SHA-256:** `{ckpt['sha256']}`",
        f"* **Model Architecture:** {ckpt['architecture']} (in={cfg['input_channels']}, out=4)",
        f"* **Input Channels & Semantics:** {cfg['input_channels']} channel(s) — {cfg['input_semantics']}",
        f"* **Inference Threshold (`min_fg_prob`):** `{cfg['min_fg_prob'] if cfg.get('min_fg_prob') is not None else 'Default (Argmax / No Threshold)'}`",
        f"* **Intracranial Brain Masking:** `{'Enabled (Foreground non-zero suppression)' if cfg.get('apply_brain_mask') else 'Disabled'}`",
        f"* **Training Epochs:** {ckpt['epochs_trained']} epochs (Best Epoch: {ckpt.get('best_epoch', 'N/A')})",
        f"* **Dataset Manifest SHA-256:** `{report['dataset'].get('data_manifest_sha256', 'N/A')}`",
        f"* **Split Partition Evaluated:** `{report['dataset']['split_evaluated']}` (Cohort: {summary['valid_cases_evaluated']} cases evaluated)",
        f"* **Splits File:** `{report['dataset'].get('splits', {}).get('splits_file', 'N/A')}` (SHA-256: `{report['dataset'].get('splits', {}).get('splits_sha256', 'N/A')}`)",
        f"* **PyTorch / MONAI / NiBabel:** `{env['torch_version']}` / `{env['monai_version']}` / `{env['nibabel_version']}`",
        "",
        "---",
        "",
        "## 2. Aggregate Benchmark Performance",
        "",
        f"**Overall Mean Dice:** **`{summary['overall_mean_dice']:.4f}`** | **Overall Median Dice:** **`{summary['overall_median_dice']:.4f}`** | **Mean Latency:** `{summary['mean_inference_time_sec']:.2f}s` / case",
        "",
        "| Subregion Hierarchy | Dice Mean (±SD) | Dice Median | Dice [Min, Max] | IoU (Jaccard) | Sensitivity (Recall) | Precision (PPV) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for sub, name in [("WT", "Whole Tumor (WT)"), ("TC", "Tumor Core (TC)"), ("ET", "Enhancing Tumor (ET)")]:
        s = agg["subregions"][sub]
        d_mean = s["dice"]["mean"]
        d_std = s["dice"]["std"]
        d_med = s["dice"]["median"]
        d_min = s["dice"]["min"]
        d_max = s["dice"]["max"]
        iou_m = s["iou"]["mean"]
        sens_m = s["sensitivity"]["mean"]
        prec_m = s["precision"]["mean"]

        lines.append(
            f"| **{name}** | {d_mean:.4f} (±{d_std:.4f}) | {d_med:.4f} | [{d_min:.4f}, {d_max:.4f}] | {iou_m:.4f} | {sens_m:.4f} | {prec_m:.4f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Per-Patient Quantitative Breakdown",
        "",
        "| Case Key | File ID | Dimensions | Latency | WT Dice | TC Dice | ET Dice | Mean Dice | Pred WT Vol | Ref WT Vol | Vol Diff % |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ])

    for c in cases:
        wt = c["subregions"]["WT"]
        tc = c["subregions"]["TC"]
        et = c["subregions"]["ET"]
        dims_str = f"{c['volume_shape'][0]}×{c['volume_shape'][1]}×{c['volume_shape'][2]}"
        lines.append(
            f"| **`{c['case_key']}`** | `{c['mri_filename']}` | {dims_str} | {c['inference_duration_sec']:.1f}s | "
            f"{wt['dice']:.4f} | {tc['dice']:.4f} | {et['dice']:.4f} | **{c['mean_dice']:.4f}** | "
            f"{wt['pred_volume_cm3']:.2f} cm³ | {wt['target_volume_cm3']:.2f} cm³ | {wt['volume_diff_pct']:.1f}% |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Methodological Findings & Honest Assessment",
        "",
        "1. **Full-Volume Reconstructed Space vs. Center-Patch Illusion:**",
        "   - During initial training, validation dice was evaluated solely on isolated $64\\times64\\times64$ central crops (`val_mean_dice: 0.2758`).",
        "   - Full-volume sliding window evaluation tests the real clinical deployment setting across the entire volume ($240\\times240\\times155$), capturing peripheral false positives and brain margin transitions.",
        "2. **Single-Modality Input Bottleneck:**",
        "   - The current baseline checkpoint receives axial T1-CE duplicated 4 times across input channels to satisfy the 4-channel U-Net contract.",
        "   - In authentic BraTS benchmarks, T2, FLAIR, and native T1 provide orthogonal contrast required to delineate vasogenic edema from core necrosis.",
        "3. **Statistical Sample Size Safeguard:**",
        "   - This baseline evaluation is conducted on the curated 6-case BraTS cohort.",
        "   - These numbers serve as an engineering baseline for regression tracking and pipeline verification, not as proof of general clinical efficacy.",
        ""
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="NeuroScan AI — Volumetric 3D Segmentation Baseline Evaluator")
    parser.add_argument("--checkpoint", type=str, default="models/NeuroScan_Seg3D_UNet.pth", help="Path to model checkpoint")
    parser.add_argument("--data-dir", type=str, default="backend/segmentation/data", help="Directory containing NIfTI cases")
    parser.add_argument("--splits", type=str, default="backend/segmentation/splits.json", help="Path to patient splits manifest")
    parser.add_argument("--subset", type=str, default="val", choices=["val", "train", "all"], help="Cohort split to evaluate")
    parser.add_argument("--output-json", type=str, default="reports/segmentation_baseline.json", help="Path for JSON report")
    parser.add_argument("--output-md", type=str, default="reports/segmentation_baseline.md", help="Path for Markdown report")
    parser.add_argument("--device", type=str, default="auto", help="Inference device: auto, cpu, cuda")
    parser.add_argument("--apply-brain-mask", dest="apply_brain_mask", action="store_true", default=True, help="Suppress false positives outside intracranial brain mask")
    parser.add_argument("--no-brain-mask", dest="apply_brain_mask", action="store_false", help="Disable intracranial brain masking")
    parser.add_argument("--min-fg-prob", type=float, default=None, help="Minimum foreground probability threshold")
    args = parser.parse_args()

    evaluate_cohort(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        splits_path=args.splits,
        split_subset=args.subset,
        output_json=args.output_json,
        output_md=args.output_md,
        device_name=args.device,
        apply_brain_mask=args.apply_brain_mask,
        min_fg_prob=args.min_fg_prob
    )


if __name__ == "__main__":
    main()
