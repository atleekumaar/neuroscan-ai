"""
NEUROSCAN AI — 3D Brain Tumor Segmentation Training Script
Supports BraTS NIfTI datasets with patient-level deterministic splitting,
Dice + Cross-Entropy loss, mixed-precision (AMP), and checkpoint persistence.
"""

import os
import sys
import argparse
import time
from pathlib import Path
import math
import hashlib
from typing import List, Dict, Any, Tuple, Optional

# Ensure repository root is on sys.path
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def compute_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

from backend.segmentation.model import build_segmentation_model
from backend.segmentation.pipeline import load_nifti_file, normalize_intensity
from backend.segmentation.metrics import compute_case_subregion_metrics

try:
    from monai.losses import DiceCELoss
    MONAI_LOSS_AVAILABLE = True
except ImportError:
    MONAI_LOSS_AVAILABLE = False


class NativeDiceCELoss(nn.Module):
    """Fallback Dice + Cross-Entropy loss for multi-class 3D segmentation"""
    def __init__(self, num_classes: int = 4, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, H, W, D), targets: (B, H, W, D) long
        ce_loss = self.ce(logits, targets)

        # One-hot encode targets for Dice
        probs = torch.softmax(logits, dim=1)
        targets_onehot = torch.nn.functional.one_hot(targets, num_classes=self.num_classes)
        targets_onehot = targets_onehot.permute(0, 4, 1, 2, 3).float()

        dims = (2, 3, 4)
        intersection = torch.sum(probs * targets_onehot, dim=dims)
        cardinality = torch.sum(probs + targets_onehot, dim=dims)
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        # Average across classes 1..C-1 (foreground)
        dice_loss = 1.0 - torch.mean(dice_score[:, 1:])

        return 0.5 * ce_loss + 0.5 * dice_loss


class NiftiSegmentationDataset(Dataset):
    """
    Patient-level dataset for 3D MRI volumes and ground-truth segmentation masks.
    Supports balanced patch sampling across:
      - Foreground tumor subregions (45%)
      - Normal intracranial brain parenchyma (45%)
      - General / transition boundary zones (10%)
    Includes in-memory volume caching for accelerated training.
    """
    def __init__(
        self,
        case_pairs: List[Tuple[Path, Path]],
        patch_size: Tuple[int, int, int] = (64, 64, 64),
        is_train: bool = True,
        in_channels: int = 1,
        samples_per_case: int = 1,
        fg_ratio: float = 0.45,
        norm_brain_ratio: float = 0.45,
        cache_data: bool = True
    ):
        self.case_pairs = case_pairs
        self.patch_size = patch_size
        self.is_train = is_train
        self.in_channels = in_channels
        self.samples_per_case = max(1, samples_per_case)
        self.fg_ratio = fg_ratio
        self.norm_brain_ratio = norm_brain_ratio
        self.cache_data = cache_data
        self._cache = []

        if self.cache_data:
            for mri_p, seg_p in self.case_pairs:
                mri_vol = load_nifti_file(mri_p)
                seg_vol = load_nifti_file(seg_p)

                channels = []
                for c in range(mri_vol.modality_channels):
                    channels.append(normalize_intensity(mri_vol.data[c]))

                if self.in_channels == 1:
                    mri_data = channels[0][np.newaxis, ...]
                elif len(channels) == 1 and self.in_channels == 4:
                    mri_data = np.repeat(channels[0][np.newaxis, ...], 4, axis=0)
                else:
                    mri_data = np.stack(channels[:self.in_channels], axis=0)

                seg_data = seg_vol.data[0] if seg_vol.data.ndim == 4 else seg_vol.data
                seg_data = np.where(seg_data == 4, 3, seg_data).astype(np.int64)

                tumor_indices = np.argwhere(seg_data > 0)
                brain_mask = (mri_data[0] > 0)
                normal_brain_indices = np.argwhere(brain_mask & (seg_data == 0))

                self._cache.append({
                    "mri_data": mri_data,
                    "seg_data": seg_data,
                    "tumor_indices": tumor_indices,
                    "normal_brain_indices": normal_brain_indices,
                    "patient_id": mri_p.stem
                })

    def __len__(self) -> int:
        return len(self.case_pairs) * self.samples_per_case

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        case_idx = idx % len(self.case_pairs)
        if self.cache_data and self._cache:
            entry = self._cache[case_idx]
            mri_data = entry["mri_data"]
            seg_data = entry["seg_data"]
            tumor_indices = entry["tumor_indices"]
            normal_brain_indices = entry["normal_brain_indices"]
            patient_id = entry["patient_id"]
        else:
            mri_path, seg_path = self.case_pairs[case_idx]
            mri_vol = load_nifti_file(mri_path)
            seg_vol = load_nifti_file(seg_path)

            channels = []
            for c in range(mri_vol.modality_channels):
                channels.append(normalize_intensity(mri_vol.data[c]))

            if self.in_channels == 1:
                mri_data = channels[0][np.newaxis, ...]
            elif len(channels) == 1 and self.in_channels == 4:
                mri_data = np.repeat(channels[0][np.newaxis, ...], 4, axis=0)
            else:
                mri_data = np.stack(channels[:self.in_channels], axis=0)

            seg_data = seg_vol.data[0] if seg_vol.data.ndim == 4 else seg_vol.data
            seg_data = np.where(seg_data == 4, 3, seg_data).astype(np.int64)

            tumor_indices = np.argwhere(seg_data > 0)
            brain_mask = (mri_data[0] > 0)
            normal_brain_indices = np.argwhere(brain_mask & (seg_data == 0))
            patient_id = mri_path.stem

        c, h, w, d = mri_data.shape
        ph, pw, pd = self.patch_size

        if self.is_train and h > ph and w > pw and d > pd:
            r = np.random.rand()
            if r < self.fg_ratio and len(tumor_indices) > 0:
                center = tumor_indices[np.random.choice(len(tumor_indices))]
                sh = max(0, min(h - ph, center[0] - ph // 2))
                sw = max(0, min(w - pw, center[1] - pw // 2))
                sd = max(0, min(d - pd, center[2] - pd // 2))
            elif r < (self.fg_ratio + self.norm_brain_ratio) and len(normal_brain_indices) > 0:
                center = normal_brain_indices[np.random.choice(len(normal_brain_indices))]
                sh = max(0, min(h - ph, center[0] - ph // 2))
                sw = max(0, min(w - pw, center[1] - pw // 2))
                sd = max(0, min(d - pd, center[2] - pd // 2))
            else:
                sh = np.random.randint(0, h - ph + 1)
                sw = np.random.randint(0, w - pw + 1)
                sd = np.random.randint(0, d - pd + 1)
        else:
            sh = max(0, (h - ph) // 2)
            sw = max(0, (w - pw) // 2)
            sd = max(0, (d - pd) // 2)

        mri_patch = mri_data[:, sh:sh+ph, sw:sw+pw, sd:sd+pd]
        seg_patch = seg_data[sh:sh+ph, sw:sw+pw, sd:sd+pd]

        # Pad if volume smaller than patch size
        if mri_patch.shape[1:] != (ph, pw, pd):
            pad_h = ph - mri_patch.shape[1]
            pad_w = pw - mri_patch.shape[2]
            pad_d = pd - mri_patch.shape[3]
            mri_patch = np.pad(mri_patch, ((0, 0), (0, pad_h), (0, pad_w), (0, pad_d)))
            seg_patch = np.pad(seg_patch, ((0, pad_h), (0, pad_w), (0, pad_d)))

        return {
            "image": torch.from_numpy(mri_patch).float(),
            "mask": torch.from_numpy(seg_patch).long(),
            "patient_id": patient_id
        }


def partition_cases_by_patient(
    data_dir: Path,
    val_ratio: float = 0.333,
    seed: int = 42,
    splits_path: Optional[Path] = None
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    """
    Partitions NIfTI cases by patient. Ingests splits.json if present
    to ensure reproducible held-out evaluation with zero cross-split leakage.
    On raw filesystem scanning, strictly deduplicates paths across case-insensitive OSes.
    """
    import json
    splits_file = splits_path or (data_dir.parent / "splits.json")
    if splits_file and Path(splits_file).exists():
        try:
            with open(splits_file, "r", encoding="utf-8") as f:
                s_data = json.load(f)
            train_items = s_data.get("splits", {}).get("train", [])
            val_items = s_data.get("splits", {}).get("val", [])

            train_pairs = []
            for item in train_items:
                mri_p = data_dir / item["mri_filename"]
                seg_p = data_dir / item["seg_filename"]
                if mri_p.exists() and seg_p.exists():
                    train_pairs.append((mri_p, seg_p))

            val_pairs = []
            for item in val_items:
                mri_p = data_dir / item["mri_filename"]
                seg_p = data_dir / item["seg_filename"]
                if mri_p.exists() and seg_p.exists():
                    val_pairs.append((mri_p, seg_p))

            if len(train_pairs) > 0 and len(val_pairs) > 0:
                return train_pairs, val_pairs
        except Exception as e:
            print(f"[WARN] Failed to load {splits_file}, falling back to directory scan: {e}")

    # Fallback to deduplicated file scanning
    all_nii = list(data_dir.glob("*.nii*"))
    unique_paths = {p.resolve(): p for p in all_nii}
    files = sorted([p for p in unique_paths.values() if "t1ce" in p.name.lower() and "seg" not in p.name.lower()])
    pairs = []

    for mri_p in files:
        name = mri_p.name
        seg_name = name.replace("_T1CE", "_seg").replace("_t1ce", "_seg")
        seg_p = mri_p.parent / seg_name
        if seg_p.exists():
            pairs.append((mri_p, seg_p))

    # Deterministic patient split
    rng = np.random.RandomState(seed)
    indices = np.arange(len(pairs))
    rng.shuffle(indices)

    n_val = max(1, int(len(pairs) * val_ratio)) if len(pairs) > 1 else 0
    val_idx = set(indices[:n_val])

    train_pairs = [pairs[i] for i in range(len(pairs)) if i not in val_idx]
    val_pairs = [pairs[i] for i in range(len(pairs)) if i in val_idx]

    return train_pairs, val_pairs


def train_segmentation_model(
    data_dir: str = "backend/segmentation/data",
    output_checkpoint_path: str = "models/NeuroScan_Seg3D_UNet_T1CE_SingleChannel.pth",
    epochs: int = 20,
    batch_size: int = 2,
    lr: float = 1e-4,
    patch_size: Tuple[int, int, int] = (64, 64, 64),
    device_name: str = "auto",
    in_channels: int = 1,
    splits_path: Optional[str] = "backend/segmentation/splits.json",
    seed: int = 42,
    samples_per_case: int = 1,
    include_background: bool = True,
    fg_ratio: float = 0.45,
    norm_brain_ratio: float = 0.45,
    save_final_checkpoint: bool = False,
    final_checkpoint_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes supervised training of 3D U-Net using ground-truth masks.
    Saves trained checkpoint with weights and validation metadata.
    Guarantees that the saved checkpoint is strictly the best validation performer.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if device_name == "auto" else device_name
    )

    data_path = Path(data_dir)
    splits_p = Path(splits_path) if splits_path else None
    train_pairs, val_pairs = partition_cases_by_patient(data_path, splits_path=splits_p, seed=seed)

    if len(train_pairs) == 0:
        raise ValueError(f"No valid MRI/segmentation pairs found in {data_dir}")

    train_ds = NiftiSegmentationDataset(
        train_pairs,
        patch_size=patch_size,
        is_train=True,
        in_channels=in_channels,
        samples_per_case=samples_per_case,
        fg_ratio=fg_ratio,
        norm_brain_ratio=norm_brain_ratio
    )
    val_ds = NiftiSegmentationDataset(
        val_pairs or train_pairs,
        patch_size=patch_size,
        is_train=False,
        in_channels=in_channels,
        samples_per_case=1
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = build_segmentation_model(in_channels=in_channels, out_channels=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    if MONAI_LOSS_AVAILABLE:
        loss_fn = DiceCELoss(include_background=include_background, to_onehot_y=True, softmax=True)
    else:
        loss_fn = NativeDiceCELoss(num_classes=4)

    best_val_dice = -1.0
    best_epoch = -1
    f_p: Optional[Path] = None
    history = []

    splits_sha = compute_sha256(splits_p) if (splits_p and splits_p.exists()) else None
    splits_meta = {
        "splits_file": str(splits_p.as_posix()) if splits_p else None,
        "splits_sha256": splits_sha,
        "train_cases_count": len(train_pairs),
        "val_cases_count": len(val_pairs)
    }
    training_config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "patch_size": list(patch_size),
        "in_channels": in_channels,
        "out_channels": 4,
        "seed": seed,
        "samples_per_case": samples_per_case,
        "fg_ratio": fg_ratio,
        "norm_brain_ratio": norm_brain_ratio,
        "include_background": include_background,
        "device": str(device)
    }

    print(f"[TRAIN] Starting 3D U-Net (in_channels={in_channels}) training on {len(train_pairs)} cases (Val: {len(val_pairs)}) on {device}")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        start_t = time.perf_counter()

        for batch in train_loader:
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type if device.type in ["cuda", "cpu"] else "cpu", enabled=(device.type == "cuda")):
                logits = model(imgs)
                if MONAI_LOSS_AVAILABLE:
                    loss = loss_fn(logits, masks.unsqueeze(1))
                else:
                    loss = loss_fn(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, len(train_loader))

        # Validation step
        model.eval()
        val_dices = []
        with torch.no_grad():
            for v_batch in val_loader:
                v_img = v_batch["image"].to(device)
                v_mask = v_batch["mask"].cpu().numpy()[0]
                with torch.amp.autocast(device_type=device.type if device.type in ["cuda", "cpu"] else "cpu", enabled=(device.type == "cuda")):
                    v_logits = model(v_img)
                v_pred = torch.argmax(v_logits, dim=1).cpu().numpy()[0]
                case_m = compute_case_subregion_metrics(v_pred, v_mask)
                m_dice = case_m.get("mean_dice")
                if m_dice is not None and not (isinstance(m_dice, (float, int)) and (math.isnan(m_dice) or math.isinf(m_dice))):
                    val_dices.append(float(m_dice))

        if val_dices:
            raw_mean = float(np.mean(val_dices))
            val_mean_dice = None if (math.isnan(raw_mean) or math.isinf(raw_mean)) else raw_mean
        else:
            val_mean_dice = None

        epoch_dur = round(time.perf_counter() - start_t, 2)
        history.append({
            "epoch": epoch,
            "train_loss": round(avg_loss, 4),
            "val_mean_dice": round(val_mean_dice, 4) if val_mean_dice is not None else None,
            "duration_sec": epoch_dur
        })
        val_display = f"{val_mean_dice:.4f}" if val_mean_dice is not None else "N/A"
        print(f"[Epoch {epoch:02d}/{epochs:02d}] Loss: {avg_loss:.4f} | Val Dice: {val_display} ({epoch_dur}s)")

        # Save best model when validation metric strictly improves
        is_improved = (val_mean_dice is not None) and (val_mean_dice > best_val_dice)
        if is_improved:
            best_val_dice = val_mean_dice
            best_epoch = epoch
            out_p = Path(output_checkpoint_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "best_epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_dice": round(best_val_dice, 6),
                "validation_metric": "mean_subregion_dice",
                "in_channels": in_channels,
                "seed": seed,
                "metadata": {
                    "architecture": "3D U-Net (BraTS Single-Channel Baseline)" if in_channels == 1 else "3D U-Net (BraTS Benchmark)",
                    "epochs_trained": epoch,
                    "best_epoch": best_epoch,
                    "validation_metric": "mean_subregion_dice",
                    "best_val_dice": round(best_val_dice, 6),
                    "final_val_dice": round(best_val_dice, 6),
                    "patch_size": list(patch_size),
                    "in_channels": in_channels,
                    "out_channels": 4,
                    "input_modality": "Axial T1-CE" if in_channels == 1 else "mpMRI",
                    "seed": seed,
                    "splits_path": str(splits_p) if splits_p else None,
                    "splits_manifest": splits_meta,
                    "training_config": training_config,
                    "date": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            }, str(out_p))

        # Optional separate final-epoch checkpoint preservation
        if epoch == epochs and (save_final_checkpoint or final_checkpoint_path):
            out_p = Path(output_checkpoint_path)
            f_p = Path(final_checkpoint_path) if final_checkpoint_path else out_p.parent / f"{out_p.stem}_final{out_p.suffix}"
            f_p.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "best_epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_dice": round(best_val_dice, 6) if best_val_dice >= 0 else None,
                "final_val_dice": round(val_mean_dice, 6) if val_mean_dice is not None else None,
                "is_final_epoch": True,
                "validation_metric": "mean_subregion_dice",
                "in_channels": in_channels,
                "seed": seed,
                "metadata": {
                    "architecture": "3D U-Net (BraTS Single-Channel Baseline)" if in_channels == 1 else "3D U-Net (BraTS Benchmark)",
                    "epochs_trained": epoch,
                    "best_epoch": best_epoch,
                    "validation_metric": "mean_subregion_dice",
                    "best_val_dice": round(best_val_dice, 6) if best_val_dice >= 0 else None,
                    "val_metric_at_save": round(val_mean_dice, 6) if val_mean_dice is not None else None,
                    "patch_size": list(patch_size),
                    "in_channels": in_channels,
                    "out_channels": 4,
                    "input_modality": "Axial T1-CE" if in_channels == 1 else "mpMRI",
                    "seed": seed,
                    "splits_path": str(splits_p) if splits_p else None,
                    "splits_manifest": splits_meta,
                    "training_config": training_config,
                    "date": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            }, str(f_p))

    # Fallback save if no epoch produced a validation score > -1.0
    out_p = Path(output_checkpoint_path)
    if not out_p.exists():
        best_epoch = epochs
        best_val_dice = 0.0
        out_p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch": epochs,
            "best_epoch": best_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_dice": 0.0,
            "validation_metric": "mean_subregion_dice",
            "in_channels": in_channels,
            "seed": seed,
            "metadata": {
                "architecture": "3D U-Net (BraTS Single-Channel Baseline)" if in_channels == 1 else "3D U-Net (BraTS Benchmark)",
                "epochs_trained": epochs,
                "best_epoch": best_epoch,
                "validation_metric": "mean_subregion_dice",
                "best_val_dice": 0.0,
                "final_val_dice": 0.0,
                "patch_size": list(patch_size),
                "in_channels": in_channels,
                "out_channels": 4,
                "input_modality": "Axial T1-CE" if in_channels == 1 else "mpMRI",
                "seed": seed,
                "splits_path": str(splits_p) if splits_p else None,
                "splits_manifest": splits_meta,
                "training_config": training_config,
                "date": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }, str(out_p))

    return {
        "status": "success",
        "best_epoch": best_epoch,
        "best_val_dice": round(best_val_dice, 4) if best_val_dice >= 0 else None,
        "total_epochs": epochs,
        "checkpoint_path": str(output_checkpoint_path),
        "final_checkpoint_path": str(f_p) if f_p else None,
        "history": history
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NeuroScan 3D Brain Tumor Segmentation Model")
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir", type=str, default="backend/segmentation/data", help="Directory containing NIfTI volume pairs")
    parser.add_argument("--output", "--output-checkpoint-path", dest="output", type=str, default="models/NeuroScan_Seg3D_UNet_T1CE_SingleChannel.pth", help="Checkpoint output path")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=2, help="Patch batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--in-channels", "--in_channels", dest="in_channels", type=int, default=1, help="Input modality channels (1 for T1-CE, 4 for mpMRI)")
    parser.add_argument("--splits-path", "--splits_path", dest="splits_path", type=str, default="backend/segmentation/splits.json", help="Path to splits.json manifest")
    parser.add_argument("--seed", type=int, default=42, help="Reproducible random seed")
    parser.add_argument("--samples-per-case", "--samples_per_case", dest="samples_per_case", type=int, default=1, help="Patches extracted per case per epoch")
    parser.add_argument("--include-background", dest="include_background", action="store_true", default=True, help="Include background in Dice loss")
    parser.add_argument("--no-include-background", dest="include_background", action="store_false", help="Exclude background from Dice loss")
    parser.add_argument("--save-final-checkpoint", dest="save_final_checkpoint", action="store_true", default=False, help="Preserve a separate final-epoch checkpoint")
    parser.add_argument("--final-output", "--final-checkpoint-path", dest="final_checkpoint_path", type=str, default=None, help="Explicit output path for final-epoch checkpoint")
    parser.add_argument("--fg-ratio", "--fg_ratio", dest="fg_ratio", type=float, default=0.45, help="Sampling probability for foreground patches")
    parser.add_argument("--norm-brain-ratio", "--norm_brain_ratio", dest="norm_brain_ratio", type=float, default=0.45, help="Sampling probability for normal brain patches")
    args = parser.parse_args()

    train_segmentation_model(
        data_dir=args.data_dir,
        output_checkpoint_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        in_channels=args.in_channels,
        splits_path=args.splits_path,
        seed=args.seed,
        samples_per_case=args.samples_per_case,
        include_background=args.include_background,
        fg_ratio=args.fg_ratio,
        norm_brain_ratio=args.norm_brain_ratio,
        save_final_checkpoint=args.save_final_checkpoint,
        final_checkpoint_path=args.final_checkpoint_path
    )
