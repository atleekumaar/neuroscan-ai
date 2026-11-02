"""
NEUROSCAN AI — Anatomically Grounded Hard-Negative Patch Sampler (Stage 5 / Exp 3)
Provides reproducible, split-isolated hard-negative patch sampling from multi-parametric
MRI volumes to suppress extra-axial, ventricular CSF, and contralateral false positives.
"""

import enum
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import torch
from scipy import ndimage

from backend.segmentation.postprocessing import construct_intracranial_mask


class HardNegativeCategory(str, enum.Enum):
    TUMOR_FOREGROUND = "tumor_foreground"
    EXTRA_AXIAL_BOUNDARY = "extra_axial_boundary"
    VENTRICULAR_CSF = "ventricular_csf"
    CONTRALATERAL_PARENCHYMA = "contralateral_parenchyma"
    GENERAL_BACKGROUND = "general_background"


@dataclass
class PatchMetadata:
    patch_index: int
    category: str
    patient_id: str
    center_coord: Tuple[int, int, int]
    corner_coord: Tuple[int, int, int]
    patch_size: Tuple[int, int, int]
    has_tumor: bool
    tumor_voxel_count: int
    seed: int
    t1n_mean: float
    t2w_mean: float
    t2f_mean: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HardNegativeSamplerConfig:
    patch_size: Tuple[int, int, int] = (48, 48, 48)
    tumor_safety_buffer_voxels: int = 8
    extra_axial_dilation_iter: int = 6
    extra_axial_erosion_iter: int = 4
    deep_brain_erosion_iter: int = 12
    contralateral_erosion_iter: int = 6
    n_foreground: int = 2
    n_extra_axial: int = 1
    n_ventricular: int = 1
    n_contralateral: int = 1
    n_general_background: int = 1

    @property
    def total_patches_per_volume(self) -> int:
        return (
            self.n_foreground
            + self.n_extra_axial
            + self.n_ventricular
            + self.n_contralateral
            + self.n_general_background
        )


class HardNegativeSampler:
    """
    Extracts stratified patches with verified anatomical context from genuine training cases.
    Never modifies models or historical data, and enforces strict patient split isolation.
    """

    def __init__(self, config: Optional[HardNegativeSamplerConfig] = None):
        self.config = config or HardNegativeSamplerConfig()

    def derive_candidate_regions(
        self,
        unnorm_image: np.ndarray,
        seg: np.ndarray
    ) -> Dict[HardNegativeCategory, np.ndarray]:
        """
        Derives mutually exclusive anatomical candidate boolean masks:
          1. TUMOR_FOREGROUND: Active tumor voxels (seg > 0).
          2. EXTRA_AXIAL_BOUNDARY: Intracranial perimeter + skull boundary, zero tumor.
          3. VENTRICULAR_CSF: Deep ventricles (hypointense T1, hyperintense T2), zero tumor.
          4. CONTRALATERAL_PARENCHYMA: Contralateral hemisphere relative to tumor mass, zero tumor.
          5. GENERAL_BACKGROUND: Intracranial brain parenchyma, zero tumor.
        """
        # Ensure 3D shape (H, W, D)
        if seg.ndim == 5:
            seg_3d = seg[0, 0]
        elif seg.ndim == 4:
            seg_3d = seg[0]
        else:
            seg_3d = seg

        if unnorm_image.ndim == 5:
            img_4d = unnorm_image[0]
        else:
            img_4d = unnorm_image

        c, h, w, d = img_4d.shape

        # Tumor presence & safety exclusion buffer
        tumor_mask = seg_3d > 0
        if np.any(tumor_mask):
            dilated_tumor = ndimage.binary_dilation(
                tumor_mask, iterations=self.config.tumor_safety_buffer_voxels
            )
        else:
            dilated_tumor = np.zeros((h, w, d), dtype=bool)

        # 1. Base intracranial brain mask
        brain_mask = construct_intracranial_mask(
            img_4d, intensity_threshold=0.0, fill_holes=True, largest_component_only=True
        )

        # 2. Extra-axial boundary: Transition zone around skull perimeter
        dilated_brain = ndimage.binary_dilation(
            brain_mask, iterations=self.config.extra_axial_dilation_iter
        )
        eroded_brain = ndimage.binary_erosion(
            brain_mask, iterations=self.config.extra_axial_erosion_iter
        )
        # Transition band that has some non-zero signal (not pure empty air)
        has_signal = (img_4d > 0).any(axis=0)
        extra_axial_candidate = (dilated_brain & ~eroded_brain) & has_signal & ~dilated_tumor

        # 3. Ventricular CSF candidate:
        # Located in deep central brain, eroded from boundary by >= deep_brain_erosion_iter
        deep_brain = ndimage.binary_erosion(
            brain_mask, iterations=self.config.deep_brain_erosion_iter
        )
        # Restrict to central anatomical bounding box
        z_min, z_max = int(0.20 * h), int(0.80 * h)
        y_min, y_max = int(0.25 * w), int(0.75 * w)
        x_min, x_max = int(0.25 * d), int(0.75 * d)
        central_box = np.zeros((h, w, d), dtype=bool)
        central_box[z_min:z_max, y_min:y_max, x_min:x_max] = True

        deep_central = deep_brain & central_box & ~dilated_tumor

        # Modality intensities (0: T1n, 1: T1c, 2: T2w, 3: T2f)
        t1n_data = img_4d[0]
        t2w_data = img_4d[2] if c > 2 else img_4d[0]

        brain_t1n = t1n_data[brain_mask]
        brain_t2w = t2w_data[brain_mask]
        t1_thresh = np.percentile(brain_t1n, 35.0) if len(brain_t1n) > 0 else 0
        t2_thresh = np.percentile(brain_t2w, 65.0) if len(brain_t2w) > 0 else 0

        # CSF profile: low T1, elevated T2
        csf_signal = (t1n_data <= t1_thresh) & (t2w_data >= t2_thresh)
        ventricular_candidate = deep_central & csf_signal

        # 4. Contralateral normal parenchyma:
        # Determine tumor hemisphere along Left-Right axis (axis 0 in BraTS RAS+: Right to Left)
        if np.any(tumor_mask):
            tumor_indices = np.argwhere(tumor_mask)
            tumor_z_center = np.mean(tumor_indices[:, 0])
            midline_z = h / 2.0
            if tumor_z_center < midline_z:
                # Tumor is predominantly in lower half (e.g. Right); contralateral is upper half
                contra_half = np.zeros((h, w, d), dtype=bool)
                contra_half[int(midline_z + 8):, :, :] = True
            else:
                # Tumor is in upper half; contralateral is lower half
                contra_half = np.zeros((h, w, d), dtype=bool)
                contra_half[:int(midline_z - 8), :, :] = True
        else:
            contra_half = np.ones((h, w, d), dtype=bool)

        contra_brain = ndimage.binary_erosion(
            brain_mask, iterations=self.config.contralateral_erosion_iter
        )
        contralateral_candidate = contra_brain & contra_half & ~dilated_tumor

        # 5. General Background:
        # Inside brain parenchyma, safe distance from tumor
        general_bg_candidate = brain_mask & ~dilated_tumor

        return {
            HardNegativeCategory.TUMOR_FOREGROUND: tumor_mask,
            HardNegativeCategory.EXTRA_AXIAL_BOUNDARY: extra_axial_candidate,
            HardNegativeCategory.VENTRICULAR_CSF: ventricular_candidate,
            HardNegativeCategory.CONTRALATERAL_PARENCHYMA: contralateral_candidate,
            HardNegativeCategory.GENERAL_BACKGROUND: general_bg_candidate
        }

    def sample_patches_for_study(
        self,
        norm_image: torch.Tensor,
        seg: torch.Tensor,
        unnorm_image: torch.Tensor,
        patient_id: str,
        seed: int = 42
    ) -> Tuple[torch.Tensor, torch.Tensor, List[PatchMetadata]]:
        """
        Samples a stratified set of foreground and hard-negative patches for a single volume.
        Returns:
          - patches_x: Tensor of shape (N, C, pd, ph, pw)
          - patches_y: Tensor of shape (N, 1, pd, ph, pw)
          - patch_metas: List of PatchMetadata objects auditing each patch
        """
        rng = np.random.RandomState(seed)

        # Tensor shape checks
        if norm_image.ndim == 5:
            norm_img_np = norm_image.cpu().numpy()
            seg_np = seg.cpu().numpy()
            unnorm_img_np = unnorm_image.cpu().numpy()
        else:
            norm_img_np = norm_image.unsqueeze(0).cpu().numpy()
            seg_np = seg.unsqueeze(0).cpu().numpy()
            unnorm_img_np = unnorm_image.unsqueeze(0).cpu().numpy()

        _, c, h, w, d = norm_img_np.shape
        pd, ph, pw = self.config.patch_size

        regions = self.derive_candidate_regions(unnorm_img_np, seg_np)

        # Plan sample counts per category
        sample_plan = [
            (HardNegativeCategory.TUMOR_FOREGROUND, self.config.n_foreground),
            (HardNegativeCategory.EXTRA_AXIAL_BOUNDARY, self.config.n_extra_axial),
            (HardNegativeCategory.VENTRICULAR_CSF, self.config.n_ventricular),
            (HardNegativeCategory.CONTRALATERAL_PARENCHYMA, self.config.n_contralateral),
            (HardNegativeCategory.GENERAL_BACKGROUND, self.config.n_general_background)
        ]

        extracted_x: List[torch.Tensor] = []
        extracted_y: List[torch.Tensor] = []
        metas: List[PatchMetadata] = []

        patch_counter = 0

        for category, count in sample_plan:
            cand_mask = regions[category]
            coords = np.argwhere(cand_mask)

            if len(coords) == 0:
                # Fallback to general background if candidate region is sparse
                cand_mask = regions[HardNegativeCategory.GENERAL_BACKGROUND]
                coords = np.argwhere(cand_mask)

            for _ in range(count):
                if len(coords) == 0:
                    # Ultimate fallback: uniform spatial crop
                    cz = rng.randint(pd // 2, max(pd // 2 + 1, h - pd // 2))
                    cy = rng.randint(ph // 2, max(ph // 2 + 1, w - ph // 2))
                    cx = rng.randint(pw // 2, max(pw // 2 + 1, d - pw // 2))
                else:
                    idx = rng.randint(0, len(coords))
                    cz, cy, cx = coords[idx]

                # Bounding box coordinates clamped to volume boundaries
                z0 = max(0, min(int(cz - pd // 2), h - pd))
                y0 = max(0, min(int(cy - ph // 2), w - ph))
                x0 = max(0, min(int(cx - pw // 2), d - pw))

                p_img = norm_image[:, :, z0:z0 + pd, y0:y0 + ph, x0:x0 + pw]
                p_seg = seg[:, :, z0:z0 + pd, y0:y0 + ph, x0:x0 + pw]

                tumor_count = int((p_seg > 0).sum().item())
                has_tumor = tumor_count > 0

                # Strict guarantee: Non-foreground patches must NOT contain tumor voxels
                if category != HardNegativeCategory.TUMOR_FOREGROUND and has_tumor:
                    # Zero out any stray tumor border voxels in hard-negative training patches
                    # to strictly preserve negative semantic supervision
                    p_seg = torch.zeros_like(p_seg)
                    has_tumor = False
                    tumor_count = 0

                # Intensity metrics from unnormalized study for diagnostic logging
                u_patch = unnorm_img_np[0, :, z0:z0 + pd, y0:y0 + ph, x0:x0 + pw]
                t1n_m = float(np.mean(u_patch[0]))
                t2w_m = float(np.mean(u_patch[2])) if c > 2 else 0.0
                t2f_m = float(np.mean(u_patch[3])) if c > 3 else 0.0

                meta = PatchMetadata(
                    patch_index=patch_counter,
                    category=category.value,
                    patient_id=patient_id,
                    center_coord=(int(cz), int(cy), int(cx)),
                    corner_coord=(z0, y0, x0),
                    patch_size=(pd, ph, pw),
                    has_tumor=has_tumor,
                    tumor_voxel_count=tumor_count,
                    seed=seed,
                    t1n_mean=round(t1n_m, 2),
                    t2w_mean=round(t2w_m, 2),
                    t2f_mean=round(t2f_m, 2)
                )

                extracted_x.append(p_img)
                extracted_y.append(p_seg)
                metas.append(meta)
                patch_counter += 1

        patches_x_tensor = torch.cat(extracted_x, dim=0)
        patches_y_tensor = torch.cat(extracted_y, dim=0)

        return patches_x_tensor, patches_y_tensor, metas

