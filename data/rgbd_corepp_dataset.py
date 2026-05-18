"""RGB-D dataset for CoRe++ encoder inference (slim port of MaskedCameraLaserData)."""

from __future__ import annotations

import copy
import json
import os

import cv2
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from torchvision.transforms import v2

from data.corepp_transforms import Pad


class RgbdSampleError(Exception):
    """Raised when a frame cannot be loaded safely (caller should skip)."""


def _label_year(label: str) -> int | None:
    """Parse cohort year from labels like ``2025-013``."""
    prefix = label.split("-", 1)[0]
    if prefix.isdigit() and len(prefix) == 4:
        return int(prefix)
    return None


def resolve_depth_bounds(
    enc_cfg: dict,
    label: str,
    *,
    default_min: float = 230.0,
    default_max: float = 350.0,
) -> tuple[float, float]:
    """Return (depth_min, depth_max) in mm for a tuber label.

    Uses ``encoder.depth_by_year`` when present (keys 2023 / 2025 or ``"2023"``),
    otherwise ``encoder.depth_min`` / ``encoder.depth_max``.
    """
    by_year = enc_cfg.get("depth_by_year")
    year = _label_year(label)
    if by_year and year is not None:
        entry = by_year.get(year) or by_year.get(str(year))
        if entry is not None:
            return float(entry["depth_min"]), float(entry["depth_max"])
    return (
        float(enc_cfg.get("depth_min", default_min)),
        float(enc_cfg.get("depth_max", default_max)),
    )


def load_intrinsics(intrinsics_file: str):
    with open(intrinsics_file) as json_file:
        data = json.load(json_file)
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        data["width"],
        data["height"],
        data["intrinsic_matrix"][0],
        data["intrinsic_matrix"][4],
        data["intrinsic_matrix"][6],
        data["intrinsic_matrix"][7],
    )
    # 3DPotatoTwin intrinsic JSON has no depth_scale; RealSense D405 uses mm (1000).
    depth_scale = float(data.get("depth_scale", 1000.0))
    return intrinsics, depth_scale


def histogram_filtering(dimg, mask, max_depth_range=50, max_depth_contribution=0.05):
    mask = mask.astype(np.uint8)
    mask_bool = mask.astype(bool)

    z = np.expand_dims(dimg, axis=2)
    z_mask = z[mask_bool]
    z_mask_filtered = z_mask[z_mask != 0]

    if z_mask_filtered.size > 1:
        z_mask_filtered_range = np.max(z_mask_filtered) - np.min(z_mask_filtered)
        if z_mask_filtered_range > max_depth_range:
            hist, bin_edges = np.histogram(z_mask_filtered, density=False)
            hist_peak = np.argmax(hist)
            lb = bin_edges[hist_peak]
            ub = bin_edges[hist_peak + 1]
            bc = np.bincount(np.absolute(z_mask_filtered.astype(np.int64)))
            peak_id = np.argmax(bc)
            if peak_id > int(lb) and peak_id < int(ub):
                pass
            else:
                bc_clip = bc[int(lb) : int(ub)]
                peak_id = int(lb) + np.argmax(bc_clip)
            pixel_counts = np.zeros((10), dtype=np.int64)
            for j in range(10):
                lower_bound = peak_id - (max_depth_range - (j * 10))
                upper_bound = lower_bound + max_depth_range
                z_final = z_mask_filtered[
                    np.where(
                        np.logical_and(
                            z_mask_filtered >= lower_bound,
                            z_mask_filtered <= upper_bound,
                        )
                    )
                ]
                pixel_counts[j] = z_final.size
            pix_id = np.argmax(pixel_counts)
            lower_bound = peak_id - (max_depth_range - (pix_id * 10))
            upper_bound = lower_bound + max_depth_range
            z_final = z_mask_filtered[
                np.where(
                    np.logical_and(
                        z_mask_filtered >= lower_bound,
                        z_mask_filtered <= upper_bound,
                    )
                )
            ]
        else:
            z_final = z_mask_filtered
        hist_f, bin_edges_f = np.histogram(z_final, density=False)
        if hist_f.sum() == 0:
            z_min = float(np.min(z_final))
            z_max = float(np.max(z_final))
        else:
            norm1 = hist_f / hist_f.sum()
            keep = np.where(norm1 >= max_depth_contribution)[0]
            if keep.size == 0:
                z_min = float(np.min(z_final))
                z_max = float(np.max(z_final))
            else:
                sel1 = bin_edges_f[keep]
                sel2 = bin_edges_f[np.minimum(keep + 1, len(bin_edges_f) - 1)]
                edges = np.unique(np.concatenate((sel1, sel2)))
                z_min = float(np.min(edges))
                z_max = float(np.max(edges))
    else:
        z_min = float(np.min(z_mask_filtered)) if z_mask_filtered.size else 0.0
        z_max = float(np.max(z_mask_filtered)) if z_mask_filtered.size else 0.0

    return z_min, z_max


def preprocess_images(rgb, depth, mask, intrinsic_file, detection_input="mask"):
    intrinsics, depth_scale = load_intrinsics(intrinsic_file)
    img_mask = np.multiply(rgb, np.expand_dims(mask, axis=2))
    dimg_mask = np.multiply(depth, mask)
    z_min, z_max = histogram_filtering(depth, mask, 50, 0.05)
    dimg_mask[dimg_mask < z_min] = 0
    dimg_mask[dimg_mask > z_max] = 0

    if not np.any(mask):
        raise RgbdSampleError("empty mask after load")

    offset = 20
    indices = np.where(mask)
    min_i = max(indices[0].min() - offset, 0)
    min_j = max(indices[1].min() - offset, 0)
    max_i = indices[0].max() + offset
    max_j = indices[1].max() + offset

    rgb = rgb[min_i:max_i, min_j:max_j, :]
    depth = depth[min_i:max_i, min_j:max_j]
    mask = mask[min_i:max_i, min_j:max_j]

    if detection_input == "mask":
        depth[depth < z_min] = 0
        depth[depth > z_max] = 0
        depth = depth * mask
        mask = np.clip(depth, 0.0, 1.0)
        rgb = rgb * np.expand_dims(mask.astype(bool), 2)

    w = max_i - min_i
    h = max_j - min_j
    if w < 1 or h < 1:
        raise RgbdSampleError(f"degenerate crop size {w}x{h}")
    return rgb, depth, mask, (min_i, min_j), (w, h)


class RgbdCoreppDataset(Dataset):
    """CoRe++-style RGB-D frames for encoder inference."""

    def __init__(
        self,
        data_root: str,
        split: str = "test",
        input_size: int = 304,
        detection_input: str = "mask",
        normalize_depth: bool = True,
        depth_min: float = 230.0,
        depth_max: float = 350.0,
        depth_by_year: dict | None = None,
        label_filter: set[str] | None = None,
    ):
        self.data_root = data_root
        self.split = split
        self.detection_input = detection_input
        self.normalize_depth = normalize_depth
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.depth_by_year = depth_by_year or {}
        self.tf = Compose([
            Pad(size=input_size),
            v2.Resize((input_size, input_size), antialias=True),
        ])

        split_path = os.path.join(data_root, "split.json")
        with open(split_path, encoding="utf-8") as f:
            split_ids = set(json.load(f)[split])
        if label_filter is not None:
            split_ids &= label_filter

        self.files: list[str] = []
        for label in sorted(split_ids):
            mask_dir = os.path.join(data_root, label, "realsense", "masks")
            if not os.path.isdir(mask_dir):
                continue
            for fname in sorted(os.listdir(mask_dir)):
                if not fname.endswith(".png"):
                    continue
                color_path = os.path.join(
                    data_root, label, "realsense", "color", fname
                )
                if os.path.isfile(color_path):
                    self.files.append(color_path)

        if not self.files:
            raise RuntimeError(
                f"No RGB-D frames found for split={split!r} under {data_root}"
            )
        print(f"RgbdCoreppDataset [{split}]: {len(self.files)} frames")
        if self.depth_by_year:
            for y, bounds in sorted(self.depth_by_year.items(), key=lambda kv: str(kv[0])):
                print(
                    f"  depth [{y}]: min={bounds['depth_min']} max={bounds['depth_max']} mm"
                )
            print(f"  depth default: min={self.depth_min} max={self.depth_max} mm")

    def __len__(self) -> int:
        return len(self.files)

    def depth_bounds_for_label(self, label: str) -> tuple[float, float]:
        year = _label_year(label)
        if year is not None and year in self.depth_by_year:
            entry = self.depth_by_year[year]
            return float(entry["depth_min"]), float(entry["depth_max"])
        if year is not None and str(year) in self.depth_by_year:
            entry = self.depth_by_year[str(year)]
            return float(entry["depth_min"]), float(entry["depth_max"])
        return self.depth_min, self.depth_max

    def __getitem__(self, idx: int) -> dict:
        image_path = self.files[idx]
        parts = image_path.replace("\\", "/").split("/")
        label = parts[-4]
        frame_id = os.path.splitext(parts[-1])[0]

        depth_path = image_path.replace("color", "depth").replace(".png", ".npy")
        mask_path = image_path.replace("color", "masks")

        intrinsic_file = os.path.join(
            self.data_root, label, "realsense", "intrinsic.json"
        )

        rgb = cv2.imread(image_path)
        if rgb is None:
            raise RgbdSampleError(f"failed to read RGB: {image_path}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        if not os.path.isfile(depth_path):
            raise RgbdSampleError(f"missing depth: {depth_path}")
        depth = np.load(depth_path).astype(np.float32)

        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            raise RgbdSampleError(f"failed to read mask: {mask_path}")
        mask = mask_img // 255

        if rgb.shape[:2] != depth.shape[:2]:
            raise RgbdSampleError(
                f"RGB/depth shape mismatch {rgb.shape[:2]} vs {depth.shape} for {image_path}"
            )

        try:
            rgb, depth, mask, _crop_origin, _crop_dim = preprocess_images(
                rgb, depth, mask, intrinsic_file, self.detection_input
            )
        except (ValueError, IndexError) as e:
            raise RgbdSampleError(f"preprocess failed for {image_path}: {e}") from e

        depth_min, depth_max = self.depth_bounds_for_label(label)
        if self.normalize_depth:
            depth_original = copy.deepcopy(depth)
            span = depth_max - depth_min
            depth = (depth_original - depth_min) / span if span > 0 else depth_original * 0.0
            depth[depth_original == 0] = 0
        else:
            depth = depth / depth_max

        rgb_t = torch.from_numpy(np.array(self.tf(rgb))).permute(2, 0, 1).float() / 255.0
        depth_t = torch.from_numpy(np.array(self.tf(depth))).unsqueeze(0).float()
        mask_t = torch.from_numpy(np.array(self.tf(mask))).unsqueeze(0).float()
        rgb_t = torch.clamp(rgb_t, 0.0, 1.0)
        depth_t = torch.nan_to_num(depth_t, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "rgb": rgb_t,
            "depth": depth_t,
            "mask": mask_t,
            "label": label,
            "frame_id": frame_id,
            "file_name": image_path,
        }
