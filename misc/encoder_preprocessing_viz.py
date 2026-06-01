"""Plotly 3D encoder preprocessing pipeline (misc/eda.ipynb encoder cell)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots

from thesis_style import plotly_fonts

POINT_CLR = "#4C72B0"
MARKER = dict(size=1.8, opacity=0.45, line=dict(width=0))
# Closer camera → larger 3D scene inside each subplot (smaller eye = nearer).
DEFAULT_CAMERA = dict(eye=dict(x=1.5, y=1.5, z=1))
VIEW_RANGE_PAD_FRAC = 0.24
EXPORT_MARGINS = dict(l=52, r=52, t=74, b=58)
# 1x4 strip: moderate canvas so each panel is not oversized in the PDF (see --scale).
FIGURE_WIDTH = 2800
FIGURE_HEIGHT = 600
HORIZONTAL_SPACING = 0.003


def load_encoder_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_raw_ply(ply_path: Path) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(ply_path))
    return np.asarray(pcd.points, dtype=np.float32)


def center_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = points.mean(axis=0, keepdims=True)
    return points - center, center.squeeze()


def normalize_points(
    points: np.ndarray, half_extent: float
) -> tuple[np.ndarray, float]:
    """Same as encoder_dataset._normalize_points / test.py process_ply."""
    max_he = float(np.abs(points).max())
    if max_he < 1e-6:
        return points.copy(), 1.0
    scale = max_he / half_extent
    return points / scale, scale


def subsample_fixed(points: np.ndarray, n: int, seed: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    count = len(pts)
    if count == 0:
        return np.zeros((n, 3), dtype=np.float32)
    rng = np.random.default_rng(seed)
    if count >= n:
        return pts[rng.choice(count, n, replace=False)]
    extra = rng.choice(count, n - count, replace=True)
    return np.concatenate([pts, pts[extra]], axis=0)


def downsample_display(pts: np.ndarray, cap: int, seed: int) -> np.ndarray:
    if len(pts) <= cap:
        return pts
    idx = np.random.default_rng(seed).choice(len(pts), cap, replace=False)
    return pts[idx]


def raw_for_display(
    raw: np.ndarray,
    grid_half_extent: float,
    seed: int,
    *,
    margin_frac: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Viz-only: place the raw cloud centroid at a random point inside ±grid_half_extent."""
    centroid = raw.mean(axis=0)
    rng = np.random.default_rng(seed)
    margin = margin_frac * grid_half_extent
    lo = -grid_half_extent + margin
    hi = grid_half_extent - margin
    target = rng.uniform(lo, hi, size=3).astype(np.float32)
    return raw - centroid + target, centroid, target


def bbox_wireframe(half_extent: float, color: str, name: str) -> go.Scatter3d:
    h = float(half_extent)
    corners = np.array(
        [
            [-h, -h, -h],
            [h, -h, -h],
            [h, h, -h],
            [-h, h, -h],
            [-h, -h, h],
            [h, -h, h],
            [h, h, h],
            [-h, h, h],
        ],
        dtype=np.float64,
    )
    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    xs, ys, zs = [], [], []
    for i, j in edges:
        xs.extend([corners[i, 0], corners[j, 0], None])
        ys.extend([corners[i, 1], corners[j, 1], None])
        zs.extend([corners[i, 2], corners[j, 2], None])
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line=dict(color=color, width=3),
        name=name,
        showlegend=False,
        hoverinfo="skip",
    )


def scene_ranges(pts: np.ndarray, pad_frac: float, fallback_half: float) -> tuple:
    if len(pts) == 0:
        h = fallback_half * (1.0 + pad_frac)
        return [-h, h], [-h, h], [-h, h]
    lo, hi = pts.min(0), pts.max(0)
    center = (lo + hi) / 2
    r = max(float((hi - lo).max()) / 2, 0.02)
    r *= 1.0 + pad_frac
    return (
        [center[0] - r, center[0] + r],
        [center[1] - r, center[1] + r],
        [center[2] - r, center[2] + r],
    )


def cube_ranges(half_extent: float, pad_frac: float = 0.12) -> tuple:
    h = float(half_extent) * (1.0 + pad_frac)
    return [-h, h], [-h, h], [-h, h]


def unified_view_ranges(
    grid_half_extent: float,
    *,
    pad_frac: float = VIEW_RANGE_PAD_FRAC,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Same origin-centered axis limits on every panel (±grid_half_extent + pad)."""
    return cube_ranges(grid_half_extent, pad_frac=pad_frac)


def _axis_style(fonts: dict[str, str | int], label: str, axis_range: list[float]) -> dict:
    tick = dict(family=fonts["family"], size=fonts["tick_size"], color="black")
    title_font = dict(family=fonts["family"], size=fonts["label_size"], color="black")
    return dict(
        title=dict(text=label, font=title_font),
        range=axis_range,
        showbackground=True,
        backgroundcolor="rgb(248,248,250)",
        gridcolor="white",
        showspikes=False,
        tickfont=tick,
    )


def _scene_axis(xr, yr, zr, fonts: dict[str, str | int]) -> dict:
    return dict(
        xaxis=_axis_style(fonts, "X (m)", xr),
        yaxis=_axis_style(fonts, "Y (m)", yr),
        zaxis=_axis_style(fonts, "Z (m)", zr),
        aspectmode="cube",
        camera=DEFAULT_CAMERA,
        dragmode="orbit",
    )


def _apply_thesis_plotly_layout(fig: go.Figure, *, panel_titles: bool = True) -> None:
    fonts = plotly_fonts()
    fig.update_layout(
        font=dict(family=fonts["family"], size=fonts["label_size"], color="black"),
        title=None,
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    if panel_titles:
        fig.update_annotations(
            font=dict(family=fonts["family"], size=fonts["title_size"], color="black"),
        )


def point_trace(
    pts: np.ndarray, color: str, name: str, *, showlegend: bool = False
) -> go.Scatter3d:
    hover_skip = len(pts) > 6000
    m = {**MARKER, "color": color, "line": dict(width=0)}
    if hover_skip:
        ht = "<extra></extra>"
    else:
        ht = (
            name + "<br>x=%{x:.4f} m<br>y=%{y:.4f} m<br>z=%{z:.4f} m<extra></extra>"
        )
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode="markers",
        marker=m,
        name=name,
        showlegend=showlegend,
        hovertemplate=ht,
    )


def build_preprocessing_stages(
    ply_path: Path,
    *,
    normalize_half_extent: float,
    grid_bbox: float,
    num_points: int,
    display_cap: int = 20_000,
    subsample_seed: int = 0,
) -> tuple[list[tuple], dict]:
    """Return (stages, stats) for the four encoder preprocessing panels."""
    raw = load_raw_ply(ply_path)
    centered, centroid = center_points(raw)
    normalized, scale_ratio = normalize_points(centered, normalize_half_extent)
    subsampled = subsample_fixed(normalized, num_points, subsample_seed)
    raw_viz, raw_centroid, raw_viz_centroid = raw_for_display(
        raw, grid_bbox, subsample_seed + 101
    )

    view_ranges = unified_view_ranges(grid_bbox)

    stages = [
        ("1. Raw", downsample_display(raw_viz, display_cap, 11), [], view_ranges),
        (
            "2. Centered",
            downsample_display(centered, display_cap, 12),
            [],
            view_ranges,
        ),
        (
            "3. Normalized (encoder)",
            downsample_display(normalized, display_cap, 13),
            [],
            view_ranges,
        ),
        ("4. Subsampled (encoder)", subsampled, [], view_ranges),
    ]

    stats = {
        "n_raw": len(raw),
        "centroid": centroid,
        "raw_centroid": raw_centroid,
        "viz_raw_centroid": raw_viz_centroid,
        "metric_half_extent": float(np.abs(centered).max()),
        "scale_ratio": scale_ratio,
        "n_subsampled": len(subsampled),
    }
    return stages, stats


def build_encoder_preprocessing_figure(
    ply_path: Path,
    *,
    normalize_half_extent: float,
    grid_bbox: float,
    num_points: int,
    display_cap: int = 20_000,
    subsample_seed: int = 14,
    width: int | None = None,
    height: int | None = None,
    horizontal_spacing: float | None = None,
) -> tuple[go.Figure, dict]:
    """Four-panel 1x4 Plotly figure for encoder preprocessing."""
    stages, stats = build_preprocessing_stages(
        ply_path,
        normalize_half_extent=normalize_half_extent,
        grid_bbox=grid_bbox,
        num_points=num_points,
        display_cap=display_cap,
        subsample_seed=subsample_seed,
    )

    fonts = plotly_fonts()
    fig = make_subplots(
        rows=1,
        cols=4,
        specs=[[{"type": "scatter3d"}] * 4],
        subplot_titles=[s[0] for s in stages],
        horizontal_spacing=(
            horizontal_spacing if horizontal_spacing is not None else HORIZONTAL_SPACING
        ),
    )

    for col, (title, pts, bboxes, ranges) in enumerate(stages, start=1):
        fig.add_trace(point_trace(pts, POINT_CLR, title), row=1, col=col)
        for box in bboxes:
            fig.add_trace(
                bbox_wireframe(box["he"], box["color"], box["name"]),
                row=1,
                col=col,
            )

    for scene_idx, (_, _, _, ranges) in enumerate(stages):
        xr, yr, zr = ranges
        key = "scene" if scene_idx == 0 else f"scene{scene_idx + 1}"
        fig.update_layout(**{key: _scene_axis(xr, yr, zr, fonts)})

    fig.update_layout(
        height=height if height is not None else FIGURE_HEIGHT,
        width=width if width is not None else FIGURE_WIDTH,
        margin=EXPORT_MARGINS,
    )
    _apply_thesis_plotly_layout(fig)
    stats["ply_path"] = str(ply_path)
    return fig, stats


def build_single_stage_figure(
    stage: tuple,
    *,
    width: int = 820,
    height: int = 720,
) -> go.Figure:
    """One preprocessing stage for per-page PDF export."""
    title, pts, bboxes, ranges = stage
    fonts = plotly_fonts()
    fig = go.Figure()
    fig.add_trace(point_trace(pts, POINT_CLR, title))
    for box in bboxes:
        fig.add_trace(bbox_wireframe(box["he"], box["color"], box["name"]))
    xr, yr, zr = ranges
    fig.update_layout(
        width=width,
        height=height,
        margin=EXPORT_MARGINS,
        scene=_scene_axis(xr, yr, zr, fonts),
        annotations=[
            dict(
                text=title,
                xref="paper",
                yref="paper",
                x=0.5,
                y=1.02,
                xanchor="center",
                yanchor="bottom",
                showarrow=False,
                font=dict(
                    family=fonts["family"],
                    size=fonts["title_size"],
                    color="black",
                ),
            ),
        ],
    )
    _apply_thesis_plotly_layout(fig, panel_titles=False)
    return fig
