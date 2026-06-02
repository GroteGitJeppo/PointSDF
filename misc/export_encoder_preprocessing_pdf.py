#!/usr/bin/env python3
"""Export encoder preprocessing Plotly 3D views from misc/eda.ipynb to PDF/HTML.

Reproduces the four-panel figure:
  1. Raw partial scan (metric coordinates)
  2. Centered (metric, origin at centroid)
  3. Normalized for PointNet++ (+ green encoder norm box)
  4. Subsampled to num_points (demo uses random; training uses FPS)

Requires: pip install plotly kaleido open3d pyyaml

Usage (from PointSDF_2/, one line — PowerShell does not use bash \\ continuations):

    python misc/export_encoder_preprocessing_pdf.py --ply misc/reviewdata/R9-9_pcd_365.ply -o misc/encoder_preprocessing.pdf --separate

Separate PDFs (default stem encoder_preprocessing):

    misc/encoder_preprocessing_1_raw.pdf
    misc/encoder_preprocessing_2_centered.pdf
    misc/encoder_preprocessing_3_normalized.pdf
    misc/encoder_preprocessing_4_subsampled.pdf
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

MISC_DIR = Path(__file__).resolve().parent
REPO_ROOT = MISC_DIR.parent
sys.path.insert(0, str(MISC_DIR))

from encoder_preprocessing_viz import (
    build_encoder_preprocessing_figure,
    build_preprocessing_stages,
    build_single_stage_figure,
    load_encoder_config,
)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "train_encoder.yaml"
DEFAULT_PLY = MISC_DIR / "reviewdata/R9-9_pcd_365.ply"
DEFAULT_OUTPUT = MISC_DIR / "encoder_preprocessing.pdf"
DEFAULT_HTML = MISC_DIR / "encoder_preprocessing.html"

STAGE_PDF_SLUGS = ("1_raw", "2_centered", "3_normalized", "4_subsampled")


def _stage_pdf_path(output: Path, slug: str) -> Path:
    return output.parent / f"{output.stem}_{slug}{output.suffix}"


def _write_pdf(fig, path: Path, *, scale: int) -> None:
    try:
        fig.write_image(str(path), format="pdf", scale=scale)
    except ValueError as exc:
        if "kaleido" in str(exc).lower():
            raise SystemExit(
                "PDF export needs kaleido: pip install kaleido"
            ) from exc
        raise


def _write_pdf_one_panel_per_page(
    stages: list,
    path: Path,
    *,
    scale: int,
) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Per-page PDF needs Pillow: pip install pillow") from exc

    pages: list[Image.Image] = []
    for stage in stages:
        fig = build_single_stage_figure(stage)
        png_bytes = fig.to_image(format="png", scale=scale)
        pages.append(Image.open(io.BytesIO(png_bytes)).convert("RGB"))

    path.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(
        path,
        save_all=True,
        append_images=pages[1:],
        resolution=150.0,
    )


def _write_separate_pdfs(
    stages: list,
    output: Path,
    *,
    scale: int,
    figure_width: int | None,
    figure_height: int | None,
) -> list[Path]:
    paths: list[Path] = []
    for stage, slug in zip(stages, STAGE_PDF_SLUGS):
        path = _stage_pdf_path(output, slug)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig = build_single_stage_figure(
            stage,
            width=figure_width if figure_width is not None else 820,
            height=figure_height if figure_height is not None else 720,
        )
        _write_pdf(fig, path, scale=scale)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export misc/eda.ipynb encoder preprocessing Plotly 3D figure "
            "(raw → centered → normalized → subsampled)."
        ),
    )
    parser.add_argument(
        "--ply",
        type=Path,
        default=DEFAULT_PLY,
        help="Partial point cloud .ply (default: misc/reviewdata/R9-9_pcd_365.ply)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="train_encoder.yaml for normalize_half_extent and num_points",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PDF path (default: misc/encoder_preprocessing.pdf)",
    )
    parser.add_argument(
        "--html",
        nargs="?",
        const=str(DEFAULT_HTML),
        default=None,
        metavar="PATH",
        help="Also write interactive HTML (default: misc/encoder_preprocessing.html)",
    )
    parser.add_argument(
        "--grid-bbox",
        type=float,
        default=0.05,
        help="View axis half-extent on all panels (default: 0.10 m)",
    )
    parser.add_argument(
        "--display-cap",
        type=int,
        default=20_000,
        help="Max points shown in panels 1–3 (default: 20000)",
    )
    parser.add_argument(
        "--subsample-seed",
        type=int,
        default=100,
        help="RNG seed for panel 4 random subsample demo (default: 14)",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=1,
        help="Kaleido scale factor for PDF export (default: 1; use 2 for higher resolution)",
    )
    parser.add_argument(
        "--figure-width",
        type=int,
        default=None,
        help="Plotly figure width in px (default: 2800)",
    )
    parser.add_argument(
        "--figure-height",
        type=int,
        default=None,
        help="Plotly figure height in px (default: 600)",
    )
    parser.add_argument(
        "--horizontal-spacing",
        type=float,
        default=None,
        help="Gap between panels as fraction of figure width (default: minimal)",
    )
    export_group = parser.add_mutually_exclusive_group()
    export_group.add_argument(
        "--one-panel-per-page",
        action="store_true",
        help="Single PDF with four pages (one stage per page)",
    )
    export_group.add_argument(
        "--separate",
        action="store_true",
        help=(
            "Write four PDF files (one stage each), e.g. "
            "encoder_preprocessing_1_raw.pdf … _4_subsampled.pdf"
        ),
    )
    args = parser.parse_args()

    ply_path = args.ply.resolve()
    if not ply_path.is_file():
        raise SystemExit(f"PLY not found: {ply_path}")

    config_path = args.config.resolve()
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")

    cfg = load_encoder_config(config_path)
    normalize_half_extent = float(cfg["normalize_half_extent"])
    num_points = int(cfg.get("num_points", 1024))

    print(f"PLY:    {ply_path}")
    print(f"Config: {config_path.name}")
    print(
        f"  normalize_half_extent={normalize_half_extent} m  "
        f"num_points={num_points}  grid_bbox=±{args.grid_bbox} m"
    )

    stages, stats = build_preprocessing_stages(
        ply_path,
        normalize_half_extent=normalize_half_extent,
        grid_bbox=args.grid_bbox,
        num_points=num_points,
        display_cap=args.display_cap,
        subsample_seed=args.subsample_seed,
    )

    out_pdf = Path(args.output).resolve()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    if args.separate:
        paths = _write_separate_pdfs(
            stages,
            out_pdf,
            scale=args.scale,
            figure_width=args.figure_width,
            figure_height=args.figure_height,
        )
        for path in paths:
            print(f"Saved PDF: {path}")
    elif args.one_panel_per_page:
        _write_pdf_one_panel_per_page(stages, out_pdf, scale=args.scale)
        print(f"Saved PDF (4 pages): {out_pdf}")
    else:
        fig, _ = build_encoder_preprocessing_figure(
            ply_path,
            normalize_half_extent=normalize_half_extent,
            grid_bbox=args.grid_bbox,
            num_points=num_points,
            display_cap=args.display_cap,
            subsample_seed=args.subsample_seed,
            horizontal_spacing=args.horizontal_spacing,
            width=args.figure_width,
            height=args.figure_height,
        )
        _write_pdf(fig, out_pdf, scale=args.scale)
        print(f"Saved PDF (1x4): {out_pdf}")

    if args.html is not None:
        fig_html, _ = build_encoder_preprocessing_figure(
            ply_path,
            normalize_half_extent=normalize_half_extent,
            grid_bbox=args.grid_bbox,
            num_points=num_points,
            display_cap=args.display_cap,
            subsample_seed=args.subsample_seed,
            horizontal_spacing=args.horizontal_spacing,
            width=args.figure_width,
            height=args.figure_height,
        )
        html_path = Path(args.html).resolve()
        html_path.parent.mkdir(parents=True, exist_ok=True)
        fig_html.write_html(str(html_path), include_plotlyjs="cdn")
        print(f"Saved HTML: {html_path}")

    print(
        f"\n  n_raw={stats['n_raw']:,}  true centroid={stats['centroid'].round(4)}  "
        f"viz raw centroid={stats['viz_raw_centroid'].round(4)}  "
        f"metric half-extent={stats['metric_half_extent']:.4f} m  "
        f"scale_ratio={stats['scale_ratio']:.4f}  "
        f"n_subsampled={stats['n_subsampled']}"
    )
    print(
        "\nNote: panel 4 uses random subsample for display; "
        "training/inference use FPS (pytorch_fpsample)."
    )


if __name__ == "__main__":
    main()
