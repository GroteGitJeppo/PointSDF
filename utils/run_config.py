"""Merge encoder settings from a training run's saved config.yaml."""

from copy import deepcopy
from pathlib import Path

import yaml

_ENCODER_KEYS = (
    'encoder',
    'so3',
    'num_points',
    'normalize_half_extent',
    'augmentation_enabled',
    'augmentation',
    'batch_size',
)


def resolve_run_dir(path: str, output_dir: str | None = None) -> Path | None:
    """Return the training run directory for a checkpoint or run path."""
    p = Path(path).resolve()
    if p.is_dir() and (p / 'config.yaml').is_file():
        return p

    if not p.is_file():
        return None

    if output_dir:
        try:
            rel = p.parent.relative_to(Path(output_dir).resolve())
            if rel.parts:
                candidate = Path(output_dir).resolve() / rel.parts[0]
                if (candidate / 'config.yaml').is_file():
                    return candidate
        except ValueError:
            pass

    run_dir = p.parent
    while run_dir.name in ('snapshots',) or run_dir.name.startswith('best_vol'):
        run_dir = run_dir.parent
    if (run_dir / 'config.yaml').is_file():
        return run_dir
    return None


def load_merged_config(base_cfg: dict, run_dir: Path | None) -> dict:
    """Overlay encoder-related keys from a run's config.yaml onto base_cfg."""
    if run_dir is None:
        return base_cfg

    run_cfg_path = run_dir / 'config.yaml'
    if not run_cfg_path.is_file():
        return base_cfg

    with open(run_cfg_path) as f:
        run_cfg = yaml.safe_load(f) or {}

    merged = deepcopy(base_cfg)
    for key in _ENCODER_KEYS:
        if key in run_cfg:
            merged[key] = run_cfg[key]
    return merged


def merge_config_from_checkpoint(base_cfg: dict, checkpoint_path: str) -> dict:
    """Resolve run dir from checkpoint path and merge encoder settings."""
    run_dir = resolve_run_dir(checkpoint_path, base_cfg.get('output_dir'))
    return load_merged_config(base_cfg, run_dir)
