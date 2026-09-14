"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Load a YAML configuration file.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ValueError
        If the YAML file is empty or does not contain a mapping.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Configuration path is not a file: {path}"
        )

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(
            f"Configuration file is empty: {path}"
        )

    if not isinstance(config, dict):
        raise ValueError(
            "Configuration must contain a YAML mapping."
        )

    return config