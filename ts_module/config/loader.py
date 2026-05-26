"""Load ModuleConfig from YAML files or plain dicts."""

from __future__ import annotations

from pathlib import Path

import yaml  # type: ignore[import-untyped]

from ts_module.config.schema import ModuleConfig


def load_from_yaml(path: str | Path) -> ModuleConfig:
    """Load and validate a ModuleConfig from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Validated ModuleConfig instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the YAML content fails validation.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping at root, got {type(data).__name__}")
    return load_from_dict(data)


def load_from_dict(data: dict) -> ModuleConfig:
    """Load and validate a ModuleConfig from a plain dict.

    Args:
        data: Dictionary containing the module configuration.

    Returns:
        Validated ModuleConfig instance.

    Raises:
        ValueError: If the data fails Pydantic validation.
    """
    return ModuleConfig.model_validate(data)
