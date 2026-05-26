"""Unit tests for configuration schema and loader (8 tests)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ts_module.config.loader import load_from_dict, load_from_yaml
from ts_module.config.schema import (
    ArmConfig,
    ContextConfig,
    ContextFeatureConfig,
    ContextMode,
    HyperparamsConfig,
    ModelType,
    ModuleConfig,
    ObjectiveConfig,
    ObjectiveType,
    RewardConfig,
    RewardType,
    SignalConfig,
)

CONFIG_YAML = Path(__file__).parent.parent.parent / "examples" / "ticket_routing" / "config.yaml"


def _base_config_dict(**overrides) -> dict:
    """Return a minimal valid config dict with optional overrides."""
    base: dict = {
        "id": "test_module",
        "name": "Test",
        "arms": [
            {"id": "arm_a", "name": "A"},
            {"id": "arm_b", "name": "B"},
        ],
        "reward": {
            "type": "binary",
            "signals": [{"name": "fcr", "weight": 1.0}],
        },
    }
    base.update(overrides)
    return base


def test_auto_model_type_binary_categorical_resolves_to_beta() -> None:
    """auto + binary reward + categorical context → model_type=beta."""
    d = _base_config_dict(
        context={"mode": "categorical", "features": [{"name": "cat", "type": "categorical"}]},
        hyperparams={"model_type": "auto"},
    )
    config = load_from_dict(d)
    assert config.hyperparams.model_type == ModelType.beta


def test_auto_model_type_binary_features_resolves_to_logistic() -> None:
    """auto + binary reward + features context → model_type=logistic."""
    d = _base_config_dict(
        context={"mode": "features", "features": [{"name": "f", "type": "continuous"}]},
        hyperparams={"model_type": "auto"},
    )
    config = load_from_dict(d)
    assert config.hyperparams.model_type == ModelType.logistic


def test_auto_model_type_continuous_resolves_to_linear() -> None:
    """auto + continuous reward → model_type=linear."""
    d = _base_config_dict(
        reward={"type": "continuous", "signals": [{"name": "revenue"}]},
        hyperparams={"model_type": "auto"},
    )
    config = load_from_dict(d)
    assert config.hyperparams.model_type == ModelType.linear


def test_duplicate_arm_ids_raise_validation_error() -> None:
    """Duplicate arm IDs must raise a Pydantic ValidationError."""
    d = _base_config_dict(
        arms=[
            {"id": "arm_a", "name": "A"},
            {"id": "arm_a", "name": "A duplicate"},
        ]
    )
    with pytest.raises(ValidationError, match="Duplicate arm IDs"):
        load_from_dict(d)


def test_custom_objective_without_formula_raises_error() -> None:
    """Objective type 'custom' without a formula raises ValidationError."""
    d = _base_config_dict(objective={"type": "custom"})
    with pytest.raises(ValidationError, match="formula"):
        load_from_dict(d)


def test_load_from_yaml_returns_module_config() -> None:
    """load_from_yaml parses the ticket_routing config.yaml correctly."""
    config = load_from_yaml(CONFIG_YAML)
    assert isinstance(config, ModuleConfig)
    assert config.id == "ticket_routing_v1"
    assert len(config.arms) == 5
    assert config.hyperparams.model_type == ModelType.beta  # auto → beta


def test_invalid_yaml_raises_error() -> None:
    """A YAML file with wrong types or missing required fields raises an error."""
    # Missing required 'reward' field
    d = {
        "id": "x",
        "name": "X",
        "arms": [{"id": "a", "name": "A"}],
        # no 'reward' key
    }
    with pytest.raises(ValidationError):
        load_from_dict(d)


def test_missing_required_fields_raise_validation_error() -> None:
    """Missing 'arms' raises ValidationError."""
    with pytest.raises(ValidationError):
        load_from_dict({"id": "x", "name": "X", "reward": {"signals": [{"name": "fcr"}]}})
