"""Integration tests: Phase 2 backward compatibility and engine wiring (3 tests)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ts_module.config.loader import load_from_dict
from ts_module.core.engine import TSEngine
from ts_module.core.models.linear import LinearModel
from ts_module.core.models.logistic import LogisticModel

CONFIG_YAML = Path(__file__).parent.parent.parent / "examples" / "ticket_routing" / "config.yaml"


def _features_binary_config(model_type: str = "logistic") -> dict:
    return {
        "id": "test",
        "name": "Test",
        "arms": [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}],
        "context": {
            "mode": "features",
            "features": [
                {"name": "cat", "type": "categorical", "categories": ["x", "y"]},
                {"name": "val", "type": "continuous"},
            ],
        },
        "reward": {"type": "binary", "signals": [{"name": "fcr"}]},
        "hyperparams": {"model_type": model_type},
    }


def test_engine_creates_logistic_for_features_binary_config() -> None:
    config = load_from_dict(_features_binary_config("logistic"))
    engine = TSEngine(config)
    assert isinstance(engine._model, LogisticModel)


def test_engine_creates_linear_for_continuous_config() -> None:
    d = {
        "id": "test",
        "name": "Test",
        "arms": [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}],
        "context": {"mode": "features", "features": [{"name": "x", "type": "continuous"}]},
        "reward": {"type": "continuous", "signals": [{"name": "revenue"}]},
        "hyperparams": {"model_type": "linear"},
    }
    config = load_from_dict(d)
    engine = TSEngine(config)
    assert isinstance(engine._model, LinearModel)


def test_min_exploration_read_from_general_hyperparams() -> None:
    """Top-level min_exploration is used by the engine, not beta.min_exploration."""
    d = _features_binary_config()
    d["hyperparams"]["min_exploration"] = 0.1
    config = load_from_dict(d)
    assert config.hyperparams.min_exploration == pytest.approx(0.1)

    # Engine must apply 10% floor: with 2 arms, each arm gets >= 0.1
    engine = TSEngine(config)
    result = engine.decide(context={"cat": "x", "val": 1.0})
    scores = result.arm_scores
    total = sum(scores.values())
    for arm_id, score in scores.items():
        share = score / total
        assert share >= 0.1 - 1e-9, f"{arm_id} share {share:.3f} < 0.10"
