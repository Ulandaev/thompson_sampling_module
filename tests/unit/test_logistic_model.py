"""Unit tests for LogisticModel (10 tests)."""

from __future__ import annotations

import numpy as np
import pytest

from ts_module.config.schema import (
    ArmConfig,
    ContextConfig,
    ContextFeatureConfig,
    ContextMode,
    LogisticHyperparams,
)
from ts_module.core.models.logistic import LogisticModel
from ts_module.core.preprocessing import FeatureProcessor


def _make_model(n_arms: int = 2) -> LogisticModel:
    arms = [ArmConfig(id=f"arm_{i}", name=f"Arm {i}") for i in range(n_arms)]
    feats = [ContextFeatureConfig(name="x", type="continuous")]
    ctx = ContextConfig(mode=ContextMode.features, features=feats)
    processor = FeatureProcessor(ctx, feature_scaling=False)
    return LogisticModel(arms=arms, feature_processor=processor, hyperparams=LogisticHyperparams())


def _context(x: float = 1.0) -> dict:
    return {"x": x}


def test_sample_returns_probability_in_unit_interval() -> None:
    model = _make_model()
    for _ in range(10):
        theta = model.sample("arm_0", _context())
        assert 0.0 <= theta <= 1.0


def test_update_success_shifts_probability_up() -> None:
    model = _make_model()
    ctx = _context(1.0)
    p_before = model.get_distribution("arm_0", ctx)["estimated_p"]
    for _ in range(20):
        model.update("arm_0", 1.0, ctx)
    p_after = model.get_distribution("arm_0", ctx)["estimated_p"]
    assert p_after > p_before


def test_update_failure_shifts_probability_down() -> None:
    model = _make_model()
    ctx = _context(1.0)
    p_before = model.get_distribution("arm_0", ctx)["estimated_p"]
    for _ in range(20):
        model.update("arm_0", 0.0, ctx)
    p_after = model.get_distribution("arm_0", ctx)["estimated_p"]
    assert p_after < p_before


def test_convergence_binary_reward() -> None:
    arms = [ArmConfig(id="arm_0", name="A")]
    feats = [ContextFeatureConfig(name="x", type="continuous")]
    ctx_cfg = ContextConfig(mode=ContextMode.features, features=feats)
    processor = FeatureProcessor(ctx_cfg, feature_scaling=False)
    model = LogisticModel(arms=arms, feature_processor=processor, hyperparams=LogisticHyperparams(prior_variance=1.0))

    rng = np.random.default_rng(42)
    true_p = 0.75
    ctx = {"x": 1.0}
    for _ in range(500):
        reward = 1.0 if rng.random() < true_p else 0.0
        model.update("arm_0", reward, ctx)

    estimated = model.get_distribution("arm_0", ctx)["estimated_p"]
    assert abs(estimated - true_p) < 0.12


def test_different_contexts_give_different_predictions() -> None:
    arms = [ArmConfig(id="arm_0", name="A")]
    feats = [ContextFeatureConfig(name="x", type="continuous")]
    ctx_cfg = ContextConfig(mode=ContextMode.features, features=feats)
    processor = FeatureProcessor(ctx_cfg, feature_scaling=False)
    model = LogisticModel(arms=arms, feature_processor=processor, hyperparams=LogisticHyperparams())

    rng = np.random.default_rng(0)
    for _ in range(100):
        reward = 1.0 if rng.random() < 0.9 else 0.0
        model.update("arm_0", reward, {"x": 5.0})
    for _ in range(100):
        reward = 1.0 if rng.random() < 0.1 else 0.0
        model.update("arm_0", reward, {"x": -5.0})

    p_high = model.get_distribution("arm_0", {"x": 5.0})["estimated_p"]
    p_low = model.get_distribution("arm_0", {"x": -5.0})["estimated_p"]
    assert p_high > p_low


def test_deterministic_with_seed() -> None:
    model = _make_model()
    ctx = _context()
    s1 = model.sample("arm_0", ctx, seed=42)
    s2 = model.sample("arm_0", ctx, seed=42)
    assert s1 == pytest.approx(s2)


def test_unknown_arm_raises_value_error() -> None:
    model = _make_model()
    with pytest.raises(ValueError, match="Unknown arm_id"):
        model.sample("nonexistent", _context())


def test_state_roundtrip() -> None:
    model = _make_model()
    ctx = _context()
    for _ in range(20):
        model.update("arm_0", 1.0, ctx)

    state = model.get_state()

    arms = [ArmConfig(id=f"arm_{i}", name=f"Arm {i}") for i in range(2)]
    feats = [ContextFeatureConfig(name="x", type="continuous")]
    ctx_cfg = ContextConfig(mode=ContextMode.features, features=feats)
    processor2 = FeatureProcessor(ctx_cfg, feature_scaling=False)
    model2 = LogisticModel(arms=arms, feature_processor=processor2, hyperparams=LogisticHyperparams())
    model2.load_state(state)

    s1 = model.sample("arm_0", ctx, seed=7)
    s2 = model2.sample("arm_0", ctx, seed=7)
    assert s1 == pytest.approx(s2, abs=1e-6)


def test_get_all_distributions_has_global_key() -> None:
    model = _make_model()
    dists = model.get_all_distributions()
    for arm_id in ["arm_0", "arm_1"]:
        assert arm_id in dists
        assert "global" in dists[arm_id]


def test_estimated_p_without_sampling() -> None:
    model = _make_model()
    ctx = _context()
    dist_before = model.get_distribution("arm_0", ctx)
    _ = model.get_distribution("arm_0", ctx)
    dist_after = model.get_distribution("arm_0", ctx)
    assert dist_before["estimated_p"] == pytest.approx(dist_after["estimated_p"])
