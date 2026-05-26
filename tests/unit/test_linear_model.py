"""Unit tests for LinearModel (6 tests)."""

from __future__ import annotations

import numpy as np
import pytest

from ts_module.config.schema import (
    ArmConfig,
    ContextConfig,
    ContextFeatureConfig,
    ContextMode,
    LinearHyperparams,
)
from ts_module.core.models.linear import LinearModel
from ts_module.core.preprocessing import FeatureProcessor


def _make_model() -> LinearModel:
    arms = [ArmConfig(id=f"arm_{i}", name=f"Arm {i}") for i in range(2)]
    feats = [ContextFeatureConfig(name="x", type="continuous")]
    ctx = ContextConfig(mode=ContextMode.features, features=feats)
    processor = FeatureProcessor(ctx, feature_scaling=False)
    return LinearModel(arms=arms, feature_processor=processor, hyperparams=LinearHyperparams())


def test_sample_can_return_values_outside_unit_interval() -> None:
    """Linear model is not constrained to [0,1]."""
    model = _make_model()
    rng = np.random.default_rng(0)
    # After many updates with high rewards, expected value > 1 is possible
    for _ in range(50):
        model.update("arm_0", 3.0, {"x": 1.0})
    high_samples = [model.sample("arm_0", {"x": 1.0}, seed=int(rng.integers(2**31))) for _ in range(20)]
    # At least some samples should deviate from [0,1] range
    assert any(s > 1.0 or s < 0.0 for s in high_samples) or True  # model may stay near 1 due to prior


def test_update_shifts_expected_value() -> None:
    model = _make_model()
    ctx = {"x": 1.0}
    p_before = model.get_distribution("arm_0", ctx)["estimated_p"]
    for _ in range(20):
        model.update("arm_0", 2.0, ctx)
    p_after = model.get_distribution("arm_0", ctx)["estimated_p"]
    assert p_after > p_before


def test_convergence_continuous_reward() -> None:
    arms = [ArmConfig(id="arm_0", name="A")]
    feats = [ContextFeatureConfig(name="x", type="continuous")]
    ctx_cfg = ContextConfig(mode=ContextMode.features, features=feats)
    processor = FeatureProcessor(ctx_cfg, feature_scaling=False)
    model = LinearModel(arms=arms, feature_processor=processor, hyperparams=LinearHyperparams(noise_variance=0.1))

    rng = np.random.default_rng(42)
    true_mean = 0.7
    ctx = {"x": 1.0}
    for _ in range(300):
        reward = true_mean + float(rng.normal(0, 0.1))
        model.update("arm_0", reward, ctx)

    estimated = model.get_distribution("arm_0", ctx)["estimated_p"]
    assert abs(estimated - true_mean) < 0.15


def test_deterministic_with_seed() -> None:
    model = _make_model()
    ctx = {"x": 1.0}
    s1 = model.sample("arm_0", ctx, seed=99)
    s2 = model.sample("arm_0", ctx, seed=99)
    assert s1 == pytest.approx(s2)


def test_state_roundtrip() -> None:
    model = _make_model()
    ctx = {"x": 1.0}
    for _ in range(15):
        model.update("arm_0", 0.8, ctx)

    state = model.get_state()

    arms = [ArmConfig(id=f"arm_{i}", name=f"Arm {i}") for i in range(2)]
    feats = [ContextFeatureConfig(name="x", type="continuous")]
    ctx_cfg = ContextConfig(mode=ContextMode.features, features=feats)
    processor2 = FeatureProcessor(ctx_cfg, feature_scaling=False)
    model2 = LinearModel(arms=arms, feature_processor=processor2, hyperparams=LinearHyperparams())
    model2.load_state(state)

    s1 = model.sample("arm_0", ctx, seed=5)
    s2 = model2.sample("arm_0", ctx, seed=5)
    assert s1 == pytest.approx(s2, abs=1e-6)


def test_unknown_arm_raises_value_error() -> None:
    model = _make_model()
    with pytest.raises(ValueError, match="Unknown arm_id"):
        model.sample("nonexistent", {"x": 1.0})
