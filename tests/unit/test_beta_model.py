"""Unit tests for BetaModel (15 tests)."""

from __future__ import annotations

import numpy as np
import pytest

from ts_module.config.schema import ArmConfig, BetaHyperparams, ContextConfig, ContextFeatureConfig, ContextMode
from ts_module.core.models.beta import BetaModel

# ── fixtures ──────────────────────────────────────────────────────────────────


def _make_arms() -> list[ArmConfig]:
    return [
        ArmConfig(id="arm_a", name="Arm A", metadata={}),
        ArmConfig(id="arm_b", name="Arm B", metadata={}),
    ]


def _make_model(mode: ContextMode = ContextMode.none, features=None) -> BetaModel:
    hyp = BetaHyperparams(alpha_init=1.0, beta_init=1.0)
    ctx = ContextConfig(
        mode=mode,
        features=features or [],
    )
    return BetaModel(arms=_make_arms(), context_config=ctx, hyperparams=hyp)


def _categorical_model() -> BetaModel:
    features = [
        ContextFeatureConfig(name="category", type="categorical"),
        ContextFeatureConfig(name="tier", type="categorical"),
    ]
    return _make_model(ContextMode.categorical, features)


# ── tests ─────────────────────────────────────────────────────────────────────


def test_sample_in_unit_interval() -> None:
    """sample() always returns a value in [0.0, 1.0]."""
    model = _make_model()
    for arm in ["arm_a", "arm_b"]:
        for seed in range(20):
            v = model.sample(arm, {}, seed=seed)
            assert 0.0 <= v <= 1.0


def test_success_increments_alpha() -> None:
    """update(reward=1.0) increments alpha by 1, leaves beta unchanged."""
    model = _make_model()
    before = model.get_distribution("arm_a", {})
    model.update("arm_a", 1.0, {})
    after = model.get_distribution("arm_a", {})
    assert after["alpha"] == before["alpha"] + 1.0
    assert after["beta"] == before["beta"]


def test_failure_increments_beta() -> None:
    """update(reward=0.0) increments beta by 1, leaves alpha unchanged."""
    model = _make_model()
    before = model.get_distribution("arm_a", {})
    model.update("arm_a", 0.0, {})
    after = model.get_distribution("arm_a", {})
    assert after["beta"] == before["beta"] + 1.0
    assert after["alpha"] == before["alpha"]


def test_negative_reward_increments_beta_by_absolute_value() -> None:
    """update(reward=-0.5) increments beta by 0.5."""
    model = _make_model()
    before = model.get_distribution("arm_a", {})
    model.update("arm_a", -0.5, {})
    after = model.get_distribution("arm_a", {})
    assert abs(after["beta"] - (before["beta"] + 0.5)) < 1e-9
    assert after["alpha"] == before["alpha"]


def test_partial_reward_increments_alpha_by_value() -> None:
    """update(reward=0.3) increments alpha by 0.3."""
    model = _make_model()
    before = model.get_distribution("arm_a", {})
    model.update("arm_a", 0.3, {})
    after = model.get_distribution("arm_a", {})
    assert abs(after["alpha"] - (before["alpha"] + 0.3)) < 1e-9
    assert after["beta"] == before["beta"]


def test_categorical_context_creates_independent_distributions() -> None:
    """Updating billing context does not affect complaint context distribution."""
    model = _categorical_model()
    ctx_billing = {"category": "billing", "tier": "smb"}
    ctx_complaint = {"category": "complaint", "tier": "smb"}

    model.update("arm_a", 1.0, ctx_billing)
    dist_complaint = model.get_distribution("arm_a", ctx_complaint)
    # complaint context should still be at initial values
    assert dist_complaint["alpha"] == 1.0
    assert dist_complaint["beta"] == 1.0


def test_global_context_key_when_mode_none() -> None:
    """With context_mode=none, all updates use the 'global' key."""
    model = _make_model(ContextMode.none)
    ctx1 = {"category": "billing"}
    ctx2 = {"category": "tech"}
    model.update("arm_a", 1.0, ctx1)
    # Different context dict should still hit the same global key
    dist = model.get_distribution("arm_a", ctx2)
    assert dist["alpha"] == 2.0  # 1 init + 1 update
    assert dist["beta"] == 1.0


def test_context_key_is_deterministic_regardless_of_dict_order() -> None:
    """Context keys must be identical regardless of dict insertion order."""
    model = _categorical_model()
    ctx_order1 = {"category": "billing", "tier": "enterprise"}
    ctx_order2 = {"tier": "enterprise", "category": "billing"}

    model.update("arm_a", 1.0, ctx_order1)
    dist = model.get_distribution("arm_a", ctx_order2)
    # Should see the update from ctx_order1
    assert dist["alpha"] == 2.0


def test_convergence_to_true_probability() -> None:
    """After 500 Bernoulli updates with p=0.75, estimated_p should be within 0.05 of 0.75."""
    model = _make_model()
    rng = np.random.default_rng(0)
    true_p = 0.75
    for _ in range(500):
        reward = 1.0 if rng.random() < true_p else 0.0
        model.update("arm_a", reward, {})
    dist = model.get_distribution("arm_a", {})
    assert abs(dist["estimated_p"] - true_p) < 0.05


def test_deterministic_sampling_with_seed() -> None:
    """sample(seed=42) always returns the same value for the same model state."""
    model = _make_model()
    v1 = model.sample("arm_a", {}, seed=42)
    v2 = model.sample("arm_a", {}, seed=42)
    assert v1 == v2


def test_different_seeds_give_different_samples() -> None:
    """sample(seed=42) and sample(seed=99) should give different values (almost always)."""
    model = _make_model()
    v1 = model.sample("arm_a", {}, seed=42)
    v2 = model.sample("arm_a", {}, seed=99)
    assert v1 != v2


def test_initial_distribution_from_hyperparams() -> None:
    """Custom alpha_init and beta_init are used when the key is first accessed."""
    hyp = BetaHyperparams(alpha_init=2.0, beta_init=5.0)
    ctx = ContextConfig(mode=ContextMode.none, features=[])
    model = BetaModel(arms=_make_arms(), context_config=ctx, hyperparams=hyp)
    dist = model.get_distribution("arm_a", {})
    assert dist["alpha"] == 2.0
    assert dist["beta"] == 5.0


def test_get_all_distributions_returns_all_arms() -> None:
    """get_all_distributions includes all known arms after updates."""
    model = _make_model()
    model.update("arm_a", 1.0, {})
    model.update("arm_b", 0.0, {})
    all_dists = model.get_all_distributions()
    assert "arm_a" in all_dists
    assert "arm_b" in all_dists


def test_state_serialization_roundtrip() -> None:
    """get_state() → load_state() restores identical alpha/beta values."""
    model = _make_model()
    model.update("arm_a", 1.0, {})
    model.update("arm_a", 0.0, {})
    model.update("arm_b", 1.0, {})

    state = model.get_state()
    new_model = _make_model()
    new_model.load_state(state)

    for arm in ["arm_a", "arm_b"]:
        original = model.get_distribution(arm, {})
        restored = new_model.get_distribution(arm, {})
        assert original["alpha"] == restored["alpha"]
        assert original["beta"] == restored["beta"]


def test_unknown_arm_raises_value_error() -> None:
    """sample() on an unknown arm_id raises ValueError."""
    model = _make_model()
    with pytest.raises(ValueError, match="Unknown arm_id"):
        model.sample("nonexistent_arm", {})
