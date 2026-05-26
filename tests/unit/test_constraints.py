"""Unit tests for ConstraintEngine (6 tests)."""

from __future__ import annotations

from ts_module.config.schema import ArmConfig, ConstraintConfig
from ts_module.core.constraints import ConstraintEngine


def _make_arms() -> list[ArmConfig]:
    return [
        ArmConfig(id="arm_a", name="A", metadata={"capacity": 3, "active": True}),
        ArmConfig(id="arm_b", name="B", metadata={"capacity": 3, "active": True}),
        ArmConfig(id="arm_c", name="C", metadata={"capacity": 3, "active": False}),
    ]


def test_capacity_excludes_overloaded_arm() -> None:
    """Arm at or above capacity is excluded from the eligible list."""
    constraint = ConstraintConfig(type="capacity", arm_field="metadata.capacity")
    engine = ConstraintEngine([constraint], _make_arms())
    # arm_a is at capacity, arm_b is under
    result = engine.filter_eligible(
        ["arm_a", "arm_b"],
        context={},
        current_loads={"arm_a": 3, "arm_b": 1},
    )
    assert "arm_a" not in result
    assert "arm_b" in result


def test_capacity_keeps_arm_under_limit() -> None:
    """Arms with load < capacity are kept in the eligible list."""
    constraint = ConstraintConfig(type="capacity", arm_field="metadata.capacity")
    engine = ConstraintEngine([constraint], _make_arms())
    result = engine.filter_eligible(
        ["arm_a", "arm_b"],
        context={},
        current_loads={"arm_a": 2, "arm_b": 2},
    )
    assert set(result) == {"arm_a", "arm_b"}


def test_min_traffic_floor_applied_to_all_arms() -> None:
    """After apply_exploration_floor, every arm gets at least min_exploration share."""
    engine = ConstraintEngine([], [])
    scores = {"arm_a": 0.9, "arm_b": 0.001, "arm_c": 0.5}
    result = engine.apply_exploration_floor(scores, min_exploration=0.1)
    for arm_id, share in result.items():
        assert share >= 0.1 - 1e-9, f"{arm_id} share {share:.4f} below floor"


def test_all_arms_excluded_returns_all_arms() -> None:
    """If every arm is filtered out, the original list is returned unchanged."""
    # Make a capacity constraint where all arms are overloaded
    constraint = ConstraintConfig(type="capacity", arm_field="metadata.capacity")
    engine = ConstraintEngine([constraint], _make_arms())
    result = engine.filter_eligible(
        ["arm_a", "arm_b"],
        context={},
        current_loads={"arm_a": 10, "arm_b": 10},
    )
    assert set(result) == {"arm_a", "arm_b"}


def test_exploration_floor_normalizes_scores() -> None:
    """apply_exploration_floor produces scores that sum to ~1.0."""
    engine = ConstraintEngine([], [])
    scores = {"arm_a": 0.9, "arm_b": 0.05, "arm_c": 0.01}
    result = engine.apply_exploration_floor(scores, min_exploration=0.03)
    assert abs(sum(result.values()) - 1.0) < 1e-9


def test_no_constraints_returns_all_arms() -> None:
    """With no constraints, filter_eligible returns all arms unchanged."""
    engine = ConstraintEngine([], _make_arms())
    arms = ["arm_a", "arm_b", "arm_c"]
    result = engine.filter_eligible(arms, context={})
    assert set(result) == set(arms)
