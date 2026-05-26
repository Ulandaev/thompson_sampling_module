"""Unit tests for ObjectiveFunction (8 tests)."""

from __future__ import annotations

import pytest

from ts_module.config.schema import ObjectiveConfig, ObjectiveType
from ts_module.core.objective import ObjectiveFunction


def _obj(obj_type: ObjectiveType, formula: str | None = None) -> ObjectiveFunction:
    return ObjectiveFunction(ObjectiveConfig(type=obj_type, formula=formula))


def test_max_probability_returns_theta() -> None:
    """max_probability score equals theta regardless of metadata."""
    obj = _obj(ObjectiveType.max_probability)
    assert obj.score("a", 0.7, {"price": 100}, {}) == 0.7
    assert obj.score("a", 0.0, {}, {}) == 0.0


def test_max_revenue_multiplies_by_price() -> None:
    """max_expected_revenue score = theta * price (default price=1.0)."""
    obj = _obj(ObjectiveType.max_expected_revenue)
    assert obj.score("a", 0.5, {"price": 200.0}, {}) == pytest.approx(100.0)
    assert obj.score("a", 0.5, {}, {}) == pytest.approx(0.5)  # default price=1


def test_max_roi_formula() -> None:
    """max_roi = (theta * value - cost) / max(cost, 0.001)."""
    obj = _obj(ObjectiveType.max_roi)
    # theta=0.8, value=100, cost=20 → (80-20)/20 = 3.0
    result = obj.score("a", 0.8, {"value": 100.0, "cost": 20.0}, {})
    assert result == pytest.approx(3.0)


def test_min_cost_per_success() -> None:
    """min_cost_per_success = -cost / max(theta, 0.001)."""
    obj = _obj(ObjectiveType.min_cost_per_success)
    result = obj.score("a", 0.5, {"cost": 10.0}, {})
    assert result == pytest.approx(-20.0)
    # Higher theta should give a higher (less negative) score
    result2 = obj.score("a", 0.8, {"cost": 10.0}, {})
    assert result2 > result


def test_select_returns_highest_score_arm() -> None:
    """select() returns the arm with the maximum score."""
    obj = _obj(ObjectiveType.max_probability)
    scores = {"arm_a": 0.3, "arm_b": 0.9, "arm_c": 0.5}
    assert obj.select(scores) == "arm_b"


def test_select_handles_single_arm() -> None:
    """select() returns the only arm when there is exactly one."""
    obj = _obj(ObjectiveType.max_probability)
    assert obj.select({"arm_only": 0.42}) == "arm_only"


def test_custom_formula_with_theta_and_arm_metadata() -> None:
    """Custom formula can access theta and arm namespace."""
    obj = _obj(ObjectiveType.custom, formula="theta * arm['price']")
    result = obj.score("a", 0.5, {"price": 10.0}, {})
    assert result == pytest.approx(5.0)


def test_invalid_custom_formula_raises_value_error() -> None:
    """A malformed custom formula raises ValueError with a descriptive message."""
    obj = _obj(ObjectiveType.custom, formula="theta ** 'bad'")
    with pytest.raises(ValueError, match="Error evaluating custom formula"):
        obj.score("a", 0.5, {}, {})
