"""Unit tests for FeedbackAggregator (6 tests)."""

from __future__ import annotations

import pytest

from ts_module.config.schema import RewardConfig, RewardType, SignalConfig
from ts_module.core.feedback import FeedbackAggregator


def _make_agg(aggregation: str = "weighted_sum") -> FeedbackAggregator:
    config = RewardConfig(
        type=RewardType.binary,
        signals=[
            SignalConfig(name="fcr", weight=1.0),
            SignalConfig(name="reopen", weight=-0.5),
            SignalConfig(name="csat", weight=0.3),
        ],
        aggregation=aggregation,
    )
    return FeedbackAggregator(config)


def test_single_signal_aggregation() -> None:
    """A single known signal returns weight * value."""
    agg = _make_agg()
    result = agg.aggregate([{"name": "fcr", "value": 1.0}])
    assert result == pytest.approx(1.0)


def test_weighted_sum_multiple_signals() -> None:
    """weighted_sum correctly combines multiple signals with their weights."""
    agg = _make_agg()
    # fcr=1.0 * 1.0 + reopen=1.0 * -0.5 = 0.5
    result = agg.aggregate([
        {"name": "fcr", "value": 1.0},
        {"name": "reopen", "value": 1.0},
    ])
    assert result == pytest.approx(0.5)


def test_negative_weight_signal() -> None:
    """Negative-weight signal reduces the aggregated reward."""
    agg = _make_agg()
    result = agg.aggregate([{"name": "reopen", "value": 1.0}])
    assert result == pytest.approx(-0.5)


def test_unknown_signal_name_ignored() -> None:
    """Signals not declared in reward_config are silently ignored."""
    agg = _make_agg()
    result = agg.aggregate([
        {"name": "unknown_signal", "value": 99.0},
        {"name": "fcr", "value": 1.0},
    ])
    assert result == pytest.approx(1.0)


def test_empty_signals_returns_zero() -> None:
    """An empty signal list returns 0.0."""
    agg = _make_agg()
    assert agg.aggregate([]) == 0.0


def test_first_positive_aggregation_mode() -> None:
    """first_positive mode returns value of first signal with value > 0."""
    agg = _make_agg(aggregation="first_positive")
    result = agg.aggregate([
        {"name": "reopen", "value": 1.0},  # positive but reopen
        {"name": "fcr", "value": 0.0},
    ])
    # first positive is reopen with value 1.0
    assert result == pytest.approx(1.0)
    # If no positive, returns 0
    result2 = agg.aggregate([{"name": "fcr", "value": 0.0}])
    assert result2 == 0.0
