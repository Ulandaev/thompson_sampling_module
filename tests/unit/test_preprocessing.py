"""Unit tests for FeatureProcessor (8 tests)."""

from __future__ import annotations

import numpy as np
import pytest

from ts_module.config.schema import ContextConfig, ContextFeatureConfig, ContextMode
from ts_module.core.preprocessing import FeatureProcessor


def _make_processor(features: list[dict], scaling: bool = True) -> FeatureProcessor:
    feat_configs = [ContextFeatureConfig(**f) for f in features]
    ctx = ContextConfig(mode=ContextMode.features, features=feat_configs)
    return FeatureProcessor(ctx, feature_scaling=scaling)


def test_categorical_one_hot_encoding() -> None:
    proc = _make_processor([{"name": "cat", "type": "categorical", "categories": ["billing", "tech", "complaint", "onboard"]}])
    vec = proc.transform({"cat": "billing"})
    assert vec.tolist() == [1.0, 0.0, 0.0, 0.0]

    vec2 = proc.transform({"cat": "tech"})
    assert vec2.tolist() == [0.0, 1.0, 0.0, 0.0]


def test_continuous_feature_passthrough_without_scaling() -> None:
    proc = _make_processor([{"name": "x", "type": "continuous"}], scaling=False)
    vec = proc.transform({"x": 14.5})
    assert vec[0] == pytest.approx(14.5)


def test_continuous_feature_standardized_with_scaling() -> None:
    proc = _make_processor([{"name": "x", "type": "continuous"}], scaling=True)
    proc.fit([{"x": 10.0}, {"x": 20.0}, {"x": 30.0}])
    # mean=20, std=~8.165
    vec = proc.transform({"x": 20.0})
    assert abs(vec[0]) < 1e-6  # mean → 0


def test_unknown_category_gives_all_zeros() -> None:
    proc = _make_processor([{"name": "cat", "type": "categorical", "categories": ["a", "b"]}])
    vec = proc.transform({"cat": "unknown_value"})
    assert vec.tolist() == [0.0, 0.0]


def test_missing_feature_gives_zeros() -> None:
    proc = _make_processor([
        {"name": "cat", "type": "categorical", "categories": ["a", "b"]},
        {"name": "x", "type": "continuous"},
    ], scaling=False)
    vec = proc.transform({})
    assert np.all(vec == 0.0)


def test_feature_dim_matches_config() -> None:
    proc = _make_processor([
        {"name": "ticket_category", "type": "categorical", "categories": ["billing", "tech", "complaint", "onboard"]},
        {"name": "time_of_day", "type": "continuous"},
    ])
    assert proc.feature_dim == 5  # 4 one-hot + 1 continuous


def test_transform_deterministic() -> None:
    proc = _make_processor([
        {"name": "cat", "type": "categorical", "categories": ["a", "b"]},
        {"name": "x", "type": "continuous"},
    ], scaling=False)
    ctx = {"cat": "a", "x": 3.14}
    assert proc.transform(ctx).tolist() == proc.transform(ctx).tolist()


def test_state_roundtrip() -> None:
    proc = _make_processor([{"name": "x", "type": "continuous"}], scaling=True)
    proc.fit([{"x": 5.0}, {"x": 10.0}, {"x": 15.0}])
    val_before = proc.transform({"x": 10.0})[0]

    state = proc.get_state()

    proc2 = _make_processor([{"name": "x", "type": "continuous"}], scaling=True)
    proc2.load_state(state)
    val_after = proc2.transform({"x": 10.0})[0]

    assert val_before == pytest.approx(val_after)
