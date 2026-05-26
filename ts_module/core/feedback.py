"""Feedback aggregator: collapse multiple reward signals into a scalar reward."""

from __future__ import annotations

import logging

from ts_module.config.schema import RewardConfig

logger = logging.getLogger(__name__)


class FeedbackAggregator:
    """Aggregates a list of named reward signals into a single float reward.

    Unknown signal names (not declared in reward_config) are silently ignored.
    """

    def __init__(self, reward_config: RewardConfig) -> None:
        """Initialize with reward configuration.

        Args:
            reward_config: Defines signal weights and aggregation strategy.
        """
        self._config = reward_config
        self._weights: dict[str, float] = {s.name: s.weight for s in reward_config.signals}

    def aggregate(self, signals: list[dict]) -> float:
        """Aggregate reward signals into a scalar value.

        Args:
            signals: List of dicts, each with "name" and "value" keys.
                     Unknown signal names are ignored.

        Returns:
            Aggregated scalar reward. Returns 0.0 if no known signals.
        """
        known = [s for s in signals if s.get("name") in self._weights]
        if not known:
            logger.debug("No known signals in feedback; reward=0.0")
            return 0.0

        mode = self._config.aggregation

        if mode == "weighted_sum":
            total = sum(s["value"] * self._weights[s["name"]] for s in known)
            logger.debug("weighted_sum reward=%.4f from %d signals", total, len(known))
            return total

        if mode == "first_positive":
            for s in known:
                if s["value"] > 0:
                    logger.debug("first_positive reward=%.4f (signal=%s)", s["value"], s["name"])
                    return float(s["value"])
            return 0.0

        if mode == "max":
            value = max(s["value"] for s in known)
            logger.debug("max reward=%.4f", value)
            return float(value)

        logger.warning("Unknown aggregation mode '%s'; falling back to weighted_sum.", mode)
        return sum(s["value"] * self._weights[s["name"]] for s in known)
