"""Objective function: score arms and select the best one."""

from __future__ import annotations

import logging
import random

from ts_module.config.schema import ObjectiveConfig, ObjectiveType

logger = logging.getLogger(__name__)


class ObjectiveFunction:
    """Computes arm scores and selects the winner based on the configured objective."""

    def __init__(self, config: ObjectiveConfig) -> None:
        """Initialize with an ObjectiveConfig.

        Args:
            config: Objective type and optional custom formula.
        """
        self._config = config

    def score(
        self,
        arm_id: str,
        theta: float,
        arm_metadata: dict,
        context: dict,
    ) -> float:
        """Compute a scalar score for an arm given its sampled theta.

        Args:
            arm_id: Identifier of the arm (used for logging).
            theta: Sampled probability from the arm's posterior.
            arm_metadata: Arm-level metadata dict from ArmConfig.
            context: Context dict from the current decision request.

        Returns:
            Scalar score. Higher = better (except min_cost_per_success which negates cost).

        Raises:
            ValueError: If a custom formula raises or is syntactically invalid.
        """
        obj = self._config.type
        if obj == ObjectiveType.max_probability:
            return theta

        if obj == ObjectiveType.max_expected_revenue:
            price = arm_metadata.get("price", 1.0)
            return theta * float(price)

        if obj == ObjectiveType.max_roi:
            value = float(arm_metadata.get("value", 1.0))
            cost = float(arm_metadata.get("cost", 0.0))
            return (theta * value - cost) / max(cost, 0.001)

        if obj == ObjectiveType.min_cost_per_success:
            cost = float(arm_metadata.get("cost", 1.0))
            return -cost / max(theta, 0.001)

        if obj == ObjectiveType.custom:
            formula: str = self._config.formula  # type: ignore[assignment]  # validated non-None
            assert formula is not None, "custom objective requires a formula (caught at config load)"
            namespace: dict = {
                "theta": theta,
                "arm": arm_metadata,
                "context": context,
                "__builtins__": {},
            }
            try:
                result = eval(formula, namespace)  # noqa: S307
                return float(result)
            except Exception as exc:
                raise ValueError(
                    f"Error evaluating custom formula '{formula}' for arm '{arm_id}': {exc}"
                ) from exc

        raise ValueError(f"Unknown objective type: {obj}")

    def select(self, scored_arms: dict[str, float]) -> str:
        """Return the arm_id with the highest score.

        Breaks ties by uniform random selection.

        Args:
            scored_arms: Mapping of arm_id → score.

        Returns:
            arm_id of the selected arm.

        Raises:
            ValueError: If scored_arms is empty.
        """
        if not scored_arms:
            raise ValueError("scored_arms cannot be empty")
        max_score = max(scored_arms.values())
        candidates = [arm_id for arm_id, s in scored_arms.items() if s == max_score]
        chosen = random.choice(candidates)
        logger.debug("select from %d arms -> '%s' (score=%.4f)", len(scored_arms), chosen, max_score)
        return chosen
