"""Constraint engine: eligibility filtering and exploration-floor scoring."""

from __future__ import annotations

import logging

from ts_module.config.schema import ArmConfig, ConstraintConfig

logger = logging.getLogger(__name__)


class ConstraintEngine:
    """Applies business constraints to filter arms and guarantee minimum exploration."""

    def __init__(
        self,
        constraints: list[ConstraintConfig],
        arms: list[ArmConfig],
    ) -> None:
        """Initialize with a list of constraints and known arms.

        Args:
            constraints: Constraint configurations to enforce.
            arms: All arm configurations (used for metadata lookups).
        """
        self._constraints = constraints
        self._arms: dict[str, ArmConfig] = {arm.id: arm for arm in arms}

    # ── helpers ────────────────────────────────────────────────────────────

    def _get_nested(self, arm: ArmConfig, field_path: str):
        """Access a dotted field path such as 'metadata.capacity'."""
        parts = field_path.split(".", 1)
        obj = getattr(arm, parts[0], None)
        if len(parts) == 1:
            return obj
        if isinstance(obj, dict):
            return obj.get(parts[1])
        return getattr(obj, parts[1], None)

    # ── public API ─────────────────────────────────────────────────────────

    def filter_eligible(
        self,
        arms: list[str],
        context: dict,
        current_loads: dict[str, int] | None = None,
    ) -> list[str]:
        """Filter the arm list by capacity and eligibility constraints.

        Always keeps at least one arm — if all arms would be excluded, returns
        the original list unchanged.

        Args:
            arms: Candidate arm IDs to filter.
            context: Current context dict (used for eligibility evaluation).
            current_loads: Current load per arm_id (for capacity checks).

        Returns:
            Filtered list of eligible arm IDs.
        """
        eligible = list(arms)
        loads = current_loads or {}

        for constraint in self._constraints:
            if constraint.type == "capacity" and constraint.arm_field:
                filtered = []
                for arm_id in eligible:
                    arm = self._arms.get(arm_id)
                    if arm is None:
                        continue
                    capacity = self._get_nested(arm, constraint.arm_field)
                    if capacity is None:
                        filtered.append(arm_id)
                        continue
                    current = loads.get(arm_id, 0)
                    if current < int(capacity):
                        filtered.append(arm_id)
                    else:
                        logger.debug(
                            "arm '%s' excluded by capacity: load=%d cap=%d",
                            arm_id,
                            current,
                            capacity,
                        )
                if filtered:
                    eligible = filtered

            elif constraint.type == "eligibility" and constraint.condition:
                filtered = []
                for arm_id in eligible:
                    arm = self._arms.get(arm_id)
                    if arm is None:
                        continue
                    ns: dict = {"arm": arm, "context": context, "__builtins__": {}}
                    try:
                        if eval(constraint.condition, ns):  # noqa: S307
                            filtered.append(arm_id)
                        else:
                            logger.debug(
                                "arm '%s' excluded by eligibility: %s",
                                arm_id,
                                constraint.condition,
                            )
                    except Exception as exc:
                        logger.warning(
                            "eligibility eval error for arm '%s': %s", arm_id, exc
                        )
                        filtered.append(arm_id)
                if filtered:
                    eligible = filtered

        if not eligible:
            logger.warning("All arms excluded by constraints; returning full candidate list.")
            return list(arms)
        return eligible

    def apply_exploration_floor(
        self,
        arm_scores: dict[str, float],
        min_exploration: float,
    ) -> dict[str, float]:
        """Ensure every arm receives at least min_exploration share of traffic.

        Raises low-scoring arms to the floor value, then normalises all scores
        so they sum to 1.0.

        Args:
            arm_scores: Raw arm_id → score mapping from the objective function.
            min_exploration: Minimum fraction of traffic guaranteed to each arm.

        Returns:
            Adjusted and normalised scores summing to 1.0.
        """
        if not arm_scores or min_exploration <= 0:
            return arm_scores

        n = len(arm_scores)
        # If min_exploration per arm would exceed 100%, distribute uniformly.
        if min_exploration * n >= 1.0:
            return {arm_id: 1.0 / n for arm_id in arm_scores}

        # Normalise raw scores to proportions first (floor at 0)
        values_pos = {arm_id: max(s, 0.0) for arm_id, s in arm_scores.items()}
        total = sum(values_pos.values())
        if total <= 0:
            return {arm_id: 1.0 / n for arm_id in arm_scores}
        normalized = {arm_id: v / total for arm_id, v in values_pos.items()}

        # Blend: (1 - n*min_exp) * normalized + min_exp
        # Guarantees every arm gets at least min_exploration fraction.
        blend = 1.0 - n * min_exploration
        return {arm_id: blend * p + min_exploration for arm_id, p in normalized.items()}
