"""TSEngine: the main public interface for Thompson Sampling decisions."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from ts_module.config.loader import load_from_yaml
from ts_module.config.schema import ModelType, ModuleConfig
from ts_module.core.constraints import ConstraintEngine
from ts_module.core.feedback import FeedbackAggregator
from ts_module.core.models.base import BaseModel
from ts_module.core.models.beta import BetaModel
from ts_module.core.models.linear import LinearModel
from ts_module.core.models.logistic import LogisticModel
from ts_module.core.objective import ObjectiveFunction
from ts_module.core.preprocessing import FeatureProcessor
from ts_module.core.state.base import SessionData
from ts_module.core.state.memory import InMemoryStateStore

logger = logging.getLogger(__name__)


@dataclass
class DecisionResult:
    """Result returned by TSEngine.decide()."""

    session_id: str
    recommended_arm: str
    confidence: float
    arm_scores: dict[str, float]
    model_state_snapshot: dict


@dataclass
class UpdateResult:
    """Result returned by TSEngine.feedback()."""

    session_id: str
    arm_updated: str
    reward_applied: float
    new_distribution: dict


class TSEngine:
    """Thompson Sampling decision engine.

    Usage::

        engine = TSEngine.from_yaml("config.yaml")
        result = engine.decide(context={"category": "billing"})
        engine.feedback(result.session_id, [{"name": "fcr", "value": 1.0}])
    """

    def __init__(self, config: ModuleConfig) -> None:
        """Initialise the engine from a validated ModuleConfig.

        Args:
            config: Fully validated module configuration.

        Raises:
            NotImplementedError: If the configured model_type is not yet implemented.
        """
        self._config = config
        self._model: BaseModel = self._build_model(config)
        self._store = InMemoryStateStore()
        self._objective = ObjectiveFunction(config.objective)
        self._constraint_engine = ConstraintEngine(config.constraints, config.arms)
        self._feedback_aggregator = FeedbackAggregator(config.reward)
        self._all_arm_ids = [arm.id for arm in config.arms]
        self._arm_metadata: dict[str, dict] = {arm.id: arm.metadata for arm in config.arms}

    @staticmethod
    def _build_model(config: ModuleConfig) -> BaseModel:
        """Instantiate the correct model class for the resolved model_type."""
        mtype = config.hyperparams.model_type
        if mtype == ModelType.beta:
            return BetaModel(
                arms=config.arms,
                context_config=config.context,
                hyperparams=config.hyperparams.beta,
            )
        if mtype == ModelType.logistic:
            processor = FeatureProcessor(
                context_config=config.context,
                feature_scaling=config.hyperparams.logistic.feature_scaling,
            )
            return LogisticModel(
                arms=config.arms,
                feature_processor=processor,
                hyperparams=config.hyperparams.logistic,
            )
        if mtype == ModelType.linear:
            processor = FeatureProcessor(
                context_config=config.context,
                feature_scaling=config.hyperparams.linear.feature_scaling,
            )
            return LinearModel(
                arms=config.arms,
                feature_processor=processor,
                hyperparams=config.hyperparams.linear,
            )
        raise NotImplementedError(
            f"Model type '{mtype}' is not implemented. Supported: beta, logistic, linear."
        )

    # ── core interface ────────────────────────────────────────────────────

    def decide(
        self,
        context: dict,
        eligible_arms: list[str] | None = None,
        session_id: str | None = None,
        seed: int | None = None,
        current_loads: dict[str, int] | None = None,
    ) -> DecisionResult:
        """Make a Thompson Sampling decision and record the session.

        Args:
            context: Context features available at decision time.
            eligible_arms: Subset of arm IDs to consider. None means all configured arms.
            session_id: Optional caller-supplied session identifier. Auto-generated if None.
            seed: RNG seed for deterministic sampling (useful in tests).
            current_loads: Current arm loads for capacity constraint evaluation.

        Returns:
            DecisionResult with the recommended arm, scores, and a model snapshot.
        """
        # 1. Candidate arms
        arms = list(eligible_arms) if eligible_arms is not None else list(self._all_arm_ids)

        # 2. Apply eligibility / capacity constraints
        arms = self._constraint_engine.filter_eligible(arms, context, current_loads)
        if not arms:
            arms = list(eligible_arms) if eligible_arms is not None else list(self._all_arm_ids)

        # 3. Derive per-arm seeds from parent seed (ensures determinism without same-sample issue)
        if seed is not None:
            parent_rng = np.random.default_rng(seed)
            raw_seeds = parent_rng.integers(0, 2**31, size=len(arms))
            arm_seeds: dict[str, int | None] = {
                arm_id: int(s) for arm_id, s in zip(sorted(arms), raw_seeds)
            }
        else:
            arm_seeds = {arm_id: None for arm_id in arms}

        # 4. Sample theta for each arm
        arm_thetas: dict[str, float] = {
            arm_id: self._model.sample(arm_id, context, seed=arm_seeds[arm_id])
            for arm_id in arms
        }

        # 5. Compute objective scores
        arm_scores: dict[str, float] = {
            arm_id: self._objective.score(
                arm_id, theta, self._arm_metadata.get(arm_id, {}), context
            )
            for arm_id, theta in arm_thetas.items()
        }

        # 6. Apply exploration floor
        min_exp: float = self._config.hyperparams.min_exploration or 0.0
        arm_scores = self._constraint_engine.apply_exploration_floor(arm_scores, min_exp)

        # 7. Select best arm
        recommended_arm = self._objective.select(arm_scores)

        # 8. Confidence = gap between best and second-best score
        sorted_scores = sorted(arm_scores.values(), reverse=True)
        confidence = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0.0

        # 9. Generate session_id if not supplied
        if session_id is None:
            session_id = str(uuid.uuid4())

        # 10. Persist session for future feedback
        self._store.save_session(
            session_id,
            SessionData(
                session_id=session_id,
                arm_id=recommended_arm,
                context=context,
                timestamp=datetime.now(timezone.utc),
            ),
        )

        # 11. Snapshot model state for auditability
        snapshot = self._model.get_state()

        logger.info(
            "decide session=%s arm='%s' confidence=%.4f ctx=%s",
            session_id,
            recommended_arm,
            confidence,
            context,
        )
        return DecisionResult(
            session_id=session_id,
            recommended_arm=recommended_arm,
            confidence=confidence,
            arm_scores=arm_scores,
            model_state_snapshot=snapshot,
        )

    def feedback(
        self,
        session_id: str,
        signals: list[dict],
    ) -> UpdateResult:
        """Apply feedback signals and update the model posterior.

        Args:
            session_id: Session ID returned by a prior decide() call.
            signals: List of {"name": str, "value": float} dicts.

        Returns:
            UpdateResult with the reward applied and new distribution params.

        Raises:
            ValueError: If session_id is not found.
        """
        session = self._store.get_session(session_id)
        if session is None:
            raise ValueError(f"Session not found: '{session_id}'")

        reward = self._feedback_aggregator.aggregate(signals)
        self._model.update(session.arm_id, reward, session.context)

        session.signals_received.extend(signals)
        session.is_finalized = True

        new_dist = self._model.get_distribution(session.arm_id, session.context)
        logger.info(
            "feedback session=%s arm='%s' reward=%.4f -> estimated_p=%.4f",
            session_id,
            session.arm_id,
            reward,
            new_dist["estimated_p"],
        )
        return UpdateResult(
            session_id=session_id,
            arm_updated=session.arm_id,
            reward_applied=reward,
            new_distribution=new_dist,
        )

    # ── inspection ────────────────────────────────────────────────────────

    def get_arm_state(self, arm_id: str, context: dict | None = None) -> dict:
        """Return current distribution params for a single arm.

        Args:
            arm_id: Arm to inspect.
            context: Context to look up; defaults to empty dict (→ "global" key).

        Returns:
            {"alpha": ..., "beta": ..., "estimated_p": ...}
        """
        return self._model.get_distribution(arm_id, context or {})

    def get_all_states(self) -> dict:
        """Return all arm × context distributions.

        Returns:
            {arm_id: {context_key: {"alpha", "beta", "estimated_p"}}}
        """
        return self._model.get_all_distributions()

    # ── factory ──────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str) -> TSEngine:
        """Create a TSEngine from a YAML configuration file.

        Args:
            path: Path to the YAML file.

        Returns:
            Initialised TSEngine.
        """
        config = load_from_yaml(path)
        return cls(config)
