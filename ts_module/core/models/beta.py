"""Beta Thompson Sampling model — one Beta(alpha, beta) per (arm, context_key)."""

from __future__ import annotations

import logging

import numpy as np

from ts_module.config.schema import ArmConfig, BetaHyperparams, ContextConfig, ContextMode
from ts_module.core.models.base import BaseModel

logger = logging.getLogger(__name__)


class BetaModel(BaseModel):
    """Beta TS model.

    Maintains an independent Beta(alpha, beta) distribution for each
    (arm_id, context_key) pair.  The context_key is derived deterministically
    from the active context features so that the same context always maps to
    the same distribution.
    """

    def __init__(
        self,
        arms: list[ArmConfig],
        context_config: ContextConfig,
        hyperparams: BetaHyperparams,
    ) -> None:
        """Initialize BetaModel.

        Args:
            arms: List of arm configurations. Unknown arm IDs raise ValueError.
            context_config: Determines how context is mapped to a key.
            hyperparams: Initial alpha/beta values and other Beta-specific settings.
        """
        self._arms: dict[str, ArmConfig] = {arm.id: arm for arm in arms}
        self._context_config = context_config
        self._hyperparams = hyperparams
        # {(arm_id, context_key): (alpha, beta)}
        self._store: dict[tuple[str, str], tuple[float, float]] = {}

    # ── context key ────────────────────────────────────────────────────────

    def _make_context_key(self, context: dict) -> str:
        """Build a deterministic string key for the given context dict.

        - mode=none  → "global"
        - mode=categorical → "{name}_{value}__{name}_{value}..." (sorted by name)
        """
        mode = self._context_config.mode
        if mode == ContextMode.none:
            return "global"
        if mode == ContextMode.categorical:
            feature_names = {f.name for f in self._context_config.features}
            relevant = {k: v for k, v in context.items() if k in feature_names}
            parts = [f"{k}_{v}" for k, v in sorted(relevant.items())]
            return "__".join(parts) if parts else "global"
        # features mode: treated like global in Phase 1 (Logistic/Linear handle it in Phase 2)
        return "global"

    # ── internal helpers ───────────────────────────────────────────────────

    def _get_or_init(self, arm_id: str, context_key: str) -> tuple[float, float]:
        """Return stored (alpha, beta) or initialize from hyperparams."""
        key = (arm_id, context_key)
        if key not in self._store:
            self._store[key] = (
                self._hyperparams.alpha_init,
                self._hyperparams.beta_init,
            )
        return self._store[key]

    # ── BaseModel interface ────────────────────────────────────────────────

    def sample(self, arm_id: str, context: dict, seed: int | None = None) -> float:
        """Sample from Beta posterior for this arm and context.

        Args:
            arm_id: Must be a known arm ID.
            context: Active context dict.
            seed: Optional seed for np.random.default_rng.

        Returns:
            Float in [0.0, 1.0] sampled from Beta(alpha, beta).
        """
        if arm_id not in self._arms:
            raise ValueError(f"Unknown arm_id: '{arm_id}'")
        context_key = self._make_context_key(context)
        alpha, beta = self._get_or_init(arm_id, context_key)
        rng = np.random.default_rng(seed)
        value = float(rng.beta(alpha, beta))
        logger.debug(
            "sample arm=%s ctx=%s alpha=%.3f beta=%.3f -> %.4f",
            arm_id,
            context_key,
            alpha,
            beta,
            value,
        )
        return value

    def update(self, arm_id: str, reward: float, context: dict) -> None:
        """Update Beta posterior with observed reward.

        - reward > 0  → alpha += reward   (success)
        - reward < 0  → beta  += |reward| (weighted failure)
        - reward == 0 → beta  += 1.0      (failure)
        """
        if arm_id not in self._arms:
            raise ValueError(f"Unknown arm_id: '{arm_id}'")
        context_key = self._make_context_key(context)
        alpha, beta = self._get_or_init(arm_id, context_key)
        if reward > 0:
            alpha += reward
        elif reward < 0:
            beta += abs(reward)
        else:
            beta += 1.0
        self._store[(arm_id, context_key)] = (alpha, beta)
        logger.debug(
            "update arm=%s ctx=%s reward=%.3f -> alpha=%.3f beta=%.3f",
            arm_id,
            context_key,
            reward,
            alpha,
            beta,
        )

    def get_distribution(self, arm_id: str, context: dict) -> dict:
        """Return distribution params for this arm and context."""
        if arm_id not in self._arms:
            raise ValueError(f"Unknown arm_id: '{arm_id}'")
        context_key = self._make_context_key(context)
        alpha, beta = self._get_or_init(arm_id, context_key)
        return {"alpha": alpha, "beta": beta, "estimated_p": alpha / (alpha + beta)}

    def get_all_distributions(self) -> dict:
        """Return nested dict of all arm × context distributions."""
        result: dict[str, dict[str, dict]] = {arm_id: {} for arm_id in self._arms}
        for (arm_id, context_key), (alpha, beta) in self._store.items():
            result.setdefault(arm_id, {})[context_key] = {
                "alpha": alpha,
                "beta": beta,
                "estimated_p": alpha / (alpha + beta),
            }
        return result

    def get_state(self) -> dict:
        """Serialize store to a JSON-compatible dict."""
        return {
            f"{arm_id}|||{ctx_key}": {"alpha": alpha, "beta": beta}
            for (arm_id, ctx_key), (alpha, beta) in self._store.items()
        }

    def load_state(self, state: dict) -> None:
        """Restore store from a serialized dict."""
        self._store = {}
        for composite_key, vals in state.items():
            arm_id, ctx_key = composite_key.split("|||", 1)
            self._store[(arm_id, ctx_key)] = (float(vals["alpha"]), float(vals["beta"]))
