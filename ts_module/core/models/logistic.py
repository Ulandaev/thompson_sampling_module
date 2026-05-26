"""Logistic Thompson Sampling model — Bayesian linear regression with sigmoid link."""

from __future__ import annotations

import logging

import numpy as np

from ts_module.config.schema import ArmConfig, LogisticHyperparams
from ts_module.core.models.base import BaseModel
from ts_module.core.preprocessing import FeatureProcessor

logger = logging.getLogger(__name__)


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))  # type: ignore[return-value]


class _BayesianLinearBase(BaseModel):
    """Shared Bayesian linear regression logic for Logistic and Linear TS.

    Maintains per-arm posterior N(w_map, A_inv) updated via Sherman-Morrison.
    Subclasses override _activate() to apply (or skip) the sigmoid link.
    """

    def __init__(
        self,
        arms: list[ArmConfig],
        feature_processor: FeatureProcessor,
        prior_variance: float,
        regularization: float,
    ) -> None:
        self._arms: dict[str, ArmConfig] = {arm.id: arm for arm in arms}
        self._processor = feature_processor
        self._prior_variance = prior_variance
        self._regularization = regularization

        d = feature_processor.feature_dim
        self._A_inv: dict[str, np.ndarray] = {}  # arm_id → d×d covariance
        self._b: dict[str, np.ndarray] = {}  # arm_id → d-vector

        for arm in arms:
            self._A_inv[arm.id] = prior_variance * np.eye(d)
            self._b[arm.id] = np.zeros(d)

    # ── subclass hooks ────────────────────────────────────────────────────

    def _activate(self, logit: float | np.ndarray) -> float:  # noqa: ARG002
        """Apply activation function to raw linear prediction."""
        raise NotImplementedError

    def _scale_reward(self, reward: float) -> float:
        """Clip or otherwise transform reward before accumulating into b."""
        return reward

    def _precision_scale(self) -> float:
        """Multiplicative factor applied to both the A precision update and b.

        For Logistic: 1.0 (reward is already in [0,1], no noise model).
        For Linear:   1.0 / noise_variance (correct Bayesian linear regression).
        """
        return 1.0

    # ── BaseModel interface ────────────────────────────────────────────────

    def sample(self, arm_id: str, context: dict, seed: int | None = None) -> float:
        if arm_id not in self._arms:
            raise ValueError(f"Unknown arm_id: '{arm_id}'")

        x = self._processor.transform(context)
        A_inv = self._A_inv[arm_id]
        b = self._b[arm_id]
        w_map = A_inv @ b

        rng = np.random.default_rng(seed)
        d = len(w_map)
        if d > 50:
            w = rng.multivariate_normal(w_map, A_inv, method="cholesky")
        else:
            w = rng.multivariate_normal(w_map, A_inv)

        theta = self._activate(float(w @ x))
        logger.debug("sample arm=%s theta=%.4f", arm_id, theta)
        return theta

    def update(self, arm_id: str, reward: float, context: dict) -> None:
        if arm_id not in self._arms:
            raise ValueError(f"Unknown arm_id: '{arm_id}'")

        x = self._processor.transform(context)
        r = self._scale_reward(reward)
        scale = self._precision_scale()

        A_inv = self._A_inv[arm_id]
        # Sherman-Morrison: (A + scale·x xᵀ)⁻¹
        Ax = A_inv @ x
        denom = 1.0 + scale * float(x @ Ax) + 1e-8
        self._A_inv[arm_id] = A_inv - scale * np.outer(Ax, Ax) / denom
        self._b[arm_id] = self._b[arm_id] + scale * r * x

        logger.debug("update arm=%s reward=%.4f", arm_id, reward)

    def get_distribution(self, arm_id: str, context: dict) -> dict:
        if arm_id not in self._arms:
            raise ValueError(f"Unknown arm_id: '{arm_id}'")

        x = self._processor.transform(context)
        A_inv = self._A_inv[arm_id]
        b = self._b[arm_id]
        w_map = A_inv @ b
        estimated_p = self._activate(float(w_map @ x))
        return {
            "estimated_p": estimated_p,
            "w_map": w_map.tolist(),
            "sigma_diag": np.diag(A_inv).tolist(),
        }

    def get_all_distributions(self) -> dict:
        return {arm_id: {"global": self.get_distribution(arm_id, {})} for arm_id in self._arms}

    def get_state(self) -> dict:
        return {
            "arms": {
                arm_id: {
                    "A_inv": self._A_inv[arm_id].tolist(),
                    "b": self._b[arm_id].tolist(),
                }
                for arm_id in self._arms
            },
            "processor": self._processor.get_state(),
        }

    def load_state(self, state: dict) -> None:
        for arm_id, vals in state.get("arms", {}).items():
            if arm_id in self._A_inv:
                self._A_inv[arm_id] = np.array(vals["A_inv"], dtype=np.float64)
                self._b[arm_id] = np.array(vals["b"], dtype=np.float64)
        if "processor" in state:
            self._processor.load_state(state["processor"])


class LogisticModel(_BayesianLinearBase):
    """Logistic Thompson Sampling — binary reward with continuous features.

    Uses an online IRLS (Iteratively Reweighted Least Squares) approximation:
    - Covariance update weighted by Fisher information p*(1-p).
    - Mean update via Newton step: m_new = m + A_inv_new @ (r - p) * x.

    This ensures that both successes and failures are properly reflected in
    the posterior: successes push estimated_p up, failures push it down.
    """

    def __init__(
        self,
        arms: list[ArmConfig],
        feature_processor: FeatureProcessor,
        hyperparams: LogisticHyperparams,
    ) -> None:
        super().__init__(
            arms=arms,
            feature_processor=feature_processor,
            prior_variance=hyperparams.prior_variance,
            regularization=hyperparams.regularization,
        )
        d = feature_processor.feature_dim
        # Store MAP mean explicitly (IRLS update; _b is unused in LogisticModel)
        self._m_log: dict[str, np.ndarray] = {arm.id: np.zeros(d) for arm in arms}

    def _activate(self, logit: float | np.ndarray) -> float:
        result = _sigmoid(np.array(logit))
        return float(np.clip(result, 1e-6, 1.0 - 1e-6))

    # ── override update / sample / distribution with IRLS ─────────────────

    def update(self, arm_id: str, reward: float, context: dict) -> None:
        if arm_id not in self._arms:
            raise ValueError(f"Unknown arm_id: '{arm_id}'")

        x = self._processor.transform(context)
        r = float(np.clip(reward, 0.0, 1.0))
        m = self._m_log[arm_id]
        A_inv = self._A_inv[arm_id]

        p = self._activate(float(m @ x))
        fisher = p * (1.0 - p) + 1e-6

        # Sherman-Morrison: precision += fisher * x xᵀ
        Ax = A_inv @ x
        denom = 1.0 + fisher * float(x @ Ax) + 1e-8
        new_A_inv = A_inv - fisher * np.outer(Ax, Ax) / denom

        # Newton step: m_new = m + A_inv_new @ (r - p) * x
        self._A_inv[arm_id] = new_A_inv
        self._m_log[arm_id] = m + new_A_inv @ ((r - p) * x)
        logger.debug("update arm=%s reward=%.4f p_before=%.4f", arm_id, reward, p)

    def sample(self, arm_id: str, context: dict, seed: int | None = None) -> float:
        if arm_id not in self._arms:
            raise ValueError(f"Unknown arm_id: '{arm_id}'")
        x = self._processor.transform(context)
        m = self._m_log[arm_id]
        A_inv = self._A_inv[arm_id]
        rng = np.random.default_rng(seed)
        d = len(m)
        w = rng.multivariate_normal(m, A_inv, method="cholesky" if d > 50 else "svd")
        return self._activate(float(w @ x))

    def get_distribution(self, arm_id: str, context: dict) -> dict:
        if arm_id not in self._arms:
            raise ValueError(f"Unknown arm_id: '{arm_id}'")
        x = self._processor.transform(context)
        m = self._m_log[arm_id]
        A_inv = self._A_inv[arm_id]
        return {
            "estimated_p": self._activate(float(m @ x)),
            "w_map": m.tolist(),
            "sigma_diag": np.diag(A_inv).tolist(),
        }

    def get_all_distributions(self) -> dict:
        return {arm_id: {"global": self.get_distribution(arm_id, {})} for arm_id in self._arms}

    def get_state(self) -> dict:
        return {
            "arms": {
                arm_id: {
                    "A_inv": self._A_inv[arm_id].tolist(),
                    "m": self._m_log[arm_id].tolist(),
                }
                for arm_id in self._arms
            },
            "processor": self._processor.get_state(),
        }

    def load_state(self, state: dict) -> None:
        for arm_id, vals in state.get("arms", {}).items():
            if arm_id in self._A_inv:
                self._A_inv[arm_id] = np.array(vals["A_inv"], dtype=np.float64)
                self._m_log[arm_id] = np.array(vals["m"], dtype=np.float64)
        if "processor" in state:
            self._processor.load_state(state["processor"])
