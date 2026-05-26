"""Linear Thompson Sampling model — Bayesian linear regression (no sigmoid)."""

from __future__ import annotations

from ts_module.config.schema import ArmConfig, LinearHyperparams
from ts_module.core.models.logistic import _BayesianLinearBase
from ts_module.core.preprocessing import FeatureProcessor


class LinearModel(_BayesianLinearBase):
    """Linear Thompson Sampling — continuous reward with continuous features."""

    def __init__(
        self,
        arms: list[ArmConfig],
        feature_processor: FeatureProcessor,
        hyperparams: LinearHyperparams,
    ) -> None:
        super().__init__(
            arms=arms,
            feature_processor=feature_processor,
            prior_variance=hyperparams.prior_variance,
            regularization=hyperparams.regularization,
        )
        self._noise_variance = hyperparams.noise_variance

    def _activate(self, logit: float) -> float:  # type: ignore[override]
        return float(logit)

    def _precision_scale(self) -> float:
        return 1.0 / self._noise_variance
