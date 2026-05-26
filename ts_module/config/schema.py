"""Pydantic v2 configuration schemas for the TS module."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, model_validator


class ContextMode(str, Enum):
    """How context is used in the model."""

    none = "none"
    categorical = "categorical"
    features = "features"


class ModelType(str, Enum):
    """Thompson Sampling model variant."""

    auto = "auto"
    beta = "beta"
    logistic = "logistic"
    linear = "linear"


class RewardType(str, Enum):
    """Type of reward signal."""

    binary = "binary"
    continuous = "continuous"
    composite = "composite"


class ObjectiveType(str, Enum):
    """How to score and rank arms."""

    max_probability = "max_probability"
    max_expected_revenue = "max_expected_revenue"
    max_roi = "max_roi"
    min_cost_per_success = "min_cost_per_success"
    custom = "custom"


class ArmConfig(BaseModel):
    """Configuration for a single arm (decision option)."""

    id: str
    name: str
    metadata: dict = {}  # noqa: B006


class ContextFeatureConfig(BaseModel):
    """Configuration for a single context feature."""

    name: str
    type: str  # "categorical" | "continuous"
    categories: list[str] | None = None
    # Required for categorical features in features mode (one-hot encoding).
    # Not needed in categorical mode (context_key handles it).


class ContextConfig(BaseModel):
    """Context configuration: what information is available at decision time."""

    mode: ContextMode = ContextMode.none
    features: list[ContextFeatureConfig] = []  # noqa: B006


class SignalConfig(BaseModel):
    """Configuration for a single reward signal."""

    name: str
    weight: float = 1.0
    timeout_hours: int = 72
    default_on_timeout: float = 0.0


class RewardConfig(BaseModel):
    """Configuration for reward aggregation."""

    type: RewardType = RewardType.binary
    signals: list[SignalConfig]
    aggregation: str = "weighted_sum"


class ObjectiveConfig(BaseModel):
    """Configuration for the objective function used to rank arms."""

    type: ObjectiveType = ObjectiveType.max_probability
    formula: str | None = None


class ConstraintConfig(BaseModel):
    """Configuration for a single business constraint."""

    type: str  # "capacity" | "min_traffic" | "eligibility"
    arm_id: str | None = None
    value: float | None = None
    arm_field: str | None = None
    condition: str | None = None


class BetaHyperparams(BaseModel):
    """Hyperparameters specific to Beta TS model."""

    alpha_init: float = 1.0
    beta_init: float = 1.0
    forgetting_rate: float = 0.97
    min_exploration: float = 0.03
    cold_start_pulls: int = 30


class LogisticHyperparams(BaseModel):
    """Hyperparameters for Logistic TS model."""

    prior_variance: float = 1.0
    # σ²: A initialised as (1/prior_variance)*I. Larger = less informative prior = learns faster.
    regularization: float = 0.01
    # λ added to A diagonal for numerical stability.
    feature_scaling: bool = True
    # Standardise continuous features (recommended).


class LinearHyperparams(BaseModel):
    """Hyperparameters for Linear TS model."""

    prior_variance: float = 1.0
    regularization: float = 0.01
    feature_scaling: bool = True
    noise_variance: float = 1.0
    # σ²_noise: update uses b += (r / noise_variance) * x.
    # Smaller → faster learning. At 1.0 behaviour matches LogisticModel (no sigmoid).


class HyperparamsConfig(BaseModel):
    """Hyperparameter configuration for the module."""

    model_type: ModelType = ModelType.auto
    min_exploration: float | None = None
    # None → resolved to beta.min_exploration for backward compatibility.
    beta: BetaHyperparams = BetaHyperparams()
    logistic: LogisticHyperparams = LogisticHyperparams()
    linear: LinearHyperparams = LinearHyperparams()
    update_mode: str = "realtime"

    @model_validator(mode="after")
    def resolve_min_exploration(self) -> HyperparamsConfig:
        """Fall back to beta.min_exploration for configs that don't set the top-level field."""
        if self.min_exploration is None:
            self.min_exploration = self.beta.min_exploration
        return self


class ModuleConfig(BaseModel):
    """Top-level configuration for a TS module instance."""

    id: str
    name: str
    arms: list[ArmConfig]
    context: ContextConfig = ContextConfig()
    reward: RewardConfig
    objective: ObjectiveConfig = ObjectiveConfig()
    constraints: list[ConstraintConfig] = []  # noqa: B006
    hyperparams: HyperparamsConfig = HyperparamsConfig()

    @model_validator(mode="after")
    def auto_select_model_type(self) -> ModuleConfig:
        """Resolve 'auto' model_type to a concrete type based on reward and context."""
        if self.hyperparams.model_type == ModelType.auto:
            if self.reward.type == RewardType.continuous:
                self.hyperparams.model_type = ModelType.linear
            elif self.reward.type == RewardType.binary:
                if self.context.mode == ContextMode.features:
                    self.hyperparams.model_type = ModelType.logistic
                else:
                    self.hyperparams.model_type = ModelType.beta
            else:
                # composite: default to beta for Phase 1
                self.hyperparams.model_type = ModelType.beta
        return self

    @model_validator(mode="after")
    def validate_arm_ids_unique(self) -> ModuleConfig:
        """Raise if any arm IDs are duplicated."""
        ids = [arm.id for arm in self.arms]
        seen: set[str] = set()
        duplicates = [i for i in ids if i in seen or seen.add(i)]  # type: ignore[func-returns-value]
        if duplicates:
            raise ValueError(f"Duplicate arm IDs found: {duplicates}")
        return self

    @model_validator(mode="after")
    def validate_custom_objective_has_formula(self) -> ModuleConfig:
        """Raise if objective type is 'custom' but no formula is provided."""
        if self.objective.type == ObjectiveType.custom and self.objective.formula is None:
            raise ValueError("Objective type 'custom' requires a formula to be specified.")
        return self

    @model_validator(mode="after")
    def validate_categorical_features_have_categories(self) -> ModuleConfig:
        """In features mode, categorical features must declare their categories list."""
        if self.context.mode != ContextMode.features:
            return self
        for feat in self.context.features:
            if feat.type == "categorical" and feat.categories is None:
                raise ValueError(
                    f"Feature '{feat.name}' is categorical in features mode but has no "
                    "'categories' list. Provide categories for one-hot encoding."
                )
        return self
