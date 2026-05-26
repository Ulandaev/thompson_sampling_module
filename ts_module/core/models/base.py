"""Abstract base class for all Thompson Sampling model variants."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Abstract interface for a Thompson Sampling model.

    All model variants (Beta, Logistic, Linear) must implement this interface.
    The TSEngine interacts exclusively through this interface, so Phase 2 models
    can be added without changing the engine.
    """

    @abstractmethod
    def sample(self, arm_id: str, context: dict, seed: int | None = None) -> float:
        """Sample theta from the posterior distribution for the given arm and context.

        Args:
            arm_id: Identifier of the arm to sample from.
            context: Context features at decision time.
            seed: Optional RNG seed for deterministic sampling (used in tests).

        Returns:
            Sampled probability theta in [0.0, 1.0].

        Raises:
            ValueError: If arm_id is not known to the model.
        """
        ...

    @abstractmethod
    def update(self, arm_id: str, reward: float, context: dict) -> None:
        """Update the posterior distribution with an observed reward.

        Args:
            arm_id: Identifier of the arm that received the reward.
            reward: Observed reward value. Positive updates alpha, negative/zero updates beta.
            context: Context that was active when the decision was made.

        Raises:
            ValueError: If arm_id is not known to the model.
        """
        ...

    @abstractmethod
    def get_distribution(self, arm_id: str, context: dict) -> dict:
        """Get current distribution parameters for a given arm and context.

        Args:
            arm_id: Identifier of the arm.
            context: Context to look up (determines the context key).

        Returns:
            Dict with keys: "alpha", "beta", "estimated_p" (= alpha / (alpha + beta)).

        Raises:
            ValueError: If arm_id is not known to the model.
        """
        ...

    @abstractmethod
    def get_all_distributions(self) -> dict:
        """Get all distributions across all arms and context keys.

        Returns:
            Nested dict: {arm_id: {context_key: {"alpha": ..., "beta": ..., "estimated_p": ...}}}
        """
        ...

    @abstractmethod
    def get_state(self) -> dict:
        """Serialize the full model state to a JSON-compatible dict.

        Returns:
            Dict that can be passed to load_state() to restore the model.
        """
        ...

    @abstractmethod
    def load_state(self, state: dict) -> None:
        """Restore the model state from a previously serialized dict.

        Args:
            state: Dict previously returned by get_state().
        """
        ...
