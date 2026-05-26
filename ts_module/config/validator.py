"""Additional configuration validation utilities."""

from __future__ import annotations

from ts_module.config.schema import ModuleConfig


def validate_context_features_present(config: ModuleConfig) -> None:
    """Raise if context mode requires features but none are declared.

    Args:
        config: Validated ModuleConfig to check.

    Raises:
        ValueError: If context mode is 'categorical' or 'features' but features list is empty.
    """
    from ts_module.config.schema import ContextMode

    if config.context.mode in (ContextMode.categorical, ContextMode.features):
        if not config.context.features:
            raise ValueError(
                f"Context mode '{config.context.mode}' requires at least one feature defined "
                "in context.features."
            )


def validate_signals_not_empty(config: ModuleConfig) -> None:
    """Raise if reward config has no signals.

    Args:
        config: Validated ModuleConfig to check.

    Raises:
        ValueError: If reward.signals is empty.
    """
    if not config.reward.signals:
        raise ValueError("reward.signals must contain at least one signal.")
