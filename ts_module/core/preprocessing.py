"""FeatureProcessor: converts context dicts to numpy feature vectors."""

from __future__ import annotations

import numpy as np

from ts_module.config.schema import ContextConfig


class FeatureProcessor:
    """Converts a context dict to a fixed-length numpy feature vector.

    Categorical features → one-hot encoding (categories declared in config).
    Continuous features  → optional z-score standardisation.

    Fitted lazily on first transform() call if fit() was not called explicitly.
    """

    def __init__(
        self,
        context_config: ContextConfig,
        feature_scaling: bool = True,
    ) -> None:
        self._features = context_config.features
        self._feature_scaling = feature_scaling
        self._fitted = False

        # Continuous feature stats (filled by fit())
        self._means: dict[str, float] = {}
        self._stds: dict[str, float] = {}

        # Build static index: feature name → slice in output vector
        self._cat_indices: dict[str, dict[str, int]] = {}  # name → {category → offset}
        self._cat_offsets: dict[str, int] = {}  # name → start index in vector
        self._cont_offsets: dict[str, int] = {}  # name → index in vector
        self._dim = 0

        for feat in self._features:
            if feat.type == "categorical":
                cats = feat.categories or []
                self._cat_offsets[feat.name] = self._dim
                self._cat_indices[feat.name] = {c: i for i, c in enumerate(cats)}
                self._dim += len(cats)
            else:  # continuous
                self._cont_offsets[feat.name] = self._dim
                self._dim += 1

    # ── public API ────────────────────────────────────────────────────────

    @property
    def feature_dim(self) -> int:
        """Dimension of the output feature vector."""
        return self._dim

    def fit(self, contexts: list[dict]) -> None:
        """Compute mean and std for continuous features.

        Args:
            contexts: List of context dicts to fit on.
                      If len == 1, std defaults to 1.0 (no normalisation possible).
        """
        if not self._feature_scaling:
            self._fitted = True
            return

        for feat in self._features:
            if feat.type != "continuous":
                continue
            values = [float(ctx[feat.name]) for ctx in contexts if feat.name in ctx]
            if not values:
                self._means[feat.name] = 0.0
                self._stds[feat.name] = 1.0
            elif len(values) == 1:
                self._means[feat.name] = values[0]
                self._stds[feat.name] = 1.0
            else:
                self._means[feat.name] = float(np.mean(values))
                std = float(np.std(values))
                self._stds[feat.name] = std if std > 1e-8 else 1.0

        self._fitted = True

    def transform(self, context: dict) -> np.ndarray:
        """Convert a context dict to a feature vector.

        Triggers lazy fit on the first call if fit() was not called.
        Unknown categorical value → all-zeros for that feature's block.
        Missing feature → zeros.
        """
        if not self._fitted:
            self.fit([context])

        vec = np.zeros(self._dim, dtype=np.float64)

        for feat in self._features:
            if feat.type == "categorical":
                val = context.get(feat.name)
                if val is not None:
                    idx = self._cat_indices[feat.name].get(str(val))
                    if idx is not None:
                        vec[self._cat_offsets[feat.name] + idx] = 1.0
            else:  # continuous
                raw = context.get(feat.name)
                if raw is not None:
                    offset = self._cont_offsets[feat.name]
                    if self._feature_scaling and self._fitted:
                        mean = self._means.get(feat.name, 0.0)
                        std = self._stds.get(feat.name, 1.0)
                        vec[offset] = (float(raw) - mean) / std
                    else:
                        vec[offset] = float(raw)

        return vec

    def fit_transform(self, contexts: list[dict]) -> list[np.ndarray]:
        """Fit then transform a list of contexts."""
        self.fit(contexts)
        return [self.transform(c) for c in contexts]

    def refit(self, contexts: list[dict]) -> None:
        """Explicit refit — replaces prior scaling stats."""
        self._fitted = False
        self.fit(contexts)

    # ── serialisation ─────────────────────────────────────────────────────

    def get_state(self) -> dict:
        """Serialise scaler state."""
        return {
            "fitted": self._fitted,
            "means": dict(self._means),
            "stds": dict(self._stds),
        }

    def load_state(self, state: dict) -> None:
        """Restore scaler state."""
        self._fitted = bool(state.get("fitted", False))
        self._means = {k: float(v) for k, v in state.get("means", {}).items()}
        self._stds = {k: float(v) for k, v in state.get("stds", {}).items()}
