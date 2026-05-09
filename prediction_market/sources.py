from __future__ import annotations

import numpy as np


class SourceLayer:
    """K noisy signals about a binary outcome θ ∈ {0, 1}: s_j = θ + μ_j + ε_j."""

    def __init__(
        self,
        mu: np.ndarray,
        noise_std: np.ndarray | None = None,
        *,
        cov: np.ndarray | None = None,
        resolution_decay: bool = False,
    ) -> None:
        """
        resolution_decay: If True, each call to signals() should pass ``step`` and ``n_steps``.
        Noise is multiplied by (1 - progress) with progress = step / max(n_steps - 1, 1),
        so σ_eff → 0 at the last step (linear σ_resolution ramp).
        """
        self.resolution_decay = bool(resolution_decay)
        self.mu = np.asarray(mu, dtype=float).reshape(-1)
        self.K = int(self.mu.size)
        if cov is not None:
            self._cov = np.asarray(cov, dtype=float)
            if self._cov.shape != (self.K, self.K):
                raise ValueError("cov must be (K, K)")
            self._noise_std = None
        else:
            std = np.ones(self.K, dtype=float) if noise_std is None else np.asarray(noise_std, dtype=float).reshape(-1)
            if std.shape != (self.K,):
                raise ValueError("noise_std must have shape (K,)")
            if np.any(std < 0):
                raise ValueError("noise_std must be non-negative")
            self._noise_std = std
            self._cov = None

    @staticmethod
    def _resolution_noise_scale(step: int, n_steps: int) -> float:
        """Linear ramp: scale 1 at step 0 → 0 at last step (inclusive)."""
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        denom = max(n_steps - 1, 1)
        progress = step / denom
        return float(max(0.0, min(1.0, 1.0 - progress)))

    def signals(
        self,
        theta: int | float,
        rng: np.random.Generator,
        *,
        step: int | None = None,
        n_steps: int | None = None,
    ) -> np.ndarray:
        theta = float(theta)
        scale = 1.0
        if self.resolution_decay:
            if step is None or n_steps is None:
                raise ValueError("signals() requires step= and n_steps= when resolution_decay=True")
            scale = self._resolution_noise_scale(step, n_steps)

        if scale <= 0.0:
            return theta + self.mu

        if self._cov is not None:
            eps = scale * rng.multivariate_normal(np.zeros(self.K), self._cov)
        else:
            eps = scale * rng.normal(0.0, self._noise_std)
        return theta + self.mu + eps
