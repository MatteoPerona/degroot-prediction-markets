from __future__ import annotations

import numpy as np


def _sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    z = np.asarray(z, dtype=float)
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


class Trader:
    """One trader: belief update (Eq. 3) and demand with deadband (Eq. 4)."""

    def __init__(
        self,
        w: np.ndarray,
        *,
        a: float = 1.0,
        lam: float = 1.0,
        alpha: float = 0.05,
        beta_info: float = 0.4,
        beta_price: float = 0.4,
        belief0: float = 0.5,
    ) -> None:
        self.w = np.asarray(w, dtype=float).reshape(-1)
        self.a = float(a)
        self.lam = float(lam)
        self.alpha = float(alpha)
        self.beta_info = float(beta_info)
        self.beta_price = float(beta_price)
        if self.beta_info + self.beta_price > 1.0 + 1e-9:
            raise ValueError("beta_info + beta_price must be <= 1")
        if self.a <= 0 or self.lam < 0:
            raise ValueError("require a > 0 and lam >= 0")
        self.belief = float(belief0)

    @property
    def K(self) -> int:
        return int(self.w.size)

    def resize_attention(self, k: int) -> None:
        """Pad or truncate w to length k (useful when wiring sources after init)."""
        w = self.w
        if w.size == k:
            return
        if w.size > k:
            self.w = w[:k].copy()
            return
        pad = np.zeros(k - w.size, dtype=float)
        self.w = np.concatenate([w, pad])

    def update_belief(self, s: np.ndarray, pi: float) -> float:
        """Convex combination of σ(w·s), market price, and prior; returns new belief."""
        s = np.asarray(s, dtype=float).reshape(-1)
        if s.size != self.w.size:
            raise ValueError(f"signal length {s.size} != attention length {self.w.size}")
        info = float(_sigmoid(float(np.dot(self.w, s))))
        prior = self.belief
        b_i, b_p = self.beta_info, self.beta_price
        self.belief = b_i * info + b_p * float(pi) + (1.0 - b_i - b_p) * prior
        return self.belief

    def trade_quantity(self, pi: float) -> float:
        """Signed demand q = (p - π)/a if |p - π| > α, else 0."""
        edge = self.belief - float(pi)
        if abs(edge) <= self.alpha:
            return 0.0
        return edge / self.a
