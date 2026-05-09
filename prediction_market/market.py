from __future__ import annotations


class Market:
    """Binary contract price π ∈ [0, 1] with linear impact (Eq. 5)."""

    def __init__(self, gamma: float = 0.15, pi0: float = 0.5) -> None:
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        self.gamma = float(gamma)
        self.pi = float(pi0)

    def reset(self, pi0: float | None = None) -> None:
        self.pi = float(0.5 if pi0 is None else pi0)

    def apply_trade(self, q: float) -> float:
        """π ← clip(π + γ q, 0, 1). Returns new price."""
        self.pi = max(0.0, min(1.0, self.pi + self.gamma * float(q)))
        return self.pi
