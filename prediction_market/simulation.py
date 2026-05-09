from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from prediction_market.market import Market
from prediction_market.sources import SourceLayer
from prediction_market.trader import Trader


@dataclass
class SimulationResult:
    n_steps: int
    pi: np.ndarray
    beliefs: np.ndarray
    actor: np.ndarray
    traded: np.ndarray
    active_mask: np.ndarray
    theta: float

    @property
    def n_traders(self) -> int:
        return int(self.beliefs.shape[1])


class PredictionMarketSimulation:
    """
    Discrete event chain: each step draws fresh source noise, picks one trader
    with probability ∝ λ_i, runs belief update then optional trade + price move.
    """

    def __init__(
        self,
        theta: int,
        sources: SourceLayer,
        traders: list[Trader],
        market: Market,
        *,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.theta = int(theta)
        self.sources = sources
        self.traders = list(traders)
        self.market = market
        self.rng = rng if rng is not None else np.random.default_rng()
        self._validate_shapes()

    def _validate_shapes(self) -> None:
        k = self.sources.K
        for i, tr in enumerate(self.traders):
            if tr.K != k:
                raise ValueError(f"trader {i}: w has length {tr.K}, expected K={k}")

    def reset(self, *, belief: float = 0.5, pi: float | None = None) -> None:
        """Reset market price and all trader beliefs (same RNG is kept)."""
        self.market.reset(pi)
        for tr in self.traders:
            tr.belief = float(belief)

    def _pick_trader(self) -> int:
        lam = np.array([max(tr.lam, 0.0) for tr in self.traders], dtype=float)
        total = float(lam.sum())
        if total <= 0:
            raise ValueError("at least one trader needs lam > 0")
        lam = lam / total
        return int(self.rng.choice(len(self.traders), p=lam))

    def peer_influence_matrix(self, active: set[int]) -> np.ndarray:
        """
        D(t) from the notes (Eq. 9), for a given active set A (traders who have traded).
        Rows/columns align with trader indices 0..N-1.
        """
        n = len(self.traders)
        d = np.zeros((n, n), dtype=float)
        if not active:
            for j in range(n):
                tr = self.traders[j]
                d[j, j] = 1.0 - tr.beta_info - tr.beta_price
            return d

        g = self.market.gamma
        weights = {k: g / self.traders[k].a for k in active}
        denom = sum(weights.values())
        if denom <= 0:
            denom = 1.0

        for j in range(n):
            tr_j = self.traders[j]
            d[j, j] = 1.0 - tr_j.beta_info - tr_j.beta_price
            for k in active:
                if k == j:
                    continue
                d[j, k] = tr_j.beta_price * (weights[k] / denom)
        return d

    def run(self, n_steps: int, *, record_initial: bool = True) -> SimulationResult:
        n = len(self.traders)
        pi_hist: list[float] = []
        belief_hist: list[np.ndarray] = []
        actors: list[int] = []
        traded_flags: list[bool] = []
        active_rows: list[np.ndarray] = []

        active: set[int] = set()

        if record_initial:
            pi_hist.append(self.market.pi)
            belief_hist.append(np.array([tr.belief for tr in self.traders], dtype=float))
            active_rows.append(self._active_array(n, active))

        for step_idx in range(n_steps):
            s = self.sources.signals(
                self.theta, self.rng, step=step_idx, n_steps=n_steps
            )
            idx = self._pick_trader()
            tr = self.traders[idx]
            tr.update_belief(s, self.market.pi)
            q = tr.trade_quantity(self.market.pi)
            traded = q != 0.0
            if traded:
                self.market.apply_trade(q)
                active.add(idx)

            pi_hist.append(self.market.pi)
            belief_hist.append(np.array([t.belief for t in self.traders], dtype=float))
            actors.append(idx)
            traded_flags.append(traded)
            active_rows.append(self._active_array(n, active))

        return SimulationResult(
            n_steps=n_steps,
            pi=np.array(pi_hist, dtype=float),
            beliefs=np.vstack(belief_hist),
            actor=np.array(actors, dtype=int),
            traded=np.array(traded_flags, dtype=bool),
            active_mask=np.vstack(active_rows),
            theta=float(self.theta),
        )

    @staticmethod
    def _active_array(n: int, active: set[int]) -> np.ndarray:
        row = np.zeros(n, dtype=float)
        for k in active:
            row[k] = 1.0
        return row
