"""Sec 5.2 frozen-topology sweep.

For each active-set size |A| in {3, 5, 8, 12, 16, 20, 25, 30}:
  - Build a homogeneous simulator with N=30 traders, K=5 sources, β_price=0.4
    (so β_info = 1 − β_price = 0.6, no prior inertia).
  - The first |A| traders are eager (deadband α=0); the remaining N−|A|
    "frozen-inactive" traders are given a deadband large enough they never
    trade. The active set is therefore deterministic.
  - Run 6000 steps, drop the first 1500 as transient, fit OU on the price
    logit, and compute ρ(D) on the frozen active subset.
  - Five seeds per |A|.

Writes:
  data/processed/frozen_topology_sweep.pkl
    list of dicts with keys: n_active, n_traders, seed, rho_D, kappa_pred
    (naive (1−ρ)/N), kappa_hat, r2, n_obs.

This reproduces the data behind topology_sweep.png. The "corrected" prediction
κ_pred = (1 − ρ(D)) · |A| / N² discussed in the paper is computed downstream
from rho_D and n_active and does not need to be cached here.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fit.method_of_moments import fit_ou_mom
from prediction_market.market import Market
from prediction_market.simulation import PredictionMarketSimulation
from prediction_market.sources import SourceLayer
from prediction_market.trader import Trader

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

N_TRADERS = 30
N_SOURCES = 5
BETA_PRICE = 0.4
BETA_INFO = 1.0 - BETA_PRICE
GAMMA = 0.05
MU_SCALE = 0.5
NOISE_STD = 0.5
N_STEPS = 6000
TRANSIENT_DROP = 1500
ACTIVE_SIZES = [3, 5, 8, 12, 16, 20, 25, 30]
SEEDS = [0, 1, 2, 3, 4]
FROZEN_ALPHA = 10.0  # > max possible |belief − price| ∈ [0, 1]


def build_simulator(*, n_active: int, seed: int) -> PredictionMarketSimulation:
    """Build a simulator with `n_active` eager traders and the rest frozen."""
    rng = np.random.default_rng(seed)
    mu = rng.uniform(-MU_SCALE, MU_SCALE, size=N_SOURCES)
    sources = SourceLayer(mu=mu, noise_std=np.full(N_SOURCES, NOISE_STD))

    traders = []
    for i in range(N_TRADERS):
        w = rng.dirichlet(np.ones(N_SOURCES))
        alpha = 0.0 if i < n_active else FROZEN_ALPHA
        traders.append(Trader(
            w=w, a=1.0, lam=1.0, alpha=alpha,
            beta_info=BETA_INFO, beta_price=BETA_PRICE, belief0=0.5,
        ))

    market = Market(gamma=GAMMA, pi0=0.5)
    return PredictionMarketSimulation(
        theta=1, sources=sources, traders=traders, market=market,
        rng=np.random.default_rng(seed + 1),
    )


def run_one(n_active: int, seed: int) -> dict:
    sim = build_simulator(n_active=n_active, seed=seed)
    result = sim.run(n_steps=N_STEPS)

    pi = np.clip(np.asarray(result.pi[TRANSIENT_DROP:], dtype=float),
                 1e-3, 1.0 - 1e-3)
    z = np.log(pi / (1.0 - pi))
    fit = fit_ou_mom(z, dt=1.0)

    active = set(range(n_active))
    D = sim.peer_influence_matrix(active=active)
    rho_D = float(np.max(np.abs(np.linalg.eigvals(D))))

    return {
        "n_active": n_active,
        "n_traders": N_TRADERS,
        "seed": seed,
        "rho_D": rho_D,
        "kappa_pred": (1.0 - rho_D) / N_TRADERS,
        "kappa_hat": float(fit.kappa_hat),
        "r2": float(fit.log_acf_r2),
        "n_obs": int(fit.n_obs),
    }


def main() -> None:
    results: list[dict] = []
    for A in ACTIVE_SIZES:
        for seed in SEEDS:
            r = run_one(A, seed)
            results.append(r)
            print(f"|A|={A:>2d} seed={seed}: ρ(D)={r['rho_D']:.4f}  "
                  f"κ̂={r['kappa_hat']:.5f}  R²={r['r2']:.3f}")

    out_path = PROC / "frozen_topology_sweep.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(results, f)
    print(f"\nWrote {out_path.relative_to(ROOT)} ({len(results)} runs)")


if __name__ == "__main__":
    main()
