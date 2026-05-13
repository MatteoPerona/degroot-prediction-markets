"""Sec 5.3 information-matrix rank sweep.

Sweep rank(Φ) ∈ {1, 2, 3, 5} (with K=5 sources, N=20 traders, β_price=0.4)
while holding all other simulator parameters fixed. For each (rank_target,
seed) draw a fresh set of source biases μ_j ~ Uniform[−0.5, +0.5] and build
the attention matrix W with the target rank:

  - rank 1: a single random simplex point w_0 is shared by every trader.
  - rank r, 1 < r < K: pick a random subset of r sources; every trader's
                       attention is a Dirichlet draw supported on those r
                       sources.
  - rank K: standard K-dim Dirichlet attention per trader.

Run 4000 steps, drop the first 1000 as transient, fit OU on the price logit,
and record:

  - psi_pred = mean_i (w_i · (theta + mu))    (predicted mean-field attractor)
  - psi_hat, kappa_hat, sigma_hat, log-ACF R^2
  - rank_Phi_numerical = numerical rank of Φ = Wᵀ Σ_B⁻¹ W via SVD

Twelve seeds per rank.

Writes:
  data/processed/rank_sweep.pkl

This reproduces the data behind rank_phi_sweep.png. The "logit_truth_proxy"
quantity in the pre-existing pickle is logit(0.9) = 2.197, included only as a
ceiling diagnostic for cross-seed plots; it is omitted here because no paper
claim depends on it.
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

N_TRADERS = 20
N_SOURCES = 5
BETA_PRICE = 0.4
BETA_INFO = 1.0 - BETA_PRICE
ALPHA = 0.0
GAMMA = 0.05
MU_SCALE = 0.5
NOISE_STD = 0.5
N_STEPS = 4000
TRANSIENT_DROP = 1000
RANK_TARGETS = [1, 2, 3, 5]
SEEDS = list(range(12))


def attention_matrix(rng: np.random.Generator, rank_target: int) -> np.ndarray:
    """Build an N × K attention matrix with the requested rank."""
    K = N_SOURCES
    if rank_target == 1:
        w0 = rng.dirichlet(np.ones(K))
        return np.tile(w0, (N_TRADERS, 1))
    if rank_target == K:
        return np.array([rng.dirichlet(np.ones(K)) for _ in range(N_TRADERS)])
    # 1 < rank < K: pick `rank_target` source indices; every row Dirichlet on them
    cols = rng.choice(K, size=rank_target, replace=False)
    W = np.zeros((N_TRADERS, K))
    for i in range(N_TRADERS):
        W[i, cols] = rng.dirichlet(np.ones(rank_target))
    return W


def numerical_rank_phi(W: np.ndarray) -> int:
    """rank(Φ) = rank(Wᵀ Σ_B⁻¹ W); Σ_B = NOISE_STD² I, so rank(Φ) = rank(W)."""
    s = np.linalg.svd(W, compute_uv=False)
    tol = max(W.shape) * s.max() * np.finfo(float).eps
    return int(np.sum(s > tol))


def run_one(rank_target: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    mu = rng.uniform(-MU_SCALE, MU_SCALE, size=N_SOURCES)
    sources = SourceLayer(mu=mu, noise_std=np.full(N_SOURCES, NOISE_STD))

    W = attention_matrix(rng, rank_target)
    traders = [
        Trader(w=W[i], a=1.0, lam=1.0, alpha=ALPHA,
               beta_info=BETA_INFO, beta_price=BETA_PRICE, belief0=0.5)
        for i in range(N_TRADERS)
    ]
    sim = PredictionMarketSimulation(
        theta=1, sources=sources, traders=traders,
        market=Market(gamma=GAMMA, pi0=0.5),
        rng=np.random.default_rng(seed + 1000),
    )
    result = sim.run(n_steps=N_STEPS)

    pi = np.clip(np.asarray(result.pi[TRANSIENT_DROP:], dtype=float),
                 1e-3, 1.0 - 1e-3)
    z = np.log(pi / (1.0 - pi))
    fit = fit_ou_mom(z, dt=1.0)

    s_mean = float(sim.theta) + mu
    psi_pred = float(np.mean(W @ s_mean))

    return {
        "rank_target": rank_target,
        "rank_Phi_numerical": numerical_rank_phi(W),
        "seed": seed,
        "psi_pred": psi_pred,
        "psi_hat": float(fit.psi_hat),
        "kappa_hat": float(fit.kappa_hat),
        "sigma_hat": float(fit.sigma_z_from_stationary),
        "r2": float(fit.log_acf_r2),
    }


def main() -> None:
    results: list[dict] = []
    for r in RANK_TARGETS:
        for seed in SEEDS:
            row = run_one(r, seed)
            results.append(row)
            print(f"rank={r} seed={seed:2d}: ψ_pred={row['psi_pred']:+.3f}  "
                  f"ψ_hat={row['psi_hat']:+.3f}  κ̂={row['kappa_hat']:.4f}  "
                  f"R²={row['r2']:.3f}  rank(Φ)_num={row['rank_Phi_numerical']}")

    out_path = PROC / "rank_sweep.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(results, f)
    print(f"\nWrote {out_path.relative_to(ROOT)} ({len(results)} runs)")

    # Quick sanity report
    psi_pred = np.array([r["psi_pred"] for r in results])
    psi_hat = np.array([r["psi_hat"] for r in results])
    print(f"corr(ψ_pred, ψ_hat) = {np.corrcoef(psi_pred, psi_hat)[0,1]:.4f}")
    for rk in RANK_TARGETS:
        ph = np.array([r["psi_hat"] for r in results if r["rank_target"] == rk])
        print(f"  rank={rk}: std(ψ_hat) (ddof=0) = {ph.std():.4f}")


if __name__ == "__main__":
    main()
