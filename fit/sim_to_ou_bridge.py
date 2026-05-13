"""Bridge between the DeGroot simulator and the OU fit.

For a sweep of population price-trust ``β_price``, build a homogeneous
simulator with random attention weights, run it, fit the OU process to the
resulting price-logit path, then compare the OU-derived ``κ̂_eff`` and ``ψ̂``
to the analytical predictions implied by the simulator's ``D(t)`` and the
source signals:

    predicted κ_eff   ∝ (1 − ρ(D))           (eq. 13 of the model formulation)
    predicted ψ̄      = mean_i (w_i · s_mean)   (mean-field log-odds attractor
                                                under no prior inertia)

Both predictions come straight from the user's matrix-spectral framework, so
agreement between predicted and fitted values means the OU framing is the
mean-field reduction of the DeGroot system that the simulator implements.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fit.method_of_moments import fit_ou_mom
from prediction_market.market import Market
from prediction_market.simulation import PredictionMarketSimulation
from prediction_market.sources import SourceLayer
from prediction_market.trader import Trader


@dataclass
class BridgeResult:
    beta_price: float
    beta_info: float
    n_traders: int
    n_sources: int
    rho_D: float
    kappa_predicted_per_step: float   # (1 − ρ(D)) / N (asynchronous rescaling)
    kappa_hat_per_step: float
    psi_predicted: float
    psi_hat: float
    sigma_hat: float
    r2: float
    n_obs: int
    seed: int


def build_simulator(
    *,
    n_traders: int,
    n_sources: int,
    beta_price: float,
    beta_info: float | None = None,
    alpha: float = 0.0,
    gamma: float = 0.05,
    mu_scale: float = 0.5,
    noise_std: float = 0.5,
    theta: int = 1,
    rng_seed: int = 0,
) -> tuple[PredictionMarketSimulation, np.ndarray]:
    """Construct simulator with homogeneous β_info, β_price; random attention."""
    rng = np.random.default_rng(rng_seed)
    if beta_info is None:
        beta_info = max(0.0, 1.0 - beta_price)  # no prior inertia

    mu = rng.uniform(-mu_scale, mu_scale, size=n_sources)
    sources = SourceLayer(mu=mu, noise_std=np.full(n_sources, noise_std))

    traders = []
    for _ in range(n_traders):
        w = rng.dirichlet(np.ones(n_sources))
        traders.append(Trader(
            w=w, a=1.0, lam=1.0, alpha=alpha,
            beta_info=beta_info, beta_price=beta_price, belief0=0.5,
        ))

    market = Market(gamma=gamma, pi0=0.5)
    sim = PredictionMarketSimulation(
        theta=theta, sources=sources, traders=traders, market=market,
        rng=np.random.default_rng(rng_seed + 1),
    )
    return sim, mu


def run_one_regime(
    *,
    n_traders: int,
    n_sources: int,
    beta_price: float,
    n_steps: int,
    transient_drop: int = 0,
    rng_seed: int = 0,
    **kwargs,
) -> BridgeResult:
    sim, mu = build_simulator(
        n_traders=n_traders, n_sources=n_sources, beta_price=beta_price,
        rng_seed=rng_seed, **kwargs,
    )
    result = sim.run(n_steps=n_steps)
    pi = np.asarray(result.pi[transient_drop:], dtype=float)
    # logit transform with mild clipping (linear-impact market doesn't saturate
    # as hard as LMSR but can still graze 0/1 in extreme regimes)
    eps = 1e-3
    pi_c = np.clip(pi, eps, 1.0 - eps)
    z = np.log(pi_c / (1.0 - pi_c))

    fit = fit_ou_mom(z, dt=1.0)

    # rho(D) at the full active set
    D = sim.peer_influence_matrix(active=set(range(n_traders)))
    eigs = np.linalg.eigvals(D)
    rho_D = float(np.max(np.abs(eigs)))

    # predicted scalar attractor: average over traders of w_i · (theta + mu)
    s_mean = float(sim.theta) + mu
    psi_predicted = float(np.mean([t.w @ s_mean for t in sim.traders]))

    # The DeGroot mixing time is per *synchronous* iteration; the simulator
    # advances asynchronously (one trader per step), so a coarse rescaling
    # divides by the number of traders.
    kappa_pred = (1.0 - rho_D) / n_traders

    beta_info_actual = sim.traders[0].beta_info

    return BridgeResult(
        beta_price=beta_price,
        beta_info=beta_info_actual,
        n_traders=n_traders,
        n_sources=n_sources,
        rho_D=rho_D,
        kappa_predicted_per_step=kappa_pred,
        kappa_hat_per_step=fit.kappa_hat,
        psi_predicted=psi_predicted,
        psi_hat=fit.psi_hat,
        sigma_hat=fit.sigma_z_from_stationary,
        r2=fit.log_acf_r2,
        n_obs=fit.n_obs,
        seed=rng_seed,
    )


@dataclass
class TimeVaryingTrace:
    """Per-window summary of a single simulation's evolving topology."""
    seed: int
    centers: np.ndarray            # step index at window center
    active_size: np.ndarray        # |A(t_c)| at each center
    rho_D: np.ndarray              # ρ(D(t_c)) at each center
    kappa_predicted: np.ndarray    # (1 − ρ(D(t_c))) / N at each center
    kappa_hat: np.ndarray          # OU fit on window centred at t_c
    psi_hat: np.ndarray            # OU fit attractor on the window


def run_time_varying_topology(
    *,
    n_traders: int = 20,
    n_sources: int = 5,
    beta_price: float = 0.4,
    alpha: float = 0.05,
    gamma: float = 0.05,
    n_steps: int = 8000,
    window: int = 500,
    stride: int = 50,
    rng_seed: int = 0,
    **kwargs,
) -> TimeVaryingTrace:
    """Run one simulation, then slide a window across the price-logit path,
    fitting OU and computing ρ(D(t)) at each window center.

    With ``alpha > 0`` the deadband prevents every trader from trading on
    step 1, so the active set ``A(t)`` grows gradually — exposing the
    Seeding → Discovery → Consensus phases predicted by the matrix
    framework.
    """
    sim, _ = build_simulator(
        n_traders=n_traders, n_sources=n_sources, beta_price=beta_price,
        alpha=alpha, gamma=gamma, rng_seed=rng_seed, **kwargs,
    )
    result = sim.run(n_steps=n_steps)

    pi = np.clip(np.asarray(result.pi, dtype=float), 1e-3, 1.0 - 1e-3)
    z = np.log(pi / (1.0 - pi))

    half = window // 2
    centers = np.arange(half, len(z) - half, stride, dtype=int)

    active_sizes, rho_Ds, kappa_preds, kappa_hats, psi_hats = [], [], [], [], []
    for c in centers:
        # active set at step c (cumulative — every trader who's traded by now)
        active_arr = result.active_mask[c]
        active = set(int(k) for k in np.where(active_arr > 0)[0])
        active_sizes.append(len(active))

        if len(active) >= 2:
            D = sim.peer_influence_matrix(active=active)
            rho_D = float(np.max(np.abs(np.linalg.eigvals(D))))
        else:
            rho_D = 0.0
        rho_Ds.append(rho_D)
        kappa_preds.append((1.0 - rho_D) / n_traders)

        z_win = z[c - half:c + half]
        try:
            fit = fit_ou_mom(z_win, dt=1.0)
            kappa_hats.append(fit.kappa_hat)
            psi_hats.append(fit.psi_hat)
        except (ValueError, np.linalg.LinAlgError):
            kappa_hats.append(np.nan)
            psi_hats.append(np.nan)

    return TimeVaryingTrace(
        seed=rng_seed,
        centers=np.asarray(centers, dtype=float),
        active_size=np.asarray(active_sizes, dtype=float),
        rho_D=np.asarray(rho_Ds, dtype=float),
        kappa_predicted=np.asarray(kappa_preds, dtype=float),
        kappa_hat=np.asarray(kappa_hats, dtype=float),
        psi_hat=np.asarray(psi_hats, dtype=float),
    )


def sweep(
    betas: list[float],
    *,
    seeds: list[int],
    n_traders: int = 20,
    n_sources: int = 5,
    n_steps: int = 4000,
    transient_drop: int = 1000,
    **kwargs,
) -> list[BridgeResult]:
    out = []
    for bp in betas:
        for seed in seeds:
            try:
                r = run_one_regime(
                    n_traders=n_traders, n_sources=n_sources,
                    beta_price=bp, n_steps=n_steps,
                    transient_drop=transient_drop, rng_seed=seed,
                    **kwargs,
                )
                out.append(r)
            except ValueError as e:
                # OU fit can fail when beta_price → 0 (no peer mixing, ACF
                # crosses zero very quickly) or when the price saturates.
                # Record-skip; the caller can see the gap in the sweep.
                print(f"  SKIP β_p={bp}, seed={seed}: {e}")
    return out
