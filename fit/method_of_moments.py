"""Method-of-moments estimator for the OU process

    dz = κ_eff (ψ̄ − z) dt + σ_z dW_t

on a discretely-sampled price-logit series.

Three moments give the three parameters:

    ψ̂        = sample mean of z
    κ̂_eff    = exponential decay rate of the empirical autocorrelation
                ρ(τ) = exp(−κ_eff · τ)
    σ̂_z      = either √(2 κ̂_eff · var(z))   (from stationary variance), or
                from realized variance of Δz with the exact OU increment
                identity:  var(Δz_τ) = σ_z² / κ · (1 − exp(−κ τ))

Both σ_z estimators are reported; agreement between them is a sanity check
on the OU shape assumption.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OUFit:
    psi_hat: float
    kappa_hat: float            # 1 / time_unit
    sigma_z_from_stationary: float
    sigma_z_from_increments: float
    var_z: float
    n_obs: int
    dt: float                   # time per bar in the same units as kappa_hat
    acf_lags: np.ndarray        # shape (L,)
    acf_values: np.ndarray      # shape (L,)
    fit_lags_used: int
    log_acf_r2: float           # R^2 of the log-ACF linear fit


def empirical_acf(z: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelation at lags 0..max_lag (inclusive)."""
    z = np.asarray(z, dtype=float)
    n = z.size
    if n < 2:
        raise ValueError("need at least 2 observations")
    z_centered = z - z.mean()
    var = np.dot(z_centered, z_centered) / n
    if var <= 0:
        raise ValueError("zero variance in series; cannot compute ACF")
    out = np.empty(max_lag + 1, dtype=float)
    out[0] = 1.0
    for k in range(1, max_lag + 1):
        out[k] = np.dot(z_centered[:-k], z_centered[k:]) / (n * var)
    return out


def fit_kappa_from_acf(
    acf: np.ndarray,
    dt: float,
    *,
    min_lag: int = 1,
    noise_floor_factor: float = 2.0,
) -> tuple[float, int, float]:
    """Fit ρ(τ) = exp(−κ τ) by linear regression of log ρ on τ.

    Uses lags ``[min_lag, ..., L*]`` where L* is the lag just before the ACF
    drops below ``noise_floor_factor / sqrt(N)`` or turns non-positive,
    whichever comes first. Returns (kappa_hat, n_lags_used, R^2).
    """
    n = acf.size  # max_lag + 1
    # Heuristic noise floor for ACF of an IID series of length N is ~2/sqrt(N).
    # We don't have N here, but the user-facing API passes max_lag <= N/4 etc.;
    # we instead just walk lags until the ACF turns non-positive.
    max_useful_lag = n - 1
    for k in range(min_lag, n):
        if acf[k] <= 0:
            max_useful_lag = k - 1
            break

    if max_useful_lag < min_lag + 1:
        raise ValueError(
            f"ACF decays/crosses zero too fast (only {max_useful_lag - min_lag + 1} "
            "positive lags); cannot fit exponential."
        )

    lags = np.arange(min_lag, max_useful_lag + 1, dtype=float)
    log_acf = np.log(acf[min_lag:max_useful_lag + 1])

    # OLS: log_acf = a + b * (lags * dt), with kappa = -b. Force intercept = 0?
    # Forcing a=0 is more principled (theory says ρ(0)=1, so log ρ(0)=0); use it.
    # b = sum(x*y) / sum(x*x) where x = lags*dt
    x = lags * dt
    y = log_acf
    b = float(np.dot(x, y) / np.dot(x, x))
    kappa = -b

    # R^2 against zero-intercept model
    y_hat = b * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum(y ** 2))  # because intercept forced to 0
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return kappa, int(max_useful_lag - min_lag + 1), r2


def estimate_sigma_z_from_stationary(kappa: float, var_z: float) -> float:
    """σ_z = √(2 κ var(z)) — uses OU stationary variance identity."""
    return float(np.sqrt(2.0 * kappa * var_z)) if kappa > 0 else float("nan")


def estimate_sigma_z_from_increments(
    z: np.ndarray, dt: float, kappa: float
) -> float:
    """Recover σ_z from realized variance of Δz using the exact OU formula.

    For an OU process z_t with mean-reversion κ and diffusion σ_z, sampled at
    intervals Δt, increments Δz_τ = z(t+Δt) − z(t) satisfy

        var(Δz_τ) = (σ_z² / κ) · (1 − exp(−κ Δt))

    Solve for σ_z given κ.
    """
    if kappa <= 0:
        return float("nan")
    dz = np.diff(np.asarray(z, dtype=float))
    var_dz = float(np.var(dz, ddof=1))
    factor = 1.0 - np.exp(-kappa * dt)
    if factor <= 0:
        return float("nan")
    return float(np.sqrt(var_dz * kappa / factor))


def fit_ou_mom(
    z: np.ndarray,
    *,
    dt: float,
    max_lag: int | None = None,
) -> OUFit:
    """Run the full method-of-moments fit on a single series.

    ``dt`` is the time between consecutive z observations in the same units
    you want κ to come out in (e.g. 1.0 if z is hourly and you want κ in 1/hr).
    ``max_lag`` defaults to ``min(50, len(z)//4)``.
    """
    z = np.asarray(z, dtype=float).ravel()
    n = z.size
    if n < 10:
        raise ValueError(f"series too short: n={n}")

    psi_hat = float(z.mean())
    var_z = float(np.var(z, ddof=1))

    if max_lag is None:
        max_lag = min(50, n // 4)
    acf = empirical_acf(z, max_lag=max_lag)
    kappa, n_lags, r2 = fit_kappa_from_acf(acf, dt=dt)

    sigma_stat = estimate_sigma_z_from_stationary(kappa, var_z)
    sigma_inc = estimate_sigma_z_from_increments(z, dt=dt, kappa=kappa)

    return OUFit(
        psi_hat=psi_hat,
        kappa_hat=kappa,
        sigma_z_from_stationary=sigma_stat,
        sigma_z_from_increments=sigma_inc,
        var_z=var_z,
        n_obs=n,
        dt=dt,
        acf_lags=np.arange(max_lag + 1, dtype=float),
        acf_values=acf,
        fit_lags_used=n_lags,
        log_acf_r2=r2,
    )


def block_bootstrap(
    z: np.ndarray,
    *,
    dt: float,
    point_estimate: OUFit | None = None,
    block_size: int | None = None,
    n_bootstrap: int = 500,
    rng: np.random.Generator | None = None,
    max_lag: int | None = None,
) -> dict[str, dict[str, float]]:
    """Block-bootstrap spread for (psi, kappa, sigma_z_stationary).

    For autocorrelated series the block size needs to comfortably exceed the
    decorrelation timescale; we default to ``max(8, 4/κ̂)`` rounded to int.

    Returns, per parameter, a dict with:
      - ``se``: bootstrap standard error (std of the resampled estimates)
      - ``p16``, ``p84``: 16th/84th percentiles (~ ±1σ if approximately Normal)
      - ``p2.5``, ``p97.5``: 2.5th/97.5th percentiles (raw percentile CI ends)
      - ``n_resamples``: number of finite resamples that contributed

    Note on interpretation: block bootstrap is biased upward for ACF-derived
    statistics (κ in particular). Treat the reported ``se`` as a *spread*
    rather than a calibrated 1σ; the percentile range is a useful sanity-
    check on how concentrated the bootstrap distribution is.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    z = np.asarray(z, dtype=float).ravel()
    n = z.size
    if point_estimate is None:
        point_estimate = fit_ou_mom(z, dt=dt, max_lag=max_lag)
    if block_size is None:
        kappa = max(point_estimate.kappa_hat, 1e-3)
        block_size = max(8, int(round(4.0 / (kappa * dt))))
    block_size = max(2, min(block_size, n // 2))
    n_blocks = int(np.ceil(n / block_size))

    psis, kappas, sigmas = [], [], []
    for _ in range(n_bootstrap):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        sample = np.concatenate([z[s:s + block_size] for s in starts])[:n]
        try:
            fit = fit_ou_mom(sample, dt=dt, max_lag=max_lag)
        except ValueError:
            continue
        psis.append(fit.psi_hat)
        kappas.append(fit.kappa_hat)
        sigmas.append(fit.sigma_z_from_stationary)

    def spread(arr: list[float]) -> dict[str, float]:
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        if a.size < 10:
            return {k: float("nan") for k in ("se", "p16", "p84", "p2.5", "p97.5")} | {
                "n_resamples": int(a.size)
            }
        return {
            "se": float(a.std(ddof=1)),
            "p16": float(np.quantile(a, 0.16)),
            "p84": float(np.quantile(a, 0.84)),
            "p2.5": float(np.quantile(a, 0.025)),
            "p97.5": float(np.quantile(a, 0.975)),
            "n_resamples": int(a.size),
        }

    return {
        "psi_hat": spread(psis),
        "kappa_hat": spread(kappas),
        "sigma_z_from_stationary": spread(sigmas),
    }
