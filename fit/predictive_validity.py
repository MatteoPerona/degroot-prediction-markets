"""Predictive-validity metrics for OU-fitted prediction-market estimates.

Given a corpus of resolved markets with realized outcomes θ ∈ {0, 1} and
three candidate predictions π̂ ∈ [0, 1] per market — the OU lifetime
attractor σ(ψ̂), the last observed price, and the uninformative null 0.5 —
compute Brier score, log-loss, and accuracy aggregated across markets, plus
paired-bootstrap CIs on the differences between predictors.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-6


def brier_score(p_hat: np.ndarray, theta: np.ndarray) -> float:
    p = np.asarray(p_hat, dtype=float)
    y = np.asarray(theta, dtype=float)
    return float(np.mean((p - y) ** 2))


def log_loss(p_hat: np.ndarray, theta: np.ndarray, eps: float = EPS) -> float:
    p = np.clip(np.asarray(p_hat, dtype=float), eps, 1.0 - eps)
    y = np.asarray(theta, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def accuracy(p_hat: np.ndarray, theta: np.ndarray) -> float:
    p = np.asarray(p_hat, dtype=float)
    y = np.asarray(theta, dtype=float)
    return float(np.mean((p >= 0.5).astype(float) == y))


def metrics(p_hat: np.ndarray, theta: np.ndarray) -> dict[str, float]:
    return {
        "brier": brier_score(p_hat, theta),
        "log_loss": log_loss(p_hat, theta),
        "accuracy": accuracy(p_hat, theta),
    }


def paired_bootstrap_difference(
    p_a: np.ndarray,
    p_b: np.ndarray,
    theta: np.ndarray,
    *,
    metric: str = "brier",
    n_bootstrap: int = 2000,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Paired bootstrap on markets: resample (π̂_a, π̂_b, θ) triples and
    compute the difference (metric(a) − metric(b)) on each resample.

    Returns dict with point difference, bootstrap SE, 2.5/97.5 percentiles,
    and the share of resamples in which a beats b (lower metric is better
    for ``brier`` and ``log_loss``; higher for ``accuracy``).
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    y = np.asarray(theta, dtype=float)
    n = p_a.size
    fn = {"brier": brier_score, "log_loss": log_loss, "accuracy": accuracy}[metric]

    point_diff = fn(p_a, y) - fn(p_b, y)

    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        diffs[i] = fn(p_a[idx], y[idx]) - fn(p_b[idx], y[idx])

    if metric == "accuracy":
        a_beats_b = float(np.mean(diffs > 0))
    else:
        a_beats_b = float(np.mean(diffs < 0))  # lower is better

    return {
        "point_difference": float(point_diff),
        "se": float(diffs.std(ddof=1)),
        "p2.5": float(np.quantile(diffs, 0.025)),
        "p97.5": float(np.quantile(diffs, 0.975)),
        "share_a_beats_b": a_beats_b,
    }


def expit(z: float | np.ndarray) -> float | np.ndarray:
    z = np.asarray(z, dtype=float)
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
