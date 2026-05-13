"""Regenerate paper figures for the expanded N=68 corpus.

Produces:
  data/processed/figures/predictive_validity.png
  data/processed/figures/ou_fits_good_markets.png
  data/processed/figures/sigma_mismatch_stratification.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = PROC / "figures"


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    with (PROC / "predictive_validity.json").open() as f:
        pv = json.load(f)
    with (PROC / "mom_fits.json").open() as f:
        fits = {f["token_id"]: f for f in json.load(f)}

    rows = pv["per_market"]
    theta = np.array([r["theta"] for r in rows])
    p_ou = np.array([r["p_ou"] for r in rows])
    p_last = np.array([r["p_last"] for r in rows])
    sigma_mm = np.array([r["sigma_mismatch"] for r in rows])
    r2 = np.array([r["log_acf_r2"] for r in rows])

    # ============ Figure 1: predictive validity scatter ============
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    sc = ax.scatter(theta + np.random.default_rng(0).normal(0, 0.02, size=theta.size),
                    p_ou, c=sigma_mm, cmap="viridis", s=60, edgecolor="black",
                    linewidth=0.5, label=r"$\hat\pi^*_{OU}=\sigma(\hat\psi)$")
    ax.scatter(theta + np.random.default_rng(1).normal(0, 0.02, size=theta.size),
               p_last, color="gray", marker="s", s=18, alpha=0.5, label=r"$\hat\pi^*_{last}$")
    ax.axhline(0.5, color="red", lw=0.6, ls=":", label="decision boundary")
    ax.set_xlabel(r"realized outcome $\theta$ (jittered)")
    ax.set_ylabel(r"predicted probability $\hat\pi^*$")
    ax.set_title(f"Predictive validity (N={len(rows)} markets)")
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["NO (θ=0)", "YES (θ=1)"])
    ax.grid(alpha=0.3)
    ax.legend(loc="center right", fontsize=8)
    cb = plt.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label(r"$|\hat\sigma_{st}-\hat\sigma_{inc}|/\hat\sigma_{st}$ (OU-shape mismatch)",
                 fontsize=8)

    # Bar chart of aggregate metrics
    ax = axes[1]
    agg = pv["aggregate"]
    labels = ["uniform (0.5)", "OU σ(ψ̂)", "last price"]
    briers = [agg["pi_uniform"]["brier"], agg["pi_ou"]["brier"], agg["pi_last"]["brier"]]
    accs = [agg["pi_uniform"]["accuracy"], agg["pi_ou"]["accuracy"], agg["pi_last"]["accuracy"]]
    x = np.arange(3)
    w = 0.35
    b1 = ax.bar(x - w/2, briers, w, color="#1976d2", label="Brier")
    ax2 = ax.twinx()
    b2 = ax2.bar(x + w/2, accs, w, color="#f57c00", alpha=0.8, label="Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Brier score (lower=better)", color="#1976d2")
    ax2.set_ylabel("Accuracy (higher=better)", color="#f57c00")
    ax.set_title("Aggregate metrics by predictor")
    ax.grid(alpha=0.3, axis="y")
    for b, v in zip(b1, briers):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.3f}",
                ha="center", fontsize=8, color="#1976d2")
    for b, v in zip(b2, accs):
        ax2.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.2f}",
                 ha="center", fontsize=8, color="#f57c00")

    fig.tight_layout()
    out = FIG / "predictive_validity.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    # ============ Figure 2: best OU fits (4 markets) ============
    # Pick 4 markets with very high R² and varied attractor values
    rs = [(r, r2[i]) for i, r in enumerate(rows)]
    rs.sort(key=lambda x: -x[1])
    # Take top 12 by R² then pick 4 spanning psi range
    top = rs[:20]
    top_sorted_by_psi = sorted(top, key=lambda x: x[0]["psi_hat"])
    pick_idx = [0, len(top_sorted_by_psi) // 3, 2 * len(top_sorted_by_psi) // 3, -1]
    chosen = [top_sorted_by_psi[i][0] for i in pick_idx]

    fig, axes = plt.subplots(len(chosen), 2, figsize=(13, 2.5 * len(chosen)))
    for ax_row, r in zip(axes, chosen):
        tok = r["token_id"]
        fit = fits[tok]
        proc = pd.read_csv(PROC / f"{tok}.csv")
        dt = pd.to_datetime(proc["t_bar"], unit="s")
        ax = ax_row[0]
        ax.plot(dt, proc["z"], lw=0.9, color="#1976d2")
        psi = fit["psi_hat"]
        sigma_stat = fit["sigma_z_stationary"]
        kappa = fit["kappa_hat_per_hour"]
        if kappa > 0:
            stat_std = np.sqrt(sigma_stat ** 2 / (2 * kappa))
            ax.axhspan(psi - stat_std, psi + stat_std, color="red", alpha=0.12)
        ax.axhline(psi, color="red", ls="--", lw=1)
        ax.set_ylabel("z = logit π")
        ax.set_title(f"{r['question'][:60]}", fontsize=9)
        ax.grid(alpha=0.3)
        ax.text(0.02, 0.95,
                f"θ={int(r['theta'])}  p_OU={r['p_ou']:.2f}\n"
                f"τ={1/kappa:.1f}h  R²={fit['log_acf_r2']:.2f}",
                transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray"))
        # ACF
        from fit.method_of_moments import empirical_acf
        max_lag = min(40, len(proc) // 4)
        if max_lag >= 5:
            acf = empirical_acf(proc["z"].to_numpy(), max_lag=max_lag)
            lags = np.arange(max_lag + 1)
            ax = ax_row[1]
            ax.vlines(lags, 0, acf, color="#1976d2", lw=1)
            ax.scatter(lags, acf, s=14, color="#1976d2", edgecolor="black", linewidth=0.3)
            tau = np.linspace(0, max_lag, 200)
            ax.plot(tau, np.exp(-kappa * tau), color="red", lw=1.2)
            ax.axhline(0, color="black", lw=0.5)
            ax.set_xlabel("lag (h)")
            ax.set_ylabel("ACF")
            ax.set_ylim(-0.2, 1.05)
            ax.grid(alpha=0.3)
    fig.suptitle("OU fits — four representative high-R² markets", fontsize=11, y=0.995)
    fig.tight_layout()
    out = FIG / "ou_fits_good_markets.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    # ============ Figure 3: stratification by sigma-mismatch + R² ============
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    tier_q = pv["by_sigma_mismatch_quartile"]
    keys = ["Q1_lowest_mismatch", "Q2", "Q3", "Q4_highest_mismatch"]
    ns = [tier_q[k]["n"] for k in keys]
    briers = [tier_q[k]["pi_ou"]["brier"] for k in keys]
    uniform_briers = [tier_q[k]["pi_uniform"]["brier"] for k in keys]
    x = np.arange(4)
    w = 0.35
    ax.bar(x - w/2, briers, w, color="#1976d2", label="OU π̂")
    ax.bar(x + w/2, uniform_briers, w, color="#9e9e9e", label="uniform 0.5")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{i+1}\n(n={n})" for i, n in enumerate(ns)])
    ax.set_xlabel("σ-mismatch quartile (Q1=lowest, Q4=highest)")
    ax.set_ylabel("Brier score")
    ax.set_title("Stratification by σ-mismatch quartile")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    for b, v in zip(ax.patches[:4], briers):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005,
                f"{v:.3f}", ha="center", fontsize=8)

    ax = axes[1]
    fq = pv["by_fit_quality"]
    keys = ["good_fit_r2>=0.94", "poor_fit_r2<0.94"]
    labels = ["log-ACF R² ≥ 0.94\n(good OU shape)", "log-ACF R² < 0.94\n(poor OU shape)"]
    ns = [fq[k]["n"] for k in keys]
    briers = [fq[k]["pi_ou"]["brier"] for k in keys]
    uni = [fq[k]["pi_uniform"]["brier"] for k in keys]
    x = np.arange(2)
    ax.bar(x - w/2, briers, w, color="#388e3c", label="OU π̂")
    ax.bar(x + w/2, uni, w, color="#9e9e9e", label="uniform 0.5")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lbl}\n(n={n})" for lbl, n in zip(labels, ns)])
    ax.set_ylabel("Brier score")
    ax.set_title("Stratification by log-ACF fit quality")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    for b, v in zip(ax.patches[:2], briers):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005,
                f"{v:.3f}", ha="center", fontsize=8)

    fig.tight_layout()
    out = FIG / "sigma_mismatch_stratification.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
