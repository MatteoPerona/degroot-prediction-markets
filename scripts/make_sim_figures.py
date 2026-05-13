"""Regenerate the simulator-validation figures (Sec 5.1, 5.2, 5.3).

Inputs:
  data/processed/bridge_results.json       (produced by fit.sim_to_ou_bridge.sweep)
  data/processed/frozen_topology_sweep.pkl (produced by scripts/run_topology_sweep.py)
  data/processed/rank_sweep.pkl            (produced by scripts/run_rank_sweep.py)

Outputs:
  data/processed/figures/bridge_sim_to_ou.png
  data/processed/figures/topology_sweep.png
  data/processed/figures/rank_phi_sweep.png
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = PROC / "figures"


def slope_through_origin(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.dot(x, y) / np.dot(x, x))


def bridge_figure() -> None:
    with (PROC / "bridge_results.json").open() as f:
        rows = json.load(f)
    bp = np.array([r["beta_price"] for r in rows])
    k_pred = np.array([r["kappa_predicted_per_step"] for r in rows])
    k_hat = np.array([r["kappa_hat_per_step"] for r in rows])
    psi_pred = np.array([r["psi_predicted"] for r in rows])
    psi_hat = np.array([r["psi_hat"] for r in rows])

    slope_k = slope_through_origin(k_pred, k_hat)
    corr_k = float(np.corrcoef(k_pred, k_hat)[0, 1])
    corr_psi = float(np.corrcoef(psi_pred, psi_hat)[0, 1])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # (a) kappa scatter
    ax = axes[0]
    sc = ax.scatter(k_pred, k_hat, c=bp, cmap="viridis", s=55,
                    edgecolor="black", linewidth=0.4)
    lim = max(k_pred.max(), k_hat.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="y = x")
    xs = np.linspace(0, lim, 50)
    ax.plot(xs, slope_k * xs, color="red", lw=1.0,
            label=f"slope={slope_k:.2f}")
    ax.set_xlabel(r"predicted $\kappa_{\rm pred}=(1-\rho(D))/N$")
    ax.set_ylabel(r"fitted $\hat\kappa_{\rm eff}$")
    ax.set_title(f"(a) mean-reversion rate (corr={corr_k:.3f})")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.colorbar(sc, ax=ax, label=r"$\beta_{\rm price}$", shrink=0.85)

    # (b) psi scatter
    ax = axes[1]
    sc = ax.scatter(psi_pred, psi_hat, c=bp, cmap="viridis", s=55,
                    edgecolor="black", linewidth=0.4)
    lo, hi = min(psi_pred.min(), psi_hat.min()), max(psi_pred.max(), psi_hat.max())
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8, label="y = x")
    ax.set_xlabel(r"predicted $\bar\psi_{\rm pred}={\rm mean}_i\,w_i\cdot\bar s$")
    ax.set_ylabel(r"fitted $\hat{\bar\psi}$")
    ax.set_title(f"(b) source-weighted attractor (corr={corr_psi:.4f})")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.colorbar(sc, ax=ax, label=r"$\beta_{\rm price}$", shrink=0.85)

    # (c) kappa vs beta_price (mean ± SD across seeds)
    ax = axes[2]
    betas_unique = sorted(set(bp.tolist()))
    means, sds, kpreds = [], [], []
    for b in betas_unique:
        mask = bp == b
        means.append(k_hat[mask].mean())
        sds.append(k_hat[mask].std(ddof=1))
        kpreds.append(k_pred[mask].mean())
    ax.errorbar(betas_unique, means, yerr=sds, color="red", marker="o",
                capsize=3, lw=1.5, label=r"$\hat\kappa$ (mean ± SD)")
    ax.plot(betas_unique, kpreds, color="blue", marker="s", lw=1.5,
            label=r"$\kappa_{\rm pred}$")
    ax.set_xlabel(r"$\beta_{\rm price}$")
    ax.set_ylabel(r"$\kappa_{\rm eff}$")
    ax.set_title(r"(c) $\hat\kappa$ vs $\beta_{\rm price}$")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("Bridge: spectral prediction matches OU fit (β_price sweep, 5 seeds)",
                 fontsize=11)
    fig.tight_layout()
    out = FIG / "bridge_sim_to_ou.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}  "
          f"(slope={slope_k:.3f}, corr_κ={corr_k:.3f}, corr_ψ={corr_psi:.4f})")


def topology_figure() -> None:
    with (PROC / "frozen_topology_sweep.pkl").open("rb") as f:
        rows = pickle.load(f)
    A = np.array([r["n_active"] for r in rows])
    rho = np.array([r["rho_D"] for r in rows])
    k_hat = np.array([r["kappa_hat"] for r in rows])
    N = int(rows[0]["n_traders"])

    k_naive = (1.0 - rho) / N
    k_corrected = (1.0 - rho) * A / (N ** 2)
    corr = float(np.corrcoef(k_corrected, k_hat)[0, 1])
    slope = slope_through_origin(k_corrected, k_hat)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # (a) rho(D) vs |A|
    ax = axes[0]
    A_unique = sorted(set(A.tolist()))
    rho_means = [rho[A == a].mean() for a in A_unique]
    ax.plot(A_unique, rho_means, "o-", color="purple", lw=1.5)
    ax.set_xlabel(r"$|A|$")
    ax.set_ylabel(r"$\rho(D)$")
    ax.set_title(f"(a) spectral radius vs active-set size "
                 f"({rho_means[0]:.2f} → {rho_means[-1]:.2f})")
    ax.grid(alpha=0.3)

    # (b) kappa_hat, corrected pred, naive pred
    ax = axes[1]
    k_hat_means = [k_hat[A == a].mean() for a in A_unique]
    k_hat_sds = [k_hat[A == a].std(ddof=1) for a in A_unique]
    k_corr_means = [k_corrected[A == a].mean() for a in A_unique]
    k_naive_means = [k_naive[A == a].mean() for a in A_unique]
    ax.errorbar(A_unique, k_hat_means, yerr=k_hat_sds, color="red",
                marker="s", capsize=3, lw=1.5, label=r"$\hat\kappa$ (fit)")
    ax.plot(A_unique, k_corr_means, color="green", marker="^", lw=1.5,
            label=r"$(1-\rho)\,|A|/N^2$ (corrected)")
    ax.plot(A_unique, k_naive_means, color="blue", marker="o", ls="--",
            lw=1.0, alpha=0.7, label=r"$(1-\rho)/N$ (naive)")
    ax.set_xlabel(r"$|A|$")
    ax.set_ylabel(r"$\kappa$")
    ax.set_title(r"(b) fitted $\hat\kappa$ and predictions")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)

    # (c) scatter of kappa_hat vs corrected prediction
    ax = axes[2]
    ax.scatter(k_corrected, k_hat, s=55, color="red", alpha=0.7,
               edgecolor="black", linewidth=0.4)
    lim = max(k_corrected.max(), k_hat.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="y = x")
    xs = np.linspace(0, lim, 50)
    ax.plot(xs, slope * xs, color="red", lw=1.0,
            label=f"slope={slope:.2f}")
    ax.set_xlabel(r"$\kappa_{\rm pred}=(1-\rho)\,|A|/N^2$")
    ax.set_ylabel(r"$\hat\kappa$")
    ax.set_title(rf"(c) $\hat\kappa$ vs corrected (corr={corr:.3f})")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("Frozen-topology sweep: how OU fit responds to |A| "
                 "(N=30, 5 seeds)", fontsize=11)
    fig.tight_layout()
    out = FIG / "topology_sweep.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}  (corr={corr:.3f}, slope={slope:.3f})")


def rank_phi_figure() -> None:
    with (PROC / "rank_sweep.pkl").open("rb") as f:
        rows = pickle.load(f)
    rank = np.array([r["rank_target"] for r in rows])
    psi_pred = np.array([r["psi_pred"] for r in rows])
    psi_hat = np.array([r["psi_hat"] for r in rows])
    corr = float(np.corrcoef(psi_pred, psi_hat)[0, 1])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # (a) psi scatter, color by rank
    ax = axes[0]
    sc = ax.scatter(psi_pred, psi_hat, c=rank, cmap="plasma", s=60,
                    edgecolor="black", linewidth=0.4)
    lo, hi = min(psi_pred.min(), psi_hat.min()), max(psi_pred.max(), psi_hat.max())
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8, label="y = x")
    ax.set_xlabel(r"predicted $\bar\psi_{\rm pred}$")
    ax.set_ylabel(r"fitted $\hat{\bar\psi}$")
    ax.set_title(f"(a) attractor recovery (corr={corr:.4f})")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    cb = plt.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label(r"$\mathrm{rank}(\Phi)$")
    cb.set_ticks(sorted(set(rank.tolist())))

    # (b) distribution of psi_hat by rank
    ax = axes[1]
    rank_unique = sorted(set(rank.tolist()))
    jitter_rng = np.random.default_rng(0)
    for rk in rank_unique:
        m = rank == rk
        x = rk + jitter_rng.uniform(-0.08, 0.08, size=int(m.sum()))
        ax.scatter(x, psi_hat[m], s=35, color="gray", alpha=0.6)
        mean = psi_hat[m].mean()
        sd = psi_hat[m].std()
        ax.errorbar([rk], [mean], yerr=[sd], color="black", marker="o",
                    capsize=5, lw=1.8, markersize=8)
    ax.set_xticks(rank_unique)
    ax.set_xlabel(r"$\mathrm{rank}(\Phi)$")
    ax.set_ylabel(r"$\hat{\bar\psi}$")
    ax.set_title(r"(b) per-rank $\hat\psi$ distribution (12 seeds)")
    ax.grid(alpha=0.3)

    # (c) std(psi_hat) vs rank, with 1/sqrt(r) reference
    ax = axes[2]
    sds = [psi_hat[rank == rk].std() for rk in rank_unique]
    ax.plot(rank_unique, sds, "o-", color="red", lw=1.5, markersize=9,
            label=r"empirical std")
    # CLT 1/sqrt(r) reference, scaled to rank=1 std
    ref = [sds[0] / np.sqrt(rk) for rk in rank_unique]
    ax.plot(rank_unique, ref, "s--", color="blue", lw=1.0, alpha=0.7,
            label=r"$1/\sqrt{r}$ reference")
    reduction = (sds[0] - sds[-1]) / sds[0] * 100
    ax.set_xticks(rank_unique)
    ax.set_xlabel(r"$\mathrm{rank}(\Phi)$")
    ax.set_ylabel(r"$\mathrm{std}(\hat{\bar\psi})$ across seeds")
    ax.set_title(f"(c) cross-seed dispersion ({reduction:.0f}% reduction)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle(r"Information-matrix rank sweep: $\mathrm{rank}(\Phi)\in\{1,2,3,5\}$, "
                 "12 seeds each", fontsize=11)
    fig.tight_layout()
    out = FIG / "rank_phi_sweep.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}  (corr={corr:.4f}, reduction={reduction:.0f}%)")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    bridge_figure()
    topology_figure()
    rank_phi_figure()


if __name__ == "__main__":
    main()
