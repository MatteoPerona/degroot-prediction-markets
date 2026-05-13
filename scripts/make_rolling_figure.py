"""Figure for the rolling-window OU experiment on real Polymarket data.

Three panels:
  (a) Aggregate κ̂(u) by u-bin across the corpus, with IQR ribbon.
      The DeGroot/topology-growth prediction is grow-then-plateau; we
      overlay a stylized expectation curve as a reference.
  (b) Distribution of per-market OLS slopes of log(κ̂) on u_norm. A null
      result would centre on zero; the DeGroot prediction is a systematic
      positive shift.
  (c) Per-market κ̂(u) trajectories (faded grey), with the cross-market
      median trajectory overlaid in bold.

Writes:
  data/processed/figures/rolling_kappa.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = PROC / "figures"


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    with (PROC / "rolling_ou.json").open() as f:
        roll = json.load(f)
    with (PROC / "rolling_summary.json").open() as f:
        summary = json.load(f)

    # ------------- panel (a): binned κ̂(u) across markets -------------
    bins = summary["bin_stats"]
    centers = np.array([0.125, 0.375, 0.625, 0.875])
    medians = np.array([b["median_kappa"] for b in bins])
    p25 = np.array([b["p25_kappa"] for b in bins])
    p75 = np.array([b["p75_kappa"] for b in bins])
    ns = [b["n_windows"] for b in bins]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    ax.fill_between(centers, p25, p75, color="#1976d2", alpha=0.22,
                    label="IQR across windows")
    ax.plot(centers, medians, "o-", color="#0d47a1", lw=2, ms=8,
            label="median κ̂ per u-bin")
    # Stylized DeGroot prediction: grow then plateau, scaled to corpus median
    median_overall = float(np.median(medians))
    pred_u = np.linspace(0, 1, 50)
    pred_curve = median_overall * (
        0.55 + 0.95 * (1.0 - np.exp(-3.0 * pred_u))
    )
    ax.plot(pred_u, pred_curve, "--", color="#c62828", lw=1.7, alpha=0.85,
            label="DeGroot prediction (illustrative)")
    for c, m, n in zip(centers, medians, ns):
        ax.text(c, m + 0.012, f"n={n}", ha="center", fontsize=8, color="#0d47a1")
    ax.set_xlabel("normalized market lifetime $u = (t - t_\\mathrm{first\\,window})/(T - W)$")
    ax.set_ylabel(r"$\hat\kappa$ (per hour)")
    ax.set_title("(a) Binned $\\hat\\kappa(u)$ across the corpus")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(p75) * 1.25)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # ------------- panel (b): distribution of per-market slopes -------------
    slopes = np.array([m["slope_log_kappa_vs_u"] for m in summary["per_market"]])
    ax = axes[1]
    ax.hist(slopes, bins=14, color="#1976d2", edgecolor="black", alpha=0.85)
    med = summary["slope_log_kappa_vs_u"]["median"]
    lo = summary["slope_log_kappa_vs_u"]["p2.5"]
    hi = summary["slope_log_kappa_vs_u"]["p97.5"]
    share_pos = summary["slope_log_kappa_vs_u"]["share_positive"]
    ax.axvline(0.0, color="black", lw=0.8, ls=":", label="null (no trend)")
    ax.axvline(med, color="#c62828", lw=1.8, label=f"median = {med:+.2f}")
    ax.axvspan(lo, hi, color="#c62828", alpha=0.15, label=f"95% CI on median [{lo:+.2f}, {hi:+.2f}]")
    ax.set_xlabel(r"OLS slope of $\log\hat\kappa$ on $u_\mathrm{norm}$ (per market)")
    ax.set_ylabel("count of markets")
    ax.set_title("(b) Per-market trends")
    ax.text(0.02, 0.96, f"share > 0: {share_pos:.0%}  ({int(round(share_pos*len(slopes)))}/{len(slopes)})",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85))
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # ------------- panel (c): per-market trajectories -------------
    ax = axes[2]
    median_per_bin: list[list[float]] = [[] for _ in centers]
    median_per_bin_us: list[list[float]] = [[] for _ in centers]
    for m in roll["per_market"]:
        kept = [w for w in m["windows"] if w["kept"]]
        if len(kept) < 3:
            continue
        u = np.array([w["u_norm"] for w in kept])
        k = np.array([w["kappa_hat_per_hour"] for w in kept])
        ax.plot(u, k, color="#666", alpha=0.30, lw=0.8)
        for w in kept:
            u_v = w["u_norm"]
            idx = min(int(u_v // 0.25), 3)
            median_per_bin[idx].append(w["kappa_hat_per_hour"])
            median_per_bin_us[idx].append(u_v)
    bin_medians = [float(np.median(b)) if b else float("nan") for b in median_per_bin]
    ax.plot(centers, bin_medians, "o-", color="#c62828", lw=2.4, ms=9,
            label="cross-market median (per u-bin)")
    ax.set_xlabel("normalized market lifetime $u$")
    ax.set_ylabel(r"$\hat\kappa$ (per hour)")
    ax.set_title("(c) Per-market trajectories")
    ax.set_xlim(-0.02, 1.02)
    # Clip y axis to a reasonable range — long-tailed κ̂ on a few markets
    upper = np.quantile(
        [w["kappa_hat_per_hour"] for m in roll["per_market"]
         for w in m["windows"] if w["kept"]],
        0.97,
    )
    ax.set_ylim(0, float(upper) * 1.1)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"Rolling-window OU fits on Polymarket: $\\hat\\kappa(t)$ does not "
        f"show the predicted Discovery$\\to$Consensus growth\n"
        f"$N$={summary['n_markets']} markets, $W$={summary['params']['W_bars']}h "
        f"window, stride $S$={summary['params']['stride_bars']}h",
        fontsize=10.5,
    )
    fig.tight_layout()
    out = FIG / "rolling_kappa.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
